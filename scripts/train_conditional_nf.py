#!/usr/bin/env python
"""Train conditional normalizing flow on weight sweep data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from synthemol.flows.conditional_flow import ConditionalMAF


def load_data(
    data_dir: Path,
    pca_dim: int = 128,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Load fingerprints and scores, apply PCA reduction.

    Args:
        data_dir: Directory containing fingerprints.npy and scores.npy
        pca_dim: Target dimensionality after PCA
        test_size: Fraction for validation set
        random_state: Random seed

    Returns:
        train_fps, val_fps, train_scores, val_scores, pca_model
    """
    print("Loading data...")
    fingerprints = np.load(data_dir / "fingerprints.npy")
    scores = np.load(data_dir / "scores.npy")

    print(f"Original fingerprint shape: {fingerprints.shape}")
    print(f"Scores shape: {scores.shape}")

    # Apply PCA for dimensionality reduction
    print(f"Applying PCA to reduce to {pca_dim} dimensions...")
    pca = PCA(n_components=pca_dim, random_state=random_state)
    fingerprints_reduced = pca.fit_transform(fingerprints)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # Split into train/val
    train_fps, val_fps, train_scores, val_scores = train_test_split(
        fingerprints_reduced, scores, test_size=test_size, random_state=random_state
    )

    print(f"Train set: {len(train_fps)} samples")
    print(f"Val set: {len(val_fps)} samples")

    return train_fps, val_fps, train_scores, val_scores, pca


def create_dataloaders(
    train_fps: np.ndarray,
    val_fps: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch dataloaders."""
    train_dataset = TensorDataset(
        torch.FloatTensor(train_fps),
        torch.FloatTensor(train_scores),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_fps),
        torch.FloatTensor(val_scores),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def train_epoch(
    model: ConditionalMAF,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for fps, scores in train_loader:
        fps = fps.to(device)
        scores = scores.to(device)

        optimizer.zero_grad()
        log_prob = model.log_prob(fps, scores)
        loss = -log_prob.mean()  # Negative log likelihood
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: ConditionalMAF,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for fps, scores in val_loader:
        fps = fps.to(device)
        scores = scores.to(device)

        log_prob = model.log_prob(fps, scores)
        loss = -log_prob.mean()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Negative Log Likelihood", fontsize=12)
    ax.set_title("Conditional NF Training Progress", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train conditional normalizing flow")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/weight_sweep"),
        help="Directory containing fingerprints.npy and scores.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/weight_sweep/nf_model"),
        help="Output directory for model and plots",
    )
    parser.add_argument("--pca_dim", type=int, default=128, help="PCA dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of MAF layers")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_fps, val_fps, train_scores, val_scores, pca = load_data(
        args.data_dir,
        pca_dim=args.pca_dim,
    )

    # Save PCA model for later use
    import pickle
    pca_path = args.output_dir / "pca_model.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {pca_path}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_fps, val_fps, train_scores, val_scores,
        batch_size=args.batch_size,
    )

    # Create model
    model = ConditionalMAF(
        input_dim=args.pca_dim,
        cond_dim=2,  # (activity, qed)
        hidden_dims=[args.hidden_dim, args.hidden_dim],
        num_layers=args.num_layers,
        cond_encoder_dims=[64, 64],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": {
                        "input_dim": args.pca_dim,
                        "cond_dim": 2,
                        "hidden_dims": [args.hidden_dim, args.hidden_dim],
                        "num_layers": args.num_layers,
                        "cond_encoder_dims": [64, 64],
                    },
                },
                args.output_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        args.output_dir / "training_curves.png",
    )

    # Test sampling
    print("\nTesting sampling...")
    model.eval()
    with torch.no_grad():
        # Sample for various target conditions
        test_conditions = torch.tensor([
            [0.7, 0.7],  # High activity, high QED
            [0.5, 0.8],  # Medium activity, high QED
            [0.8, 0.5],  # High activity, medium QED
        ]).to(device)

        for i, cond in enumerate(test_conditions):
            samples = model.sample(cond, num_samples=5)
            print(f"Condition {cond.cpu().numpy()}: Generated {samples.shape[0]} samples")

    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
