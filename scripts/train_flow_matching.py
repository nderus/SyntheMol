#!/usr/bin/env python
"""Train conditional flow matching model on synthesis routes."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from synthemol.flows.flow_matching import ConditionalFlowMatching


def load_data(
    data_dir: Path,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Load route embeddings and properties."""
    print("Loading data...")
    route_embeddings = np.load(data_dir / "route_embeddings.npy")
    properties = np.load(data_dir / "properties.npy")

    print(f"Route embeddings: {route_embeddings.shape}")
    print(f"Properties: {properties.shape}")

    # Normalize route embeddings (important for flow matching)
    # Use standardization
    mean = route_embeddings.mean(axis=0, keepdims=True)
    std = route_embeddings.std(axis=0, keepdims=True) + 1e-8
    route_embeddings_norm = (route_embeddings - mean) / std

    # Split
    train_routes, val_routes, train_props, val_props = train_test_split(
        route_embeddings_norm, properties,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Train: {len(train_routes)}, Val: {len(val_routes)}")

    return train_routes, val_routes, train_props, val_props, mean, std


def create_dataloaders(
    train_routes: np.ndarray,
    val_routes: np.ndarray,
    train_props: np.ndarray,
    val_props: np.ndarray,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch dataloaders."""
    train_dataset = TensorDataset(
        torch.FloatTensor(train_routes),
        torch.FloatTensor(train_props),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_routes),
        torch.FloatTensor(val_props),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(
    model: ConditionalFlowMatching,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for routes, props in train_loader:
        routes = routes.to(device)
        props = props.to(device)

        optimizer.zero_grad()
        loss, metrics = model(routes, props)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / num_batches}


@torch.no_grad()
def validate(
    model: ConditionalFlowMatching,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for routes, props in val_loader:
        routes = routes.to(device)
        props = props.to(device)

        loss, metrics = model(routes, props)

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / num_batches}


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
):
    """Plot training curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Flow Matching Loss", fontsize=12)
    ax.set_title("Conditional Flow Matching Training", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def test_generation(
    model: ConditionalFlowMatching,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
):
    """Test generation with various conditions."""
    print("\nTesting generation...")
    model.eval()

    test_conditions = [
        (0.3, 0.5, "Low activity, medium QED"),
        (0.7, 0.7, "High activity, high QED"),
        (0.9, 0.9, "Very high both"),
        (0.5, 0.8, "Medium activity, high QED"),
    ]

    for act, qed, desc in test_conditions:
        cond = torch.tensor([[act, qed]], device=device, dtype=torch.float32)
        samples = model.sample(cond, num_samples=5, num_steps=50)

        # Denormalize
        samples_denorm = samples.cpu().numpy() * std + mean

        print(f"\n{desc} (act={act}, qed={qed}):")
        print(f"  Generated {samples.shape[0]} route embeddings")
        print(f"  Embedding norm (denorm): {np.linalg.norm(samples_denorm, axis=1).mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train flow matching model")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/flow_matching"),
        help="Directory containing prepared data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/flow_matching/model"),
        help="Output directory for model",
    )
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of MLP layers")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_routes, val_routes, train_props, val_props, mean, std = load_data(args.data_dir)

    # Save normalization parameters
    np.save(args.output_dir / "mean.npy", mean)
    np.save(args.output_dir / "std.npy", std)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_routes, val_routes, train_props, val_props,
        batch_size=args.batch_size,
    )

    # Create model
    input_dim = train_routes.shape[1]  # 4096
    model = ConditionalFlowMatching(
        input_dim=input_dim,
        cond_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sigma=0.01,  # Small noise for source distribution
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting training...")
    pbar = tqdm(range(1, args.epochs + 1), desc="Training")

    for epoch in pbar:
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])

        scheduler.step()

        pbar.set_postfix({
            "train": f"{train_metrics['loss']:.4f}",
            "val": f"{val_metrics['loss']:.4f}",
        })

        # Early stopping and checkpointing
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "config": {
                    "input_dim": input_dim,
                    "cond_dim": 2,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "sigma": 0.01,
                },
            }, args.output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Log every 20 epochs
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}: Train={train_metrics['loss']:.4f}, Val={val_metrics['loss']:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, args.output_dir / "training_curves.png")

    # Test generation
    test_generation(model, device, mean, std)

    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
