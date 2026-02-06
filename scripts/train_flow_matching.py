#!/usr/bin/env python
"""Train conditional flow matching model on synthesis routes.

Supports:
- Scaffold-based train/val split (prevents data leakage)
- Reaction conditioning (optional)
- Standard random split (fallback)
"""

import argparse
import pickle
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
    use_scaffold_split: bool = True,
    use_reactions: bool = True,
) -> tuple:
    """Load route embeddings, properties, and optionally reaction indices.

    Args:
        data_dir: Directory containing prepared data
        test_size: Fraction of data to use for validation (if no scaffold split)
        random_state: Random seed for reproducibility
        use_scaffold_split: Whether to use pre-computed scaffold split
        use_reactions: Whether to load reaction indices

    Returns:
        Tuple of (train_routes, val_routes, train_props, val_props,
                  train_reactions, val_reactions, mean, std, num_reactions)
    """
    print("Loading data...")
    route_embeddings = np.load(data_dir / "route_embeddings.npy")
    properties = np.load(data_dir / "properties.npy")

    print(f"Route embeddings: {route_embeddings.shape}")
    print(f"Properties: {properties.shape}")

    # Load reaction indices if available
    reaction_indices = None
    num_reactions = 124  # Default
    if use_reactions and (data_dir / "reaction_indices.npy").exists():
        reaction_indices = np.load(data_dir / "reaction_indices.npy")
        print(f"Reaction indices: {reaction_indices.shape}")

        # Load reaction mapping to get actual count
        if (data_dir / "reaction_id_to_idx.pkl").exists():
            with open(data_dir / "reaction_id_to_idx.pkl", "rb") as f:
                reaction_id_to_idx = pickle.load(f)
            num_reactions = len(reaction_id_to_idx)
            print(f"Number of unique reactions: {num_reactions}")

    # Normalize route embeddings (important for flow matching)
    mean = route_embeddings.mean(axis=0, keepdims=True)
    std = route_embeddings.std(axis=0, keepdims=True) + 1e-8
    route_embeddings_norm = (route_embeddings - mean) / std

    # Check for pre-computed scaffold split
    train_indices_path = data_dir / "train_indices.npy"
    val_indices_path = data_dir / "val_indices.npy"

    if use_scaffold_split and train_indices_path.exists() and val_indices_path.exists():
        print("Using pre-computed scaffold split...")
        train_indices = np.load(train_indices_path)
        val_indices = np.load(val_indices_path)

        train_routes = route_embeddings_norm[train_indices]
        val_routes = route_embeddings_norm[val_indices]
        train_props = properties[train_indices]
        val_props = properties[val_indices]

        if reaction_indices is not None:
            train_reactions = reaction_indices[train_indices]
            val_reactions = reaction_indices[val_indices]
        else:
            train_reactions = None
            val_reactions = None

        print(f"Train: {len(train_routes)} (scaffold split), Val: {len(val_routes)}")
    else:
        print("Using random train/val split...")
        if reaction_indices is not None:
            (
                train_routes, val_routes,
                train_props, val_props,
                train_reactions, val_reactions,
            ) = train_test_split(
                route_embeddings_norm, properties, reaction_indices,
                test_size=test_size,
                random_state=random_state,
            )
        else:
            train_routes, val_routes, train_props, val_props = train_test_split(
                route_embeddings_norm, properties,
                test_size=test_size,
                random_state=random_state,
            )
            train_reactions = None
            val_reactions = None

        print(f"Train: {len(train_routes)}, Val: {len(val_routes)}")

    return (
        train_routes, val_routes,
        train_props, val_props,
        train_reactions, val_reactions,
        mean, std, num_reactions,
    )


def create_dataloaders(
    train_routes: np.ndarray,
    val_routes: np.ndarray,
    train_props: np.ndarray,
    val_props: np.ndarray,
    train_reactions: np.ndarray | None = None,
    val_reactions: np.ndarray | None = None,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch dataloaders.

    Args:
        train_routes: Training route embeddings
        val_routes: Validation route embeddings
        train_props: Training properties
        val_props: Validation properties
        train_reactions: Optional training reaction indices
        val_reactions: Optional validation reaction indices
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if train_reactions is not None:
        train_dataset = TensorDataset(
            torch.FloatTensor(train_routes),
            torch.FloatTensor(train_props),
            torch.LongTensor(train_reactions),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_routes),
            torch.FloatTensor(val_props),
            torch.LongTensor(val_reactions),
        )
    else:
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
    use_reactions: bool = False,
) -> dict:
    """Train for one epoch.

    Args:
        model: Flow matching model
        train_loader: Training dataloader
        optimizer: Optimizer
        device: Device to use
        use_reactions: Whether to use reaction conditioning

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        if use_reactions and len(batch) == 3:
            routes, props, reactions = batch
            routes = routes.to(device)
            props = props.to(device)
            reactions = reactions.to(device)
        else:
            routes, props = batch[:2]
            routes = routes.to(device)
            props = props.to(device)
            reactions = None

        optimizer.zero_grad()
        loss, metrics = model(routes, props, reaction_idx=reactions)
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
    use_reactions: bool = False,
) -> dict:
    """Validate the model.

    Args:
        model: Flow matching model
        val_loader: Validation dataloader
        device: Device to use
        use_reactions: Whether to use reaction conditioning

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        if use_reactions and len(batch) == 3:
            routes, props, reactions = batch
            routes = routes.to(device)
            props = props.to(device)
            reactions = reactions.to(device)
        else:
            routes, props = batch[:2]
            routes = routes.to(device)
            props = props.to(device)
            reactions = None

        loss, metrics = model(routes, props, reaction_idx=reactions)

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
    parser.add_argument(
        "--use_scaffold_split",
        action="store_true",
        default=True,
        help="Use scaffold-based train/val split (default: True)",
    )
    parser.add_argument(
        "--no_scaffold_split",
        action="store_true",
        help="Disable scaffold split, use random split instead",
    )
    parser.add_argument(
        "--use_reactions",
        action="store_true",
        default=True,
        help="Use reaction conditioning (default: True)",
    )
    parser.add_argument(
        "--no_reactions",
        action="store_true",
        help="Disable reaction conditioning",
    )
    parser.add_argument(
        "--reaction_embed_dim",
        type=int,
        default=64,
        help="Dimension of reaction embedding",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Handle negation flags
    use_scaffold_split = args.use_scaffold_split and not args.no_scaffold_split
    use_reactions = args.use_reactions and not args.no_reactions

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Scaffold split: {use_scaffold_split}")
    print(f"Reaction conditioning: {use_reactions}")

    # Load data
    (
        train_routes, val_routes,
        train_props, val_props,
        train_reactions, val_reactions,
        mean, std, num_reactions,
    ) = load_data(
        args.data_dir,
        use_scaffold_split=use_scaffold_split,
        use_reactions=use_reactions,
    )

    # Check if we actually have reaction data
    has_reactions = train_reactions is not None and use_reactions
    if use_reactions and train_reactions is None:
        print("Warning: Reaction conditioning requested but no reaction data found")
        has_reactions = False

    # Save normalization parameters
    np.save(args.output_dir / "mean.npy", mean)
    np.save(args.output_dir / "std.npy", std)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_routes, val_routes, train_props, val_props,
        train_reactions=train_reactions if has_reactions else None,
        val_reactions=val_reactions if has_reactions else None,
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
        num_reactions=num_reactions if has_reactions else 124,
        reaction_embed_dim=args.reaction_embed_dim if has_reactions else 64,
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
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_reactions=has_reactions,
        )
        val_metrics = validate(
            model, val_loader, device,
            use_reactions=has_reactions,
        )

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
                    "num_reactions": num_reactions if has_reactions else 124,
                    "reaction_embed_dim": args.reaction_embed_dim if has_reactions else 64,
                    "use_reactions": has_reactions,
                    "use_scaffold_split": use_scaffold_split,
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
