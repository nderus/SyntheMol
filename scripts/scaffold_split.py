#!/usr/bin/env python
"""Utility for scaffold-based data splitting.

Prevents data leakage by ensuring molecules with the same Murcko scaffold
are in the same split (train or test).
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def get_scaffold(smiles: str, generic: bool = False) -> str:
    """Get Murcko scaffold from SMILES.

    Args:
        smiles: Input SMILES string
        generic: If True, return generic scaffold (no side chains)

    Returns:
        Scaffold SMILES string
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    try:
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
            )
            return Chem.MolToSmiles(scaffold)
        else:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return smiles


def scaffold_split(
    smiles_list: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
    generic_scaffolds: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Split molecules by scaffold to prevent data leakage.

    Ensures that all molecules with the same scaffold are in the same split.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        random_state: Random seed for reproducibility
        generic_scaffolds: Whether to use generic scaffolds

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(random_state)

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(tqdm(smiles_list, desc="Computing scaffolds")):
        scaffold = get_scaffold(smi, generic=generic_scaffolds)
        scaffold_to_indices[scaffold].append(idx)

    # Get scaffolds sorted by size (largest first for more balanced splits)
    scaffolds = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffolds)

    # Calculate target counts
    total = len(smiles_list)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    # Assign scaffolds to splits
    train_indices = []
    val_indices = []
    test_indices = []

    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]

        if len(train_indices) < train_target:
            train_indices.extend(indices)
        elif len(val_indices) < val_target:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    print(f"\nScaffold split statistics:")
    print(f"  Total molecules: {total}")
    print(f"  Unique scaffolds: {len(scaffold_to_indices)}")
    print(f"  Train: {len(train_indices)} ({100*len(train_indices)/total:.1f}%)")
    print(f"  Val: {len(val_indices)} ({100*len(val_indices)/total:.1f}%)")
    print(f"  Test: {len(test_indices)} ({100*len(test_indices)/total:.1f}%)")

    # Check for scaffold overlap (should be zero)
    train_scaffolds = set(get_scaffold(smiles_list[i], generic_scaffolds) for i in train_indices)
    val_scaffolds = set(get_scaffold(smiles_list[i], generic_scaffolds) for i in val_indices)
    test_scaffolds = set(get_scaffold(smiles_list[i], generic_scaffolds) for i in test_indices)

    train_val_overlap = len(train_scaffolds & val_scaffolds)
    train_test_overlap = len(train_scaffolds & test_scaffolds)
    val_test_overlap = len(val_scaffolds & test_scaffolds)

    print(f"\nScaffold overlap check:")
    print(f"  Train-Val overlap: {train_val_overlap} scaffolds")
    print(f"  Train-Test overlap: {train_test_overlap} scaffolds")
    print(f"  Val-Test overlap: {val_test_overlap} scaffolds")

    if train_val_overlap + train_test_overlap + val_test_overlap == 0:
        print("  No scaffold leakage detected!")
    else:
        print("  WARNING: Scaffold leakage detected!")

    return train_indices, val_indices, test_indices


def scaffold_split_binary(
    smiles_list: list[str],
    train_ratio: float = 0.9,
    random_state: int = 42,
    generic_scaffolds: bool = False,
) -> tuple[list[int], list[int]]:
    """Binary scaffold split (train/test only).

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction of data for training
        random_state: Random seed for reproducibility
        generic_scaffolds: Whether to use generic scaffolds

    Returns:
        Tuple of (train_indices, test_indices)
    """
    train_indices, val_indices, test_indices = scaffold_split(
        smiles_list,
        train_ratio=train_ratio,
        val_ratio=0.0,
        random_state=random_state,
        generic_scaffolds=generic_scaffolds,
    )

    # Combine val and test for binary split
    test_indices = val_indices + test_indices

    return train_indices, test_indices


def analyze_scaffolds(smiles_list: list[str], generic: bool = False) -> dict:
    """Analyze scaffold distribution in a dataset.

    Args:
        smiles_list: List of SMILES strings
        generic: Whether to use generic scaffolds

    Returns:
        Dictionary with scaffold statistics
    """
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(tqdm(smiles_list, desc="Analyzing scaffolds")):
        scaffold = get_scaffold(smi, generic=generic)
        scaffold_to_indices[scaffold].append(idx)

    scaffold_sizes = [len(indices) for indices in scaffold_to_indices.values()]

    stats = {
        "num_molecules": len(smiles_list),
        "num_scaffolds": len(scaffold_to_indices),
        "scaffold_ratio": len(scaffold_to_indices) / len(smiles_list),
        "min_scaffold_size": min(scaffold_sizes),
        "max_scaffold_size": max(scaffold_sizes),
        "mean_scaffold_size": np.mean(scaffold_sizes),
        "median_scaffold_size": np.median(scaffold_sizes),
        "singleton_scaffolds": sum(1 for s in scaffold_sizes if s == 1),
    }

    # Top 10 most common scaffolds
    top_scaffolds = sorted(
        scaffold_to_indices.items(),
        key=lambda x: -len(x[1]),
    )[:10]

    stats["top_scaffolds"] = [
        (scaffold, len(indices))
        for scaffold, indices in top_scaffolds
    ]

    return stats


def main():
    parser = argparse.ArgumentParser(description="Scaffold-based data splitting")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file with SMILES column",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for split indices",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="product_smiles",
        help="Name of SMILES column",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction for validation",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--generic",
        action="store_true",
        help="Use generic scaffolds",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze scaffolds, don't split",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    smiles_list = df[args.smiles_col].dropna().tolist()
    print(f"Loaded {len(smiles_list)} molecules")

    if args.analyze_only:
        # Just analyze scaffold distribution
        stats = analyze_scaffolds(smiles_list, generic=args.generic)
        print("\n=== Scaffold Analysis ===")
        for key, value in stats.items():
            if key != "top_scaffolds":
                print(f"  {key}: {value}")
        print("\n  Top scaffolds:")
        for scaffold, count in stats["top_scaffolds"]:
            print(f"    {scaffold[:50]}...: {count} molecules")
    else:
        # Perform split
        train_indices, val_indices, test_indices = scaffold_split(
            smiles_list,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_state=args.random_state,
            generic_scaffolds=args.generic,
        )

        # Save indices
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.output_dir / "train_indices.npy", np.array(train_indices))
            np.save(args.output_dir / "val_indices.npy", np.array(val_indices))
            np.save(args.output_dir / "test_indices.npy", np.array(test_indices))

            # Save scaffolds for each split
            train_scaffolds = set(get_scaffold(smiles_list[i], args.generic) for i in train_indices)
            with open(args.output_dir / "train_scaffolds.pkl", "wb") as f:
                pickle.dump(train_scaffolds, f)

            print(f"\nSaved indices to {args.output_dir}")


if __name__ == "__main__":
    main()
