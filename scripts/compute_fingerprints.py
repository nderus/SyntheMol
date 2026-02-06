#!/usr/bin/env python
"""Compute Morgan fingerprints for weight sweep molecules and building blocks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def compute_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius (default 2 = ECFP4)
        n_bits: Number of bits in fingerprint

    Returns:
        Numpy array of shape (n_molecules, n_bits)
    """
    fingerprints = []
    valid_indices = []

    for i, smiles in enumerate(tqdm(smiles_list, desc="Computing fingerprints")):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
            valid_indices.append(i)
        else:
            print(f"Warning: Could not parse SMILES at index {i}: {smiles}")

    return np.array(fingerprints), valid_indices


def main():
    parser = argparse.ArgumentParser(description="Compute Morgan fingerprints")
    parser.add_argument(
        "--weight_sweep_path",
        type=Path,
        default=Path("data/weight_sweep/all_results.csv"),
        help="Path to weight sweep results CSV",
    )
    parser.add_argument(
        "--building_blocks_path",
        type=Path,
        default=Path("data/Models/antibiotic_chemprop/building_blocks_with_qed.csv"),
        help="Path to building blocks CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/weight_sweep"),
        help="Output directory for fingerprint files",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default 2 = ECFP4)",
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=2048,
        help="Number of bits in fingerprint",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process weight sweep molecules
    print("Loading weight sweep data...")
    weight_sweep_df = pd.read_csv(args.weight_sweep_path)
    print(f"Loaded {len(weight_sweep_df)} molecules from weight sweep")

    print("\nComputing fingerprints for weight sweep molecules...")
    weight_sweep_fps, valid_indices = compute_fingerprints(
        weight_sweep_df["smiles"].tolist(),
        radius=args.radius,
        n_bits=args.n_bits,
    )
    print(f"Computed {len(weight_sweep_fps)} valid fingerprints")

    # Save weight sweep fingerprints
    fp_path = args.output_dir / "fingerprints.npy"
    np.save(fp_path, weight_sweep_fps)
    print(f"Saved fingerprints to {fp_path}")

    # Save valid indices for filtering the dataframe if needed
    indices_path = args.output_dir / "valid_indices.npy"
    np.save(indices_path, np.array(valid_indices))
    print(f"Saved valid indices to {indices_path}")

    # Also save the scores corresponding to valid fingerprints
    scores_df = weight_sweep_df.iloc[valid_indices][["activity_score", "qed_score"]].copy()
    scores_path = args.output_dir / "scores.npy"
    np.save(scores_path, scores_df.values)
    print(f"Saved scores to {scores_path}")

    # Load and process building blocks
    print("\nLoading building blocks...")
    bb_df = pd.read_csv(args.building_blocks_path)
    print(f"Loaded {len(bb_df)} building blocks")

    # Find the SMILES column
    smiles_col = None
    for col in ["SMILES", "smiles", "Smiles"]:
        if col in bb_df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        print(f"Available columns: {bb_df.columns.tolist()}")
        raise ValueError("Could not find SMILES column in building blocks file")

    print(f"\nComputing fingerprints for building blocks (using column '{smiles_col}')...")
    bb_fps, bb_valid_indices = compute_fingerprints(
        bb_df[smiles_col].tolist(),
        radius=args.radius,
        n_bits=args.n_bits,
    )
    print(f"Computed {len(bb_fps)} valid fingerprints")

    # Save building block fingerprints
    bb_fp_path = args.output_dir / "building_block_fingerprints.npy"
    np.save(bb_fp_path, bb_fps)
    print(f"Saved building block fingerprints to {bb_fp_path}")

    # Save building block SMILES for retrieval
    bb_smiles = bb_df.iloc[bb_valid_indices][smiles_col].values
    bb_smiles_path = args.output_dir / "building_block_smiles.npy"
    np.save(bb_smiles_path, bb_smiles)
    print(f"Saved building block SMILES to {bb_smiles_path}")

    # Save building block indices
    bb_indices_path = args.output_dir / "building_block_valid_indices.npy"
    np.save(bb_indices_path, np.array(bb_valid_indices))
    print(f"Saved building block indices to {bb_indices_path}")

    print("\nDone! Summary:")
    print(f"  Weight sweep: {len(weight_sweep_fps)} molecules, shape {weight_sweep_fps.shape}")
    print(f"  Building blocks: {len(bb_fps)} molecules, shape {bb_fps.shape}")


if __name__ == "__main__":
    main()
