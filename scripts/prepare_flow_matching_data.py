#!/usr/bin/env python
"""Prepare synthesis route data for flow matching training."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def main():
    output_dir = Path("data/flow_matching")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all synthesis routes from weight sweep logs
    print("Loading synthesis routes from logs...")
    all_routes = []

    for log_path in sorted(Path("data/weight_sweep").glob("activity_*/logs.pkl")):
        print(f"  Loading {log_path}")
        with open(log_path, "rb") as f:
            logs = pickle.load(f)

        for entry in logs:
            bb_ids = entry.get("Building Block IDs")
            if bb_ids and len(bb_ids) >= 2:
                all_routes.append({
                    "bb1_smiles": bb_ids[0],
                    "bb2_smiles": bb_ids[1],
                    "activity": entry["Activity Score"],
                    "qed": entry["QED Score"],
                    "product_smiles": entry["Best Molecule SMILES"],
                })

    print(f"Total routes with 2+ building blocks: {len(all_routes)}")

    # Compute fingerprints for all unique building blocks
    print("\nExtracting unique building blocks...")
    unique_bbs = set()
    for route in all_routes:
        unique_bbs.add(route["bb1_smiles"])
        unique_bbs.add(route["bb2_smiles"])

    unique_bbs = list(unique_bbs)
    print(f"Unique building blocks: {len(unique_bbs)}")

    print("\nComputing building block fingerprints...")
    bb_to_idx = {}
    bb_fingerprints = []
    bb_smiles_list = []

    for i, smiles in enumerate(tqdm(unique_bbs)):
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            bb_to_idx[smiles] = len(bb_fingerprints)
            bb_fingerprints.append(fp)
            bb_smiles_list.append(smiles)

    bb_fingerprints = np.array(bb_fingerprints, dtype=np.float32)
    print(f"Valid building blocks: {len(bb_fingerprints)}")

    # Create training data: (bb1_fp, bb2_fp, activity, qed)
    print("\nCreating training data...")
    valid_routes = []
    bb1_fps = []
    bb2_fps = []
    activities = []
    qeds = []
    product_smiles = []

    for route in tqdm(all_routes):
        bb1_smiles = route["bb1_smiles"]
        bb2_smiles = route["bb2_smiles"]

        if bb1_smiles in bb_to_idx and bb2_smiles in bb_to_idx:
            bb1_fps.append(bb_fingerprints[bb_to_idx[bb1_smiles]])
            bb2_fps.append(bb_fingerprints[bb_to_idx[bb2_smiles]])
            activities.append(route["activity"])
            qeds.append(route["qed"])
            product_smiles.append(route["product_smiles"])
            valid_routes.append({
                "bb1_idx": bb_to_idx[bb1_smiles],
                "bb2_idx": bb_to_idx[bb2_smiles],
                "bb1_smiles": bb1_smiles,
                "bb2_smiles": bb2_smiles,
            })

    bb1_fps = np.array(bb1_fps, dtype=np.float32)
    bb2_fps = np.array(bb2_fps, dtype=np.float32)
    activities = np.array(activities, dtype=np.float32)
    qeds = np.array(qeds, dtype=np.float32)

    print(f"Valid training routes: {len(valid_routes)}")

    # Concatenate bb1 and bb2 fingerprints to form route embedding
    route_embeddings = np.concatenate([bb1_fps, bb2_fps], axis=1)
    properties = np.stack([activities, qeds], axis=1)

    print(f"\nRoute embeddings shape: {route_embeddings.shape}")
    print(f"Properties shape: {properties.shape}")

    # Save everything
    print("\nSaving data...")

    # Training data
    np.save(output_dir / "route_embeddings.npy", route_embeddings)
    np.save(output_dir / "properties.npy", properties)
    np.save(output_dir / "bb1_fps.npy", bb1_fps)
    np.save(output_dir / "bb2_fps.npy", bb2_fps)

    # Building block index for retrieval
    np.save(output_dir / "bb_fingerprints.npy", bb_fingerprints)
    np.save(output_dir / "bb_smiles.npy", np.array(bb_smiles_list, dtype=object))

    # Route metadata
    routes_df = pd.DataFrame({
        "bb1_idx": [r["bb1_idx"] for r in valid_routes],
        "bb2_idx": [r["bb2_idx"] for r in valid_routes],
        "bb1_smiles": [r["bb1_smiles"] for r in valid_routes],
        "bb2_smiles": [r["bb2_smiles"] for r in valid_routes],
        "activity": activities,
        "qed": qeds,
        "product_smiles": product_smiles,
    })
    routes_df.to_csv(output_dir / "routes.csv", index=False)

    # Save bb_to_idx mapping
    with open(output_dir / "bb_to_idx.pkl", "wb") as f:
        pickle.dump(bb_to_idx, f)

    print(f"\nData saved to {output_dir}")
    print(f"  route_embeddings.npy: {route_embeddings.shape}")
    print(f"  properties.npy: {properties.shape}")
    print(f"  bb_fingerprints.npy: {bb_fingerprints.shape}")
    print(f"  routes.csv: {len(routes_df)} routes")

    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Activity: {activities.mean():.3f} ± {activities.std():.3f} [{activities.min():.3f}, {activities.max():.3f}]")
    print(f"QED: {qeds.mean():.3f} ± {qeds.std():.3f} [{qeds.min():.3f}, {qeds.max():.3f}]")

    # Check for high-scoring routes
    high_both = np.sum((activities > 0.5) & (qeds > 0.7))
    print(f"High-scoring routes (act>0.5, qed>0.7): {high_both} ({100*high_both/len(activities):.1f}%)")


if __name__ == "__main__":
    main()
