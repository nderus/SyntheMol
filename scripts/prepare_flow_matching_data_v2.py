#!/usr/bin/env python
"""Prepare synthesis route data for flow matching training (v2).

Improvements over v1:
1. Uses full 139,493 building blocks instead of MCTS-limited subset
2. Extracts reaction IDs for reaction-conditioned generation
3. Implements scaffold-based train/val split to prevent data leakage
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from synthemol.reactions import REAL_REACTIONS


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def get_scaffold(smiles: str) -> str:
    """Get Murcko scaffold from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold
    except Exception:
        return smiles


def scaffold_split(
    smiles_list: list[str],
    train_ratio: float = 0.9,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Split molecules by scaffold to prevent data leakage.

    Args:
        smiles_list: List of product SMILES
        train_ratio: Fraction of data to use for training
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, test_indices)
    """
    np.random.seed(random_state)

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        scaffold_to_indices[scaffold].append(idx)

    # Shuffle scaffolds
    scaffolds = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffolds)

    # Assign entire scaffolds to train or test
    train_indices, test_indices = [], []
    train_count = int(len(smiles_list) * train_ratio)

    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]
        if len(train_indices) < train_count:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    print(f"Scaffold split: {len(scaffold_to_indices)} unique scaffolds")
    print(f"  Train: {len(train_indices)} samples ({100*len(train_indices)/len(smiles_list):.1f}%)")
    print(f"  Test: {len(test_indices)} samples ({100*len(test_indices)/len(smiles_list):.1f}%)")

    return train_indices, test_indices


def build_reaction_mapping() -> dict[str, int]:
    """Build mapping from reaction ID to index."""
    # Get all unique reaction IDs from REAL_REACTIONS
    all_reaction_ids = sorted(set(r.id for r in REAL_REACTIONS))
    reaction_id_to_idx = {rid: idx for idx, rid in enumerate(all_reaction_ids)}

    print(f"Reaction mapping: {len(reaction_id_to_idx)} unique reactions")
    return reaction_id_to_idx


def main():
    parser = argparse.ArgumentParser(description="Prepare flow matching data (v2)")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/flow_matching_v2"),
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--bb_path",
        type=Path,
        default=Path("synthemol/resources/real/building_blocks.csv"),
        help="Path to full building blocks CSV",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for scaffold split",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build reaction ID mapping
    reaction_id_to_idx = build_reaction_mapping()

    # Load full building block set
    print(f"\nLoading full building blocks from {args.bb_path}...")
    bb_df = pd.read_csv(args.bb_path)
    print(f"Total building blocks: {len(bb_df)}")

    # Compute fingerprints for all building blocks
    print("\nComputing fingerprints for all building blocks...")
    full_bb_fingerprints = []
    full_bb_smiles = []
    full_bb_to_idx = {}

    for _, row in tqdm(bb_df.iterrows(), total=len(bb_df)):
        smiles = row["smiles"]
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            full_bb_to_idx[smiles] = len(full_bb_fingerprints)
            full_bb_fingerprints.append(fp)
            full_bb_smiles.append(smiles)

    full_bb_fingerprints = np.array(full_bb_fingerprints, dtype=np.float32)
    print(f"Valid building blocks with fingerprints: {len(full_bb_fingerprints)}")

    # Save full BB data
    np.save(args.output_dir / "full_bb_fingerprints.npy", full_bb_fingerprints)
    np.save(args.output_dir / "full_bb_smiles.npy", np.array(full_bb_smiles, dtype=object))
    with open(args.output_dir / "full_bb_to_idx.pkl", "wb") as f:
        pickle.dump(full_bb_to_idx, f)

    # Load synthesis routes from weight sweep logs
    print("\nLoading synthesis routes from logs...")
    all_routes = []

    for log_path in sorted(Path("data/weight_sweep").glob("activity_*/logs.pkl")):
        print(f"  Loading {log_path}")
        with open(log_path, "rb") as f:
            logs = pickle.load(f)

        for entry in logs:
            bb_ids = entry.get("Building Block IDs")
            if bb_ids and len(bb_ids) >= 2:
                # Extract reaction ID from logs
                # The reaction info is stored in Construction logs
                reaction_id = None
                construction_log = entry.get("Construction", {})
                if isinstance(construction_log, dict):
                    reaction_id = construction_log.get("Reaction 1 ID")
                    if reaction_id is None:
                        reaction_id = construction_log.get("Reaction ID")

                # Try to get from alternate locations
                if reaction_id is None:
                    reaction_id = entry.get("Reaction 1 ID")
                if reaction_id is None:
                    reaction_id = entry.get("Reaction ID")

                all_routes.append({
                    "bb1_smiles": bb_ids[0],
                    "bb2_smiles": bb_ids[1],
                    "activity": entry["Activity Score"],
                    "qed": entry["QED Score"],
                    "product_smiles": entry["Best Molecule SMILES"],
                    "reaction_id": str(reaction_id) if reaction_id is not None else None,
                })

    print(f"Total routes with 2+ building blocks: {len(all_routes)}")

    # Get unique building blocks used in training data
    print("\nExtracting unique building blocks from training data...")
    training_bbs = set()
    for route in all_routes:
        training_bbs.add(route["bb1_smiles"])
        training_bbs.add(route["bb2_smiles"])
    print(f"Unique building blocks in training data: {len(training_bbs)}")

    # Create training data with building blocks from training logs
    # (but we'll use full BB set for generation/retrieval)
    print("\nCreating training data...")
    bb_to_idx = {}
    bb_fingerprints = []
    bb_smiles_list = []

    for smiles in tqdm(training_bbs):
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            bb_to_idx[smiles] = len(bb_fingerprints)
            bb_fingerprints.append(fp)
            bb_smiles_list.append(smiles)

    bb_fingerprints = np.array(bb_fingerprints, dtype=np.float32)
    print(f"Valid training building blocks: {len(bb_fingerprints)}")

    # Create training samples
    valid_routes = []
    bb1_fps = []
    bb2_fps = []
    activities = []
    qeds = []
    product_smiles = []
    reaction_indices = []

    # Track routes with/without reaction info
    routes_with_reaction = 0
    routes_without_reaction = 0

    for route in tqdm(all_routes):
        bb1_smiles = route["bb1_smiles"]
        bb2_smiles = route["bb2_smiles"]

        if bb1_smiles in bb_to_idx and bb2_smiles in bb_to_idx:
            bb1_fps.append(bb_fingerprints[bb_to_idx[bb1_smiles]])
            bb2_fps.append(bb_fingerprints[bb_to_idx[bb2_smiles]])
            activities.append(route["activity"])
            qeds.append(route["qed"])
            product_smiles.append(route["product_smiles"])

            # Get reaction index
            reaction_id = route["reaction_id"]
            if reaction_id is not None and reaction_id in reaction_id_to_idx:
                reaction_idx = reaction_id_to_idx[reaction_id]
                routes_with_reaction += 1
            else:
                # Use 0 as default for unknown reactions
                reaction_idx = 0
                routes_without_reaction += 1
            reaction_indices.append(reaction_idx)

            valid_routes.append({
                "bb1_idx": bb_to_idx[bb1_smiles],
                "bb2_idx": bb_to_idx[bb2_smiles],
                "bb1_smiles": bb1_smiles,
                "bb2_smiles": bb2_smiles,
                "reaction_id": reaction_id,
                "reaction_idx": reaction_idx,
            })

    print(f"\nValid training routes: {len(valid_routes)}")
    print(f"  Routes with reaction info: {routes_with_reaction}")
    print(f"  Routes without reaction info: {routes_without_reaction}")

    bb1_fps = np.array(bb1_fps, dtype=np.float32)
    bb2_fps = np.array(bb2_fps, dtype=np.float32)
    activities = np.array(activities, dtype=np.float32)
    qeds = np.array(qeds, dtype=np.float32)
    reaction_indices = np.array(reaction_indices, dtype=np.int64)

    # Concatenate bb1 and bb2 fingerprints to form route embedding
    route_embeddings = np.concatenate([bb1_fps, bb2_fps], axis=1)
    properties = np.stack([activities, qeds], axis=1)

    print(f"\nRoute embeddings shape: {route_embeddings.shape}")
    print(f"Properties shape: {properties.shape}")
    print(f"Reaction indices shape: {reaction_indices.shape}")

    # Scaffold split
    print("\nPerforming scaffold-based split...")
    train_indices, val_indices = scaffold_split(
        product_smiles,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
    )

    # Save split indices
    np.save(args.output_dir / "train_indices.npy", np.array(train_indices))
    np.save(args.output_dir / "val_indices.npy", np.array(val_indices))

    # Save all data
    print("\nSaving data...")

    # Training data
    np.save(args.output_dir / "route_embeddings.npy", route_embeddings)
    np.save(args.output_dir / "properties.npy", properties)
    np.save(args.output_dir / "reaction_indices.npy", reaction_indices)
    np.save(args.output_dir / "bb1_fps.npy", bb1_fps)
    np.save(args.output_dir / "bb2_fps.npy", bb2_fps)

    # Building block index for retrieval (training BBs)
    np.save(args.output_dir / "bb_fingerprints.npy", bb_fingerprints)
    np.save(args.output_dir / "bb_smiles.npy", np.array(bb_smiles_list, dtype=object))

    # Route metadata
    routes_df = pd.DataFrame({
        "bb1_idx": [r["bb1_idx"] for r in valid_routes],
        "bb2_idx": [r["bb2_idx"] for r in valid_routes],
        "bb1_smiles": [r["bb1_smiles"] for r in valid_routes],
        "bb2_smiles": [r["bb2_smiles"] for r in valid_routes],
        "reaction_id": [r["reaction_id"] for r in valid_routes],
        "reaction_idx": [r["reaction_idx"] for r in valid_routes],
        "activity": activities,
        "qed": qeds,
        "product_smiles": product_smiles,
    })
    routes_df.to_csv(args.output_dir / "routes.csv", index=False)

    # Save mappings
    with open(args.output_dir / "bb_to_idx.pkl", "wb") as f:
        pickle.dump(bb_to_idx, f)
    with open(args.output_dir / "reaction_id_to_idx.pkl", "wb") as f:
        pickle.dump(reaction_id_to_idx, f)

    # Save train/val scaffolds for novelty checking
    train_smiles = [product_smiles[i] for i in train_indices]
    train_scaffolds = set(get_scaffold(s) for s in train_smiles)
    with open(args.output_dir / "train_scaffolds.pkl", "wb") as f:
        pickle.dump(train_scaffolds, f)

    print(f"\nData saved to {args.output_dir}")
    print(f"  route_embeddings.npy: {route_embeddings.shape}")
    print(f"  properties.npy: {properties.shape}")
    print(f"  reaction_indices.npy: {reaction_indices.shape}")
    print(f"  bb_fingerprints.npy: {bb_fingerprints.shape}")
    print(f"  full_bb_fingerprints.npy: {full_bb_fingerprints.shape}")
    print(f"  routes.csv: {len(routes_df)} routes")
    print(f"  train_indices.npy: {len(train_indices)} samples")
    print(f"  val_indices.npy: {len(val_indices)} samples")

    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Activity: {activities.mean():.3f} +/- {activities.std():.3f} [{activities.min():.3f}, {activities.max():.3f}]")
    print(f"QED: {qeds.mean():.3f} +/- {qeds.std():.3f} [{qeds.min():.3f}, {qeds.max():.3f}]")

    # Reaction distribution
    print(f"\nReaction distribution:")
    unique_rxns, rxn_counts = np.unique(reaction_indices, return_counts=True)
    for rxn_idx, count in sorted(zip(unique_rxns, rxn_counts), key=lambda x: -x[1])[:10]:
        # Find reaction ID from index
        rxn_id = None
        for rid, idx in reaction_id_to_idx.items():
            if idx == rxn_idx:
                rxn_id = rid
                break
        print(f"  Reaction {rxn_id} (idx={rxn_idx}): {count} routes ({100*count/len(reaction_indices):.1f}%)")

    # Check for high-scoring routes
    high_both = np.sum((activities > 0.5) & (qeds > 0.7))
    print(f"\nHigh-scoring routes (act>0.5, qed>0.7): {high_both} ({100*high_both/len(activities):.1f}%)")


if __name__ == "__main__":
    main()
