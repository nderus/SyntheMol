"""Synthesis route generator using flow matching.

This module generates novel, synthesizable molecules by:
1. Sampling synthesis routes from the trained flow matching model
2. Decoding route embeddings to building block pairs
3. Finding valid building blocks via nearest neighbor retrieval
4. Applying chemical reactions to produce molecules
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from synthemol.flows.flow_matching import ConditionalFlowMatching
from synthemol.reactions import REAL_REACTIONS


@dataclass
class GeneratedMolecule:
    """Container for a generated molecule with synthesis route."""

    smiles: str
    bb1_smiles: str
    bb2_smiles: str
    reaction_id: str
    target_activity: float
    target_qed: float
    predicted_activity: Optional[float] = None
    actual_qed: Optional[float] = None
    is_novel: bool = False
    bb1_distance: float = 0.0
    bb2_distance: float = 0.0


class SynthesisGenerator:
    """Generate synthesizable molecules using flow matching.

    Pipeline:
    1. User specifies target properties (activity, QED)
    2. Flow model generates route embeddings
    3. Split embedding into bb1 and bb2 fingerprints
    4. Find nearest valid building blocks via KNN
    5. Try all compatible reactions
    6. Return valid product molecules with synthesis routes
    """

    def __init__(
        self,
        model_path: Path,
        data_dir: Path,
        bb_scores_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        """Initialize the generator.

        Args:
            model_path: Path to trained flow matching model
            data_dir: Directory with flow matching data (bb_fingerprints, etc.)
            bb_scores_path: Optional path to building block activity scores
            device: Device for model inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fp_dim = 2048  # Morgan fingerprint dimension

        # Load flow matching model
        print(f"Loading flow matching model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        self.model = ConditionalFlowMatching(
            input_dim=config["input_dim"],
            cond_dim=config["cond_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            sigma=config["sigma"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load normalization parameters
        model_dir = model_path.parent
        self.mean = np.load(model_dir / "mean.npy")
        self.std = np.load(model_dir / "std.npy")

        # Load building block data
        print(f"Loading building block data from {data_dir}...")
        self.bb_fingerprints = np.load(data_dir / "bb_fingerprints.npy")
        self.bb_smiles = np.load(data_dir / "bb_smiles.npy", allow_pickle=True)
        print(f"Loaded {len(self.bb_fingerprints)} building blocks")

        # Build KNN index for each half of the route embedding
        print("Building nearest neighbor indices...")
        self.bb_knn = NearestNeighbors(n_neighbors=20, metric="euclidean", n_jobs=-1)
        self.bb_knn.fit(self.bb_fingerprints)

        # Load reactions
        print("Loading reactions...")
        self.reactions = REAL_REACTIONS
        print(f"Loaded {len(self.reactions)} reaction templates")

        # Load training routes for novelty checking
        self.training_smiles = set()
        routes_path = data_dir / "routes.csv"
        if routes_path.exists():
            routes_df = pd.read_csv(routes_path)
            self.training_smiles = set(routes_df["product_smiles"].dropna())
            print(f"Loaded {len(self.training_smiles)} training molecules for novelty check")

        # Load building block scores if available
        self.bb_scores = None
        if bb_scores_path and bb_scores_path.exists():
            bb_scores_df = pd.read_csv(bb_scores_path)
            # Create mapping from SMILES to score
            self.bb_scores = {}
            smiles_col = "SMILES" if "SMILES" in bb_scores_df.columns else "smiles"
            score_col = "chemprop_ensemble_preds"
            if score_col in bb_scores_df.columns:
                for _, row in bb_scores_df.iterrows():
                    self.bb_scores[row[smiles_col]] = row[score_col]
                print(f"Loaded scores for {len(self.bb_scores)} building blocks")

        print("Generator initialized!")

    def _decode_route_embedding(
        self,
        route_embedding: np.ndarray,
        k_neighbors: int = 10,
    ) -> list[tuple[str, str, float, float]]:
        """Decode route embedding to building block pairs.

        Args:
            route_embedding: (4096,) normalized route embedding
            k_neighbors: Number of nearest neighbors to consider

        Returns:
            List of (bb1_smiles, bb2_smiles, bb1_dist, bb2_dist) tuples
        """
        # Denormalize
        route_denorm = route_embedding * self.std.flatten() + self.mean.flatten()

        # Split into bb1 and bb2 fingerprints
        bb1_fp = route_denorm[:self.fp_dim]
        bb2_fp = route_denorm[self.fp_dim:]

        # Find nearest building blocks
        bb1_dists, bb1_indices = self.bb_knn.kneighbors([bb1_fp], n_neighbors=k_neighbors)
        bb2_dists, bb2_indices = self.bb_knn.kneighbors([bb2_fp], n_neighbors=k_neighbors)

        # Generate all combinations of top-k building blocks
        pairs = []
        for i, (bb1_idx, bb1_dist) in enumerate(zip(bb1_indices[0], bb1_dists[0])):
            for j, (bb2_idx, bb2_dist) in enumerate(zip(bb2_indices[0], bb2_dists[0])):
                bb1_smiles = str(self.bb_smiles[bb1_idx])
                bb2_smiles = str(self.bb_smiles[bb2_idx])
                pairs.append((bb1_smiles, bb2_smiles, bb1_dist, bb2_dist))

        return pairs

    def _try_reactions(
        self,
        bb1_smiles: str,
        bb2_smiles: str,
    ) -> list[tuple[str, str]]:
        """Try all reactions with the given building blocks.

        Args:
            bb1_smiles: First building block SMILES
            bb2_smiles: Second building block SMILES

        Returns:
            List of (product_smiles, reaction_id) tuples
        """
        products = []

        for reaction in self.reactions:
            # Try bb1 + bb2 order
            if reaction.num_reactants == 2:
                try:
                    if reaction.has_match([bb1_smiles, bb2_smiles]):
                        product_list = reaction.run_reactants([bb1_smiles, bb2_smiles])
                        for product in product_list:
                            if product and Chem.MolFromSmiles(product):
                                products.append((product, reaction.id))
                except Exception:
                    pass

                # Try bb2 + bb1 order
                try:
                    if reaction.has_match([bb2_smiles, bb1_smiles]):
                        product_list = reaction.run_reactants([bb2_smiles, bb1_smiles])
                        for product in product_list:
                            if product and Chem.MolFromSmiles(product):
                                products.append((product, reaction.id))
                except Exception:
                    pass

        return products

    @torch.no_grad()
    def generate(
        self,
        target_activity: float,
        target_qed: float,
        num_samples: int = 100,
        k_neighbors: int = 5,
        num_steps: int = 50,
        deduplicate: bool = True,
    ) -> list[GeneratedMolecule]:
        """Generate molecules for given target properties.

        Args:
            target_activity: Target activity score (0-1)
            target_qed: Target QED score (0-1)
            num_samples: Number of route embeddings to sample
            k_neighbors: Building block neighbors to consider per embedding
            num_steps: ODE integration steps
            deduplicate: Whether to remove duplicates

        Returns:
            List of GeneratedMolecule objects
        """
        # Sample route embeddings from flow model
        cond = torch.tensor(
            [[target_activity, target_qed]],
            device=self.device,
            dtype=torch.float32,
        )
        route_embeddings = self.model.sample(
            cond,
            num_samples=num_samples,
            num_steps=num_steps,
        ).cpu().numpy()

        # Generate molecules from each route embedding
        molecules = []
        seen_smiles = set()

        for route_emb in route_embeddings:
            # Decode to building block pairs
            bb_pairs = self._decode_route_embedding(route_emb, k_neighbors=k_neighbors)

            for bb1_smiles, bb2_smiles, bb1_dist, bb2_dist in bb_pairs:
                # Try reactions
                products = self._try_reactions(bb1_smiles, bb2_smiles)

                for product_smiles, reaction_id in products:
                    if deduplicate and product_smiles in seen_smiles:
                        continue
                    seen_smiles.add(product_smiles)

                    # Compute actual QED
                    mol = Chem.MolFromSmiles(product_smiles)
                    actual_qed = Descriptors.qed(mol) if mol else None

                    # Check novelty
                    is_novel = product_smiles not in self.training_smiles

                    # Get predicted activity if we have building block scores
                    predicted_activity = None
                    if self.bb_scores:
                        bb1_score = self.bb_scores.get(bb1_smiles, 0)
                        bb2_score = self.bb_scores.get(bb2_smiles, 0)
                        # Simple average as proxy (actual would need chemprop)
                        predicted_activity = (bb1_score + bb2_score) / 2

                    molecules.append(GeneratedMolecule(
                        smiles=product_smiles,
                        bb1_smiles=bb1_smiles,
                        bb2_smiles=bb2_smiles,
                        reaction_id=reaction_id,
                        target_activity=target_activity,
                        target_qed=target_qed,
                        predicted_activity=predicted_activity,
                        actual_qed=actual_qed,
                        is_novel=is_novel,
                        bb1_distance=bb1_dist,
                        bb2_distance=bb2_dist,
                    ))

        return molecules

    def generate_diverse(
        self,
        activity_range: tuple[float, float] = (0.3, 0.9),
        qed_range: tuple[float, float] = (0.5, 0.9),
        n_points: int = 10,
        samples_per_point: int = 50,
        k_neighbors: int = 5,
    ) -> list[GeneratedMolecule]:
        """Generate diverse molecules across property space.

        Args:
            activity_range: (min, max) for activity targets
            qed_range: (min, max) for QED targets
            n_points: Grid points per dimension
            samples_per_point: Samples per target condition
            k_neighbors: Building block neighbors

        Returns:
            List of unique generated molecules
        """
        all_molecules = []

        activity_targets = np.linspace(activity_range[0], activity_range[1], n_points)
        qed_targets = np.linspace(qed_range[0], qed_range[1], n_points)

        print(f"Generating molecules for {n_points}x{n_points} target grid...")

        for act in tqdm(activity_targets):
            for qed in qed_targets:
                molecules = self.generate(
                    target_activity=float(act),
                    target_qed=float(qed),
                    num_samples=samples_per_point,
                    k_neighbors=k_neighbors,
                    deduplicate=True,
                )
                all_molecules.extend(molecules)

        # Global deduplication
        seen = set()
        unique_molecules = []
        for mol in all_molecules:
            if mol.smiles not in seen:
                seen.add(mol.smiles)
                unique_molecules.append(mol)

        print(f"Generated {len(unique_molecules)} unique molecules")
        novel_count = sum(1 for m in unique_molecules if m.is_novel)
        print(f"Novel molecules (not in training): {novel_count} ({100*novel_count/len(unique_molecules):.1f}%)")

        return unique_molecules

    def evaluate_molecules(self, molecules: list[GeneratedMolecule]) -> dict:
        """Compute statistics for generated molecules."""
        if not molecules:
            return {}

        qeds = [m.actual_qed for m in molecules if m.actual_qed is not None]
        novel_count = sum(1 for m in molecules if m.is_novel)

        stats = {
            "num_molecules": len(molecules),
            "num_novel": novel_count,
            "novelty_pct": 100 * novel_count / len(molecules),
        }

        if qeds:
            stats["qed_mean"] = np.mean(qeds)
            stats["qed_std"] = np.std(qeds)
            stats["qed_min"] = np.min(qeds)
            stats["qed_max"] = np.max(qeds)

        # Count high QED molecules
        high_qed = sum(1 for q in qeds if q > 0.7)
        stats["high_qed_count"] = high_qed
        stats["high_qed_pct"] = 100 * high_qed / len(qeds) if qeds else 0

        return stats
