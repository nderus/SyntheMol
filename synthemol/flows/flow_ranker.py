"""Flow-based molecule ranking using conditional log probability."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from synthemol.flows.conditional_flow import ConditionalMAF


@dataclass
class RankedMolecule:
    """Container for a ranked molecule with metadata."""

    smiles: str
    activity_score: float
    qed_score: float
    log_prob: float
    rank: int


class FlowMoleculeRanker:
    """Rank molecules by their log probability under the flow given target conditions.

    This approach uses the trained conditional NF as a scoring function:
    - Given target (activity, qed), compute log p(fingerprint | target) for all molecules
    - Higher log prob means the molecule's fingerprint is more likely given the target
    - Return molecules ranked by log probability
    """

    def __init__(
        self,
        model_path: Path,
        pca_path: Path,
        molecule_fps_path: Path,
        molecule_data_path: Path,
        device: str = "cuda",
    ):
        """Initialize the ranker.

        Args:
            model_path: Path to trained NF model checkpoint
            pca_path: Path to fitted PCA model
            molecule_fps_path: Path to molecule fingerprints (weight sweep)
            molecule_data_path: Path to molecule data CSV with SMILES and scores
            device: Device for model inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load NF model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        self.model = ConditionalMAF(
            input_dim=config["input_dim"],
            cond_dim=config["cond_dim"],
            hidden_dims=config["hidden_dims"],
            num_layers=config["num_layers"],
            cond_encoder_dims=config["cond_encoder_dims"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load PCA model
        print(f"Loading PCA model from {pca_path}...")
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)

        # Load molecule data
        print(f"Loading molecule fingerprints from {molecule_fps_path}...")
        fps = np.load(molecule_fps_path)
        print(f"Loaded {len(fps)} molecule fingerprints")

        print(f"Loading molecule data from {molecule_data_path}...")
        self.molecule_df = pd.read_csv(molecule_data_path)
        print(f"Loaded {len(self.molecule_df)} molecules")

        # Apply PCA and convert to tensor
        print("Transforming fingerprints to PCA space...")
        fps_pca = self.pca.transform(fps)
        self.fps_tensor = torch.FloatTensor(fps_pca).to(self.device)

        print("Ranker initialized!")

    @torch.no_grad()
    def rank_molecules(
        self,
        target_activity: float,
        target_qed: float,
        top_k: int = 100,
        batch_size: int = 1000,
    ) -> list[RankedMolecule]:
        """Rank all molecules by log probability given target properties.

        Args:
            target_activity: Target activity score (0-1)
            target_qed: Target QED score (0-1)
            top_k: Number of top molecules to return
            batch_size: Batch size for log prob computation

        Returns:
            List of RankedMolecule objects sorted by log probability
        """
        condition = torch.tensor(
            [[target_activity, target_qed]],
            device=self.device,
            dtype=torch.float32,
        )

        # Compute log probabilities in batches
        all_log_probs = []
        n_molecules = len(self.fps_tensor)

        for i in range(0, n_molecules, batch_size):
            batch_fps = self.fps_tensor[i : i + batch_size]
            batch_cond = condition.expand(len(batch_fps), -1)
            log_probs = self.model.log_prob(batch_fps, batch_cond)
            all_log_probs.append(log_probs.cpu().numpy())

        all_log_probs = np.concatenate(all_log_probs)

        # Get top-k indices
        top_indices = np.argsort(all_log_probs)[::-1][:top_k]

        # Build result list
        molecules = []
        for rank, idx in enumerate(top_indices):
            row = self.molecule_df.iloc[idx]
            molecules.append(
                RankedMolecule(
                    smiles=str(row["smiles"]),
                    activity_score=float(row["activity_score"]),
                    qed_score=float(row["qed_score"]),
                    log_prob=float(all_log_probs[idx]),
                    rank=rank + 1,
                )
            )

        return molecules

    def rank_pareto_front(
        self,
        activity_range: tuple[float, float] = (0.3, 0.9),
        qed_range: tuple[float, float] = (0.5, 0.9),
        n_points: int = 10,
        top_k_per_point: int = 20,
    ) -> list[RankedMolecule]:
        """Rank molecules across a grid of target conditions.

        Args:
            activity_range: (min, max) for activity targets
            qed_range: (min, max) for QED targets
            n_points: Number of points along each axis
            top_k_per_point: Top molecules per target condition

        Returns:
            List of unique molecules ranked across all conditions
        """
        all_molecules = []

        activity_targets = np.linspace(activity_range[0], activity_range[1], n_points)
        qed_targets = np.linspace(qed_range[0], qed_range[1], n_points)

        print(f"Ranking molecules for {n_points}x{n_points} target grid...")

        for act in tqdm(activity_targets):
            for qed in qed_targets:
                molecules = self.rank_molecules(
                    target_activity=float(act),
                    target_qed=float(qed),
                    top_k=top_k_per_point,
                )
                all_molecules.extend(molecules)

        # Global deduplication - keep best rank for each molecule
        best_molecules = {}
        for mol in all_molecules:
            if mol.smiles not in best_molecules or mol.log_prob > best_molecules[mol.smiles].log_prob:
                best_molecules[mol.smiles] = mol

        # Sort by log probability
        unique_molecules = sorted(best_molecules.values(), key=lambda m: -m.log_prob)

        # Re-rank
        for i, mol in enumerate(unique_molecules):
            mol.rank = i + 1

        print(f"Found {len(unique_molecules)} unique molecules")

        return unique_molecules

    def evaluate_ranking(
        self,
        molecules: list[RankedMolecule],
        target_activity: float,
        target_qed: float,
    ) -> dict:
        """Evaluate ranking quality for given target.

        Returns:
            Dictionary with evaluation metrics
        """
        n_mols = len(molecules)
        if n_mols == 0:
            return {}

        activities = [m.activity_score for m in molecules]
        qeds = [m.qed_score for m in molecules]
        log_probs = [m.log_prob for m in molecules]

        # Compute activity/QED errors relative to target
        activity_errors = [abs(a - target_activity) for a in activities]
        qed_errors = [abs(q - target_qed) for q in qeds]

        stats = {
            "num_molecules": n_mols,
            "activity_mean": np.mean(activities),
            "activity_std": np.std(activities),
            "qed_mean": np.mean(qeds),
            "qed_std": np.std(qeds),
            "log_prob_mean": np.mean(log_probs),
            "log_prob_std": np.std(log_probs),
            "activity_error_mean": np.mean(activity_errors),
            "qed_error_mean": np.mean(qed_errors),
        }

        # Count molecules in target region
        in_target = sum(
            1
            for a, q in zip(activities, qeds)
            if abs(a - target_activity) < 0.2 and abs(q - target_qed) < 0.2
        )
        stats["in_target_region"] = in_target
        stats["in_target_pct"] = 100 * in_target / n_mols

        return stats
