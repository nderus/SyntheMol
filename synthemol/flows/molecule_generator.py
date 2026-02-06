"""Molecule generation pipeline using conditional normalizing flow with retrieval."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from synthemol.flows.conditional_flow import ConditionalMAF


@dataclass
class GeneratedMolecule:
    """Container for a generated molecule with metadata."""

    smiles: str
    target_activity: float
    target_qed: float
    predicted_activity: Optional[float] = None
    predicted_qed: Optional[float] = None
    distance_to_target: float = 0.0
    molecule_idx: int = -1


class FlowMoleculeGenerator:
    """Generate molecules by sampling from conditional NF and retrieving nearest neighbors.

    The pipeline:
    1. User specifies target (activity, qed)
    2. NF samples fingerprints in PCA space conditioned on targets
    3. Find k-nearest neighbors in the weight sweep molecule fingerprint space
    4. Return molecules with their known properties
    """

    def __init__(
        self,
        model_path: Path,
        pca_path: Path,
        molecule_fps_path: Path,
        molecule_data_path: Path,
        device: str = "cuda",
    ):
        """Initialize the generator.

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
        self.molecule_fps = np.load(molecule_fps_path)
        print(f"Loaded {len(self.molecule_fps)} molecule fingerprints")

        print(f"Loading molecule data from {molecule_data_path}...")
        self.molecule_df = pd.read_csv(molecule_data_path)
        print(f"Loaded {len(self.molecule_df)} molecules")

        # Verify alignment
        if len(self.molecule_fps) != len(self.molecule_df):
            print(f"Warning: Fingerprint count ({len(self.molecule_fps)}) != molecule count ({len(self.molecule_df)})")
            # Use valid indices if available
            valid_indices_path = molecule_fps_path.parent / "valid_indices.npy"
            if valid_indices_path.exists():
                valid_indices = np.load(valid_indices_path)
                self.molecule_df = self.molecule_df.iloc[valid_indices].reset_index(drop=True)
                print(f"Filtered to {len(self.molecule_df)} molecules using valid indices")

        # Build nearest neighbor index in PCA space
        print("Building nearest neighbor index...")
        self.molecule_fps_pca = self.pca.transform(self.molecule_fps)
        self.nn_index = NearestNeighbors(n_neighbors=50, metric="euclidean", n_jobs=-1)
        self.nn_index.fit(self.molecule_fps_pca)

        print("Generator initialized!")

    @torch.no_grad()
    def generate(
        self,
        target_activity: float,
        target_qed: float,
        num_samples: int = 100,
        k_neighbors: int = 5,
        temperature: float = 1.0,
        deduplicate: bool = True,
    ) -> list[GeneratedMolecule]:
        """Generate molecules for given target properties.

        Args:
            target_activity: Target activity score (0-1)
            target_qed: Target QED score (0-1)
            num_samples: Number of NF samples to generate
            k_neighbors: Number of nearest neighbors per sample
            temperature: Sampling temperature (higher = more diverse)
            deduplicate: Whether to remove duplicate molecules

        Returns:
            List of GeneratedMolecule objects
        """
        # Sample from NF
        condition = torch.tensor([[target_activity, target_qed]], device=self.device, dtype=torch.float32)
        sampled_fps = self.model.sample(
            condition,
            num_samples=num_samples,
            temperature=temperature,
        ).cpu().numpy()

        # Find nearest neighbors for each sample
        distances, indices = self.nn_index.kneighbors(sampled_fps, n_neighbors=k_neighbors)

        # Collect unique molecules
        molecules = []
        seen_smiles = set()

        for sample_idx in range(num_samples):
            for neighbor_idx in range(k_neighbors):
                mol_idx = indices[sample_idx, neighbor_idx]
                distance = distances[sample_idx, neighbor_idx]

                row = self.molecule_df.iloc[mol_idx]
                smiles = str(row["smiles"])

                if deduplicate and smiles in seen_smiles:
                    continue
                seen_smiles.add(smiles)

                molecules.append(
                    GeneratedMolecule(
                        smiles=smiles,
                        target_activity=target_activity,
                        target_qed=target_qed,
                        predicted_activity=float(row["activity_score"]),
                        predicted_qed=float(row["qed_score"]),
                        distance_to_target=float(distance),
                        molecule_idx=int(mol_idx),
                    )
                )

        # Sort by distance to target
        molecules.sort(key=lambda m: m.distance_to_target)

        return molecules

    def generate_pareto_front(
        self,
        activity_range: tuple[float, float] = (0.3, 0.9),
        qed_range: tuple[float, float] = (0.5, 0.9),
        n_points: int = 20,
        samples_per_point: int = 50,
        k_neighbors: int = 3,
    ) -> list[GeneratedMolecule]:
        """Generate molecules spanning the Pareto frontier.

        Samples from a grid of target conditions to explore the
        activity-QED tradeoff space.

        Args:
            activity_range: (min, max) for activity targets
            qed_range: (min, max) for QED targets
            n_points: Number of points along each axis
            samples_per_point: NF samples per target condition
            k_neighbors: Neighbors per sample

        Returns:
            List of molecules spanning the Pareto frontier
        """
        all_molecules = []

        activity_targets = np.linspace(activity_range[0], activity_range[1], n_points)
        qed_targets = np.linspace(qed_range[0], qed_range[1], n_points)

        print(f"Generating molecules for {n_points}x{n_points} target grid...")

        for act in tqdm(activity_targets):
            for qed in qed_targets:
                molecules = self.generate(
                    target_activity=act,
                    target_qed=qed,
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

        return unique_molecules

    @staticmethod
    def compute_qed(smiles: str) -> Optional[float]:
        """Compute QED score for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.qed(mol)
        return None

    def evaluate_molecules(
        self,
        molecules: list[GeneratedMolecule],
    ) -> dict:
        """Compute statistics for generated molecules.

        Returns:
            Dictionary with evaluation metrics
        """
        n_mols = len(molecules)
        if n_mols == 0:
            return {}

        stats = {
            "num_molecules": n_mols,
            "num_unique": len(set(mol.smiles for mol in molecules)),
            "avg_distance": np.mean([mol.distance_to_target for mol in molecules]),
            "std_distance": np.std([mol.distance_to_target for mol in molecules]),
        }

        # Get actual QED and activity from retrieved molecules
        if any(mol.predicted_activity is not None for mol in molecules):
            pred_activities = [
                mol.predicted_activity
                for mol in molecules
                if mol.predicted_activity is not None
            ]
            pred_qeds = [
                mol.predicted_qed
                for mol in molecules
                if mol.predicted_qed is not None
            ]

            stats["activity_mean"] = np.mean(pred_activities)
            stats["activity_std"] = np.std(pred_activities)
            stats["activity_min"] = np.min(pred_activities)
            stats["activity_max"] = np.max(pred_activities)

            stats["qed_mean"] = np.mean(pred_qeds)
            stats["qed_std"] = np.std(pred_qeds)
            stats["qed_min"] = np.min(pred_qeds)
            stats["qed_max"] = np.max(pred_qeds)

        return stats
