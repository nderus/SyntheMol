"""Synthesis route generator using flow matching.

This module generates novel, synthesizable molecules by:
1. Sampling synthesis routes from the trained flow matching model
2. Decoding route embeddings to building block pairs
3. Finding valid building blocks via nearest neighbor retrieval
4. Applying chemical reactions to produce molecules

Improvements in v2:
- Support for full building block space (139k BBs)
- Comprehensive novelty checking (Tanimoto, scaffold)
- Reaction conditioning support
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from synthemol.flows.flow_matching import ConditionalFlowMatching
from synthemol.reactions import REAL_REACTIONS


@dataclass
class NoveltyMetrics:
    """Container for novelty assessment results."""

    exact_novel: bool  # Not an exact SMILES match
    canonical_novel: bool  # Not a canonical SMILES match
    max_tanimoto: float  # Maximum Tanimoto similarity to training set
    structurally_novel: bool  # max_tanimoto < threshold
    scaffold_novel: bool  # Scaffold not in training set
    nearest_training_smiles: Optional[str] = None  # Most similar training molecule


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
    novelty_metrics: Optional[NoveltyMetrics] = None


class SynthesisGenerator:
    """Generate synthesizable molecules using flow matching.

    Pipeline:
    1. User specifies target properties (activity, QED)
    2. Flow model generates route embeddings
    3. Split embedding into bb1 and bb2 fingerprints
    4. Find nearest valid building blocks via KNN
    5. Try all compatible reactions
    6. Return valid product molecules with synthesis routes

    V2 improvements:
    - Support for full 139k building block space
    - Comprehensive novelty checking (Tanimoto, scaffold)
    - Reaction conditioning support
    """

    def __init__(
        self,
        model_path: Path,
        data_dir: Path,
        bb_scores_path: Optional[Path] = None,
        device: str = "cuda",
        use_full_bb_space: bool = True,
        novelty_tanimoto_threshold: float = 0.85,
    ):
        """Initialize the generator.

        Args:
            model_path: Path to trained flow matching model
            data_dir: Directory with flow matching data (bb_fingerprints, etc.)
            bb_scores_path: Optional path to building block activity scores
            device: Device for model inference
            use_full_bb_space: Whether to use full 139k building blocks (vs training subset)
            novelty_tanimoto_threshold: Tanimoto threshold for structural novelty
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fp_dim = 2048  # Morgan fingerprint dimension
        self.novelty_tanimoto_threshold = novelty_tanimoto_threshold

        # Load flow matching model
        print(f"Loading flow matching model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        # Support both old and new model configs
        num_reactions = config.get("num_reactions", 124)
        reaction_embed_dim = config.get("reaction_embed_dim", 64)
        self.use_reactions = config.get("use_reactions", False)

        self.model = ConditionalFlowMatching(
            input_dim=config["input_dim"],
            cond_dim=config["cond_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            sigma=config["sigma"],
            num_reactions=num_reactions,
            reaction_embed_dim=reaction_embed_dim,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load normalization parameters
        model_dir = model_path.parent
        self.mean = np.load(model_dir / "mean.npy")
        self.std = np.load(model_dir / "std.npy")

        # Load building block data
        print(f"Loading building block data from {data_dir}...")

        # Check for full BB set
        full_bb_path = data_dir / "full_bb_fingerprints.npy"
        if use_full_bb_space and full_bb_path.exists():
            print("Using full building block space (139k BBs)...")
            self.bb_fingerprints = np.load(full_bb_path)
            self.bb_smiles = np.load(data_dir / "full_bb_smiles.npy", allow_pickle=True)
        else:
            if use_full_bb_space:
                print("Warning: Full BB data not found, using training subset")
            self.bb_fingerprints = np.load(data_dir / "bb_fingerprints.npy")
            self.bb_smiles = np.load(data_dir / "bb_smiles.npy", allow_pickle=True)

        print(f"Loaded {len(self.bb_fingerprints)} building blocks")

        # Build KNN index for building block retrieval
        print("Building nearest neighbor indices...")
        self.bb_knn = NearestNeighbors(n_neighbors=20, metric="euclidean", n_jobs=-1)
        self.bb_knn.fit(self.bb_fingerprints)

        # Load reactions
        print("Loading reactions...")
        self.reactions = REAL_REACTIONS
        print(f"Loaded {len(self.reactions)} reaction templates")

        # Load reaction ID to index mapping if available
        self.reaction_id_to_idx = {}
        reaction_mapping_path = data_dir / "reaction_id_to_idx.pkl"
        if reaction_mapping_path.exists():
            with open(reaction_mapping_path, "rb") as f:
                self.reaction_id_to_idx = pickle.load(f)
            print(f"Loaded {len(self.reaction_id_to_idx)} reaction mappings")

        # Load training data for novelty checking
        self.training_smiles = set()
        self.training_canonical = set()
        self.training_scaffolds = set()
        self.training_fps = []

        routes_path = data_dir / "routes.csv"
        if routes_path.exists():
            routes_df = pd.read_csv(routes_path)
            product_smiles = routes_df["product_smiles"].dropna().tolist()
            self.training_smiles = set(product_smiles)
            print(f"Loaded {len(self.training_smiles)} training molecules for novelty check")

            # Compute canonical SMILES
            print("Computing canonical SMILES for training set...")
            for smi in tqdm(product_smiles, desc="Canonical"):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        canonical = Chem.MolToSmiles(mol)
                        self.training_canonical.add(canonical)
                except Exception:
                    pass

            # Compute scaffolds
            print("Computing scaffolds for training set...")
            for smi in tqdm(product_smiles, desc="Scaffolds"):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                        self.training_scaffolds.add(scaffold)
                except Exception:
                    pass
            print(f"  Unique scaffolds in training: {len(self.training_scaffolds)}")

            # Compute fingerprints for Tanimoto comparison (sample for efficiency)
            print("Computing fingerprints for training set...")
            max_fps = 10000  # Limit for efficiency
            sample_smiles = product_smiles[:max_fps] if len(product_smiles) > max_fps else product_smiles
            for smi in tqdm(sample_smiles, desc="Fingerprints"):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        self.training_fps.append((fp, smi))
                except Exception:
                    pass
            print(f"  Fingerprints computed for {len(self.training_fps)} molecules")

        # Load pre-computed scaffolds if available
        scaffolds_path = data_dir / "train_scaffolds.pkl"
        if scaffolds_path.exists():
            with open(scaffolds_path, "rb") as f:
                self.training_scaffolds = pickle.load(f)
            print(f"Loaded {len(self.training_scaffolds)} training scaffolds")

        # Load building block scores if available
        self.bb_scores = None
        if bb_scores_path and bb_scores_path.exists():
            bb_scores_df = pd.read_csv(bb_scores_path)
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

    def check_novelty(self, smiles: str) -> NoveltyMetrics:
        """Comprehensive novelty check for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            NoveltyMetrics with detailed novelty assessment
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return NoveltyMetrics(
                exact_novel=True,
                canonical_novel=True,
                max_tanimoto=0.0,
                structurally_novel=True,
                scaffold_novel=True,
            )

        # 1. Exact SMILES match
        exact_match = smiles in self.training_smiles

        # 2. Canonical SMILES match (handles tautomers, etc.)
        try:
            canonical = Chem.MolToSmiles(mol)
            canonical_match = canonical in self.training_canonical
        except Exception:
            canonical_match = exact_match

        # 3. Tanimoto similarity to training set
        max_sim = 0.0
        nearest_smiles = None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for train_fp, train_smi in self.training_fps:
                sim = DataStructs.TanimotoSimilarity(fp, train_fp)
                if sim > max_sim:
                    max_sim = sim
                    nearest_smiles = train_smi
        except Exception:
            pass

        # 4. Scaffold novelty
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            scaffold_novel = scaffold not in self.training_scaffolds
        except Exception:
            scaffold_novel = True

        return NoveltyMetrics(
            exact_novel=not exact_match,
            canonical_novel=not canonical_match,
            max_tanimoto=max_sim,
            structurally_novel=max_sim < self.novelty_tanimoto_threshold,
            scaffold_novel=scaffold_novel,
            nearest_training_smiles=nearest_smiles,
        )

    @torch.no_grad()
    def generate(
        self,
        target_activity: float,
        target_qed: float,
        num_samples: int = 100,
        k_neighbors: int = 5,
        num_steps: int = 50,
        deduplicate: bool = True,
        reaction_idx: Optional[int] = None,
        check_full_novelty: bool = False,
    ) -> list[GeneratedMolecule]:
        """Generate molecules for given target properties.

        Args:
            target_activity: Target activity score (0-1)
            target_qed: Target QED score (0-1)
            num_samples: Number of route embeddings to sample
            k_neighbors: Building block neighbors to consider per embedding
            num_steps: ODE integration steps
            deduplicate: Whether to remove duplicates
            reaction_idx: Optional reaction index for conditioning
            check_full_novelty: Whether to compute full novelty metrics (slower)

        Returns:
            List of GeneratedMolecule objects
        """
        # Sample route embeddings from flow model
        cond = torch.tensor(
            [[target_activity, target_qed]],
            device=self.device,
            dtype=torch.float32,
        )

        # Prepare reaction conditioning
        reaction_tensor = None
        if reaction_idx is not None and self.use_reactions:
            reaction_tensor = torch.tensor([reaction_idx], device=self.device)

        route_embeddings = self.model.sample(
            cond,
            num_samples=num_samples,
            num_steps=num_steps,
            reaction_idx=reaction_tensor,
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

                for product_smiles, rxn_id in products:
                    if deduplicate and product_smiles in seen_smiles:
                        continue
                    seen_smiles.add(product_smiles)

                    # Compute actual QED
                    mol = Chem.MolFromSmiles(product_smiles)
                    actual_qed = Descriptors.qed(mol) if mol else None

                    # Check novelty
                    if check_full_novelty:
                        novelty_metrics = self.check_novelty(product_smiles)
                        is_novel = novelty_metrics.exact_novel and novelty_metrics.structurally_novel
                    else:
                        novelty_metrics = None
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
                        reaction_id=rxn_id,
                        target_activity=target_activity,
                        target_qed=target_qed,
                        predicted_activity=predicted_activity,
                        actual_qed=actual_qed,
                        is_novel=is_novel,
                        bb1_distance=bb1_dist,
                        bb2_distance=bb2_dist,
                        novelty_metrics=novelty_metrics,
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

    def evaluate_molecules(
        self,
        molecules: list[GeneratedMolecule],
        compute_full_novelty: bool = False,
    ) -> dict:
        """Compute statistics for generated molecules.

        Args:
            molecules: List of generated molecules
            compute_full_novelty: Whether to compute full novelty metrics

        Returns:
            Dictionary of evaluation statistics
        """
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

        # Compute detailed novelty metrics if available
        mols_with_novelty = [m for m in molecules if m.novelty_metrics is not None]
        if mols_with_novelty:
            # Exact novelty
            exact_novel = sum(1 for m in mols_with_novelty if m.novelty_metrics.exact_novel)
            stats["exact_novelty_pct"] = 100 * exact_novel / len(mols_with_novelty)

            # Structural novelty (Tanimoto-based)
            structural_novel = sum(
                1 for m in mols_with_novelty if m.novelty_metrics.structurally_novel
            )
            stats["structural_novelty_pct"] = 100 * structural_novel / len(mols_with_novelty)

            # Scaffold novelty
            scaffold_novel = sum(
                1 for m in mols_with_novelty if m.novelty_metrics.scaffold_novel
            )
            stats["scaffold_novelty_pct"] = 100 * scaffold_novel / len(mols_with_novelty)

            # Tanimoto statistics
            tanimotos = [m.novelty_metrics.max_tanimoto for m in mols_with_novelty]
            stats["tanimoto_mean"] = np.mean(tanimotos)
            stats["tanimoto_std"] = np.std(tanimotos)
            stats["tanimoto_max"] = np.max(tanimotos)
        elif compute_full_novelty:
            # Compute novelty on the fly
            print("Computing full novelty metrics...")
            exact_novel = 0
            structural_novel = 0
            scaffold_novel = 0
            tanimotos = []

            for mol in tqdm(molecules, desc="Novelty check"):
                novelty = self.check_novelty(mol.smiles)
                if novelty.exact_novel:
                    exact_novel += 1
                if novelty.structurally_novel:
                    structural_novel += 1
                if novelty.scaffold_novel:
                    scaffold_novel += 1
                tanimotos.append(novelty.max_tanimoto)

            stats["exact_novelty_pct"] = 100 * exact_novel / len(molecules)
            stats["structural_novelty_pct"] = 100 * structural_novel / len(molecules)
            stats["scaffold_novelty_pct"] = 100 * scaffold_novel / len(molecules)
            stats["tanimoto_mean"] = np.mean(tanimotos)
            stats["tanimoto_std"] = np.std(tanimotos)
            stats["tanimoto_max"] = np.max(tanimotos)

        # Reaction distribution
        reaction_counts = {}
        for mol in molecules:
            rxn_id = mol.reaction_id
            reaction_counts[rxn_id] = reaction_counts.get(rxn_id, 0) + 1
        stats["unique_reactions"] = len(reaction_counts)
        stats["top_reactions"] = sorted(
            reaction_counts.items(), key=lambda x: -x[1]
        )[:5]

        return stats
