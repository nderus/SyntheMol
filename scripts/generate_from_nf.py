#!/usr/bin/env python
"""Generate molecules using trained conditional normalizing flow."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from synthemol.flows.molecule_generator import FlowMoleculeGenerator


def plot_pareto_front(
    molecules: list,
    save_path: Path,
    title: str = "Generated Molecules - Property Space",
):
    """Plot generated molecules in activity-QED space."""
    activities = []
    qeds = []
    target_activities = []
    target_qeds = []

    for mol in molecules:
        activities.append(mol.predicted_activity if mol.predicted_activity else 0.0)
        qeds.append(mol.predicted_qed if mol.predicted_qed else 0.0)
        target_activities.append(mol.target_activity)
        target_qeds.append(mol.target_qed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot actual properties
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        activities, qeds,
        c=np.array(activities) + np.array(qeds),
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    ax1.set_xlabel("Activity Score", fontsize=12)
    ax1.set_ylabel("QED Score", fontsize=12)
    ax1.set_title("Generated Molecules - Actual Properties", fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="Combined Score")

    # Plot target vs actual
    ax2 = axes[1]
    ax2.scatter(
        target_activities, activities,
        alpha=0.3, s=10, label="Activity"
    )
    ax2.scatter(
        target_qeds, qeds,
        alpha=0.3, s=10, label="QED"
    )
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect match")
    ax2.set_xlabel("Target Value", fontsize=12)
    ax2.set_ylabel("Actual Value", fontsize=12)
    ax2.set_title("Target vs Actual Properties", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def compare_with_baseline(
    nf_molecules: list,
    baseline_path: Path,
    save_path: Path,
):
    """Compare NF-generated molecules with SyntheMol baseline."""
    # Load baseline molecules
    baseline_df = pd.read_csv(baseline_path)
    print(f"Loaded {len(baseline_df)} baseline molecules")

    # Extract scores
    baseline_activities = baseline_df["activity_score"].values
    baseline_qeds = baseline_df["qed_score"].values

    # Get NF molecule scores
    nf_activities = np.array([mol.predicted_activity for mol in nf_molecules if mol.predicted_activity is not None])
    nf_qeds = np.array([mol.predicted_qed for mol in nf_molecules if mol.predicted_qed is not None])

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Activity distribution
    ax1 = axes[0]
    ax1.hist(baseline_activities, bins=30, alpha=0.5, label="SyntheMol (all)", density=True)
    ax1.hist(nf_activities, bins=30, alpha=0.5, label="NF-Retrieved", density=True)
    ax1.set_xlabel("Activity Score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Activity Score Distribution", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QED distribution
    ax2 = axes[1]
    ax2.hist(baseline_qeds, bins=30, alpha=0.5, label="SyntheMol (all)", density=True)
    ax2.hist(nf_qeds, bins=30, alpha=0.5, label="NF-Retrieved", density=True)
    ax2.set_xlabel("QED Score", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("QED Score Distribution", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Pareto front comparison
    ax3 = axes[2]
    ax3.scatter(
        baseline_activities, baseline_qeds,
        alpha=0.2, s=5, label="SyntheMol (all)", c="blue"
    )
    ax3.scatter(
        nf_activities, nf_qeds,
        alpha=0.7, s=20, label="NF-Retrieved", c="orange", edgecolors="black", linewidths=0.5
    )
    ax3.set_xlabel("Activity Score", fontsize=12)
    ax3.set_ylabel("QED Score", fontsize=12)
    ax3.set_title("Pareto Front Comparison", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")

    # Print statistics
    print("\n=== Comparison Statistics ===")
    print(f"SyntheMol (all): {len(baseline_df)} molecules")
    print(f"  Activity: {baseline_activities.mean():.3f} +/- {baseline_activities.std():.3f}")
    print(f"  QED: {baseline_qeds.mean():.3f} +/- {baseline_qeds.std():.3f}")

    print(f"\nNF-Retrieved: {len(nf_molecules)} molecules")
    print(f"  Activity: {nf_activities.mean():.3f} +/- {nf_activities.std():.3f}")
    print(f"  QED: {nf_qeds.mean():.3f} +/- {nf_qeds.std():.3f}")

    # Analyze how well NF retrieves high-scoring molecules
    high_activity_baseline = baseline_activities[baseline_activities > 0.5]
    high_qed_baseline = baseline_qeds[baseline_qeds > 0.7]
    high_both_baseline = baseline_df[(baseline_df["activity_score"] > 0.5) & (baseline_df["qed_score"] > 0.7)]

    high_activity_nf = nf_activities[nf_activities > 0.5]
    high_qed_nf = nf_qeds[nf_qeds > 0.7]
    high_both_nf = sum(1 for mol in nf_molecules if mol.predicted_activity > 0.5 and mol.predicted_qed > 0.7)

    print("\n=== High-Scoring Molecule Analysis ===")
    print(f"Baseline: {len(high_activity_baseline)} with activity > 0.5 ({100*len(high_activity_baseline)/len(baseline_df):.1f}%)")
    print(f"NF:       {len(high_activity_nf)} with activity > 0.5 ({100*len(high_activity_nf)/len(nf_activities):.1f}%)")
    print(f"Baseline: {len(high_qed_baseline)} with QED > 0.7 ({100*len(high_qed_baseline)/len(baseline_df):.1f}%)")
    print(f"NF:       {len(high_qed_nf)} with QED > 0.7 ({100*len(high_qed_nf)/len(nf_qeds):.1f}%)")
    print(f"Baseline: {len(high_both_baseline)} with both ({100*len(high_both_baseline)/len(baseline_df):.1f}%)")
    print(f"NF:       {high_both_nf} with both ({100*high_both_nf/len(nf_molecules):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate molecules using conditional NF")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("data/weight_sweep/nf_model"),
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/weight_sweep"),
        help="Directory containing molecule data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/nf_generated"),
        help="Output directory",
    )
    parser.add_argument(
        "--target_activity",
        type=float,
        default=0.7,
        help="Target activity score",
    )
    parser.add_argument(
        "--target_qed",
        type=float,
        default=0.7,
        help="Target QED score",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of NF samples per target",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=5,
        help="Neighbors per sample",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Generate Pareto front exploration",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with SyntheMol baseline",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = FlowMoleculeGenerator(
        model_path=args.model_dir / "best_model.pt",
        pca_path=args.model_dir / "pca_model.pkl",
        molecule_fps_path=args.data_dir / "fingerprints.npy",
        molecule_data_path=args.data_dir / "all_results.csv",
        device=args.device,
    )

    if args.pareto:
        # Generate Pareto front exploration
        print("\nGenerating Pareto front exploration...")
        molecules = generator.generate_pareto_front(
            activity_range=(0.3, 0.9),
            qed_range=(0.5, 0.9),
            n_points=15,
            samples_per_point=args.num_samples,
            k_neighbors=args.k_neighbors,
        )
    else:
        # Single target generation
        print(f"\nGenerating molecules for target (activity={args.target_activity}, qed={args.target_qed})...")
        molecules = generator.generate(
            target_activity=args.target_activity,
            target_qed=args.target_qed,
            num_samples=args.num_samples,
            k_neighbors=args.k_neighbors,
        )

    print(f"\nGenerated {len(molecules)} molecules")

    # Evaluate
    stats = generator.evaluate_molecules(molecules)
    print("\n=== Generation Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save results
    results_df = pd.DataFrame([
        {
            "smiles": mol.smiles,
            "target_activity": mol.target_activity,
            "target_qed": mol.target_qed,
            "activity_score": mol.predicted_activity,
            "qed_score": mol.predicted_qed,
            "distance": mol.distance_to_target,
        }
        for mol in molecules
    ])
    results_path = args.output_dir / "generated_molecules.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Plot
    plot_pareto_front(molecules, args.output_dir / "property_space.png")

    # Compare with baseline if requested
    if args.compare:
        baseline_path = args.data_dir / "all_results.csv"
        if baseline_path.exists():
            compare_with_baseline(
                molecules,
                baseline_path,
                args.output_dir / "comparison.png",
            )
        else:
            print(f"Warning: Baseline file not found at {baseline_path}")


if __name__ == "__main__":
    main()
