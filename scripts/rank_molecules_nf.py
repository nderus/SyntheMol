#!/usr/bin/env python
"""Rank molecules using conditional normalizing flow log probability."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthemol.flows.flow_ranker import FlowMoleculeRanker


def plot_ranking_results(
    molecules: list,
    save_path: Path,
):
    """Plot ranked molecules in property space."""
    activities = [m.activity_score for m in molecules]
    qeds = [m.qed_score for m in molecules]
    log_probs = [m.log_prob for m in molecules]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Property space colored by log probability
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        activities, qeds,
        c=log_probs,
        cmap="viridis",
        alpha=0.7,
        s=30,
    )
    ax1.set_xlabel("Activity Score", fontsize=12)
    ax1.set_ylabel("QED Score", fontsize=12)
    ax1.set_title("Flow-Ranked Molecules (colored by log prob)", fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="Log Probability")

    # Property space colored by rank
    ax2 = axes[1]
    ranks = [m.rank for m in molecules]
    scatter2 = ax2.scatter(
        activities, qeds,
        c=ranks,
        cmap="viridis_r",
        alpha=0.7,
        s=30,
    )
    ax2.set_xlabel("Activity Score", fontsize=12)
    ax2.set_ylabel("QED Score", fontsize=12)
    ax2.set_title("Flow-Ranked Molecules (colored by rank)", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="Rank (1=best)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def compare_with_baseline(
    ranked_molecules: list,
    baseline_path: Path,
    save_path: Path,
):
    """Compare flow-ranked molecules with full baseline."""
    # Load baseline
    baseline_df = pd.read_csv(baseline_path)
    baseline_activities = baseline_df["activity_score"].values
    baseline_qeds = baseline_df["qed_score"].values

    # Get ranked molecule properties
    ranked_activities = np.array([m.activity_score for m in ranked_molecules])
    ranked_qeds = np.array([m.qed_score for m in ranked_molecules])

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Activity distribution
    ax1 = axes[0]
    ax1.hist(baseline_activities, bins=30, alpha=0.5, label="All molecules", density=True)
    ax1.hist(ranked_activities, bins=30, alpha=0.5, label="Flow-ranked", density=True)
    ax1.set_xlabel("Activity Score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Activity Score Distribution", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QED distribution
    ax2 = axes[1]
    ax2.hist(baseline_qeds, bins=30, alpha=0.5, label="All molecules", density=True)
    ax2.hist(ranked_qeds, bins=30, alpha=0.5, label="Flow-ranked", density=True)
    ax2.set_xlabel("QED Score", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("QED Score Distribution", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Pareto front
    ax3 = axes[2]
    ax3.scatter(
        baseline_activities, baseline_qeds,
        alpha=0.1, s=5, label="All molecules", c="blue"
    )
    ax3.scatter(
        ranked_activities, ranked_qeds,
        alpha=0.8, s=25, label="Flow-ranked", c="orange", edgecolors="black", linewidths=0.3
    )
    ax3.set_xlabel("Activity Score", fontsize=12)
    ax3.set_ylabel("QED Score", fontsize=12)
    ax3.set_title("Property Space Coverage", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")

    # Statistics
    print("\n=== Comparison Statistics ===")
    print(f"All molecules: {len(baseline_df)}")
    print(f"  Activity: {baseline_activities.mean():.3f} +/- {baseline_activities.std():.3f}")
    print(f"  QED: {baseline_qeds.mean():.3f} +/- {baseline_qeds.std():.3f}")

    print(f"\nFlow-ranked: {len(ranked_molecules)}")
    print(f"  Activity: {ranked_activities.mean():.3f} +/- {ranked_activities.std():.3f}")
    print(f"  QED: {ranked_qeds.mean():.3f} +/- {ranked_qeds.std():.3f}")

    # High-scoring analysis
    high_activity_all = np.sum(baseline_activities > 0.5)
    high_qed_all = np.sum(baseline_qeds > 0.7)
    high_both_all = np.sum((baseline_activities > 0.5) & (baseline_qeds > 0.7))

    high_activity_ranked = np.sum(ranked_activities > 0.5)
    high_qed_ranked = np.sum(ranked_qeds > 0.7)
    high_both_ranked = np.sum((ranked_activities > 0.5) & (ranked_qeds > 0.7))

    print("\n=== High-Scoring Molecule Enrichment ===")
    print(f"Activity > 0.5:")
    print(f"  All: {high_activity_all} ({100*high_activity_all/len(baseline_df):.1f}%)")
    print(f"  Flow-ranked: {high_activity_ranked} ({100*high_activity_ranked/len(ranked_molecules):.1f}%)")
    print(f"  Enrichment: {(high_activity_ranked/len(ranked_molecules)) / (high_activity_all/len(baseline_df)):.2f}x")

    print(f"QED > 0.7:")
    print(f"  All: {high_qed_all} ({100*high_qed_all/len(baseline_df):.1f}%)")
    print(f"  Flow-ranked: {high_qed_ranked} ({100*high_qed_ranked/len(ranked_molecules):.1f}%)")
    print(f"  Enrichment: {(high_qed_ranked/len(ranked_molecules)) / (high_qed_all/len(baseline_df)):.2f}x")

    print(f"Both (activity>0.5 AND qed>0.7):")
    print(f"  All: {high_both_all} ({100*high_both_all/len(baseline_df):.1f}%)")
    print(f"  Flow-ranked: {high_both_ranked} ({100*high_both_ranked/len(ranked_molecules):.1f}%)")
    if high_both_all > 0:
        print(f"  Enrichment: {(high_both_ranked/len(ranked_molecules)) / (high_both_all/len(baseline_df)):.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Rank molecules using conditional NF")
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
        default=Path("data/nf_ranked"),
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
        "--top_k",
        type=int,
        default=100,
        help="Number of top molecules to return",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Rank across Pareto front grid",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with baseline",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ranker
    ranker = FlowMoleculeRanker(
        model_path=args.model_dir / "best_model.pt",
        pca_path=args.model_dir / "pca_model.pkl",
        molecule_fps_path=args.data_dir / "fingerprints.npy",
        molecule_data_path=args.data_dir / "all_results.csv",
        device=args.device,
    )

    if args.pareto:
        # Rank across Pareto front
        print("\nRanking molecules across Pareto front...")
        molecules = ranker.rank_pareto_front(
            activity_range=(0.3, 0.9),
            qed_range=(0.5, 0.9),
            n_points=10,
            top_k_per_point=50,
        )
    else:
        # Single target ranking
        print(f"\nRanking molecules for target (activity={args.target_activity}, qed={args.target_qed})...")
        molecules = ranker.rank_molecules(
            target_activity=args.target_activity,
            target_qed=args.target_qed,
            top_k=args.top_k,
        )

    print(f"\nRanked {len(molecules)} molecules")

    # Print top molecules
    print("\nTop 20 molecules:")
    for mol in molecules[:20]:
        print(f"  Rank {mol.rank:3d}: act={mol.activity_score:.3f}, qed={mol.qed_score:.3f}, log_prob={mol.log_prob:.1f}")

    # Evaluate
    if not args.pareto:
        stats = ranker.evaluate_ranking(molecules, args.target_activity, args.target_qed)
        print("\n=== Ranking Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Save results
    results_df = pd.DataFrame([
        {
            "rank": mol.rank,
            "smiles": mol.smiles,
            "activity_score": mol.activity_score,
            "qed_score": mol.qed_score,
            "log_prob": mol.log_prob,
        }
        for mol in molecules
    ])
    results_path = args.output_dir / "ranked_molecules.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Plot
    plot_ranking_results(molecules, args.output_dir / "ranking_results.png")

    # Compare with baseline
    if args.compare:
        baseline_path = args.data_dir / "all_results.csv"
        if baseline_path.exists():
            compare_with_baseline(
                molecules,
                baseline_path,
                args.output_dir / "comparison.png",
            )


if __name__ == "__main__":
    main()
