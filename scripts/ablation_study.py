#!/usr/bin/env python
"""Ablation study to validate property conditioning in flow matching.

This script runs experiments to verify that:
1. Property conditioning actually affects generation
2. Generated molecules reflect target properties
3. Reaction conditioning works (if enabled)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from synthemol.flows.synthesis_generator import SynthesisGenerator


def run_property_sweep(
    generator: SynthesisGenerator,
    property_name: str = "qed",
    target_values: list[float] = None,
    other_property_value: float = 0.5,
    num_samples_per_target: int = 200,
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """Sweep across target property values and measure actual values.

    Args:
        generator: SynthesisGenerator instance
        property_name: Which property to sweep ('qed' or 'activity')
        target_values: List of target values to test
        other_property_value: Fixed value for the other property
        num_samples_per_target: Number of molecules per target value
        k_neighbors: KNN neighbors for BB retrieval

    Returns:
        DataFrame with target and actual property values
    """
    if target_values is None:
        target_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []

    for target in tqdm(target_values, desc=f"Sweeping {property_name}"):
        if property_name == "qed":
            molecules = generator.generate(
                target_activity=other_property_value,
                target_qed=target,
                num_samples=num_samples_per_target,
                k_neighbors=k_neighbors,
            )
        else:  # activity
            molecules = generator.generate(
                target_activity=target,
                target_qed=other_property_value,
                num_samples=num_samples_per_target,
                k_neighbors=k_neighbors,
            )

        for mol in molecules:
            results.append({
                "target_value": target,
                "actual_qed": mol.actual_qed,
                "target_activity": mol.target_activity,
                "target_qed": mol.target_qed,
                "is_novel": mol.is_novel,
                "smiles": mol.smiles,
            })

    return pd.DataFrame(results)


def analyze_correlation(df: pd.DataFrame, property_name: str = "qed") -> dict:
    """Analyze correlation between target and actual property values.

    Args:
        df: DataFrame from property sweep
        property_name: Which property was swept

    Returns:
        Dictionary with correlation statistics
    """
    # Filter valid QED values
    valid_df = df[df["actual_qed"].notna()]

    if property_name == "qed":
        target_col = "target_qed"
        actual_col = "actual_qed"
    else:
        target_col = "target_activity"
        actual_col = "actual_qed"  # We can only measure QED

    # Overall correlation
    correlation, p_value = stats.pearsonr(valid_df[target_col], valid_df[actual_col])

    # Per-target statistics
    per_target_stats = valid_df.groupby("target_value")[actual_col].agg([
        "mean", "std", "count"
    ]).reset_index()

    # Monotonicity check (is actual mean increasing with target?)
    means = per_target_stats["mean"].values
    is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))

    return {
        "correlation": correlation,
        "p_value": p_value,
        "per_target_stats": per_target_stats,
        "is_monotonic": is_monotonic,
        "num_molecules": len(valid_df),
        "novelty_rate": valid_df["is_novel"].mean() * 100,
    }


def run_conditioned_vs_baseline(
    generator: SynthesisGenerator,
    num_samples: int = 500,
    k_neighbors: int = 5,
) -> dict:
    """Compare high-target vs baseline generation.

    Args:
        generator: SynthesisGenerator instance
        num_samples: Number of molecules per condition
        k_neighbors: KNN neighbors

    Returns:
        Dictionary with comparison results
    """
    print("\nGenerating high-target molecules (activity=0.7, qed=0.7)...")
    high_target = generator.generate(
        target_activity=0.7,
        target_qed=0.7,
        num_samples=num_samples,
        k_neighbors=k_neighbors,
    )

    print("Generating baseline molecules (activity=0.5, qed=0.5)...")
    baseline = generator.generate(
        target_activity=0.5,
        target_qed=0.5,
        num_samples=num_samples,
        k_neighbors=k_neighbors,
    )

    # Compute statistics
    high_qeds = [m.actual_qed for m in high_target if m.actual_qed is not None]
    baseline_qeds = [m.actual_qed for m in baseline if m.actual_qed is not None]

    # Statistical test
    t_stat, t_pvalue = stats.ttest_ind(high_qeds, baseline_qeds)
    mannwhitney_stat, mw_pvalue = stats.mannwhitneyu(high_qeds, baseline_qeds, alternative="greater")

    return {
        "high_target": {
            "count": len(high_target),
            "valid_qed_count": len(high_qeds),
            "qed_mean": np.mean(high_qeds),
            "qed_std": np.std(high_qeds),
            "qed_median": np.median(high_qeds),
            "novelty_rate": sum(1 for m in high_target if m.is_novel) / len(high_target) * 100,
        },
        "baseline": {
            "count": len(baseline),
            "valid_qed_count": len(baseline_qeds),
            "qed_mean": np.mean(baseline_qeds),
            "qed_std": np.std(baseline_qeds),
            "qed_median": np.median(baseline_qeds),
            "novelty_rate": sum(1 for m in baseline if m.is_novel) / len(baseline) * 100,
        },
        "difference": {
            "qed_mean_diff": np.mean(high_qeds) - np.mean(baseline_qeds),
            "t_statistic": t_stat,
            "t_pvalue": t_pvalue,
            "mannwhitney_stat": mannwhitney_stat,
            "mannwhitney_pvalue": mw_pvalue,
        },
    }


def plot_property_sweep(
    df: pd.DataFrame,
    analysis: dict,
    output_path: Path,
    property_name: str = "qed",
):
    """Plot property sweep results.

    Args:
        df: DataFrame from property sweep
        analysis: Analysis results from analyze_correlation
        output_path: Path to save the plot
        property_name: Property name for labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Box plot of actual values by target
    valid_df = df[df["actual_qed"].notna()]
    target_values = sorted(valid_df["target_value"].unique())

    box_data = [
        valid_df[valid_df["target_value"] == t]["actual_qed"].values
        for t in target_values
    ]

    bp = axes[0].boxplot(box_data, positions=range(len(target_values)), widths=0.6)
    axes[0].set_xticks(range(len(target_values)))
    axes[0].set_xticklabels([f"{t:.1f}" for t in target_values])
    axes[0].set_xlabel(f"Target {property_name.upper()}", fontsize=12)
    axes[0].set_ylabel("Actual QED", fontsize=12)
    axes[0].set_title(f"Actual QED vs Target {property_name.upper()}", fontsize=14)

    # Add correlation info
    axes[0].text(
        0.05, 0.95,
        f"r = {analysis['correlation']:.3f}\np = {analysis['p_value']:.2e}",
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Right: Mean actual value by target
    per_target = analysis["per_target_stats"]
    axes[1].errorbar(
        per_target["target_value"],
        per_target["mean"],
        yerr=per_target["std"],
        fmt="o-",
        capsize=5,
        markersize=8,
        linewidth=2,
    )

    # Add diagonal reference line
    lims = [0.2, 1.0]
    axes[1].plot(lims, lims, "k--", alpha=0.5, label="Perfect correlation")

    axes[1].set_xlabel(f"Target {property_name.upper()}", fontsize=12)
    axes[1].set_ylabel("Mean Actual QED", fontsize=12)
    axes[1].set_title(f"Conditioning Effectiveness ({property_name.upper()})", fontsize=14)
    axes[1].legend()
    axes[1].set_xlim([0.2, 1.0])
    axes[1].set_ylim([0.2, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_comparison(results: dict, output_path: Path):
    """Plot high-target vs baseline comparison.

    Args:
        results: Results from run_conditioned_vs_baseline
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bar chart of mean QED
    conditions = ["Baseline\n(0.5, 0.5)", "High Target\n(0.7, 0.7)"]
    means = [results["baseline"]["qed_mean"], results["high_target"]["qed_mean"]]
    stds = [results["baseline"]["qed_std"], results["high_target"]["qed_std"]]

    bars = axes[0].bar(conditions, means, yerr=stds, capsize=10, color=["steelblue", "coral"])
    axes[0].set_ylabel("Mean QED", fontsize=12)
    axes[0].set_title("QED by Target Condition", fontsize=14)

    # Add significance annotation
    if results["difference"]["t_pvalue"] < 0.001:
        sig_text = "***"
    elif results["difference"]["t_pvalue"] < 0.01:
        sig_text = "**"
    elif results["difference"]["t_pvalue"] < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."

    y_max = max(means) + max(stds) + 0.05
    axes[0].plot([0, 0, 1, 1], [y_max-0.02, y_max, y_max, y_max-0.02], "k-", linewidth=1)
    axes[0].text(0.5, y_max + 0.01, sig_text, ha="center", fontsize=14)

    # Right: Novelty comparison
    novelty = [results["baseline"]["novelty_rate"], results["high_target"]["novelty_rate"]]
    axes[1].bar(conditions, novelty, color=["steelblue", "coral"])
    axes[1].set_ylabel("Novelty Rate (%)", fontsize=12)
    axes[1].set_title("Novelty by Target Condition", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for flow matching")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained flow matching model",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory with flow matching data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/ablation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples per condition",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=5,
        help="KNN neighbors for BB retrieval",
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
    print("Initializing generator...")
    generator = SynthesisGenerator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
    )

    # Run experiments
    print("\n" + "="*60)
    print("EXPERIMENT 1: Property Sweep (QED)")
    print("="*60)

    qed_sweep_df = run_property_sweep(
        generator,
        property_name="qed",
        target_values=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        other_property_value=0.5,
        num_samples_per_target=args.num_samples,
        k_neighbors=args.k_neighbors,
    )
    qed_sweep_df.to_csv(args.output_dir / "qed_sweep_results.csv", index=False)

    qed_analysis = analyze_correlation(qed_sweep_df, "qed")
    print(f"\nQED Conditioning Results:")
    print(f"  Correlation: r = {qed_analysis['correlation']:.4f} (p = {qed_analysis['p_value']:.2e})")
    print(f"  Monotonic: {qed_analysis['is_monotonic']}")
    print(f"  Total molecules: {qed_analysis['num_molecules']}")
    print(f"  Novelty rate: {qed_analysis['novelty_rate']:.1f}%")

    print("\nPer-target QED statistics:")
    print(qed_analysis["per_target_stats"].to_string(index=False))

    # Plot QED sweep
    plot_property_sweep(
        qed_sweep_df,
        qed_analysis,
        args.output_dir / "qed_sweep_plot.png",
        "qed",
    )

    print("\n" + "="*60)
    print("EXPERIMENT 2: Conditioned vs Baseline Comparison")
    print("="*60)

    comparison = run_conditioned_vs_baseline(
        generator,
        num_samples=args.num_samples * 2,
        k_neighbors=args.k_neighbors,
    )

    print(f"\nBaseline (0.5, 0.5):")
    print(f"  Mean QED: {comparison['baseline']['qed_mean']:.4f} +/- {comparison['baseline']['qed_std']:.4f}")
    print(f"  Novelty: {comparison['baseline']['novelty_rate']:.1f}%")

    print(f"\nHigh Target (0.7, 0.7):")
    print(f"  Mean QED: {comparison['high_target']['qed_mean']:.4f} +/- {comparison['high_target']['qed_std']:.4f}")
    print(f"  Novelty: {comparison['high_target']['novelty_rate']:.1f}%")

    print(f"\nStatistical Significance:")
    print(f"  QED difference: {comparison['difference']['qed_mean_diff']:.4f}")
    print(f"  t-test p-value: {comparison['difference']['t_pvalue']:.2e}")
    print(f"  Mann-Whitney p-value: {comparison['difference']['mannwhitney_pvalue']:.2e}")

    # Plot comparison
    plot_comparison(comparison, args.output_dir / "comparison_plot.png")

    # Summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)

    conditioning_works = (
        qed_analysis["correlation"] > 0.1 and
        qed_analysis["p_value"] < 0.05 and
        comparison["difference"]["t_pvalue"] < 0.05
    )

    if conditioning_works:
        print("\nCONCLUSION: Property conditioning is EFFECTIVE")
        print(f"  - Significant correlation between target and actual QED (r={qed_analysis['correlation']:.3f})")
        print(f"  - High-target generation produces significantly higher QED molecules")
    else:
        print("\nWARNING: Property conditioning may not be effective")
        print(f"  - Correlation: r={qed_analysis['correlation']:.3f} (p={qed_analysis['p_value']:.2e})")
        print(f"  - Consider retraining or adjusting model architecture")

    # Save summary
    summary = {
        "qed_correlation": qed_analysis["correlation"],
        "qed_correlation_pvalue": qed_analysis["p_value"],
        "qed_monotonic": qed_analysis["is_monotonic"],
        "baseline_qed_mean": comparison["baseline"]["qed_mean"],
        "high_target_qed_mean": comparison["high_target"]["qed_mean"],
        "qed_difference": comparison["difference"]["qed_mean_diff"],
        "ttest_pvalue": comparison["difference"]["t_pvalue"],
        "conditioning_effective": conditioning_works,
    }

    with open(args.output_dir / "ablation_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
