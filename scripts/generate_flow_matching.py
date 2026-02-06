#!/usr/bin/env python
"""Generate molecules using flow matching and evaluate results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from synthemol.flows.synthesis_generator import SynthesisGenerator


def plot_generation_results(
    molecules: list,
    save_path: Path,
):
    """Plot generated molecules in property space."""
    qeds = [m.actual_qed for m in molecules if m.actual_qed]
    target_qeds = [m.target_qed for m in molecules]
    novel_mask = [m.is_novel for m in molecules]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # QED distribution
    ax1 = axes[0]
    ax1.hist(qeds, bins=30, alpha=0.7, edgecolor="black")
    ax1.axvline(np.mean(qeds), color="red", linestyle="--", label=f"Mean: {np.mean(qeds):.3f}")
    ax1.set_xlabel("QED Score", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("QED Distribution of Generated Molecules", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Target vs Actual QED
    ax2 = axes[1]
    colors = ["orange" if novel else "blue" for novel in novel_mask]
    ax2.scatter(target_qeds[:len(qeds)], qeds, c=colors, alpha=0.3, s=10)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect match")
    ax2.set_xlabel("Target QED", fontsize=12)
    ax2.set_ylabel("Actual QED", fontsize=12)
    ax2.set_title("Target vs Actual QED (orange=novel)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Novelty pie chart
    ax3 = axes[2]
    novel_count = sum(novel_mask)
    not_novel_count = len(novel_mask) - novel_count
    ax3.pie(
        [novel_count, not_novel_count],
        labels=[f"Novel ({novel_count})", f"Training ({not_novel_count})"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#95a5a6"],
    )
    ax3.set_title("Molecule Novelty", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def analyze_molecules(molecules: list) -> dict:
    """Compute detailed statistics for generated molecules."""
    stats = {
        "total": len(molecules),
        "novel": sum(1 for m in molecules if m.is_novel),
        "novelty_pct": 100 * sum(1 for m in molecules if m.is_novel) / len(molecules),
    }

    # QED statistics
    qeds = [m.actual_qed for m in molecules if m.actual_qed]
    if qeds:
        stats["qed_mean"] = np.mean(qeds)
        stats["qed_std"] = np.std(qeds)
        stats["qed_min"] = np.min(qeds)
        stats["qed_max"] = np.max(qeds)
        stats["high_qed_count"] = sum(1 for q in qeds if q > 0.7)
        stats["high_qed_pct"] = 100 * stats["high_qed_count"] / len(qeds)

    # Compute additional properties
    mw_list = []
    logp_list = []
    hbd_list = []
    hba_list = []

    for mol in molecules:
        rdkit_mol = Chem.MolFromSmiles(mol.smiles)
        if rdkit_mol:
            mw_list.append(Descriptors.MolWt(rdkit_mol))
            logp_list.append(Descriptors.MolLogP(rdkit_mol))
            hbd_list.append(rdMolDescriptors.CalcNumHBD(rdkit_mol))
            hba_list.append(rdMolDescriptors.CalcNumHBA(rdkit_mol))

    if mw_list:
        stats["mw_mean"] = np.mean(mw_list)
        stats["mw_std"] = np.std(mw_list)
        stats["logp_mean"] = np.mean(logp_list)
        stats["logp_std"] = np.std(logp_list)
        stats["hbd_mean"] = np.mean(hbd_list)
        stats["hba_mean"] = np.mean(hba_list)

        # Lipinski rule of 5 compliance
        lipinski_pass = sum(
            1 for mw, logp, hbd, hba in zip(mw_list, logp_list, hbd_list, hba_list)
            if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
        )
        stats["lipinski_pct"] = 100 * lipinski_pass / len(mw_list)

    # Reaction diversity
    reactions = [m.reaction_id for m in molecules]
    stats["unique_reactions"] = len(set(reactions))

    # Building block diversity
    bb1s = [m.bb1_smiles for m in molecules]
    bb2s = [m.bb2_smiles for m in molecules]
    stats["unique_bb1"] = len(set(bb1s))
    stats["unique_bb2"] = len(set(bb2s))

    return stats


def compare_with_synthemol(
    fm_molecules: list,
    synthemol_path: Path,
    save_path: Path,
):
    """Compare flow matching results with SyntheMol baseline."""
    # Load SyntheMol results
    sm_df = pd.read_csv(synthemol_path)
    sm_qeds = sm_df["qed_score"].values
    sm_activities = sm_df["activity_score"].values

    # Flow matching QEDs
    fm_qeds = np.array([m.actual_qed for m in fm_molecules if m.actual_qed])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # QED comparison
    ax1 = axes[0]
    ax1.hist(sm_qeds, bins=30, alpha=0.5, label=f"SyntheMol (n={len(sm_qeds)})", density=True)
    ax1.hist(fm_qeds, bins=30, alpha=0.5, label=f"Flow Matching (n={len(fm_qeds)})", density=True)
    ax1.set_xlabel("QED Score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("QED Distribution Comparison", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Property space scatter
    ax2 = axes[1]
    ax2.scatter(sm_activities, sm_qeds, alpha=0.1, s=5, label="SyntheMol", c="blue")

    # For flow matching, we don't have activity predictions, so just show QED
    fm_target_act = [m.target_activity for m in fm_molecules if m.actual_qed]
    fm_actual_qed = [m.actual_qed for m in fm_molecules if m.actual_qed]
    ax2.scatter(fm_target_act, fm_actual_qed, alpha=0.3, s=15, label="Flow Matching", c="orange")

    ax2.set_xlabel("Activity Score (target for FM)", fontsize=12)
    ax2.set_ylabel("QED Score", fontsize=12)
    ax2.set_title("Property Space Comparison", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison to {save_path}")

    # Print comparison stats
    print("\n=== Comparison with SyntheMol ===")
    print(f"SyntheMol: {len(sm_df)} molecules")
    print(f"  QED: {sm_qeds.mean():.3f} ± {sm_qeds.std():.3f}")
    print(f"  High QED (>0.7): {np.sum(sm_qeds > 0.7)} ({100*np.sum(sm_qeds > 0.7)/len(sm_qeds):.1f}%)")

    print(f"\nFlow Matching: {len(fm_molecules)} molecules")
    print(f"  QED: {fm_qeds.mean():.3f} ± {fm_qeds.std():.3f}")
    print(f"  High QED (>0.7): {np.sum(fm_qeds > 0.7)} ({100*np.sum(fm_qeds > 0.7)/len(fm_qeds):.1f}%)")
    print(f"  Novel: {sum(1 for m in fm_molecules if m.is_novel)} ({100*sum(1 for m in fm_molecules if m.is_novel)/len(fm_molecules):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate molecules with flow matching")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("data/flow_matching/model/best_model.pt"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/flow_matching"),
        help="Directory with flow matching data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/flow_matching/generated"),
        help="Output directory",
    )
    parser.add_argument(
        "--target_activity",
        type=float,
        default=0.7,
        help="Target activity",
    )
    parser.add_argument(
        "--target_qed",
        type=float,
        default=0.7,
        help="Target QED",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of route samples",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Generate diverse molecules across property space",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with SyntheMol baseline",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SynthesisGenerator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
    )

    if args.diverse:
        print("\nGenerating diverse molecules across property space...")
        molecules = generator.generate_diverse(
            activity_range=(0.3, 0.9),
            qed_range=(0.5, 0.9),
            n_points=8,
            samples_per_point=args.num_samples,
            k_neighbors=5,
        )
    else:
        print(f"\nGenerating molecules for target (activity={args.target_activity}, qed={args.target_qed})...")
        molecules = generator.generate(
            target_activity=args.target_activity,
            target_qed=args.target_qed,
            num_samples=args.num_samples,
            k_neighbors=5,
        )

    print(f"\nGenerated {len(molecules)} unique molecules")

    # Analyze
    stats = analyze_molecules(molecules)
    print("\n=== Generation Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Save results
    results_df = pd.DataFrame([
        {
            "smiles": m.smiles,
            "bb1_smiles": m.bb1_smiles,
            "bb2_smiles": m.bb2_smiles,
            "reaction_id": m.reaction_id,
            "target_activity": m.target_activity,
            "target_qed": m.target_qed,
            "actual_qed": m.actual_qed,
            "is_novel": m.is_novel,
        }
        for m in molecules
    ])
    results_path = args.output_dir / "generated_molecules.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved {len(results_df)} molecules to {results_path}")

    # Plot
    plot_generation_results(molecules, args.output_dir / "generation_results.png")

    # Compare with SyntheMol if requested
    if args.compare:
        synthemol_path = Path("data/weight_sweep/all_results.csv")
        if synthemol_path.exists():
            compare_with_synthemol(
                molecules,
                synthemol_path,
                args.output_dir / "comparison.png",
            )


if __name__ == "__main__":
    main()
