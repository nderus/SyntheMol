#!/usr/bin/env python
"""
Weight Sweep Experiment for Pareto Frontier Exploration

Runs SyntheMol with different fixed weight configurations to understand
how objective weights affect the score tradeoffs.

Usage:
    python scripts/weight_sweep.py --activity_weight 0.5 --n_rollout 2000

Or run all weights:
    python scripts/weight_sweep.py --run_all --n_rollout 2000
"""

import argparse
import pickle
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
import torch

from synthemol.constants import OLD_REACTIONS, OLD_REACTION_ORDER
from synthemol.reactions import CHEMICAL_SPACE_TO_REACTIONS
from synthemol.generate.generator import Generator
from synthemol.generate.scorer import MoleculeScorer
from synthemol.generate.score_weights import ScoreWeights


def run_generation(
    activity_weight: float,
    n_rollout: int = 2000,
    output_dir: Path = Path("data/weight_sweep"),
    seed: int = 42,
):
    """Run generation with fixed weights."""

    qed_weight = 1.0 - activity_weight
    output_dir = Path(output_dir)
    run_dir = output_dir / f"activity_{activity_weight:.1f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running with Activity={activity_weight:.1f}, QED={qed_weight:.1f}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    # Load building blocks with QED
    print("Loading building blocks...")
    bb_data = pd.read_csv(
        "data/Models/antibiotic_chemprop/building_blocks_with_qed.csv",
        dtype={"Reagent_ID": str}
    )
    print(f"Loaded {len(bb_data)} building blocks")

    # Create mappings
    building_block_smiles_to_id = dict(zip(bb_data["smiles"], bb_data["Reagent_ID"]))
    building_block_smiles = set(bb_data["smiles"])
    building_block_smiles_to_scores = {
        row["smiles"]: [row["chemprop_ensemble_preds"], row["qed_score"]]
        for _, row in bb_data.iterrows()
    }

    # Set up reactions (filter to OLD_REACTIONS)
    print("Setting up reactions...")
    all_reactions = CHEMICAL_SPACE_TO_REACTIONS["real"]
    reactions = [deepcopy(r) for r in all_reactions if r.reaction_id in OLD_REACTIONS]
    reactions = sorted(reactions, key=lambda r: OLD_REACTION_ORDER.index(r.reaction_id))
    print(f"Using {len(reactions)} reactions")

    # Apply old reaction fixes
    for reaction in reactions:
        reaction.id = reaction.reaction_id
        if reaction.id == 240790:
            reaction.post_reactions = tuple()
        if reaction.id == 271948:
            reaction.post_reactions = tuple()
            reaction.reactants = reaction.reactants[::-1]

    # Set building blocks for reactions
    for reaction in reactions:
        for reactant in reaction.reactants:
            reactant.all_building_blocks = building_block_smiles

    # Load allowed building blocks
    print("Loading reaction to building blocks mapping...")
    with open("data/Data/4_real_space/reaction_to_building_blocks_filtered.pkl", "rb") as f:
        reaction_to_building_blocks = pickle.load(f)

    for reaction in reactions:
        for reactant_index, reactant in enumerate(reaction.reactants):
            reactant.allowed_building_blocks = reaction_to_building_blocks[reaction.id][reactant_index]

    # Set up FIXED score weights (immutable=True means no dynamic adjustment)
    score_weights = ScoreWeights(
        base_score_weights=[activity_weight, qed_weight],
        immutable=True,  # Fixed weights, no dynamic adjustment
        score_names=["Activity", "QED"],
        score_signs=[1, 1],
    )

    # Set up scorer
    print("Setting up scorer...")
    device = torch.device("cpu")
    scorer = MoleculeScorer(
        score_types=["chemprop", "qed"],
        score_weights=score_weights,
        model_paths=[Path("data/Models/antibiotic_chemprop"), None],
        fingerprint_types=[None, None],
        h2o_solvents=False,
        device=device,
        smiles_to_scores=building_block_smiles_to_scores,
        wavelength_color=None,
    )

    # Create generator
    print("Creating generator...")
    generator = Generator(
        search_type="mcts",
        chemical_space_to_building_block_smiles_to_id={"real": building_block_smiles_to_id},
        max_reactions=1,
        scorer=scorer,
        score_weights=score_weights,
        success_comparators=None,  # No dynamic adjustment
        explore_weight=10.0,
        num_expand_nodes=None,
        rl_base_temperature=0.1,
        rl_temperature_similarity_target=None,
        rl_train_frequency=10,
        reactions=tuple(reactions),
        rng_seed=seed,
        no_building_block_diversity=False,
        store_nodes=False,
        verbose=False,
        rl_model=None,
        replicate=True,
        wandb_log=False,
        log_path=run_dir / "logs.pkl",
    )

    # Run generation
    print(f"Starting generation with {n_rollout} rollouts...")
    nodes = generator.generate(n_rollout=n_rollout)

    # Extract results
    results = []
    for node in nodes:
        if node.num_molecules == 1:
            scores = scorer.compute_individual_scores(node.molecules[0])
            results.append({
                "smiles": node.molecules[0],
                "activity_score": scores[0],
                "qed_score": scores[1],
                "combined_score": node.property_score,
                "activity_weight": activity_weight,
                "qed_weight": qed_weight,
                "node_id": node.node_id,
                "rollout_num": node.rollout_num,
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(run_dir / "molecules.csv", index=False)

    print(f"\nGenerated {len(results_df)} unique molecules")
    print(f"Activity scores: mean={results_df['activity_score'].mean():.3f}, max={results_df['activity_score'].max():.3f}")
    print(f"QED scores: mean={results_df['qed_score'].mean():.3f}, max={results_df['qed_score'].max():.3f}")
    print(f"Results saved to {run_dir / 'molecules.csv'}")

    return results_df


def run_all_weights(n_rollout: int = 2000, output_dir: Path = Path("data/weight_sweep")):
    """Run sweep across all weight configurations."""

    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = []

    for w in weights:
        df = run_generation(
            activity_weight=w,
            n_rollout=n_rollout,
            output_dir=output_dir,
        )
        all_results.append(df)

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "all_results.csv", index=False)
    print(f"\nCombined results saved to {output_dir / 'all_results.csv'}")

    # Summary statistics
    print("\n" + "="*60)
    print("WEIGHT SWEEP SUMMARY")
    print("="*60)
    for w in weights:
        subset = combined[combined["activity_weight"] == w]
        print(f"\nActivity Weight = {w:.1f}:")
        print(f"  Molecules: {len(subset)}")
        print(f"  Activity: mean={subset['activity_score'].mean():.3f}, max={subset['activity_score'].max():.3f}")
        print(f"  QED: mean={subset['qed_score'].mean():.3f}, max={subset['qed_score'].max():.3f}")
        # Pareto-ish metric: molecules with both scores > 0.5
        both_good = ((subset['activity_score'] > 0.5) & (subset['qed_score'] > 0.5)).sum()
        print(f"  Both > 0.5: {both_good} ({100*both_good/len(subset):.1f}%)")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weight sweep experiment")
    parser.add_argument("--activity_weight", type=float, help="Activity weight (0-1)")
    parser.add_argument("--n_rollout", type=int, default=2000, help="Number of rollouts")
    parser.add_argument("--output_dir", type=str, default="data/weight_sweep", help="Output directory")
    parser.add_argument("--run_all", action="store_true", help="Run all weight configurations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.run_all:
        run_all_weights(n_rollout=args.n_rollout, output_dir=Path(args.output_dir))
    elif args.activity_weight is not None:
        run_generation(
            activity_weight=args.activity_weight,
            n_rollout=args.n_rollout,
            output_dir=Path(args.output_dir),
            seed=args.seed,
        )
    else:
        parser.print_help()
