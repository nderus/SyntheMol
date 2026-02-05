#!/bin/bash
# Run weight sweep experiment in parallel
# Usage: ./scripts/run_weight_sweep.sh [n_rollout]

N_ROLLOUT=${1:-2000}
OUTPUT_DIR="data/weight_sweep"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=== Weight Sweep Experiment ==="
echo "Rollouts per weight: $N_ROLLOUT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate conda environment and run in parallel
WEIGHTS=(0.1 0.3 0.5 0.7 0.9)

for w in "${WEIGHTS[@]}"; do
    echo "Starting weight=$w..."
    mamba run -n synthemol python scripts/weight_sweep.py \
        --activity_weight $w \
        --n_rollout $N_ROLLOUT \
        --output_dir "$OUTPUT_DIR" \
        > "$LOG_DIR/weight_${w}.log" 2>&1 &
done

echo ""
echo "All jobs started in background."
echo "Monitor with: tail -f $LOG_DIR/*.log"
echo "Check progress with: ps aux | grep weight_sweep"
echo ""

# Wait for all jobs
wait

echo "All jobs completed!"
echo ""

# Combine results
echo "Combining results..."
mamba run -n synthemol python << 'EOF'
import pandas as pd
from pathlib import Path

output_dir = Path("data/weight_sweep")
weights = [0.1, 0.3, 0.5, 0.7, 0.9]

all_dfs = []
for w in weights:
    csv_path = output_dir / f"activity_{w:.1f}" / "molecules.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
        print(f"Weight {w}: {len(df)} molecules")

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_dir / "all_results.csv", index=False)
    print(f"\nCombined: {len(combined)} molecules saved to {output_dir / 'all_results.csv'}")
EOF
