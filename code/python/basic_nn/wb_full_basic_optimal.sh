#!/bin/bash
# ============================================================
# wb_full_basic_optimal.sh
# Train final SFNN models with best Ray Tune hyperparameters
# West Bengal measles — all k values
# Adapted from full_basic_optimal.sh (Madden et al. 2024)
#
# KEY CHANGES FROM ORIGINAL:
#   1. Uses Python/pandas to parse CSV — not awk
#      (awk condition $3=="True" never matched WB CSVs)
#   2. Absolute server paths throughout
#   3. All 5 k values run in parallel
#   4. v1_data_loc included
#   5. year_test_cutoff=2017
#   6. Reads pre-built parquet — no data rebuilding
# ============================================================

eval "$(conda shell.bash hook)"
conda activate finalmlenv

BASE="$HOME/Msc_project"
SCRIPT="$BASE/code/python/basic_nn/wb_full_basic.py"
HP_DIR="$BASE/output/data/raytune_hp_optim"
OUT_DIR="$BASE/output/data/basic_nn_optimal"
LOG_DIR="$BASE/output/logs"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

# GPU detection
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    NO_CUDA=""
    echo "GPU detected"
else
    NO_CUDA="--no-cuda"
    echo "No GPU — CPU only"
fi

echo ""
echo "=============================================="
echo "WB SFNN FINAL TRAINING"
echo "=============================================="
echo "Script:  $SCRIPT"
echo "HP dir:  $HP_DIR"
echo "Output:  $OUT_DIR"
echo "Epochs:  200"
echo "=============================================="
echo ""

# Verify HP files exist
k_values=(1 4 12 20 34)
ALL_OK=true
for k in "${k_values[@]}"; do
    f="$HP_DIR/raytune_hp_optim_k_${k}.csv"
    if [ -f "$f" ]; then
        echo "  k=$k HP file — OK"
    else
        echo "  k=$k HP file — MISSING: $f"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo "ERROR: Missing HP files. Run Ray Tune first."
    exit 1
fi
echo ""

# ── EXTRACT BEST PARAMS AND RUN IN PARALLEL ───────────────────
# Uses Python to parse CSV correctly — not awk
# Runs all k values simultaneously in background

for k in "${k_values[@]}"; do
(
    HP_FILE="$HP_DIR/raytune_hp_optim_k_${k}.csv"
    LOG_FILE="$LOG_DIR/final_train_k${k}.log"

    echo "Starting k=$k at $(date '+%H:%M:%S')"

    # Extract best hyperparameters using Python
    PARAMS=$(python3 - << PYEOF
import pandas as pd
df   = pd.read_csv("$HP_FILE")
df   = df.sort_values("test_mse").reset_index(drop=True)
best = df.iloc[0]
print(
    f"--t-lag {int(best['t_lag'])} "
    f"--hidden-dim {int(best['hidden_dim'])} "
    f"--num-hidden-layers {int(best['num_hidden_layers'])} "
    f"--lr {best['lr']:.10f} "
    f"--weight-decay {best['weight_decay']:.10f}"
)
PYEOF
)

    if [ -z "$PARAMS" ]; then
        echo "ERROR: Could not extract params for k=$k"
        exit 1
    fi

    echo "  k=$k params: $PARAMS"

    CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT" \
        --k              "$k"     \
        --num-epochs     200      \
        --year-test-cutoff 2017   \
        --batch-size     64       \
        --log-interval   9999     \
        --save-model              \
        --save-data-loc  "$OUT_DIR/" \
        $PARAMS                   \
        $NO_CUDA                  \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  k=$k DONE at $(date '+%H:%M:%S')"
        # Print final MSE from summary
        python3 -c "
import pandas as pd
df = pd.read_csv('$OUT_DIR/${k}_summary.csv')
r  = df.iloc[0]
print(f'  k=$k: train_mse={r[\"train_mse\"]:.6f}  test_mse={r[\"test_mse\"]:.6f}')
" 2>/dev/null
    else
        echo "  k=$k FAILED (exit code $EXIT_CODE)"
        echo "  Last 5 lines of log:"
        tail -5 "$LOG_FILE"
    fi

) &
done

# Wait for all k values to finish
wait

echo ""
echo "=============================================="
echo "ALL FINAL MODELS TRAINED"
echo "=============================================="
echo ""
echo "Results summary:"
python3 - << 'PYEOF'
import pandas as pd
import os

out_dir = os.path.expanduser("~/Msc_project/output/data/basic_nn_optimal")

print(f"{'k':>4}  {'t_lag':>6}  {'hidden':>8}  {'layers':>7}  "
      f"{'train_mse':>11}  {'test_mse':>10}")
print("-" * 55)

for k in [1, 4, 12, 20, 34]:
    path = f"{out_dir}/{k}_summary.csv"
    if os.path.exists(path):
        df   = pd.read_csv(path)
        r    = df.iloc[0]
        print(f"{k:>4}  {int(r['t_lag']):>6}  {int(r['hidden_dim']):>8}  "
              f"{int(r['num_hidden_layers']):>7}  "
              f"{r['train_mse']:>11.6f}  {r['test_mse']:>10.6f}")
    else:
        print(f"{k:>4}  MISSING")
PYEOF

echo ""
echo "Output files: $OUT_DIR/"
echo "Logs:         $LOG_DIR/final_train_k*.log"