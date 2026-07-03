#!/bin/bash
# ============================================================
# wb_full_basic_optimal_V1V2.sh
# Train final SFNN models with V1+V2 features
# Uses prefit_cases_V1V2/ parquets
# Output: basic_nn_optimal_V1V2/
# ============================================================

eval "$(conda shell.bash hook)"
conda activate finalmlenv

BASE="$HOME/Msc_project"
SCRIPT="$BASE/code/python/basic_nn/wb_full_basic.py"
HP_DIR="$BASE/output/data/raytune_hp_optim"
OUT_DIR="$BASE/output/data/basic_nn_optimal_V1V2"
LOG_DIR="$BASE/output/logs"
PREFIT="$BASE/output/data/prefit_cases_V1V2"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    NO_CUDA=""
    echo "GPU detected"
else
    NO_CUDA="--no-cuda"
    echo "No GPU"
fi

echo ""
echo "=============================================="
echo "WB SFNN FINAL TRAINING — V1+V2 VERSION"
echo "=============================================="
echo "Parquets: $PREFIT"
echo "Output:   $OUT_DIR"
echo "Epochs:   200"
echo "=============================================="
echo ""

# Verify HP files and parquet files
k_values=(1 4 12 20 34)
ALL_OK=true

for k in "${k_values[@]}"; do
    hp_f="$HP_DIR/raytune_hp_optim_k_${k}.csv"
    pq_f="$PREFIT/k${k}_tlag52.gzip"

    [ -f "$hp_f" ] || { echo "MISSING HP: $hp_f"; ALL_OK=false; }
    [ -f "$pq_f" ] || { echo "MISSING parquet: $pq_f"; ALL_OK=false; }
done

if [ "$ALL_OK" = false ]; then
    echo "ERROR: Missing files. Run data loader and Ray Tune first."
    exit 1
fi
echo "All input files present."
echo ""

# ── EXTRACT BEST PARAMS AND RUN IN PARALLEL ─────────────────
for k in "${k_values[@]}"; do
(
    HP_FILE="$HP_DIR/raytune_hp_optim_k_${k}.csv"
    LOG_FILE="$LOG_DIR/final_train_V1V2_k${k}.log"

    echo "Starting k=$k at $(date '+%H:%M:%S')"

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

    # Override t-lag to 52 — V1V2 parquets only have tlag=52
    # (k=1 best t_lag=26 from Ray Tune but V1V2 only has tlag=52)
    PARAMS=$(echo "$PARAMS" | sed 's/--t-lag [0-9]*/--t-lag 52/')

    CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT" \
        --k              "$k"     \
        --num-epochs     200      \
        --year-test-cutoff 2017   \
        --batch-size     64       \
        --log-interval   9999     \
        --save-model              \
        --prefit-dir     "$PREFIT" \
        --save-data-loc  "$OUT_DIR/" \
        $PARAMS                   \
        $NO_CUDA                  \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  k=$k DONE at $(date '+%H:%M:%S')"
        python3 -c "
import pandas as pd, os
path = '$OUT_DIR/${k}_summary.csv'
if os.path.exists(path):
    df = pd.read_csv(path)
    r  = df.iloc[0]
    print(f'  k=$k: train_mse={r[\"train_mse\"]:.6f}  test_mse={r[\"test_mse\"]:.6f}')
" 2>/dev/null
    else
        echo "  k=$k FAILED (exit code $EXIT_CODE)"
        tail -5 "$LOG_FILE"
    fi

) &
done

wait

echo ""
echo "=============================================="
echo "ALL V1+V2 MODELS TRAINED"
echo "=============================================="
echo ""
echo "Results summary:"
python3 - << 'PYEOF'
import pandas as pd, os

out_dir = os.path.expanduser(
    "~/Msc_project/output/data/basic_nn_optimal_V1V2")

print(f"{'k':>4}  {'t_lag':>6}  {'hidden':>8}  {'layers':>7}  "
      f"{'train_mse':>11}  {'test_mse':>10}")
print("-" * 55)

for k in [1, 4, 12, 20, 34]:
    path = f"{out_dir}/{k}_summary.csv"
    if os.path.exists(path):
        df  = pd.read_csv(path)
        r   = df.iloc[0]
        print(f"{k:>4}  {int(r['t_lag']):>6}  "
              f"{int(r['hidden_dim']):>8}  "
              f"{int(r['num_hidden_layers']):>7}  "
              f"{r['train_mse']:>11.6f}  "
              f"{r['test_mse']:>10.6f}")
    else:
        print(f"{k:>4}  MISSING")
PYEOF

echo ""
echo "Output: $OUT_DIR"
echo "Logs:   $LOG_DIR/final_train_V1V2_k*.log"
