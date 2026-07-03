#!/bin/bash
# ============================================================
# run_raytune.sh
# Ray Tune hyperparameter search for SFNN — West Bengal measles
# Server: /home/brain/Msc_project/
# Running inside tmux — no nohup needed
# 64 cores — Ray uses all cores internally per k run
# ============================================================

# ── CONDA SETUP ───────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate finalmlenv

# ── PATHS ─────────────────────────────────────────────────────
PROJECT_DIR="/.../.../.../"
SCRIPT="$PROJECT_DIR/code/python/full_basic_raytune.py"
LOG_DIR="$PROJECT_DIR/output/logs"
PREFIT="$PROJECT_DIR/output/data/prefit_cases1"
RESULTS="$PROJECT_DIR/output/data/raytune_hp_optim"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS"

# ── GPU CHECK ─────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPUS_PER_TRIAL=1
    echo "GPU detected — gpus-per-trial: 1"
else
    GPUS_PER_TRIAL=0
    echo "No GPU — running CPU only"
fi

# ── VERIFY PARQUET FILES ──────────────────────────────────────
echo ""
echo "Checking parquet files..."

REQUIRED_FILES=(
    "k1_tlag26.gzip"   "k1_tlag52.gzip"
    "k4_tlag26.gzip"   "k4_tlag52.gzip"
    "k12_tlag26.gzip"  "k12_tlag52.gzip"
    "k20_tlag26.gzip"  "k20_tlag52.gzip"
    "k34_tlag52.gzip"
)

ALL_PRESENT=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$PREFIT/$f" ]; then
        echo "  $f — OK"
    else
        echo "  $f — MISSING"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "ERROR: Missing parquet files."
    echo "Run prevac_measles_data_loader.py first."
    exit 1
fi

# ── SETTINGS ──────────────────────────────────────────────────
k_values=(1 4 12 20 34)   # k=52 excluded — no parquet exists
NUM_SAMPLES=20
MAX_EPOCHS=50

# ── PRINT RUN PLAN ────────────────────────────────────────────
echo ""
echo "=============================================="
echo "RAY TUNE RUN PLAN"
echo "=============================================="
echo "k values:       ${k_values[*]}"
echo "Num samples:    $NUM_SAMPLES"
echo "Max epochs:     $MAX_EPOCHS"
echo "GPUs per trial: $GPUS_PER_TRIAL"
echo "CPUs per trial: 2  (32 parallel trials)"
echo "Total trials per k:"
echo "  k=1,4,12,20 : 2 t_lag × 3 hidden × 3 layers × 20 samples = 360"
echo "  k=34        : 1 t_lag × 3 hidden × 3 layers × 20 samples = 180"
echo "Logs:           $LOG_DIR/raytune_k{k}.log"
echo "Results:        $RESULTS/raytune_hp_optim_k_{k}.csv"
echo "=============================================="
echo ""
echo "Running in tmux — safe to detach with Ctrl+B D"
echo ""

# ── MAIN LOOP ─────────────────────────────────────────────────
OVERALL_START=$(date +%s)

for k in "${k_values[@]}"; do

    K_START=$(date +%s)

    echo "=============================================="
    echo "k=$k started at $(date '+%H:%M:%S')"
    echo "=============================================="

    python "$SCRIPT" \
        --k              "$k"              \
        --num-samples    "$NUM_SAMPLES"    \
        --max-num-epochs "$MAX_EPOCHS"     \
        --gpus-per-trial "$GPUS_PER_TRIAL" \
        2>&1 | tee "$LOG_DIR/raytune_k${k}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    K_END=$(date +%s)
    K_ELAPSED=$(( K_END - K_START ))
    K_MINS=$(( K_ELAPSED / 60 ))

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "k=$k DONE in ${K_MINS} minutes"
        # Print best result inline
        RESULT_FILE="$RESULTS/raytune_hp_optim_k_${k}.csv"
        if [ -f "$RESULT_FILE" ]; then
            python3 -c "
import pandas as pd
df   = pd.read_csv('$RESULT_FILE')
best = df.iloc[0]
print('Best result:')
print(f'  test_mse:          {best[\"test_mse\"]:.6f}')
print(f'  t_lag:             {int(best[\"t_lag\"])}')
print(f'  hidden_dim:        {int(best[\"hidden_dim\"])}')
print(f'  num_hidden_layers: {int(best[\"num_hidden_layers\"])}')
print(f'  lr:                {best[\"lr\"]:.6f}')
print(f'  weight_decay:      {best[\"weight_decay\"]:.6f}')
"
        fi
    else
        echo "k=$k FAILED (exit code $EXIT_CODE)"
        echo "Last 10 lines of log:"
        tail -10 "$LOG_DIR/raytune_k${k}.log"
    fi

    echo ""

done

# ── FINAL SUMMARY ─────────────────────────────────────────────
OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo "=============================================="
echo "ALL TUNING COMPLETE"
echo "Total time: ${HOURS}h ${MINS}m"
echo "=============================================="
echo ""
echo "BEST HYPERPARAMETERS PER k:"
echo "----------------------------------------------"

for k in "${k_values[@]}"; do
    RESULT_FILE="$RESULTS/raytune_hp_optim_k_${k}.csv"
    if [ -f "$RESULT_FILE" ]; then
        python3 -c "
import pandas as pd
df   = pd.read_csv('$RESULT_FILE')
best = df.iloc[0]
print(f'k={$k}:  test_mse={best[\"test_mse\"]:.4f}  '
      f't_lag={int(best[\"t_lag\"])}  '
      f'hidden={int(best[\"hidden_dim\"])}  '
      f'layers={int(best[\"num_hidden_layers\"])}  '
      f'lr={best[\"lr\"]:.5f}  '
      f'wd={best[\"weight_decay\"]:.5f}')
"
    else
        echo "k=$k: no result file"
    fi
done

echo ""
echo "Logs:    $LOG_DIR/"
echo "Results: $RESULTS/"