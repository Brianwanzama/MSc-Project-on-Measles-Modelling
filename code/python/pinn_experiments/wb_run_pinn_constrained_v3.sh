#!/bin/bash
# ============================================================
# wb_run_pinn_constrained_v3.sh
# Fully constrained PINN — β_max=31.6, amp_max=5.0
# k=34, 20 runs | NaivePINN GPU 0 | TSIRPINN GPU 1
#
# CONSTRAINTS:
#   β_base = sigmoid(vert) × 31.6    → β_base ∈ (0, 31.6)
#   amp1_c = tanh(amp1) × 5.0        → amp1 ∈ (-5, 5)
#   amp2_c = tanh(amp2) × 5.0        → amp2 ∈ (-5, 5)
#   β(t)   = clamp(β_base+amp1_c×sin+amp2_c×cos, min=0.01)
#   β(t) ∈ (0.01, 41.6)  — bounded and strictly positive
#
# SCIENTIFIC JUSTIFICATION:
#   β_base=31.6: endemic equilibrium N/S = 7.6M/240K
#   amp±5:       allows ±16% seasonal variation in β
#                consistent with measles seasonality literature
# ============================================================

eval "$(conda shell.bash hook)"
conda activate finalmlenv

BASE="$HOME/Msc_project"
NAIVE="$BASE/code/python/pinn_experiments/wb_naivepinn_constrained_v3.py"
TSIR="$BASE/code/python/pinn_experiments/wb_tsirpinn_constrained_v3.py"
WRITE="$BASE/output/models/pinn_experiments/wb_pinn_constrained_v3/"
LOG="$BASE/output/logs/pinn_constrained_v3"

mkdir -p "$WRITE" "$LOG"

K=34
TLAG=52
EPOCHS=2500
N_RUNS=20
MAX_JOBS=4

echo "============================================"
echo "FULLY CONSTRAINED PINN v3"
echo "β_base = sigmoid(vert) × 31.6"
echo "amp    = tanh(amp) × 5.0"
echo "β(t) clamped to (0.01, ~41.6)"
echo "k=$K | $N_RUNS runs | NaivePINN GPU0 | TSIRPINN GPU1"
echo "============================================"

current_jobs=0
START=$(date +%s)

for ((i=1; i<=N_RUNS; i++)); do

    CUDA_VISIBLE_DEVICES=0 python3 "$NAIVE" \
        --run-num $i --k $K --tlag $TLAG \
        --num-epochs $EPOCHS --wd-fnn 0.025 \
        --write-loc "$WRITE" \
        > "$LOG/naive_k${K}_run${i}.log" 2>&1 &
    current_jobs=$((current_jobs+1))

    CUDA_VISIBLE_DEVICES=1 python3 "$TSIR" \
        --run-num $i --k $K --tlag $TLAG \
        --num-epochs $EPOCHS --wd-fnn 0.025 \
        --write-loc "$WRITE" \
        > "$LOG/tsir_k${K}_run${i}.log" 2>&1 &
    current_jobs=$((current_jobs+1))

    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
        echo "  Batch done — run=$i at $(date '+%H:%M:%S')"
    fi
done

wait

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "Done in $(( ELAPSED/60 )) minutes"

naive_done=$(ls "$WRITE" 2>/dev/null | \
    grep "naivepinn_constrained_v3_k${K}" | \
    grep "test_pred" | wc -l)
tsir_done=$(ls "$WRITE" 2>/dev/null | \
    grep "tsirpinn_constrained_v3_k${K}" | \
    grep "test_pred" | wc -l)

echo "naivepinn_constrained_v3: $naive_done/$N_RUNS"
echo "tsirpinn_constrained_v3:  $tsir_done/$N_RUNS"
