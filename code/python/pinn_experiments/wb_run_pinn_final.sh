#!/bin/bash
# ============================================================
# wb_run_pinn_final.sh
# FINAL PINN RUN — 100 runs each model
# β(t) = sigmoid(ν)×31.6 + tanh(α₁)×5·sin + tanh(α₂)×5·cos
# λ_I=100 | λ_ODE=0.01 | normalised losses
# NaivePINN GPU 0 | TSIRPINN GPU 1
# ============================================================

eval "$(conda shell.bash hook)"
conda activate finalmlenv

BASE="$HOME/Msc_project"
NAIVE="$BASE/code/python/pinn_experiments/wb_naivepinn_constrained_v3.py"
TSIR="$BASE/code/python/pinn_experiments/wb_tsirpinn_constrained_v3.py"
WRITE="$BASE/output/models/pinn_experiments/wb_pinn_final/"
LOG="$BASE/output/logs/pinn_final"

mkdir -p "$WRITE" "$LOG"

K=34
TLAG=52
EPOCHS=2500
N_RUNS=100
MAX_JOBS=4

START=$(date +%s)
START_STR=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================"
echo "FINAL PINN RUN — 100 RUNS EACH"
echo "Started: $START_STR"
echo "--------------------------------------------"
echo "β(t) = sigmoid(ν)×31.6"
echo "     + tanh(α₁)×5·sin(2πt/26)"
echo "     + tanh(α₂)×5·cos(2πt/26)"
echo "λ_I=100 | λ_ODE=0.01 | normalised losses"
echo "NaivePINN → GPU 0 | TSIRPINN → GPU 1"
echo "k=$K | $N_RUNS runs | $EPOCHS epochs"
echo "============================================"
echo ""

current_jobs=0

for ((i=1; i<=N_RUNS; i++)); do

    RUN_START=$(date '+%H:%M:%S')
    echo ">>> Run $i/$N_RUNS | $RUN_START"

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

        ELAPSED=$(( $(date +%s) - START ))
        NAIVE_DONE=$(ls "$WRITE" 2>/dev/null | \
            grep "naivepinn" | grep "test_pred" | wc -l)
        TSIR_DONE=$(ls "$WRITE" 2>/dev/null | \
            grep "tsirpinn" | grep "test_pred" | wc -l)
        TOTAL_DONE=$(( NAIVE_DONE + TSIR_DONE ))
        REMAINING=$(( N_RUNS * 2 - TOTAL_DONE ))

        if [ "$TOTAL_DONE" -gt 0 ]; then
            PER_RUN=$(( ELAPSED / TOTAL_DONE ))
            ETA_MIN=$(( REMAINING * PER_RUN / 60 ))
        else
            ETA_MIN="?"
        fi

        echo ""
        echo "--------------------------------------------"
        echo "Batch complete | $(date '+%H:%M:%S')"
        echo "NaivePINN: $NAIVE_DONE/$N_RUNS"
        echo "TSIRPINN:  $TSIR_DONE/$N_RUNS"
        echo "Elapsed:   $(( ELAPSED/60 )) min"
        echo "ETA:       ~${ETA_MIN} min remaining"
        echo "--------------------------------------------"
        echo ""
    fi
done

wait

END_STR=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$(( $(date +%s) - START ))
HOURS=$(( ELAPSED/3600 ))
MINS=$(( (ELAPSED%3600)/60 ))
SECS=$(( ELAPSED%60 ))

NAIVE_DONE=$(ls "$WRITE" 2>/dev/null | \
    grep "naivepinn" | grep "test_pred" | wc -l)
TSIR_DONE=$(ls "$WRITE" 2>/dev/null | \
    grep "tsirpinn" | grep "test_pred" | wc -l)

echo ""
echo "============================================"
echo "COMPLETED"
echo "Started:  $START_STR"
echo "Finished: $END_STR"
echo "Elapsed:  ${HOURS}h ${MINS}m ${SECS}s"
echo "NaivePINN: $NAIVE_DONE/$N_RUNS"
echo "TSIRPINN:  $TSIR_DONE/$N_RUNS"
echo "============================================"
