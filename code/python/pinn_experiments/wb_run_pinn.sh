#!/bin/bash
# ============================================================
# wb_run_pinn.sh
# Run NaivePINN and TSIRPINN for West Bengal measles
# South Twenty Four Parganas | k=1,4,12,20,34 | 20 runs each
#
# CHANGES FROM run.sh (London):
#   1. City: South Twenty Four Parganas
#   2. k values: 1, 4, 12, 20, 34
#   3. tlag: 52
#   4. year_test_cutoff: 2017.0
#   5. epochs: 500
#   6. write_loc: wb_pinn/
#   7. GPU: naivepinn on GPU 0, tsirpinn on GPU 1
#   8. MAX_JOBS: 4 (2 per GPU)
#   9. 20 runs per model per k (not 100)
# ============================================================

eval "$(conda shell.bash hook)"
conda activate finalmlenv

BASE="$HOME/Msc_project"
NAIVE_SCRIPT="$BASE/code/python/pinn_experiments/wb_naivepinn.py"
TSIR_SCRIPT="$BASE/code/python/pinn_experiments/wb_tsirpinn.py"
WRITE_LOC="$BASE/output/models/pinn_experiments/wb_pinn/"
LOG_DIR="$BASE/output/logs/pinn"

mkdir -p "$WRITE_LOC"
mkdir -p "$LOG_DIR"

CITY="South Twenty Four Parganas"
K_VALUES=(1 4 12 20 34)
TLAG=52
YEAR_CUTOFF=2017.0
EPOCHS=2500
WD=0.025
N_RUNS=100
MAX_JOBS=4

echo "=============================================="
echo "WB PINN EXPERIMENTS"
echo "=============================================="
echo "City:     $CITY"
echo "k values: ${K_VALUES[*]}"
echo "tlag:     $TLAG"
echo "Epochs:   $EPOCHS"
echo "Runs:     $N_RUNS per model per k"
echo "GPU 0:    naivepinn"
echo "GPU 1:    tsirpinn"
echo "Output:   $WRITE_LOC"
echo "=============================================="
echo ""

current_jobs=0
START=$(date +%s)

for k in "${K_VALUES[@]}"; do
    echo "=== k=$k ==="

    for ((i=1; i<=N_RUNS; i++)); do

        # NaivePINN on GPU 0
        CUDA_VISIBLE_DEVICES=0 python3 "$NAIVE_SCRIPT" \
            --run-num          $i           \
            --k                $k           \
            --tlag             $TLAG        \
            --year-test-cutoff $YEAR_CUTOFF \
            --city             "$CITY"      \
            --num-epochs       $EPOCHS      \
            --wd-fnn           $WD          \
            --write-loc        "$WRITE_LOC" \
            > "$LOG_DIR/naivepinn_k${k}_run${i}.log" 2>&1 &

        current_jobs=$((current_jobs + 1))

        # TSIR-PINN on GPU 1
        CUDA_VISIBLE_DEVICES=1 python3 "$TSIR_SCRIPT" \
            --run-num          $i           \
            --k                $k           \
            --tlag             $TLAG        \
            --year-test-cutoff $YEAR_CUTOFF \
            --city             "$CITY"      \
            --num-epochs       $EPOCHS      \
            --wd-fnn           $WD          \
            --write-loc        "$WRITE_LOC" \
            > "$LOG_DIR/tsirpinn_k${k}_run${i}.log" 2>&1 &

        current_jobs=$((current_jobs + 1))

        # Wait when max jobs reached
        if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
            wait
            current_jobs=0
            echo "  Completed batch — k=$k run=$i"
        fi
    done

    # Wait for all runs of this k to finish
    wait
    current_jobs=0

    # Count output files
    naive_done=$(ls "$WRITE_LOC" 2>/dev/null | \
        grep "naivepinn_k${k}_" | \
        grep "_test_predictions" | wc -l)
    tsir_done=$(ls "$WRITE_LOC" 2>/dev/null | \
        grep "tsirpinn_k${k}_" | \
        grep "_test_predictions" | wc -l)

    echo "  k=$k done: naive=$naive_done/20 tsir=$tsir_done/20"
    echo ""
done

wait

END=$(date +%s)
ELAPSED=$(( END - START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo "=============================================="
echo "ALL PINN EXPERIMENTS COMPLETE"
echo "Time: ${HOURS}h ${MINS}m"
echo "=============================================="
echo ""

# Summary of output files
echo "Output files:"
for k in "${K_VALUES[@]}"; do
    naive=$(ls "$WRITE_LOC" | grep "naivepinn_k${k}_" | \
            grep "_test_predictions" | wc -l)
    tsir=$(ls "$WRITE_LOC"  | grep "tsirpinn_k${k}_" | \
            grep "_test_predictions" | wc -l)
    echo "  k=$k: naivepinn=$naive/20 | tsirpinn=$tsir/20"
done

echo ""
echo "Run wb_pinn_comparison.py to compute RMSE comparison."
