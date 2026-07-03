#!/bin/bash
# ============================================================
# wb_run_pinn_sweep.sh
#
# TSIR-PINN Loss Weight Sensitivity Sweep
# Faithful to Madden original — wb_tsirpinn.py + lambda args
#
# THREE RATIOS straddling beta_eq = 35.07:
#   ratio=10    BELOW  (Madden default — lambda_I=10, ODE=1)
#   ratio=35    AT     (lambda_I=1, ODE=0.028571)
#   ratio=10000 ABOVE  (lambda_I=1, ODE=0.0001)
#
# GPU ALLOCATION:
#   GPU 0: ratio 35    (100 runs — runs in parallel)
#   GPU 1: ratio 10    (100 runs) then ratio 10000 (100 runs)
#
# WALL CLOCK: ~80h (ratio 10) + ~80h (ratio 10000) = ~160h
#             with ratio 35 on GPU 0 in parallel
#             Total: ~160h ≈ 6.7 days
#
# FOREGROUND + TEE: every epoch visible in tmux AND in log
# ============================================================

ENV_NAME="finalmlenv"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

BASE="$HOME/Msc_project"
SCRIPT="$BASE/code/python/pinn_experiments/wb_tsirpinn_sweep.py"
WRITE_BASE="$BASE/output/models/pinn_experiments/wb_pinn_sweep/tsirpinn_sweep"
LOG_BASE="$BASE/output/logs/pinn_sweep/tsirpinn_sweep"

K=34
TLAG=52
EPOCHS=2500
N_RUNS=100
BETA_EQ="35.07"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found"; exit 1
fi

GLOBAL_START=$(date +%s)
GLOBAL_START_STR=$(date '+%Y-%m-%d %H:%M:%S')

echo "================================================================"
echo "  TSIR-PINN LOSS WEIGHT SWEEP"
echo "  Faithful to Madden (2024) — wb_tsirpinn.py"
echo "================================================================"
echo "  Started:   $GLOBAL_START_STR"
echo "  beta_eq:   $BETA_EQ"
echo "  Ratios:    10 (BELOW) | 35 (AT) | 10000 (ABOVE)"
echo "  N_RUNS:    $N_RUNS per ratio"
echo "  GPU 0:     ratio 35  (background)"
echo "  GPU 1:     ratio 10 then ratio 10000 (foreground)"
echo "================================================================"
echo ""

# ════════════════════════════════════════════════════════════
# GPU 0 — ratio 35 (AT boundary) — background
# ════════════════════════════════════════════════════════════
WRITE_35="$WRITE_BASE/ratio_35"
LOG_35="$LOG_BASE/ratio_35"
mkdir -p "$WRITE_35" "$LOG_35"

echo "--- Launching ratio 35 on GPU 0 (background) ---"
(
  for ((i=1; i<=N_RUNS; i++)); do
      CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT" \
          --run-num $i \
          --city    "South Twenty Four Parganas" \
          --k       $K --tlag $TLAG \
          --num-epochs $EPOCHS \
          --wd-fnn  0.025 \
          --lambda-i   1.0 \
          --lambda-ode 0.028571 \
          --ratio-label "ratio_35" \
          --write-loc "$WRITE_35/" \
          > "$LOG_35/run_${i}.log" 2>&1
      DONE=$(ls "$WRITE_35" 2>/dev/null | \
          grep "_run_summary" | wc -l)
      echo "[GPU0 ratio_35] Run $i done | $DONE/$N_RUNS"
  done
) &
GPU0_PID=$!
echo "  GPU0 PID: $GPU0_PID"
echo ""

# ════════════════════════════════════════════════════════════
# GPU 1 — ratio 10 (BELOW) — foreground with tee
# ════════════════════════════════════════════════════════════
WRITE_10="$WRITE_BASE/ratio_10"
LOG_10="$LOG_BASE/ratio_10"
mkdir -p "$WRITE_10" "$LOG_10"

echo "================================================================"
echo "  RATIO 10 | BELOW beta_eq=$BETA_EQ | GPU 1 | FOREGROUND"
echo "  Madden default weights — expected: vert diverges"
echo "  Watch: vert column growing without bound"
echo "================================================================"
echo ""

R_START=$(date +%s)
R_START_STR=$(date '+%Y-%m-%d %H:%M:%S')

for ((i=1; i<=N_RUNS; i++)); do

    echo "--- Ratio 10 | Run $i/$N_RUNS ---"

    CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT" \
        --run-num $i \
        --city    "South Twenty Four Parganas" \
        --k       $K --tlag $TLAG \
        --num-epochs $EPOCHS \
        --wd-fnn  0.025 \
        --lambda-i   10.0 \
        --lambda-ode 1.0 \
        --ratio-label "ratio_10" \
        --write-loc "$WRITE_10/" \
        2>&1 | tee "$LOG_10/run_${i}.log"

    DONE=$(ls "$WRITE_10" 2>/dev/null | \
        grep "_run_summary" | wc -l)
    R_ELAPSED=$(( $(date +%s) - R_START ))
    G_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
    if [ "$DONE" -gt 0 ]; then
        PER_RUN=$(( R_ELAPSED / DONE ))
        ETA=$(( (N_RUNS - DONE) * PER_RUN / 60 ))
    else
        ETA="?"
    fi
    echo ""
    echo "  ✓ Ratio 10 | Run $i | $DONE/$N_RUNS | " \
         "ETA: ~${ETA}m | Global: $(( G_ELAPSED/60 ))m"
    echo ""

done

R_END_STR=$(date '+%Y-%m-%d %H:%M:%S')
R_ELAPSED=$(( $(date +%s) - R_START ))
DONE_10=$(ls "$WRITE_10" 2>/dev/null | \
    grep "_run_summary" | wc -l)

echo "================================================================"
echo "  RATIO 10 COMPLETE"
echo "  Started: $R_START_STR  Finished: $R_END_STR"
echo "  Time: $(( R_ELAPSED/60 ))m | Runs: $DONE_10/$N_RUNS"
echo "================================================================"
echo ""

# ════════════════════════════════════════════════════════════
# GPU 1 — ratio 10000 (ABOVE) — foreground with tee
# ════════════════════════════════════════════════════════════
WRITE_10K="$WRITE_BASE/ratio_10000"
LOG_10K="$LOG_BASE/ratio_10000"
mkdir -p "$WRITE_10K" "$LOG_10K"

echo "================================================================"
echo "  RATIO 10000 | ABOVE beta_eq=$BETA_EQ | GPU 1 | FOREGROUND"
echo "  Well above boundary — expected: vert still diverges"
echo "  if structural failure confirmed, or converges if ratio helps"
echo "================================================================"
echo ""

R_START=$(date +%s)
R_START_STR=$(date '+%Y-%m-%d %H:%M:%S')

for ((i=1; i<=N_RUNS; i++)); do

    echo "--- Ratio 10000 | Run $i/$N_RUNS ---"

    CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT" \
        --run-num $i \
        --city    "South Twenty Four Parganas" \
        --k       $K --tlag $TLAG \
        --num-epochs $EPOCHS \
        --wd-fnn  0.025 \
        --lambda-i   1.0 \
        --lambda-ode 0.0001 \
        --ratio-label "ratio_10000" \
        --write-loc "$WRITE_10K/" \
        2>&1 | tee "$LOG_10K/run_${i}.log"

    DONE=$(ls "$WRITE_10K" 2>/dev/null | \
        grep "_run_summary" | wc -l)
    R_ELAPSED=$(( $(date +%s) - R_START ))
    G_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
    if [ "$DONE" -gt 0 ]; then
        PER_RUN=$(( R_ELAPSED / DONE ))
        ETA=$(( (N_RUNS - DONE) * PER_RUN / 60 ))
    else
        ETA="?"
    fi
    echo ""
    echo "  ✓ Ratio 10000 | Run $i | $DONE/$N_RUNS | " \
         "ETA: ~${ETA}m | Global: $(( G_ELAPSED/60 ))m"
    echo ""

done

# Wait for ratio 35 background
echo "--- Waiting for ratio 35 (GPU 0) to finish ---"
wait $GPU0_PID 2>/dev/null || wait

# ── Final summary ─────────────────────────────────────────────
G_END_STR=$(date '+%Y-%m-%d %H:%M:%S')
G_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
HH=$(( G_ELAPSED / 3600 ))
MM=$(( (G_ELAPSED % 3600) / 60 ))
SS=$(( G_ELAPSED % 60 ))

DONE_10=$(ls "$WRITE_10"  2>/dev/null | grep "_run_summary" | wc -l)
DONE_35=$(ls "$WRITE_35"  2>/dev/null | grep "_run_summary" | wc -l)
DONE_10K=$(ls "$WRITE_10K" 2>/dev/null | grep "_run_summary" | wc -l)

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE"
echo "================================================================"
echo "  Started:   $GLOBAL_START_STR"
echo "  Finished:  $G_END_STR"
echo "  Elapsed:   ${HH}h ${MM}m ${SS}s"
echo ""
echo "  Runs:"
echo "    Ratio    10 (BELOW): $DONE_10/$N_RUNS"
echo "    Ratio    35 (AT):    $DONE_35/$N_RUNS"
echo "    Ratio 10000 (ABOVE): $DONE_10K/$N_RUNS"
echo ""
echo "  Next: python wb_collect_sweep_results.py"
echo "        Rscript wb_pinn_sweep_fig.R"
echo "================================================================"
