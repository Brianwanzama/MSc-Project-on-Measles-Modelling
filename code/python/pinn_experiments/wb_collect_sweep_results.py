"""
wb_collect_sweep_results.py

Collects _run_summary.parquet files from the TSIR-PINN
loss weight sweep and produces two CSVs for the R figure:

  experiments/tables/loss_weight_sweep/
    sweep_all_runs.csv     one row per run (raw data for figure)
    sweep_summary.csv      one row per ratio (summary stats)

Usage:
    conda activate finalmlenv
    python wb_collect_sweep_results.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path("/home/brain/Msc_project/")
SWEEP_DIR = (BASE / "output/models/pinn_experiments"
             / "wb_pinn_sweep/tsirpinn_sweep")
OUT_DIR   = BASE / "experiments/tables/loss_weight_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BETA_EQ     = 35.07
LOG_BETA_EQ = float(np.log(BETA_EQ))   # 3.5573
TSIR_BM     = 33.0                      # thesis TSIR benchmark
RATIOS      = [10, 35, 10000]

# ── Collect all run summaries ─────────────────────────────────
print("Collecting run summaries...")
all_rows = []

for ratio in RATIOS:
    ratio_dir = SWEEP_DIR / f"ratio_{ratio}"
    if not ratio_dir.exists():
        print(f"  Ratio {ratio:>6}: directory not found")
        continue

    files = sorted(ratio_dir.glob("*_run_summary.parquet"))
    print(f"  Ratio {ratio:>6}: {len(files)} runs found")

    for f in files:
        try:
            row = pd.read_parquet(f).iloc[0].to_dict()
            all_rows.append(row)
        except Exception as e:
            print(f"    Error reading {f.name}: {e}")

if len(all_rows) == 0:
    print("\nNo results found. Check sweep completed.")
    exit(1)

df = pd.DataFrame(all_rows)

# ── Derived columns ───────────────────────────────────────────
df['ratio']          = df['ratio'].round(0).astype(int)
df['log_beta_eq']    = LOG_BETA_EQ
df['dist_from_log_eq'] = (df['final_vert'] - LOG_BETA_EQ).abs()
df['above_eq']       = (df['final_vert'] > LOG_BETA_EQ).astype(int)
df['beat_tsir']      = (df['test_rmse'] < TSIR_BM).astype(int)

# Dead zone epoch — read from fit_info parquet
# Defined as first epoch where vert stops changing (delta < 1e-4)
def get_dead_zone_epoch(ratio, run_num, city):
    city_safe = city.replace(" ", "_")
    rl        = f"ratio_{ratio}"
    stem      = (SWEEP_DIR / f"ratio_{ratio}"
                 / f"tsirpinn_sweep_{rl}_k34_tlag52"
                   f"_city{city_safe}_run_{run_num}"
                   f"_fit_info.parquet")
    if not stem.exists():
        return np.nan
    try:
        fi   = pd.read_parquet(stem)
        vert = [float(v[0]) if hasattr(v, '__len__') else float(v)
                for v in fi['vert']]
        for i in range(1, len(vert)):
            if abs(vert[i] - vert[i-1]) < 1e-4:
                return i + 1   # 1-indexed epoch
        return np.nan
    except Exception:
        return np.nan

print("Computing dead zone epochs (this may take a moment)...")
df['dead_zone_epoch'] = df.apply(
    lambda r: get_dead_zone_epoch(
        int(r['ratio']), int(r['run']), r['city']),
    axis=1
)

# ── Summary stats per ratio ───────────────────────────────────
summary_rows = []
for ratio in RATIOS:
    dr = df[df['ratio'] == ratio]
    if len(dr) == 0:
        continue

    summary_rows.append({
        'ratio':              ratio,
        'n_runs':             len(dr),
        'regime':             dr['regime'].iloc[0],
        # vert
        'median_vert':        dr['final_vert'].median(),
        'mean_vert':          dr['final_vert'].mean(),
        'q25_vert':           dr['final_vert'].quantile(0.25),
        'q75_vert':           dr['final_vert'].quantile(0.75),
        'q10_vert':           dr['final_vert'].quantile(0.10),
        'q90_vert':           dr['final_vert'].quantile(0.90),
        'sd_vert':            dr['final_vert'].std(),
        # amp1
        'median_amp1':        dr['final_amp1'].median(),
        'mean_amp1':          dr['final_amp1'].mean(),
        'q25_amp1':           dr['final_amp1'].quantile(0.25),
        'q75_amp1':           dr['final_amp1'].quantile(0.75),
        'q10_amp1':           dr['final_amp1'].quantile(0.10),
        'q90_amp1':           dr['final_amp1'].quantile(0.90),
        'sd_amp1':            dr['final_amp1'].std(),
        # rmse
        'median_rmse':        dr['test_rmse'].median(),
        'mean_rmse':          dr['test_rmse'].mean(),
        'q25_rmse':           dr['test_rmse'].quantile(0.25),
        'q75_rmse':           dr['test_rmse'].quantile(0.75),
        'beat_tsir_frac':     dr['beat_tsir'].mean(),
        # dead zone
        'median_dead_zone':   dr['dead_zone_epoch'].median(),
        'mean_dead_zone':     dr['dead_zone_epoch'].mean(),
        # reference
        'log_beta_eq':        LOG_BETA_EQ,
        'beta_eq':            BETA_EQ,
    })

df_summary = pd.DataFrame(summary_rows)

# ── Save ──────────────────────────────────────────────────────
df.to_csv(          OUT_DIR / "sweep_all_runs.csv",  index=False)
df_summary.to_csv(  OUT_DIR / "sweep_summary.csv",   index=False)

print(f"\nSaved to: {OUT_DIR}")
print("  sweep_all_runs.csv")
print("  sweep_summary.csv")

# ── Print summary ─────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"SWEEP SUMMARY  |  beta_eq={BETA_EQ}  "
      f"log(beta_eq)={LOG_BETA_EQ:.4f}")
print(f"{'='*72}")
print(f"{'Ratio':>7} {'n':>4} {'Regime':>6} "
      f"{'Med.vert':>10} {'Med.amp1':>10} "
      f"{'Med.RMSE':>10} {'DeadZone':>10}")
print("-"*72)
for _, r in df_summary.iterrows():
    print(f"{int(r['ratio']):>7} "
          f"{int(r['n_runs']):>4} "
          f"{r['regime']:>6} "
          f"{r['median_vert']:>10.4f} "
          f"{r['median_amp1']:>10.4f} "
          f"{r['median_rmse']:>10.4f} "
          f"{r['median_dead_zone']:>10.1f}")

print(f"\nlog(beta_eq) = {LOG_BETA_EQ:.4f}")
print(f"TSIR benchmark RMSE = {TSIR_BM}")
print("\nNext step: Rscript wb_pinn_sweep_fig.R")
