"""
wb_extract_s_latent.py

Extracts S_latent from the saved NaivePINN ode_model.pt
and S_pred from the TSIR-PINN train_predictions.parquet,
then saves a single CSV for the R figure.

The S_latent comparison is the confirmatory diagnostic:
  - If S_latent diverges from S_obs → NaivePINN found its
    own susceptible reconstruction, not constrained by S_obs
  - If S_pred (TSIR-PINN) collapses away from S_obs →
    confirms the soft-constraint failure mechanism

Output:
  experiments/tables/s_latent_comparison.csv

Usage:
    conda activate finalmlenv
    python wb_extract_s_latent.py
"""

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path

BASE    = Path("/home/brain/Msc_project/")
PREFIT  = BASE / "output/data/prefit_cases1/"
OUT_DIR = BASE / "experiments/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PATHS ─────────────────────────────────────────────────────
# NaivePINN final run — use run 1 as representative
NAIVE_DIR = (BASE / "output/models/pinn_experiments"
             / "wb_pinn_final")
NAIVE_ODE = (NAIVE_DIR /
             "naivepinn_constrained_v3_k34_tlag52_"
             "citySouth_Twenty_Four_Parganas_run_1"
             "_ode_model.pt")
NAIVE_PRED = (NAIVE_DIR /
              "naivepinn_constrained_v3_k34_tlag52_"
              "citySouth_Twenty_Four_Parganas_run_1"
              "_train_predictions.parquet")

# TSIR-PINN sweep — ratio 10 run 1
TSIR_DIR  = (BASE / "output/models/pinn_experiments"
             / "wb_pinn_sweep/tsirpinn_sweep/ratio_10")
TSIR_PRED = (TSIR_DIR /
             "tsirpinn_sweep_ratio_10_k34_tlag52_"
             "citySouth_Twenty_Four_Parganas_run_1"
             "_train_predictions.parquet")

CITY = "South Twenty Four Parganas"
K    = 34

# ── LOAD OBSERVED DATA ────────────────────────────────────────
print("Loading observed susceptible series...")
df_feat = pd.read_parquet(
    str(PREFIT / f"k{K}_tlag52.gzip"))
df_city = df_feat[
    (df_feat['city'] == CITY) &
    (df_feat['time'] < 2017.0)
].copy()
df_city['biweek'] = (
    np.round((df_city['time'] - 2008.0) * 26)
    .astype(int) + 1
)
df_city = df_city.sort_values('biweek').reset_index(drop=True)

S_obs      = df_city['susc'].values
time_orig  = df_city['time'].values
biweeks    = df_city['biweek'].values
n_train    = len(S_obs)

print(f"Training biweeks: {n_train}")
print(f"S_obs mean: {S_obs.mean():.0f}")

# ── EXTRACT S_LATENT FROM NAIVEPINN ───────────────────────────
print("\nExtracting S_latent from NaivePINN ode_model...")

if not NAIVE_ODE.exists():
    print(f"  File not found: {NAIVE_ODE}")
    print("  Trying wb_pinn_sweep/naivepinn/ directory...")
    NAIVE_DIR2 = (BASE / "output/models/pinn_experiments"
                  / "wb_pinn_sweep/naivepinn")
    candidates = sorted(NAIVE_DIR2.glob(
        "*naivepinn*run_1*ode_model.pt"))
    if candidates:
        NAIVE_ODE  = candidates[0]
        NAIVE_PRED = candidates[0].parent / (
            candidates[0].name
            .replace("_ode_model.pt",
                     "_train_predictions.parquet"))
        print(f"  Found: {NAIVE_ODE.name}")
    else:
        print("  No NaivePINN ode_model found. "
              "Check paths manually.")
        NAIVE_ODE = None

S_latent_vals = None
if NAIVE_ODE and NAIVE_ODE.exists():
    state_dict = torch.load(
        str(NAIVE_ODE), map_location='cpu')

    # S_latent key in the state dict
    s_key = None
    for k in state_dict.keys():
        if 'S_latent' in k or 's_latent' in k:
            s_key = k
            break

    if s_key:
        s_raw = state_dict[s_key].numpy()
        # S_latent = exp(s_raw + 4.5) * 1e3
        S_latent_vals = np.exp(s_raw + 4.5) * 1e3
        print(f"  S_latent key: {s_key}")
        print(f"  S_latent shape: {s_raw.shape}")
        print(f"  S_latent mean: {S_latent_vals.mean():.0f}")
        print(f"  S_latent min:  {S_latent_vals.min():.0f}")
        print(f"  S_latent max:  {S_latent_vals.max():.0f}")
    else:
        print("  S_latent key not found in state dict")
        print("  Available keys:", list(state_dict.keys()))

# ── LOAD NAIVEPINN S_PRED (from predictions parquet) ──────────
S_pred_naive = None
if NAIVE_PRED and NAIVE_PRED.exists():
    df_naive = pd.read_parquet(str(NAIVE_PRED))
    df_naive = df_naive.sort_values(
        'time_original').reset_index(drop=True)
    S_pred_naive = df_naive['S_pred'].values
    print(f"\nNaivePINN S_pred mean: {S_pred_naive.mean():.0f}")

# ── LOAD TSIR-PINN S_PRED ─────────────────────────────────────
print("\nLoading TSIR-PINN S_pred...")
S_pred_tsir = None
if TSIR_PRED.exists():
    df_tsir = pd.read_parquet(str(TSIR_PRED))
    df_tsir = df_tsir.sort_values(
        'time_original').reset_index(drop=True)
    S_pred_tsir = df_tsir['S_pred'].values
    print(f"TSIR-PINN S_pred mean: {S_pred_tsir.mean():.0f}")
    print(f"TSIR-PINN S_pred min:  {S_pred_tsir.min():.0f}")
    print(f"TSIR-PINN S_pred max:  {S_pred_tsir.max():.0f}")
    print(f"S_pred/S_obs ratio:    "
          f"{S_pred_tsir.mean()/S_obs.mean():.4f}")
else:
    print(f"  Not found: {TSIR_PRED}")

# ── BUILD OUTPUT DATAFRAME ────────────────────────────────────
print("\nBuilding comparison dataframe...")

n = min(n_train,
        len(S_latent_vals) if S_latent_vals is not None else 9999,
        len(S_pred_naive)  if S_pred_naive  is not None else 9999,
        len(S_pred_tsir)   if S_pred_tsir   is not None else 9999)

rows = []
for i in range(n):
    row = {
        'biweek':      int(biweeks[i]),
        'time':        float(time_orig[i]),
        'S_obs':       float(S_obs[i]),
    }
    if S_latent_vals is not None and i < len(S_latent_vals):
        row['S_latent_naive'] = float(S_latent_vals[i])
    if S_pred_naive is not None and i < len(S_pred_naive):
        row['S_pred_naive']   = float(S_pred_naive[i])
    if S_pred_tsir is not None and i < len(S_pred_tsir):
        row['S_pred_tsir']    = float(S_pred_tsir[i])
    rows.append(row)

df_out = pd.DataFrame(rows)

# ── PRINT DIAGNOSTIC ──────────────────────────────────────────
print("\n" + "="*60)
print("SUSCEPTIBLE COMPARISON DIAGNOSTIC")
print("="*60)
print(f"{'Series':25} {'Mean':>10} {'Min':>10} {'Max':>10}")
print("-"*60)
print(f"{'S_obs (TSIR recon)':25} "
      f"{df_out['S_obs'].mean():>10.0f} "
      f"{df_out['S_obs'].min():>10.0f} "
      f"{df_out['S_obs'].max():>10.0f}")

if 'S_latent_naive' in df_out:
    print(f"{'S_latent (NaivePINN)':25} "
          f"{df_out['S_latent_naive'].mean():>10.0f} "
          f"{df_out['S_latent_naive'].min():>10.0f} "
          f"{df_out['S_latent_naive'].max():>10.0f}")
    ratio = df_out['S_latent_naive'].mean() / df_out['S_obs'].mean()
    print(f"  S_latent / S_obs ratio: {ratio:.3f}")
    if ratio < 0.5:
        print("  → S_latent COLLAPSED below S_obs")
    elif ratio > 2.0:
        print("  → S_latent EXPANDED above S_obs")
    else:
        print("  → S_latent roughly tracks S_obs")

if 'S_pred_tsir' in df_out:
    print(f"{'S_pred (TSIR-PINN)':25} "
          f"{df_out['S_pred_tsir'].mean():>10.0f} "
          f"{df_out['S_pred_tsir'].min():>10.0f} "
          f"{df_out['S_pred_tsir'].max():>10.0f}")
    ratio = df_out['S_pred_tsir'].mean() / df_out['S_obs'].mean()
    print(f"  S_pred / S_obs ratio: {ratio:.3f}")
    if ratio < 0.1:
        print("  → S_pred COLLAPSED to near zero ← MECHANISM CONFIRMED")
    elif ratio > 0.9:
        print("  → S_pred tracks S_obs (healthy convergence)")
    else:
        print(f"  → S_pred at {ratio:.0%} of S_obs")

# ── SAVE ──────────────────────────────────────────────────────
out_path = OUT_DIR / "s_latent_comparison.csv"
df_out.to_csv(str(out_path), index=False)
print(f"\nSaved: {out_path}")
print("Next: Rscript wb_s_latent_fig.R")
