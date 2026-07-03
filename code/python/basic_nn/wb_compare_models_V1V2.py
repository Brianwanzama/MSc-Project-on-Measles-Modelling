# ============================================================
# wb_compare_models_V1V2.py
# TSIR vs SFNN RMSE comparison — West Bengal measles
# Uses V1+V2 TSIR predictions (Ferrari et al. 2012 Eq. 3)
#
# NOTE: SFNN predictions unchanged — SFNN already has
#       v1_lag features and susc from V1-only reconstruction.
#       We compare against V1V2 TSIR only here.
#       A full SFNN rerun with V1V2 susceptibles would require
#       rebuilding parquet files — done separately if needed.
# ============================================================

import os
import numpy as np
import pandas as pd

BASE     = "/.../.../.../"
NN_DIR   = BASE + "output/data/basic_nn_optimal/"
TSIR_CSV = BASE + "output/data/basic_nn_optimal/tsir_preds_processed_V1V2.csv"
OUT_DIR  = BASE + "output/data/comparison_V1V2/"

os.makedirs(OUT_DIR, exist_ok=True)

k_values = [1, 4, 12, 20, 34]

# ============================================================
# STEP 1: LOAD AND REVERSE-TRANSFORM SFNN PREDICTIONS
# ============================================================

print("=" * 60)
print("STEP 1: Loading SFNN predictions")
print("=" * 60)

sfnn_all = []

for k in k_values:

    output_path    = NN_DIR + f"{k}_output.parquet"
    transform_path = NN_DIR + f"{k}_transform.parquet"

    if not os.path.exists(output_path):
        print(f"  k={k}: MISSING {output_path}")
        continue

    output    = pd.read_parquet(output_path)
    transform = pd.read_parquet(transform_path)

    print(f"\n  k={k}: {len(output)} rows | "
          f"train={(output['train_test']=='train').sum()} | "
          f"test={(output['train_test']=='test').sum()}")

    output = output.merge(
        transform[['time', 'city', 'cases_mean', 'cases_std']],
        on=['time', 'city'], how='left'
    )

    output['pred_log']   = (output['pred'] *
                             output['cases_std'] +
                             output['cases_mean'])
    output['pred_cases'] = np.exp(output['pred_log']) - 1
    output['pred_cases'] = output['pred_cases'].clip(lower=0)

    output['obs_log']   = (output['cases'] *
                            output['cases_std'] +
                            output['cases_mean'])
    output['obs_cases'] = np.exp(output['obs_log']) - 1
    output['obs_cases'] = output['obs_cases'].clip(lower=0)

    output['k'] = k
    test_out = output[output['train_test'] == 'test'].copy()
    sfnn_all.append(test_out)

    print(f"  k={k}: pred [{test_out['pred_cases'].min():.1f}, "
          f"{test_out['pred_cases'].max():.1f}]")

sfnn_final = pd.concat(sfnn_all, ignore_index=True)
print(f"\nSFNN predictions: {len(sfnn_final)} rows")

# ============================================================
# STEP 2: LOAD V1V2 TSIR PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Loading V1+V2 TSIR predictions")
print("=" * 60)

tsir = pd.read_csv(TSIR_CSV)
tsir = tsir[tsir['k'].isin(k_values)].copy()
tsir = tsir.rename(columns={'tsir': 'tsir_cases'})
tsir = tsir.dropna(subset=['time'])

print(f"  TSIR rows: {len(tsir)}")
print(f"  k values:  {sorted(tsir['k'].unique().tolist())}")
print(f"  Districts: {tsir['city'].nunique()}")

# ============================================================
# STEP 3: MERGE
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Merging")
print("=" * 60)

sfnn_final['time_r'] = np.round(sfnn_final['time'], 4)
tsir['time_r']       = np.round(tsir['time'],       4)

merged = sfnn_final.merge(
    tsir[['time_r', 'city', 'k', 'tsir_cases']],
    on=['time_r', 'city', 'k'],
    how='inner'
)

print(f"  Merged rows: {len(merged)}")
for k in k_values:
    n = len(merged[merged['k'] == k])
    print(f"  k={k}: {n} rows")

# ============================================================
# STEP 4: COMPUTE RMSE
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

rmse_results = []

for k in k_values:
    df_k = merged[merged['k'] == k]
    for city in sorted(df_k['city'].unique()):
        df_c = df_k[df_k['city'] == city]
        rmse_sfnn = rmse(df_c['obs_cases'].values,
                         df_c['pred_cases'].values)
        rmse_tsir = rmse(df_c['obs_cases'].values,
                         df_c['tsir_cases'].values)
        rmse_results.append({
            'k':         k,
            'city':      city,
            'rmse_sfnn': rmse_sfnn,
            'rmse_tsir': rmse_tsir,
            'n_obs':     len(df_c),
        })

rmse_df = pd.DataFrame(rmse_results)

# ============================================================
# STEP 5: SUMMARY — V1V2 TSIR vs SFNN
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: RMSE COMPARISON — SFNN vs TSIR (V1+V2)")
print("=" * 60)

print(f"\n{'k':>4}  {'SFNN_RMSE':>10}  {'TSIR_V1V2':>10}  "
      f"{'Winner':>8}  {'improvement':>12}")
print("-" * 55)

summary_v1v2 = []
for k in k_values:
    df_k      = rmse_df[rmse_df['k'] == k]
    mean_sfnn = df_k['rmse_sfnn'].mean()
    mean_tsir = df_k['rmse_tsir'].mean()
    winner    = "SFNN" if mean_sfnn < mean_tsir else "TSIR"
    pct       = (mean_tsir - mean_sfnn) / mean_tsir * 100
    print(f"{k:>4}  {mean_sfnn:>10.3f}  {mean_tsir:>10.3f}  "
          f"{winner:>8}  {pct:>+11.1f}%")
    summary_v1v2.append({
        'k': k, 'mean_rmse_sfnn': mean_sfnn,
        'mean_rmse_tsir_V1V2': mean_tsir,
        'winner': winner, 'sfnn_improvement_pct': pct
    })

# ============================================================
# STEP 6: COMPARE V1-ONLY vs V1V2 TSIR
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: V1-only TSIR vs V1+V2 TSIR comparison")
print("=" * 60)

tsir_v1_csv = BASE + "output/data/basic_nn_optimal/tsir_preds_processed.csv"
if os.path.exists(tsir_v1_csv):
    tsir_v1 = pd.read_csv(tsir_v1_csv)
    tsir_v1 = tsir_v1[tsir_v1['k'].isin(k_values)].copy()
    tsir_v1 = tsir_v1.rename(columns={'tsir': 'tsir_v1_cases'})
    tsir_v1 = tsir_v1.dropna(subset=['time'])
    tsir_v1['time_r'] = np.round(tsir_v1['time'], 4)

    merged_both = sfnn_final.merge(
        tsir[['time_r','city','k','tsir_cases']],
        on=['time_r','city','k'], how='inner'
    ).merge(
        tsir_v1[['time_r','city','k','tsir_v1_cases']],
        on=['time_r','city','k'], how='inner'
    )

    print(f"\n{'k':>4}  {'TSIR_V1':>10}  {'TSIR_V1V2':>10}  "
          f"{'SFNN':>10}  {'V1V2_better?':>13}")
    print("-" * 55)

    for k in k_values:
        df_k = merged_both[merged_both['k'] == k]
        r_v1   = rmse(df_k['obs_cases'].values,
                      df_k['tsir_v1_cases'].values)
        r_v1v2 = rmse(df_k['obs_cases'].values,
                      df_k['tsir_cases'].values)
        r_sfnn = rmse(df_k['obs_cases'].values,
                      df_k['pred_cases'].values)
        better = "YES" if r_v1v2 < r_v1 else "no"
        print(f"{k:>4}  {r_v1:>10.3f}  {r_v1v2:>10.3f}  "
              f"{r_sfnn:>10.3f}  {better:>13}")

# ============================================================
# STEP 7: SAVE
# ============================================================

summary_df = pd.DataFrame(summary_v1v2)
rmse_df.to_csv(OUT_DIR + "rmse_comparison_V1V2.csv",   index=False)
summary_df.to_csv(OUT_DIR + "rmse_summary_V1V2.csv",   index=False)
merged.to_csv(OUT_DIR + "merged_predictions_V1V2.csv", index=False)

print(f"\nSaved to: {OUT_DIR}")
print("  rmse_comparison_V1V2.csv")
print("  rmse_summary_V1V2.csv")
print("  merged_predictions_V1V2.csv")
