# ============================================================
# wb_compare_models.py
# TSIR vs SFNN RMSE comparison — West Bengal measles
# ============================================================

import os
import numpy as np
import pandas as pd

BASE     = "/.../.../.../"
NN_DIR   = BASE + "output/data/basic_nn_optimal/"
TSIR_CSV = BASE + "output/data/basic_nn_optimal/tsir_preds_processed.csv"
OUT_DIR  = BASE + "output/data/comparison/"

os.makedirs(OUT_DIR, exist_ok=True)

k_values = [1, 4, 12, 20, 34]

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

    print(f"  k={k}: pred_cases [{test_out['pred_cases'].min():.1f}, "
          f"{test_out['pred_cases'].max():.1f}]")
    print(f"  k={k}: obs_cases  [{test_out['obs_cases'].min():.1f}, "
          f"{test_out['obs_cases'].max():.1f}]")

sfnn_final = pd.concat(sfnn_all, ignore_index=True)
sfnn_final.to_csv(OUT_DIR + "sfnn_predictions.csv", index=False)
print(f"\nSFNN predictions saved: {len(sfnn_final)} rows")

print("\n" + "=" * 60)
print("STEP 2: Loading TSIR predictions")
print("=" * 60)

tsir = pd.read_csv(TSIR_CSV)
tsir = tsir[tsir['k'].isin(k_values)].copy()
tsir = tsir.rename(columns={'tsir': 'tsir_cases'})
tsir = tsir.dropna(subset=['time'])

print(f"  TSIR rows: {len(tsir)}")
print(f"  k values:  {sorted(tsir['k'].unique().tolist())}")
print(f"  Districts: {tsir['city'].nunique()}")

print("\n" + "=" * 60)
print("STEP 3: Merging SFNN and TSIR predictions")
print("=" * 60)

sfnn_final['time_r'] = np.round(sfnn_final['time'], 4)
tsir['time_r']       = np.round(tsir['time'],       4)

merged = sfnn_final.merge(
    tsir[['time_r', 'city', 'k', 'tsir_cases']],
    on=['time_r', 'city', 'k'],
    how='inner'
)

print(f"  Merged rows: {len(merged)}")
print(f"  k values:    {sorted(merged['k'].unique().tolist())}")
print(f"  Districts:   {merged['city'].nunique()}")
for k in k_values:
    n = len(merged[merged['k'] == k])
    print(f"  k={k}: {n} rows (expected {19*54}={19*54})")

print("\n" + "=" * 60)
print("STEP 4: Computing RMSE")
print("=" * 60)

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

print("\n" + "=" * 60)
print("STEP 5: RMSE COMPARISON — SFNN vs TSIR")
print("=" * 60)

print(f"\n{'k':>4}  {'SFNN_RMSE':>10}  {'TSIR_RMSE':>10}  "
      f"{'Winner':>8}  {'improvement':>12}")
print("-" * 55)

summary = []
for k in k_values:
    df_k      = rmse_df[rmse_df['k'] == k]
    mean_sfnn = df_k['rmse_sfnn'].mean()
    mean_tsir = df_k['rmse_tsir'].mean()
    winner    = "SFNN" if mean_sfnn < mean_tsir else "TSIR"
    pct       = (mean_tsir - mean_sfnn) / mean_tsir * 100
    print(f"{k:>4}  {mean_sfnn:>10.3f}  {mean_tsir:>10.3f}  "
          f"{winner:>8}  {pct:>+11.1f}%")
    summary.append({
        'k':                   k,
        'mean_rmse_sfnn':      mean_sfnn,
        'mean_rmse_tsir':      mean_tsir,
        'winner':              winner,
        'sfnn_improvement_pct':pct,
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 60)
print("STEP 6: RMSE PER DISTRICT (k=1)")
print("=" * 60)

k1 = rmse_df[rmse_df['k'] == 1].sort_values('rmse_sfnn')
print(f"\n{'District':<35}  {'SFNN':>8}  {'TSIR':>8}  {'Winner':>8}")
print("-" * 65)
for _, row in k1.iterrows():
    winner = "SFNN" if row['rmse_sfnn'] < row['rmse_tsir'] else "TSIR"
    print(f"{row['city']:<35}  {row['rmse_sfnn']:>8.3f}  "
          f"{row['rmse_tsir']:>8.3f}  {winner:>8}")

rmse_df.to_csv(OUT_DIR + "rmse_comparison.csv",   index=False)
summary_df.to_csv(OUT_DIR + "rmse_summary.csv",   index=False)
merged.to_csv(OUT_DIR + "merged_predictions.csv", index=False)

print(f"\nSaved to: {OUT_DIR}")
print("  rmse_comparison.csv   — per district per k")
print("  rmse_summary.csv      — mean per k")
print("  merged_predictions.csv — all predictions")
