# ============================================================
# data_process_explain.py
# Prepare X_train / X_test matrices for SHAP explanation
# West Bengal measles — aligned with final V1 SFNN
#
# CHANGES FROM ORIGINAL:
#   1. year_test_cutoff: 2016 -> 2017
#   2. Keep susc_lag_* columns (drop raw 'susc' only)
#   3. Keep nbc_lag_* and nc_*_lag_* columns
#      (WB model trained WITH these — must keep for SHAP)
#   4. Add 'split' to drop list
#   5. k=52 removed
#   6. Correct t_lag per k from Ray Tune results
# ============================================================

import argparse
import os
import sys

sys.path.insert(0,
    '/.../.../.../python/data_processing/')
sys.path.insert(0,
    '/.../.../.../code/python/basic_nn/')

import numpy as np
import pandas as pd


def process_data(cases, year_test_cutoff=2017):

    cases = cases.copy()
    cases['year'] = cases['time'].apply(lambda x: int(x))

    cases_train = cases[
        cases['year'] < year_test_cutoff].reset_index(drop=True)
    cases_test  = cases[
        cases['year'] >= year_test_cutoff].reset_index(drop=True)

    id_train = cases_train[['time', 'city']]
    id_test  = cases_test[['time', 'city']]

    y_train = cases_train['cases_trans'].to_numpy().reshape(-1, 1)
    y_test  = cases_test['cases_trans'].to_numpy().reshape(-1, 1)

    # ── SELECT FEATURES — identical regex to full_basic_functions.py
    # process_data() uses filter(regex="lag_|dist_")
    # Must use EXACT same selection so columns match trained model
    X_train = cases_train.filter(regex="lag_|dist_")
    X_test  = cases_test.filter(regex="lag_|dist_")

    # ── DIAGNOSTICS ────────────────────────────────────────
    v1_cols   = [c for c in X_train.columns if 'v1_lag'   in c]
    susc_cols = [c for c in X_train.columns if 'susc_lag' in c]
    nbc_cols  = [c for c in X_train.columns if 'nbc_lag'  in c]
    nc_cols   = [c for c in X_train.columns
                 if 'nc_' in c and '_lag_' in c]
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  v1_lag cols:   {len(v1_cols)}  -> {v1_cols}")
    print(f"  susc_lag cols: {len(susc_cols)}")
    print(f"  nbc_lag cols:  {len(nbc_cols)}")
    print(f"  nc_lag cols:   {len(nc_cols)}")
    assert X_train.shape[1] == X_test.shape[1], \
        f"Train/test feature mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"
    assert list(X_train.columns) == list(X_test.columns), \
        "Column name/order mismatch between train and test" 

    return X_train, y_train, X_test, y_test, id_train, id_test


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for SHAP explanation — WB SFNN')
    parser.add_argument('--k',
                        type=int, default=1)
    parser.add_argument('--t-lag',
                        type=int, default=26)
    parser.add_argument('--year-test-cutoff',
                        type=int, default=2017)
    parser.add_argument('--save-data-loc',
                        type=str,
                        default="/home/brain/Msc_project/"
                                "output/data/basic_nn_optimal/explain/")
    parser.add_argument('--cases-data-loc',
                        type=str,
                        default="/home/brain/Msc_project/"
                                "output/data/prefit_cases1/")
    args = parser.parse_args()

    os.makedirs(args.save_data_loc, exist_ok=True)

    parquet_path = (args.cases_data_loc
                    + f"k{args.k}_tlag{args.t_lag}.gzip")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Parquet not found: {parquet_path}")

    print(f"\nk={args.k} | tlag={args.t_lag} | "
          f"cutoff={args.year_test_cutoff}")

    cases = pd.read_parquet(parquet_path)

    X_train, y_train, X_test, y_test, \
        id_train, id_test = process_data(
            cases, args.year_test_cutoff)

    X_train.to_parquet(
        args.save_data_loc + f"{args.k}_X_train.parquet")
    X_test.to_parquet(
        args.save_data_loc + f"{args.k}_X_test.parquet")
    id_train.to_parquet(
        args.save_data_loc + f"{args.k}_id_train.parquet")
    id_test.to_parquet(
        args.save_data_loc + f"{args.k}_id_test.parquet")

    print(f"  Saved to: {args.save_data_loc}")


if __name__ == '__main__':
    main()
