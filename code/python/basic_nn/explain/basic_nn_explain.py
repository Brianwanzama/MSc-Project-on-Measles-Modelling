# ============================================================
# basic_nn_explain.py
# Shapley Value Sampling attribution for WB SFNN
#
# CHANGES FROM ORIGINAL:
#   1. k=52 default -> k=1
#   2. t_lag default: 104 -> 26
#   3. hidden_dim default: 721 -> 64
#   4. num_hidden_layers default: 3 -> 1
#   5. model_read_loc: output/models/ -> output/data/
#   6. fa_groups: added susc_lag group (group 10)
#                 kept nbc/nc groups (groups 11,12)
#                 v1_lag group updated to group 13
#                 group_size = t_lag - k + 1 (correct)
#   7. nc group size: (group_size + 1) * 10
#      -> group_size * 10 + 10 distances = correct for WB
# ============================================================

import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from captum.attr import ShapleyValueSampling

import numpy as np
import pandas as pd

import sys
sys.path.insert(0,
    '/.../.../.../code/python/basic_nn/')
import full_basic_functions as fbf


def main():
    parser = argparse.ArgumentParser(
        description='WB SFNN SHAP explanation')

    parser.add_argument('--batch-size',
                        type=int, default=64)
    parser.add_argument('--k',
                        type=int, default=1)
    parser.add_argument('--t-lag',
                        type=int, default=26)
    parser.add_argument('--hidden-dim',
                        type=int, default=64)
    parser.add_argument('--num-hidden-layers',
                        type=int, default=1)
    parser.add_argument('--write-data-loc',
                        type=str,
                        default="/home/brain/Msc_project/"
                                "output/data/basic_nn_optimal/explain/")
    parser.add_argument('--model-read-loc',
                        type=str,
                        # FIX: output/data/ not output/models/
                        default="/home/brain/Msc_project/"
                                "output/data/basic_nn_optimal/")
    parser.add_argument('--data-read-loc',
                        type=str,
                        default="/home/brain/Msc_project/"
                                "output/data/basic_nn_optimal/explain/")
    parser.add_argument('--verbose',
                        action='store_true', default=False)

    args = parser.parse_args()

    if args.verbose:
        print(f"\n{'='*55}")
        print(f"SHAP Explanation | k={args.k} | "
              f"tlag={args.t_lag}")
        print(f"hidden={args.hidden_dim} | "
              f"layers={args.num_hidden_layers}")
        print(f"{'='*55}\n")

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # ── LOAD DATA ────────────────────────────────────────────
    X_train  = pd.read_parquet(
        args.data_read_loc + f"{args.k}_X_train.parquet")
    X_test   = pd.read_parquet(
        args.data_read_loc + f"{args.k}_X_test.parquet")
    id_train = pd.read_parquet(
        args.data_read_loc + f"{args.k}_id_train.parquet")
    id_test  = pd.read_parquet(
        args.data_read_loc + f"{args.k}_id_test.parquet")

    num_features = X_train.shape[1]
    print(f"Features: {num_features}")

    # ── LOAD MODEL ───────────────────────────────────────────
    model = fbf.NeuralNetwork(
        input_dim         = num_features,
        hidden_dim        = args.hidden_dim,
        output_dim        = 1,
        num_hidden_layers = args.num_hidden_layers
    ).to(device)

    model_path = args.model_read_loc + f"{args.k}_model.pt"
    model.load_state_dict(
        torch.load(model_path,
                   map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}")

    # ── SHAP ─────────────────────────────────────────────────
    svs = ShapleyValueSampling(model)

    test_input_tensor  = torch.from_numpy(
        X_test.to_numpy()).float().to(device)
    train_input_tensor = torch.from_numpy(
        X_train.to_numpy()).float().to(device)

    # ── FEATURE GROUPS ────────────────────────────────────────
    # group_size = number of lag columns per feature series
    # = t_lag - k + 1
    # e.g. k=1, tlag=26: group_size = 26
    # e.g. k=4, tlag=52: group_size = 49
    group_size = args.t_lag - args.k + 1

    print(f"group_size: {group_size}")
    print(f"Columns in X_test ({X_test.shape[1]}):")
    for i, col in enumerate(X_test.columns):
        print(f"  {i:>4}: {col}")

    # Build fa_groups to match EXACT column order in X_test
    # after data_process_explain.py drops non-feature columns
    #
    # Column order in our parquet (after cleaning):
    #   [0]          cases_lag_k .. cases_lag_tlag   (group_size cols)
    #   [1]          pop_lag_*                        (1-2 cols)
    #   [2]          births_lag_*                     (1-2 cols)
    #   [3..9]       big_city_1..7 lags               (7 * group_size cols)
    #   [10]         big_city distances               (7 cols)
    #   [11]         nbc_lag_* (nearest big city lags)(group_size cols)
    #   [12]         nc_*_lag_* (10 nearest cities)   (10 * group_size cols)
    #   [12 cont]    nc distances                     (10 cols)
    #   [13]         susc_lag_*                       (group_size cols)
    #   [14]         v1_lag_*                         (2 cols)
    #
    # NOTE: actual column order may differ — the print above
    # will show you exactly. Adjust groups if assert fails.

    # Count actual columns per group from X_test
    cols = list(X_test.columns)

    own_cases   = [c for c in cols if c.startswith('cases_lag_')]
    pop_cols    = [c for c in cols if 'pop_lag_'    in c]
    birth_cols  = [c for c in cols if 'births_lag_' in c]
    susc_cols   = [c for c in cols if 'susc_lag_'   in c]
    v1_cols     = [c for c in cols if 'v1_lag_'     in c]
    nbc_cols    = [c for c in cols if 'nbc_lag_'    in c]
    nc_lag_cols = [c for c in cols
                   if 'nc_' in c and '_lag_' in c]
    dist_cols   = [c for c in cols if 'dist' in c]

    # Big city lag columns — all remaining lag cols
    assigned = set(own_cases + pop_cols + birth_cols
                   + susc_cols + v1_cols + nbc_cols
                   + nc_lag_cols + dist_cols)
    big_city_lag_cols = [c for c in cols
                         if c not in assigned
                         and '_lag_' in c]

    print(f"\nGroup sizes:")
    print(f"  own cases lags:    {len(own_cases)}")
    print(f"  pop lags:          {len(pop_cols)}")
    print(f"  births lags:       {len(birth_cols)}")
    print(f"  big city lags:     {len(big_city_lag_cols)}")
    print(f"  nbc lags:          {len(nbc_cols)}")
    print(f"  nc lags:           {len(nc_lag_cols)}")
    print(f"  dist cols:         {len(dist_cols)}")
    print(f"  susc lags:         {len(susc_cols)}")
    print(f"  v1 lags:           {len(v1_cols)}")
    total_assigned = (len(own_cases) + len(pop_cols)
                      + len(birth_cols) + len(big_city_lag_cols)
                      + len(nbc_cols) + len(nc_lag_cols)
                      + len(dist_cols) + len(susc_cols)
                      + len(v1_cols))
    print(f"  TOTAL:             {total_assigned} "
          f"(X_test has {len(cols)})")

    # Build group assignment in column order
    # Assign each column to a group number
    col_to_group = {}
    for c in own_cases:        col_to_group[c] = 0
    for c in pop_cols:         col_to_group[c] = 1
    for c in birth_cols:       col_to_group[c] = 2
    for c in big_city_lag_cols:col_to_group[c] = 3
    for c in dist_cols:        col_to_group[c] = 4
    for c in nbc_cols:         col_to_group[c] = 5
    for c in nc_lag_cols:      col_to_group[c] = 6
    for c in susc_cols:        col_to_group[c] = 7
    for c in v1_cols:          col_to_group[c] = 8

    # Assign any remaining unassigned columns
    for c in cols:
        if c not in col_to_group:
            print(f"  WARNING: unassigned column: {c}")
            col_to_group[c] = 9  # misc group

    fa_groups = np.array([col_to_group[c] for c in cols])
    fa_groups = torch.tensor(fa_groups).to(device)

    # ── ASSERT before running expensive SHAP ─────────────────
    assert len(fa_groups) == X_test.shape[1], \
        (f"fa_groups length {len(fa_groups)} != "
         f"X_test columns {X_test.shape[1]}")
    print(f"\nfa_groups check passed: {len(fa_groups)} == "
          f"{X_test.shape[1]}")

    distinct_idx  = np.unique(
        fa_groups.cpu().numpy(), return_index=True)[1]
    distinct_cols = X_test.columns[distinct_idx]
    print(f"Distinct groups: {len(distinct_idx)}")
    print(f"Group names: {list(distinct_cols)}")

    # ── RUN SHAP ─────────────────────────────────────────────
    print("\nRunning ShapleyValueSampling...")
    svs_attr_test = svs.attribute(
        test_input_tensor,
        feature_mask=fa_groups)
    svs_attr_test = svs_attr_test.reshape(
        svs_attr_test.shape[0], -1)

    svs_pd = pd.DataFrame(
        svs_attr_test.cpu().detach().numpy())
    svs_pd.columns = X_test.columns
    svs_pd = svs_pd[distinct_cols]
    svs_pd['city'] = id_test['city'].values
    svs_pd['time'] = id_test['time'].values

    out_path = (args.write_data_loc
                + f"{args.k}_svs_explain.csv")
    svs_pd.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {svs_pd.shape}")


if __name__ == '__main__':
    main()
