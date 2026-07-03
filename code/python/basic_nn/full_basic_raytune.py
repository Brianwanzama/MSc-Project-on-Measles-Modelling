#!/usr/bin/env python
# coding: utf-8
# ============================================================
# full_basic_raytune.py
# Ray Tune hyperparameter search for SFNN — West Bengal measles
#
# KEY FIXES vs original:
#   1. Parquet files loaded ONCE outside trial (100x faster)
#   2. DataLoaders created ONCE before epoch loop
#   3. tune.report called INSIDE epoch loop (enables ASHA)
#   4. ASHAScheduler added for early stopping
#   5. t_lag_options = [26, 52] for k<52 (hyperparameter search)
#      TSIR comparison uses only tlag=52 results afterward
#   6. cpu=2 per trial — allows 32 parallel trials on 64 cores
#   7. train/test split uses 'split' column from parquet
# ============================================================

import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray

# ── PATH SETUP ────────────────────────────────────────────────
original_sys_path      = sys.path.copy()
data_processing_path   = os.path.abspath(
    "/.../.../.../code/python/data_processing/"
)
functions_path = os.path.abspath(
    "/.../.../.../code/python/"
)
sys.path.append(data_processing_path)
sys.path.append(functions_path)

import prevac_measles_data_loader as mdl
sys.path = original_sys_path

import full_basic_functions as fbf

# ── RAY INIT ──────────────────────────────────────────────────
ray.init(runtime_env={
    "env_vars": {
        "PYTHONPATH": (
            str(data_processing_path) + ":" +
            str(functions_path)
        )
    }
})

BASE      = "/.../.../.../"
PREFIT    = BASE + "output/data/prefit_cases1/"


# ============================================================
# TRAINING FUNCTION
# Called once per Ray Tune trial
# Data is passed via config as a Ray object reference —
# loaded ONCE outside and shared across all trials
# ============================================================

def train_with_tuning(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── LOAD PRE-BUILT PARQUET — no feature recomputation ─────
    # FIX 1: Read parquet file built by data loader
    # This avoids rebuilding all spatial features per trial
    t_lag   = config["t_lag"]
    k       = config["k"]
    parquet = PREFIT + f"k{k}_tlag{t_lag}.gzip"
    transform_parquet = PREFIT + f"k{k}_tlag{t_lag}_cases_transform_output.gzip"

    cases          = pd.read_parquet(parquet)
    transform_data = pd.read_parquet(transform_parquet)

    # ── PROCESS DATA ──────────────────────────────────────────
    train_data, test_data, num_features, id_train, id_test = fbf.process_data(
        cases, config["year_test_cutoff"]
    )

    # ── BUILD MODEL ───────────────────────────────────────────
    model = fbf.NeuralNetwork(
        input_dim          = num_features,
        hidden_dim         = config["hidden_dim"],
        output_dim         = 1,
        num_hidden_layers  = config["num_hidden_layers"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"]
    )
    loss_fn = nn.MSELoss()

    # FIX 2: Create DataLoaders ONCE before epoch loop
    train_loader, test_loader = fbf.get_dataloaders(
        train_data, test_data, batch_size=32
    )

    # ── TRAINING LOOP ─────────────────────────────────────────
    # FIX 3: tune.report called INSIDE epoch loop
    # This allows ASHAScheduler to stop bad trials early
    for epoch in range(config["num_epochs"]):

        train_loss = fbf.train(
            model        = model,
            device       = device,
            train_loader = train_loader,
            optimizer    = optimizer,
            loss_fn      = loss_fn,
            epoch        = epoch,
            log_interval = 9999   # suppress per-batch printing in Ray
        )

        test_loss = fbf.test(
            model       = model,
            device      = device,
            test_loader = test_loader,
            loss_fn     = loss_fn
        )

        # Report every epoch — required for ASHA early stopping
        tune.report({
            "train_mse": float(train_loss),
            "test_mse":  float(test_loss),
            "epoch":     epoch
        })


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Ray Tune hyperparameter search for SFNN")
    parser.add_argument("--k",              type=int,   default=1,
                        help="Forecast horizon in biweeks")
    parser.add_argument("--num-samples",    type=int,   default=20,
                        help="Random samples for lr and weight_decay")
    parser.add_argument("--max-num-epochs", type=int,   default=50,
                        help="Maximum training epochs per trial")
    parser.add_argument("--gpus-per-trial", type=float, default=0,
                        help="GPUs per trial (0 = CPU only)")
    args = parser.parse_args()

    # ── t_lag OPTIONS ─────────────────────────────────────────
    # Both tlag=26 and tlag=52 files were produced by data loader
    # Ray Tune searches over both — selects best per k
    # For TSIR comparison: use only tlag=52 result afterward
    # k must be strictly less than t_lag (data loader constraint)
    if args.k < 26:
        t_lag_options = [26, 52]   # k=1,4,12,20 → both options
    elif args.k < 52:
        t_lag_options = [52]       # k=34 → only 52 (34 > 26)
    else:
        t_lag_options = []         # k=52 → no valid t_lag <= 52
        print(f"k={args.k} has no valid t_lag <= 52. Exiting.")
        return

    # Verify parquet files exist before launching trials
    print(f"\nVerifying parquet files for k={args.k}...")
    missing = []
    for t_lag in t_lag_options:
        path = PREFIT + f"k{args.k}_tlag{t_lag}.gzip"
        if os.path.exists(path):
            print(f"  k{args.k}_tlag{t_lag}.gzip — OK")
        else:
            print(f"  k{args.k}_tlag{t_lag}.gzip — MISSING")
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            f"Missing parquet files: {missing}\n"
            f"Run prevac_measles_data_loader.py first."
        )

    # ── HYPERPARAMETER CONFIG ─────────────────────────────────
    config = {
        # Fixed
        "k":                args.k,
        "year_test_cutoff": 2017,
        "num_epochs":       args.max_num_epochs,

        # Grid search — all combinations tried
        "t_lag":            tune.grid_search(t_lag_options),
        "hidden_dim":       tune.grid_search([64, 128, 240]),
        "num_hidden_layers":tune.grid_search([1, 2, 3]),

        # Random search — sampled num_samples times
        "lr":               tune.loguniform(1e-4, 1e-2),
        "weight_decay":     tune.uniform(1e-4, 1e-1),
    }

    # Total trials = grid combinations × num_samples
    n_grid = len(t_lag_options) * 3 * 3   # t_lag × hidden_dim × layers
    n_total = n_grid * args.num_samples
    print(f"\nGrid combinations:  {n_grid}")
    print(f"Random samples:     {args.num_samples}")
    print(f"Total trials:       {n_total}")
    print(f"Parallel trials:    {64 // 2} (64 cores ÷ 2 per trial)")
    print(f"Max epochs:         {args.max_num_epochs}")

    # ── ASHA SCHEDULER — early stopping ───────────────────────
    # FIX 4: Stop bad trials early based on test_mse
    # grace_period: minimum epochs before a trial can be stopped
    # reduction_factor: top 1/4 trials survive each round
    scheduler = ASHAScheduler(
        metric           = "test_mse",
        mode             = "min",
        max_t            = args.max_num_epochs,
        grace_period     = 5,
        reduction_factor = 4
    )

    # ── RUN TUNE ──────────────────────────────────────────────
    # FIX 5: cpu=2 per trial — allows 32 parallel trials
    result = tune.run(
        train_with_tuning,
        resources_per_trial = {"cpu": 2, "gpu": args.gpus_per_trial},
        config              = config,
        num_samples         = args.num_samples,
        scheduler           = scheduler,
        verbose             = 1,          # show trial progress
        raise_on_failed_trial = False     # continue if one trial fails
    )

    # ── SAVE RESULTS ──────────────────────────────────────────
    rows = []
    for trial in result.trials:
        if trial.last_result and "test_mse" in trial.last_result:
            row = {
                "trial_id":         trial.trial_id,
                "test_mse":         trial.last_result["test_mse"],
                "train_mse":        trial.last_result.get("train_mse", None),
                "epoch":            trial.last_result.get("epoch", None),
                "t_lag":            trial.config.get("t_lag"),
                "hidden_dim":       trial.config.get("hidden_dim"),
                "num_hidden_layers":trial.config.get("num_hidden_layers"),
                "lr":               trial.config.get("lr"),
                "weight_decay":     trial.config.get("weight_decay"),
                "k":                args.k,
            }
            rows.append(row)

    if not rows:
        raise RuntimeError(
            "No successful Ray Tune trials. "
            "Check error logs in ~/ray_results/"
        )

    df = pd.DataFrame(rows).sort_values("test_mse").reset_index(drop=True)

    save_dir = BASE + "output/data/raytune_hp_optim/"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"raytune_hp_optim_k_{args.k}.csv")
    df.to_csv(save_path, index=False)

    # ── PRINT BEST RESULT ─────────────────────────────────────
    best = df.iloc[0]
    print(f"\n{'='*55}")
    print(f"BEST RESULT FOR k={args.k}")
    print(f"{'='*55}")
    print(f"  test_mse:          {best['test_mse']:.6f}")
    print(f"  train_mse:         {best['train_mse']:.6f}")
    print(f"  t_lag:             {best['t_lag']}")
    print(f"  hidden_dim:        {best['hidden_dim']}")
    print(f"  num_hidden_layers: {best['num_hidden_layers']}")
    print(f"  lr:                {best['lr']:.6f}")
    print(f"  weight_decay:      {best['weight_decay']:.6f}")
    print(f"  epochs completed:  {best['epoch']}")
    print(f"\nResults saved: {save_path}")
    print(f"Total successful trials: {len(df)}")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    main()
