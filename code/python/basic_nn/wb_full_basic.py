# ============================================================
# wb_full_basic.py
# Final SFNN training with best Ray Tune hyperparameters
# West Bengal measles — all 19 districts
# Adapted from full_basic.py (Madden et al. 2024)
#
# KEY CHANGES FROM ORIGINAL:
#   1. Reads pre-built parquet — no create_measles_data() call
#   2. Paths       -> /home/brain/Msc_project/
#   3. year_test_cutoff -> 2017
#   4. Uses 'split' column for train/test
#   5. Saves predictions with time/city alignment
#   6. predict() added for RMSE comparison vs TSIR
# ============================================================

import argparse
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────
BASE    = "/.../.../.../"
PREFIT  = BASE + "output/data/prefit_cases1/"
OUTDIR  = BASE + "output/data/basic_nn_optimal/"

sys.path.append(BASE + "code/python/basic_nn/")
import full_basic_functions as fbf


def main():

    parser = argparse.ArgumentParser(
        description="WB SFNN final training with best hyperparameters")

    parser.add_argument("--k",                type=int,   default=1)
    parser.add_argument("--t-lag",            type=int,   default=52)
    parser.add_argument("--hidden-dim",       type=int,   default=64)
    parser.add_argument("--num-hidden-layers",type=int,   default=1)
    parser.add_argument("--lr",               type=float, default=0.001)
    parser.add_argument("--weight-decay",     type=float, default=0.01)
    parser.add_argument("--num-epochs",       type=int,   default=200)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--year-test-cutoff", type=int,   default=2017)
    parser.add_argument("--log-interval",     type=int,   default=100)
    parser.add_argument("--save-model",       action="store_true", default=False)
    parser.add_argument("--save-data-loc",    type=str,   default=OUTDIR)
    parser.add_argument("--prefit-dir",       type=str,   default=PREFIT,
                        help="directory containing prefit parquet files")
    parser.add_argument("--no-cuda",          action="store_true", default=False)
    parser.add_argument("--seed",             type=int,   default=1)
    parser.add_argument("--verbose",          action="store_true", default=False)
    parser.add_argument("--dry-run",          action="store_true", default=False)

    args = parser.parse_args()

    os.makedirs(args.save_data_loc, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if not args.no_cuda
        and torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"SFNN Final Training | k={args.k} | tlag={args.t_lag}")
    print(f"hidden={args.hidden_dim} | layers={args.num_hidden_layers}")
    print(f"lr={args.lr:.6f} | wd={args.weight_decay:.6f}")
    print(f"epochs={args.num_epochs} | device={device}")
    print(f"{'='*55}\n")

    # ── LOAD PRE-BUILT PARQUET ─────────────────────────────────
    prefit_dir     = args.prefit_dir.rstrip("/") + "/"
    parquet_path   = prefit_dir + f"k{args.k}_tlag{args.t_lag}.gzip"
    transform_path = prefit_dir + f"k{args.k}_tlag{args.t_lag}_cases_transform_output.gzip"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Parquet not found: {parquet_path}\n"
            f"Run prevac_measles_data_loader.py first.")

    print(f"Loading: {parquet_path}")
    cases          = pd.read_parquet(parquet_path)
    transform_data = pd.read_parquet(transform_path)

    print(f"  Rows: {len(cases)} | "
          f"Train: {(cases['split']=='train').sum()} | "
          f"Test:  {(cases['split']=='test').sum()}")

    # ── PROCESS DATA ──────────────────────────────────────────
    train_data, test_data, num_features, id_train, id_test = \
        fbf.process_data(cases, args.year_test_cutoff)

    train_loader, test_loader = fbf.get_dataloaders(
        train_data, test_data, batch_size=args.batch_size)

    print(f"  Features: {num_features}")

    # ── BUILD MODEL ───────────────────────────────────────────
    model = fbf.NeuralNetwork(
        input_dim         = num_features,
        hidden_dim        = args.hidden_dim,
        output_dim        = 1,
        num_hidden_layers = args.num_hidden_layers
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay)

    loss_fn = nn.MSELoss()

    # ── TRAINING LOOP ─────────────────────────────────────────
    train_losses = []
    test_losses  = []

    for epoch in range(1, args.num_epochs + 1):

        train_loss_i = fbf.train(
            model        = model,
            device       = device,
            train_loader = train_loader,
            optimizer    = optimizer,
            loss_fn      = loss_fn,
            epoch        = epoch,
            log_interval = args.log_interval
        )
        train_losses.append(train_loss_i)

        test_loss_i = fbf.test(
            model       = model,
            device      = device,
            test_loader = test_loader,
            loss_fn     = loss_fn
        )
        test_losses.append(test_loss_i)

        if args.dry_run:
            break

    print(f"\nFinal train MSE: {train_losses[-1]:.6f}")
    print(f"Final test  MSE: {test_losses[-1]:.6f}")

    # ── PREDICTIONS ───────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_train = model(
            train_data.X.to(device)).cpu().detach().numpy()
        pred_test  = model(
            test_data.X.to(device)).cpu().detach().numpy()

    train_mse = np.mean(
        (pred_train - train_data.y.detach().numpy())**2)
    test_mse  = np.mean(
        (pred_test  - test_data.y.detach().numpy())**2)

    if args.verbose:
        print(f"Train MSE (manual): {train_mse:.6f}")
        print(f"Test  MSE (manual): {test_mse:.6f}")

    # ── SAVE ──────────────────────────────────────────────────
    if args.save_model:
        stem = args.save_data_loc + str(args.k)

        # Save model weights
        torch.save(model.state_dict(), stem + "_model.pt")

        # Save predictions with time/city labels
        # Needed for RMSE comparison vs TSIR
        id_train_out = id_train.copy()
        id_test_out  = id_test.copy()
        id_train_out['train_test'] = 'train'
        id_test_out['train_test']  = 'test'

        output = pd.concat(
            [id_train_out, id_test_out], ignore_index=True)
        output['pred']  = np.concatenate(
            [pred_train.flatten(), pred_test.flatten()])
        output['cases'] = np.concatenate([
            train_data.y.detach().numpy().flatten(),
            test_data.y.detach().numpy().flatten()
        ])

        output.to_parquet(stem + "_output.parquet")
        transform_data.to_parquet(stem + "_transform.parquet")

        # Save loss curves
        loss_df = pd.DataFrame({
            'epoch':      list(range(1, len(train_losses)+1)),
            'train_loss': train_losses,
            'test_loss':  test_losses
        })
        loss_df.to_csv(stem + "_loss.csv", index=False)

        # Save summary
        summary = pd.DataFrame([{
            'k':                args.k,
            't_lag':            args.t_lag,
            'hidden_dim':       args.hidden_dim,
            'num_hidden_layers':args.num_hidden_layers,
            'lr':               args.lr,
            'weight_decay':     args.weight_decay,
            'num_epochs':       args.num_epochs,
            'train_mse':        float(train_mse),
            'test_mse':         float(test_mse),
        }])
        summary.to_csv(stem + "_summary.csv", index=False)

        print(f"\nSaved:")
        print(f"  {stem}_model.pt")
        print(f"  {stem}_output.parquet")
        print(f"  {stem}_transform.parquet")
        print(f"  {stem}_loss.csv")
        print(f"  {stem}_summary.csv")


if __name__ == "__main__":
    main()
