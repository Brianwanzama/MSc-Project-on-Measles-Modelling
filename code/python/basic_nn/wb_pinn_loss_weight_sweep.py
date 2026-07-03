"""
wb_pinn_loss_weight_sweep.py  (FIXED — no in-place autograd error)

Loss Weight Sensitivity Analysis for the PINN Instability Condition.

Runs TSIR-PINN across 8 lambda_I / lambda_ODE ratios:
    {1, 5, 10, 35, 100, 500, 1000, 10000}

For each ratio, 100 independent training runs.
Records RMSE, alpha1 saturation frequency, per-run test predictions.

FIX APPLIED:
    - forward() rebuilt using torch.cat instead of in-place
      index assignment — resolves autograd version error
    - S_tr, S_te, t_tr, t_te detached in train_one_run
    - clamp min changed from 0.0 to 1e-6 to keep gradients flowing

KEY OUTPUT FILES:
    loss_weight_sweep_summary.csv
    loss_weight_sweep_all_runs.csv
    runs_ratio_*.csv
    sweep_predictions_test.csv

Usage:
    conda activate finalmlenv
    python wb_pinn_loss_weight_sweep.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────
BASE     = Path("/home/brain/Msc_project/")
DATA_DIR = BASE / "output/data/basic_nn_optimal/"
OUT_DIR  = BASE / "experiments/tables/loss_weight_sweep/"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY     = "South Twenty Four Parganas"
K        = 34
N_RUNS   = 100
N_EPOCHS = 1000
LR       = 1e-3
DEVICE   = torch.device("cuda" if torch.cuda.is_available()
                         else "cpu")

print(f"Device: {DEVICE}")

# ── WEST BENGAL PARAMETERS ────────────────────────────────────
N_POP   = 8_438_494
S_BAR   = 240_617
BETA_EQ = N_POP / S_BAR    # = 35.07
TSIR_BM = 33.0              # TSIR V1V2 benchmark RMSE

print(f"beta_eq = {BETA_EQ:.3f}")

# ── LOSS WEIGHT RATIOS ────────────────────────────────────────
RATIOS = [1, 5, 10, 35, 100, 500, 1000, 10000]

# alpha1 saturation: tanh(alpha1) < -0.95
SAT_THRESH = -0.95


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_city_data(city, k):
    import pyarrow.parquet as pq

    df_out = pq.read_table(
        str(DATA_DIR / f"{k}_output.parquet")).to_pandas()
    df_tfm = pq.read_table(
        str(DATA_DIR / f"{k}_transform.parquet")).to_pandas()

    city_col = next(
        c for c in df_out.columns
        if "city" in c.lower() or "district" in c.lower())

    df = (df_out[df_out[city_col] == city]
          .merge(df_tfm[df_tfm[city_col] == city]
                 [["time", "cases_mean", "cases_std"]],
                 on="time")
          .sort_values("time")
          .reset_index(drop=True))

    df["I_raw"] = np.maximum(
        np.exp(df["cases"] * df["cases_std"]
               + df["cases_mean"]) - 1, 0)

    train = df[df["time"] < 2017.0]
    test  = df[df["time"] >= 2017.0]

    return (train["I_raw"].values.astype(np.float32),
            test["I_raw"].values.astype(np.float32),
            train["time"].values,
            test["time"].values)


def load_susceptibles(city):
    p = BASE / "output/data/tsir/tsir_susceptibles.csv"
    if not p.exists():
        print("  Susceptible CSV not found — using S_bar constant")
        return None
    df = pd.read_csv(p)
    city_col = next(c for c in df.columns if "city" in c.lower())
    return (df[df[city_col] == city]
            .sort_values("time")["susc"]
            .values.astype(np.float32))


# ═══════════════════════════════════════════════════════════════
# PINN MODEL
# FIX: forward() uses torch.cat instead of in-place assignment
# ═══════════════════════════════════════════════════════════════

class TSIRPINN(nn.Module):
    def __init__(self, beta_eq, n_pop):
        super().__init__()
        self.beta_eq = beta_eq
        self.n_pop   = n_pop
        self.nu      = nn.Parameter(torch.tensor(0.0))
        self.alpha1  = nn.Parameter(torch.tensor(0.0))
        self.alpha2  = nn.Parameter(torch.tensor(0.0))

    def beta_t(self, t):
        phase = 2 * np.pi * t / 26.0
        return (torch.sigmoid(self.nu) * self.beta_eq
                + torch.tanh(self.alpha1) * 5.0 * torch.sin(phase)
                + torch.tanh(self.alpha2) * 5.0 * torch.cos(phase))

    def forward(self, I_obs, S_fixed, t_idx):
        """
        FIX: Build I_pred as a list of step tensors then cat.
        This avoids in-place index assignment which breaks autograd.
        Each step is its own node in the computation graph.
        """
        beta = self.beta_t(t_idx)   # shape (T,)

        # Start from observed I at t=0
        steps = [I_obs[0:1]]        # list of 1-element tensors

        for j in range(1, len(I_obs)):
            nxt = torch.clamp(
                beta[j] * S_fixed[j - 1]
                * steps[-1][0].pow(0.98) / self.n_pop,
                min=1e-6              # avoid zero — keeps gradients
            )
            steps.append(nxt.unsqueeze(0))

        # Stack into a single tensor — no in-place ops
        I_pred = torch.cat(steps, dim=0)   # shape (T,)

        # ODE residuals for infectious compartment
        dI  = I_pred[1:] - I_pred[:-1]
        r_I = (dI
               - beta[1:] * S_fixed[1:] * I_pred[1:] / self.n_pop
               + I_pred[1:])

        return {"I_pred": I_pred, "r_I": r_I}


# ═══════════════════════════════════════════════════════════════
# SINGLE TRAINING RUN
# FIX: detach S and t tensors so they don't enter autograd graph
# ═══════════════════════════════════════════════════════════════

def train_one_run(model, I_tr, S_tr, t_tr,
                  I_te, S_te, t_te,
                  lam_I, lam_ODE, n_epochs, lr):

    opt = Adam(model.parameters(), lr=lr)

    # FIX: detach fixed inputs — they are data, not parameters
    S_tr = S_tr.detach()
    S_te = S_te.detach()
    t_tr = t_tr.detach()
    t_te = t_te.detach()

    # Normalisation constants from epoch 0
    model.train()
    with torch.no_grad():
        o0    = model(I_tr, S_tr, t_tr)
        n_I   = max(torch.mean((o0["I_pred"] - I_tr)**2).item(),
                    1e-8)
        n_ODE = max(torch.mean(o0["r_I"]**2).item(), 1e-8)

    for epoch in range(n_epochs):
        opt.zero_grad()
        o    = model(I_tr, S_tr, t_tr)
        loss = (lam_I   * torch.mean((o["I_pred"] - I_tr)**2) / n_I
              + lam_ODE * torch.mean(o["r_I"]**2) / n_ODE)
        loss.backward()
        opt.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        o_te    = model(I_te, S_te, t_te)
        pred    = o_te["I_pred"].cpu().numpy()
        rmse    = float(np.sqrt(
            np.mean((pred - I_te.cpu().numpy())**2)))
        a1      = model.alpha1.item()
        a1_tanh = float(torch.tanh(model.alpha1).item())

    return {
        "rmse":       rmse,
        "alpha1":     a1,
        "alpha1_tanh": a1_tanh,
        "nu":         model.nu.item(),
        "alpha2":     model.alpha2.item(),
        "pred_test":  pred,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(f"TSIR-PINN Loss Weight Sweep  |  beta_eq = {BETA_EQ:.2f}")
    print(f"Ratios: {RATIOS}  |  {N_RUNS} runs each")
    print("=" * 60 + "\n")

    # ── Load data ─────────────────────────────────────────────
    print("Loading data...")
    I_tr, I_te, t_tr_times, t_te_times = load_city_data(CITY, K)
    S    = load_susceptibles(CITY)
    n_tr = len(I_tr)
    n_te = len(I_te)

    if S is None:
        S = np.full(n_tr + n_te, S_BAR, dtype=np.float32)

    S_tr   = torch.tensor(S[:n_tr]).to(DEVICE)
    S_te   = torch.tensor(S[n_tr:n_tr + n_te]).to(DEVICE)
    I_tr_t = torch.tensor(I_tr).to(DEVICE)
    I_te_t = torch.tensor(I_te).to(DEVICE)
    t_tr_t = torch.tensor(
        (np.arange(1, n_tr + 1, dtype=np.float32) % 26) + 1
    ).to(DEVICE)
    t_te_t = torch.tensor(
        (np.arange(n_tr + 1, n_tr + n_te + 1,
                   dtype=np.float32) % 26) + 1
    ).to(DEVICE)

    print(f"Train: {n_tr} biweeks | Test: {n_te} biweeks\n")

    all_summaries = []
    all_runs      = []
    all_preds     = []

    # ── Sweep ─────────────────────────────────────────────────
    for ratio in RATIOS:
        lam_I   = 1.0
        lam_ODE = 1.0 / ratio

        print(f"Ratio {ratio:>6}  "
              f"(lam_I={lam_I:.4f}, lam_ODE={lam_ODE:.6f})")

        runs = []

        for run in range(N_RUNS):
            torch.manual_seed(run * 7 + ratio * 13)

            model = TSIRPINN(BETA_EQ, float(N_POP)).to(DEVICE)
            nn.init.normal_(model.nu,     0.0, 1.0)
            nn.init.normal_(model.alpha1, 0.0, 0.5)
            nn.init.normal_(model.alpha2, 0.0, 0.5)

            res = train_one_run(
                model,
                I_tr_t, S_tr, t_tr_t,
                I_te_t, S_te, t_te_t,
                lam_I, lam_ODE, N_EPOCHS, LR)

            # Store per-run test predictions for trajectory figure
            for t_val, pred_val, obs_val in zip(
                    t_te_times, res["pred_test"], I_te):
                all_preds.append({
                    "ratio":    ratio,
                    "run":      run,
                    "time":     float(t_val),
                    "pred_raw": float(pred_val),
                    "obs_raw":  float(obs_val),
                    "period":   "test",
                    "city":     CITY,
                })

            rec = {k: v for k, v in res.items()
                   if k != "pred_test"}
            rec.update({"ratio": ratio, "run": run})
            runs.append(rec)
            all_runs.append(rec)

            if (run + 1) % 25 == 0:
                rmses = [r["rmse"] for r in runs]
                sats  = [1 if r["alpha1_tanh"] < SAT_THRESH
                           else 0 for r in runs]
                print(f"  {run + 1:3d}/100 | "
                      f"Med RMSE={np.median(rmses):.2f} | "
                      f"Sat={np.mean(sats):.2f}")

        # ── Summary statistics ────────────────────────────────
        rmses  = np.array([r["rmse"]        for r in runs])
        a1vals = np.array([r["alpha1_tanh"] for r in runs])
        sf     = float(np.mean(a1vals < SAT_THRESH))

        smry = {
            "ratio":              ratio,
            "lambda_I":           lam_I,
            "lambda_ODE":         lam_ODE,
            "median_rmse":        float(np.median(rmses)),
            "mean_rmse":          float(np.mean(rmses)),
            "q10_rmse":           float(np.percentile(rmses, 10)),
            "q25_rmse":           float(np.percentile(rmses, 25)),
            "q75_rmse":           float(np.percentile(rmses, 75)),
            "q90_rmse":           float(np.percentile(rmses, 90)),
            "sat_frequency":      sf,
            "mean_alpha1_tanh":   float(np.mean(a1vals)),
            "median_alpha1_tanh": float(np.median(a1vals)),
            "n_runs":             N_RUNS,
            "beat_tsir_count":    int(np.sum(rmses < TSIR_BM)),
            "beat_tsir_frac":     float(np.mean(rmses < TSIR_BM)),
        }
        all_summaries.append(smry)

        # Save per-ratio run file immediately
        pd.DataFrame(runs).to_csv(
            OUT_DIR / f"runs_ratio_{ratio}.csv", index=False)

        print(f"  Done: RMSE={smry['median_rmse']:.2f} | "
              f"sat={smry['sat_frequency']:.3f} | "
              f"beats TSIR={smry['beat_tsir_frac']:.2f}\n")

    # ── Save all outputs ──────────────────────────────────────
    pd.DataFrame(all_summaries).to_csv(
        OUT_DIR / "loss_weight_sweep_summary.csv", index=False)
    pd.DataFrame(all_runs).to_csv(
        OUT_DIR / "loss_weight_sweep_all_runs.csv", index=False)
    pd.DataFrame(all_preds).to_csv(
        OUT_DIR / "sweep_predictions_test.csv", index=False)

    print("=" * 60)
    print("FILES SAVED:")
    print(f"  {OUT_DIR}/loss_weight_sweep_summary.csv")
    print(f"  {OUT_DIR}/loss_weight_sweep_all_runs.csv")
    print(f"  {OUT_DIR}/runs_ratio_*.csv")
    print(f"  {OUT_DIR}/sweep_predictions_test.csv")
    print("=" * 60)

    print(f"\n{'Ratio':>8} {'Med.RMSE':>10} "
          f"{'Sat.Freq':>10} {'Beats TSIR':>12}")
    print("-" * 45)
    for s in all_summaries:
        print(f"{s['ratio']:>8} {s['median_rmse']:>10.2f} "
              f"{s['sat_frequency']:>10.3f} "
              f"{s['beat_tsir_frac']:>12.3f}")

    print(f"\nbeta_eq = {BETA_EQ:.2f}")
    print("Theory predicts transition near ratio = 35")


if __name__ == "__main__":
    main()