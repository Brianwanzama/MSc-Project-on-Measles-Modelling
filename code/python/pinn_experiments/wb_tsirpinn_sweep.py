# ============================================================
# wb_tsirpinn_sweep.py
#
# Madden et al. (2024) TSIR-PINN — West Bengal adaptation
# Loss Weight Sensitivity Sweep
#
# BASE: wb_tsirpinn.py (your WB extension of Madden original)
#
# CHANGES FROM wb_tsirpinn.py (3 additions only):
#   1. --lambda-i, --lambda-ode, --ratio-label arguments
#      so the sweep can vary the loss weight ratio
#   2. S_hp, I_hp, ode_hp driven by args instead of hardcoded
#      (default values identical to original: 0.1, 10, 1)
#   3. Per-epoch logging + run_summary.parquet save
#
# THE MODEL IS COMPLETELY UNCHANGED:
#   - beta(t) = vert + amp1*sin + amp2*cos [unconstrained]
#   - S = SI[:,0:1] from network prediction
#   - Full Jacobian with create_graph=True
#   - loss = loss_S*S_hp + loss_I*I_hp + loss_ode*ode_hp
#   - Adam model (lr=wd_fnn), Adam ode_model (lr=0.1)
#
# SCIENTIFIC PURPOSE:
#   Show that Madden's original TSIR-PINN fails in
#   post-vaccination WB regardless of loss weight ratio,
#   because the structural S/I scale mismatch
#   (S_bar/I_bar ~ 4185) breaks the gradient coupling
#   that would allow the ratio to control training.
#
#   At all ratios: vert diverges, RMSE stays high
#   This is the structural failure — not a calibration issue
#
# THREE RATIOS:
#   10    Madden default  (below beta_eq=35.07)
#   35    at boundary
#   10000 well above boundary
#
# VERIFIED: beta_eq = N/S_bar = 8438494/240617 = 35.07
# ============================================================

import argparse
import os
import math
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

BASE   = "/home/brain/Msc_project/"
PREFIT = BASE + "output/data/prefit_cases1/"
WRITE  = BASE + "output/models/pinn_experiments/wb_pinn_sweep/"

BETA_EQ     = 35.07
LOG_BETA_EQ = float(np.log(BETA_EQ))   # 3.5573


# ═══════════════════════════════════════════════════════════════
# DATA LOADING  (identical to wb_tsirpinn.py)
# ═══════════════════════════════════════════════════════════════

def readin_data(k=34, tlag=52, year_test_cutoff=2017.0):
    path    = PREFIT + f"k{k}_tlag{tlag}.gzip"
    full_df = pd.read_parquet(path)
    if 'split' in full_df.columns:
        train_df = full_df[full_df['split'] == 'train'].copy()
        test_df  = full_df[full_df['split'] == 'test'].copy()
    else:
        train_df = full_df[
            full_df['time'] <  year_test_cutoff].copy()
        test_df  = full_df[
            full_df['time'] >= year_test_cutoff].copy()
    return train_df, test_df


def get_cities(data, cities):
    return data[data['city'].isin(cities)].copy()


def process(data):
    data = data.copy()
    data['time_original'] = data['time']
    data['time'] = np.round(
        (data['time'] - 2008.0) * 26).astype(int) + 1
    return data


def get_data(data):
    case_cols = [c for c in data.columns if "cases_lag_" in c]
    susc_cols = [c for c in data.columns if "susc_lag_"  in c]
    return data[["time", "susc", "cases", "births", "pop"]
                + case_cols + susc_cols]


def get_X_y(data):
    S  = data['susc'].to_numpy().reshape(-1, 1)
    I  = data['cases'].to_numpy().reshape(-1, 1)
    t  = data['time'].to_numpy().reshape(-1, 1)
    N  = data['pop'].to_numpy().reshape(-1, 1)
    Bi = data['births'].to_numpy().reshape(-1, 1)
    X  = data.drop(
            columns=['cases', 'time', 'susc', 'pop', 'births']
         ).to_numpy()
    return S, I, t, N, Bi, X


class Data(Dataset):
    def __init__(self, t, S, I, Bi, N, X):
        self.t     = torch.from_numpy(t).float().reshape(1,-1).t()
        self.t_ode = torch.from_numpy(t).float().reshape(1,-1).t()
        self.S     = torch.from_numpy(S).float().reshape(1,-1).t()
        self.I     = torch.from_numpy(I).float().reshape(1,-1).t()
        self.Bi    = torch.from_numpy(Bi).float().reshape(1,-1).t()
        self.N     = torch.from_numpy(N).float().reshape(1,-1).t()
        self.X     = torch.from_numpy(X).float()
        self.len   = self.t.shape[0]

    def __getitem__(self, index):
        return (self.t[index], self.t_ode[index],
                self.S[index], self.I[index],
                self.Bi[index], self.N[index],
                self.X[index])

    def __len__(self):
        return self.len


# ═══════════════════════════════════════════════════════════════
# NETWORK  (identical to wb_tsirpinn.py)
# ═══════════════════════════════════════════════════════════════

class fourier_map(nn.Module):
    def __init__(self, size_in, B):
        super().__init__()
        self.size_B   = B.shape[1]
        self.B        = B
        self.size_in  = size_in * self.size_B * 2
        self.size_out = self.size_in
        self.weights  = nn.Parameter(
            torch.Tensor(self.size_out, self.size_in))
        self.bias     = nn.Parameter(
            torch.Tensor(self.size_out))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                        self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x_sin = torch.sin(torch.mm(x, self.B))
        x_cos = torch.cos(torch.mm(x, self.B))
        x     = torch.cat((x_sin, x_cos), 1)
        return torch.add(
            torch.mm(x, self.weights.t()), self.bias)


def seasonal_beta_torch(t, vert, amp1, amp2, T):
    """Identical to wb_tsirpinn.py — unconstrained"""
    return (vert
            + amp1 * torch.sin(2 * torch.pi * (t / T))
            + amp2 * torch.cos(2 * torch.pi * (t / T)))


class derivative_layer(nn.Module):
    """Identical to wb_tsirpinn.py — S from network pred"""
    def __init__(self, T):
        super().__init__()
        self.T    = T
        self.vert = nn.Parameter(torch.randn(1))
        self.amp1 = nn.Parameter(torch.randn(1))
        self.amp2 = nn.Parameter(torch.randn(1))

    def forward(self, t, SI, Bi, N):
        S    = SI[:, 0:1]    # network-predicted S
        I    = SI[:, 1:2]    # network-predicted I
        beta = seasonal_beta_torch(
                   t, self.vert, self.amp1, self.amp2, self.T)
        der_S = Bi - (beta * S * I) / N
        der_I = (beta * S * I) / N - I
        return torch.cat((der_S, der_I), 1)


class ode_nn(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.der = derivative_layer(T)

    def forward(self, t, SI, Bi, N):
        return self.der(t, SI, Bi, N)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim_t, input_dim_X,
                 hidden_dim, output_dim, B):
        super().__init__()
        self.gelu     = nn.GELU()
        self.softplus = nn.Softplus()
        self.fm = fourier_map(input_dim_t, B)
        self.f1 = nn.Linear(B.shape[1]*2 + input_dim_X,
                             hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        self.f6 = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, X):
        t_B = self.gelu(self.fm(t))
        x   = self.gelu(self.f1(torch.cat((t_B, X), dim=1)))
        x   = self.gelu(self.f2(x))
        x   = self.gelu(self.f3(x))
        x   = self.softplus(self.f6(x))
        return x


def get_B(num_features=50, scale=1):
    return torch.FloatTensor(
        np.random.randn(1, num_features) * scale)


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def get_regime(ratio):
    if   ratio < BETA_EQ * 0.9: return "BELOW"
    elif ratio > BETA_EQ * 1.1: return "ABOVE"
    else:                        return "AT"


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    # ── Original wb_tsirpinn.py arguments ─────────────────────
    parser.add_argument("--run-num",          type=int,
                        default=1)
    parser.add_argument("--k",                type=int,
                        default=34)
    parser.add_argument("--tlag",             type=int,
                        default=52)
    parser.add_argument("--year-test-cutoff", type=float,
                        default=2017.0)
    parser.add_argument("--city",             type=str,
                        default="South Twenty Four Parganas")
    parser.add_argument("--num-epochs",       type=int,
                        default=2500)
    parser.add_argument("--wd-fnn",           type=float,
                        default=0.025)
    parser.add_argument("--write-loc",        type=str,
                        default=WRITE)
    parser.add_argument("--no-cuda",
                        action="store_true", default=False)
    # ── ADDED: sweep arguments ────────────────────────────────
    parser.add_argument("--lambda-i",    type=float,
                        default=10.0,
                        help="I_hp — incidence weight "
                             "(original wb_tsirpinn default=10)")
    parser.add_argument("--lambda-ode",  type=float,
                        default=1.0,
                        help="ode_hp — ODE weight "
                             "(original wb_tsirpinn default=1)")
    parser.add_argument("--ratio-label", type=str,
                        default="",
                        help="Label for output folder")
    args = parser.parse_args()

    os.makedirs(args.write_loc, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    np.random.seed(42 + args.run_num)
    torch.manual_seed(42 + args.run_num)

    ratio         = args.lambda_i / args.lambda_ode
    regime        = get_regime(ratio)
    run_start     = time.time()
    run_start_str = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"{'='*62}", flush=True)
    print(f"TSIR-PINN SWEEP | wb_tsirpinn.py + lambda args",
          flush=True)
    print(f"Started:    {run_start_str}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Run:        {args.run_num}  k={args.k}",
          flush=True)
    print(f"lambda_I:   {args.lambda_i}  "
          f"(original=10)", flush=True)
    print(f"lambda_ODE: {args.lambda_ode}  "
          f"(original=1)", flush=True)
    print(f"Ratio:      {ratio:.2f}  "
          f"beta_eq={BETA_EQ}  "
          f"Regime: {regime}", flush=True)
    print(f"Epochs:     {args.num_epochs}", flush=True)
    print(f"beta(t):    vert+amp1*sin+amp2*cos [UNCONSTRAINED]",
          flush=True)
    print(f"S in ODE:   SI[:,0:1] from network [ORIGINAL]",
          flush=True)
    print(f"{'='*62}", flush=True)

    # ── Load data ─────────────────────────────────────────────
    train_df, test_df = readin_data(
        k=args.k, tlag=args.tlag,
        year_test_cutoff=args.year_test_cutoff)

    train_city = get_cities(train_df, [args.city])
    test_city  = get_cities(test_df,  [args.city])

    processed_train     = process(train_city)
    processed_test      = process(test_city)
    time_original_train = processed_train[
        'time_original'].to_numpy()
    time_original_test  = processed_test[
        'time_original'].to_numpy()

    cases_train = get_data(processed_train)
    cases_test  = get_data(processed_test)

    S_train, I_train, t_train, N_train, Bi_train, X_train = \
        get_X_y(cases_train)
    S_test, I_test, t_test, N_test, Bi_test, X_test = \
        get_X_y(cases_test)

    print(f"Train: {len(cases_train)} | "
          f"Test: {len(cases_test)}", flush=True)
    print(f"S_bar={S_train.mean():.0f} | "
          f"I_bar={I_train.mean():.2f} | "
          f"S/I={S_train.mean()/I_train.mean():.0f}x | "
          f"beta_eq={N_train.mean()/S_train.mean():.2f}",
          flush=True)
    print(f"{'─'*62}", flush=True)

    train_data   = Data(t_train, S_train, I_train,
                        Bi_train, N_train, X_train)
    test_data    = Data(t_test,  S_test,  I_test,
                        Bi_test,  N_test,  X_test)
    train_loader = DataLoader(train_data,
                              batch_size=64, shuffle=True)

    # ── Models ────────────────────────────────────────────────
    B         = get_B(scale=0.1).to(device)
    model     = NeuralNetwork(
        1, train_data.X.shape[1], 128, 3, B).to(device)
    ode_model = ode_nn(T=26).to(device)

    loss_fn       = nn.L1Loss()
    optimizer     = torch.optim.Adam(
        model.parameters(), lr=args.wd_fnn)
    optimizer_ode = torch.optim.Adam(
        ode_model.parameters(), lr=0.1)

    # ── LOSS WEIGHTS ──────────────────────────────────────────
    # Original wb_tsirpinn.py: S_hp=0.1, I_hp=10, ode_hp=1
    # Sweep: I_hp and ode_hp driven by args
    # S_hp fixed at 0.1 — identical to original
    S_hp   = 0.1
    I_hp   = args.lambda_i    # ADDED — was hardcoded 10
    ode_hp = args.lambda_ode  # ADDED — was hardcoded 1

    print(f"S_hp=0.1  I_hp={I_hp}  ode_hp={ode_hp}  "
          f"ratio={ratio:.2f}", flush=True)

    # ── History ───────────────────────────────────────────────
    ode_loss_vals = []
    S_loss_vals   = []
    I_loss_vals   = []
    S_test_vals   = []
    I_test_vals   = []
    vert_vals     = []
    amp1_vals     = []
    amp2_vals     = []

    # ── Training  (identical to wb_tsirpinn.py) ───────────────
    model.train()
    ode_model.train()

    for epoch in range(args.num_epochs):
        ep_S = ep_I = ep_ode = 0

        for t, t_ode, S, I, Bi, N, X in train_loader:
            t  = t.to(device);  t.requires_grad = True
            S  = S.to(device);  S.requires_grad = True
            I  = I.to(device);  I.requires_grad = True
            Bi = Bi.to(device)
            N  = N.to(device)
            X  = X.to(device)

            optimizer.zero_grad()
            optimizer_ode.zero_grad()

            pred   = model(t, X)
            S_pred = pred[:, 0:1]
            I_pred = pred[:, 1:2]

            # Jacobian — identical to original
            u_x = torch.autograd.functional.jacobian(
                      model, (t, X), create_graph=True)
            u_t = u_x[0]
            u_s = torch.diagonal(
                      torch.squeeze(u_t[:,0:1],1)).reshape(-1,1)
            u_i = torch.diagonal(
                      torch.squeeze(u_t[:,1:2],1)).reshape(-1,1)

            der   = ode_model(t=t, SI=pred, Bi=Bi, N=N)
            der_S = der[:, 0:1]
            der_I = der[:, 1:2]

            # Losses — identical to original
            loss_S   = loss_fn(S_pred, S)
            loss_I   = loss_fn(I_pred, I)
            loss_ode = (loss_fn(der_S, u_s)
                      + loss_fn(der_I, u_i))
            loss     = (loss_S   * S_hp
                      + loss_I   * I_hp
                      + loss_ode * ode_hp)
            loss.retain_grad()

            ep_S   += loss_S.item()
            ep_I   += loss_I.item()
            ep_ode += loss_ode.item()

            loss.backward()
            optimizer.step()
            optimizer_ode.step()

        # ── End-of-epoch ──────────────────────────────────────
        params = list(ode_model.parameters())
        vert_v = params[0].cpu().detach().numpy().flatten()
        amp1_v = params[1].cpu().detach().numpy().flatten()
        amp2_v = params[2].cpu().detach().numpy().flatten()

        vert_vals.append(vert_v)
        amp1_vals.append(amp1_v)
        amp2_vals.append(amp2_v)

        with torch.no_grad():
            tp = model(test_data.t.to(device),
                       test_data.X.to(device)).cpu()
            S_test_vals.append(
                loss_fn(tp[:,0:1], test_data.S).item())
            I_test_vals.append(
                loss_fn(tp[:,1:2], test_data.I).item())

        ode_loss_vals.append(ep_ode)
        S_loss_vals.append(ep_S)
        I_loss_vals.append(ep_I)

        # ── LOG EVERY EPOCH ───────────────────────────────────
        elapsed = time.time() - run_start
        frac    = (epoch + 1) / args.num_epochs
        eta     = elapsed / frac * (1 - frac) if frac > 0 else 0
        ts      = time.strftime('%H:%M:%S')

        print(
            f"[{ts}] "
            f"Ep {epoch+1:>4}/{args.num_epochs} | "
            f"S={ep_S:.4f} "
            f"I={ep_I:.4f} "
            f"ODE={ep_ode:.6f} | "
            f"testI={I_test_vals[-1]:.4f} | "
            f"vert={vert_v[0]:.4f} "
            f"amp1={amp1_v[0]:.4f} | "
            f"elapsed={fmt_time(elapsed)} "
            f"ETA={fmt_time(eta)}",
            flush=True
        )

    # ── Final ─────────────────────────────────────────────────
    run_end     = time.time()
    run_end_str = time.strftime('%Y-%m-%d %H:%M:%S')
    elapsed     = run_end - run_start

    final_vert = float(vert_vals[-1][0])
    final_amp1 = float(amp1_vals[-1][0])
    final_amp2 = float(amp2_vals[-1][0])

    model.eval()
    with torch.no_grad():
        te_pred = model(
            test_data.t.to(device),
            test_data.X.to(device)).cpu().numpy()

    I_pred_test = te_pred[:, 1]
    test_rmse   = float(np.sqrt(
        np.mean((I_pred_test - I_test.flatten())**2)))

    print(f"\n{'='*62}", flush=True)
    print(f"RUN COMPLETE", flush=True)
    print(f"Started:      {run_start_str}", flush=True)
    print(f"Finished:     {run_end_str}", flush=True)
    print(f"Elapsed:      {fmt_time(elapsed)}", flush=True)
    print(f"Test RMSE:    {test_rmse:.4f}", flush=True)
    print(f"Final vert:   {final_vert:.4f}", flush=True)
    print(f"Final amp1:   {final_amp1:.4f}", flush=True)
    print(f"Ratio:        {ratio:.2f} | Regime: {regime}",
          flush=True)
    print(f"beta_eq:      {BETA_EQ}", flush=True)
    print(f"{'='*62}\n", flush=True)

    # ── Save ──────────────────────────────────────────────────
    city_safe = args.city.replace(" ", "_")
    rl        = args.ratio_label if args.ratio_label \
                else f"ratio{int(ratio)}"
    stem      = (args.write_loc
                 + f"tsirpinn_sweep_{rl}"
                 + f"_k{args.k}_tlag{args.tlag}"
                 + f"_city{city_safe}_run_{args.run_num}")

    torch.save(model.state_dict(),
               stem + "_feature_model.pt")
    torch.save(ode_model.state_dict(),
               stem + "_ode_model.pt")

    pd.DataFrame({
        'ode_loss':    ode_loss_vals,
        'S_loss':      S_loss_vals,
        'I_loss':      I_loss_vals,
        'S_test_loss': S_test_vals,
        'I_test_loss': I_test_vals,
        'vert':        vert_vals,
        'amp1':        amp1_vals,
        'amp2':        amp2_vals,
    }).to_parquet(stem + "_fit_info.parquet")

    with torch.no_grad():
        tr_pred = model(
            train_data.t.to(device),
            train_data.X.to(device)).cpu().numpy()

    tr_df = pd.DataFrame(tr_pred,
                         columns=['S_pred','I_pred','R_pred'])
    tr_df['time']          = train_data.t.numpy().flatten()
    tr_df['time_original'] = time_original_train
    tr_df['S']             = S_train.flatten()
    tr_df['I']             = I_train.flatten()
    tr_df.to_parquet(stem + "_train_predictions.parquet")

    te_df = pd.DataFrame(te_pred,
                         columns=['S_pred','I_pred','R_pred'])
    te_df['time']          = test_data.t.numpy().flatten()
    te_df['time_original'] = time_original_test
    te_df['S']             = S_test.flatten()
    te_df['I']             = I_test.flatten()
    te_df.to_parquet(stem + "_test_predictions.parquet")

    # Run summary for collection script
    pd.DataFrame([{
        'run':          args.run_num,
        'ratio':        ratio,
        'ratio_label':  rl,
        'regime':       regime,
        'lambda_i':     args.lambda_i,
        'lambda_ode':   args.lambda_ode,
        'test_rmse':    test_rmse,
        'final_vert':   final_vert,
        'final_amp1':   final_amp1,
        'final_amp2':   final_amp2,
        'beta_eq':      BETA_EQ,
        'log_beta_eq':  LOG_BETA_EQ,
        'start_time':   run_start_str,
        'end_time':     run_end_str,
        'elapsed_sec':  elapsed,
        'city':         args.city,
        'k':            args.k,
    }]).to_parquet(stem + "_run_summary.parquet")

    print(f"Saved: {stem}", flush=True)


if __name__ == "__main__":
    main()
