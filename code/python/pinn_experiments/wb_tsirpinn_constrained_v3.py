# ============================================================
# wb_tsirpinn_retuned.py
# TSIRPINN — West Bengal — RETUNED LOSS WEIGHTS
#
# PROBLEM WITH ORIGINAL:
#   S ~ 300,000, I ~ 15-30 → ODE trivially satisfied by I=0
#   97/100 runs collapse to zero infectious predictions
#
# FIX:
#   1. Normalise all losses by compartment scale
#   2. S_hp=1.0, I_hp=100, ode_hp=0.01
#   3. ODE residual normalised by S_mean and I_mean
# ============================================================

import argparse
import os
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

BASE   = "/home/brain/Msc_project/"
PREFIT = BASE + "output/data/prefit_cases1/"
WRITE  = BASE + "output/models/pinn_experiments/wb_pinn_final/"

def readin_data(k=34, tlag=52, year_test_cutoff=2017.0):
    path    = PREFIT + f"k{k}_tlag{tlag}.gzip"
    full_df = pd.read_parquet(path)
    if 'split' in full_df.columns:
        train_df = full_df[full_df['split'] == 'train'].copy()
        test_df  = full_df[full_df['split'] == 'test'].copy()
    else:
        train_df = full_df[
            full_df['time'] < year_test_cutoff].copy()
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
    case_cols = [col for col in data.columns
                 if "cases_lag_" in col]
    susc_cols = [col for col in data.columns
                 if "susc_lag_" in col]
    return data[["time","susc","cases","births","pop"]
                + case_cols + susc_cols]

def get_X_y(data):
    S  = data['susc'].to_numpy().reshape(-1, 1)
    I  = data['cases'].to_numpy().reshape(-1, 1)
    t  = data['time'].to_numpy().reshape(-1, 1)
    N  = data['pop'].to_numpy().reshape(-1, 1)
    Bi = data['births'].to_numpy().reshape(-1, 1)
    X  = data.drop(
            columns=['cases','time','susc','pop','births']
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

class fourier_map(nn.Module):
    def __init__(self, size_in, B):
        super().__init__()
        self.size_B  = B.shape[1]
        self.B       = B
        self.size_in = size_in * self.size_B * 2
        self.size_out= self.size_in
        self.weights = nn.Parameter(
            torch.Tensor(self.size_out, self.size_in))
        self.bias    = nn.Parameter(
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
        return torch.add(torch.mm(x, self.weights.t()), self.bias)

# β_max: upper bound for endemic measles β
# sigmoid maps vert ∈ (-∞,∞) → (0,1) → β_base ∈ (0, β_max)
# amp1, amp2 allow seasonal variation around β_base
# Endemic equilibrium β = N/S = 7,600,000/240,617 ≈ 31.6
# This is the β that sustains endemic transmission
# given the post-vaccination susceptible pool
# β constraints — post-vaccination endemic WB
# BETA_MAX=31.6: endemic equilibrium β = N/S
# AMP_MAX=5.0:   max seasonal variation ±5
BETA_MAX = 31.6
AMP_MAX  = 5.0

def seasonal_beta_torch(t, vert, amp1, amp2, T):
    beta_base = torch.sigmoid(vert) * BETA_MAX
    amp1_c    = torch.tanh(amp1) * AMP_MAX
    amp2_c    = torch.tanh(amp2) * AMP_MAX
    beta = (beta_base
            + amp1_c * torch.sin(2 * torch.pi * (t / T))
            + amp2_c * torch.cos(2 * torch.pi * (t / T)))
    return torch.clamp(beta, min=0.01)

class derivative_layer(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T    = T
        self.vert = nn.Parameter(torch.randn(1))
        self.amp1 = nn.Parameter(torch.randn(1))
        self.amp2 = nn.Parameter(torch.randn(1))

    def forward(self, t, SI, Bi, N, S_scale, I_scale):
        S    = SI[:, 0:1]
        I    = SI[:, 1:2]
        beta = seasonal_beta_torch(
                   t, self.vert, self.amp1, self.amp2, self.T)
        der_S = Bi - (beta * S * I) / N
        der_I = (beta * S * I) / N - I

        # Normalise by scale
        der_S_norm = der_S / S_scale
        der_I_norm = der_I / I_scale

        return torch.cat((der_S, der_I), 1), \
               torch.cat((der_S_norm, der_I_norm), 1)

class ode_nn(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.der = derivative_layer(T)

    def forward(self, t, SI, Bi, N, S_scale, I_scale):
        return self.der(t, SI, Bi, N, S_scale, I_scale)

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
    B = torch.FloatTensor(
        np.random.randn(1, num_features) * scale)
    return B

def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    os.makedirs(args.write_loc, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    np.random.seed(42 + args.run_num)
    torch.manual_seed(42 + args.run_num)

    print(f"Device: {device} | k={args.k} | "
          f"run={args.run_num} | epochs={args.num_epochs}")

    train_df, test_df = readin_data(
        k=args.k, tlag=args.tlag,
        year_test_cutoff=args.year_test_cutoff)

    processed_train     = process(
        get_cities(train_df, [args.city]))
    processed_test      = process(
        get_cities(test_df,  [args.city]))
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

    # ── SCALE FACTORS ─────────────────────────────────────────
    S_mean_val = float(np.mean(S_train)) + 1e-8
    I_mean_val = float(np.mean(I_train)) + 1e-8
    S_scale    = torch.tensor(S_mean_val).float().to(device)
    I_scale    = torch.tensor(I_mean_val).float().to(device)

    print(f"S mean: {S_mean_val:.0f} | "
          f"I mean: {I_mean_val:.2f} | "
          f"ratio: {S_mean_val/I_mean_val:.0f}x")

    train_data = Data(t_train, S_train, I_train,
                      Bi_train, N_train, X_train)
    test_data  = Data(t_test,  S_test,  I_test,
                      Bi_test,  N_test,  X_test)
    train_loader = DataLoader(train_data, batch_size=64,
                              shuffle=True)

    B     = get_B(scale=0.1).to(device)
    model = NeuralNetwork(
        1, train_data.X.shape[1], 128, 3, B).to(device)
    ode_model = ode_nn(T=26).to(device)

    loss_fn       = nn.L1Loss()
    optimizer     = torch.optim.Adam(
        model.parameters(), lr=args.wd_fnn)
    optimizer_ode = torch.optim.Adam(
        ode_model.parameters(), lr=0.1)

    # ── RETUNED WEIGHTS ───────────────────────────────────────
    S_hp   = 1.0    # normalised S loss weight
    I_hp   = 100.0  # normalised I loss weight (10x original)
    ode_hp = 0.01   # normalised ODE weight (100x reduction)

    print(f"Loss weights: S_hp={S_hp} "
          f"I_hp={I_hp} ode_hp={ode_hp}")

    ode_loss_vals = []
    S_loss_vals   = []
    I_loss_vals   = []
    S_test_vals   = []
    I_test_vals   = []
    vert_vals, amp1_vals, amp2_vals = [], [], []

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

            u_x   = torch.autograd.functional.jacobian(
                model, (t, X), create_graph=True)
            u_t   = u_x[0]
            u_s   = torch.diagonal(
                torch.squeeze(u_t[:,0:1],1)).reshape(-1,1)
            u_i   = torch.diagonal(
                torch.squeeze(u_t[:,1:2],1)).reshape(-1,1)

            der, der_norm = ode_model(
                t=t, SI=pred, Bi=Bi, N=N,
                S_scale=S_scale, I_scale=I_scale)
            der_S_norm = der_norm[:, 0:1]
            der_I_norm = der_norm[:, 1:2]

            # ── NORMALISED LOSSES ─────────────────────────────
            loss_S   = loss_fn(S_pred / S_scale,
                               S / S_scale)
            loss_I   = loss_fn(I_pred / I_scale,
                               I / I_scale)
            loss_ode = (loss_fn(der_S_norm,
                                u_s / S_scale)
                       + loss_fn(der_I_norm,
                                 u_i / I_scale))

            loss = (loss_S * S_hp
                    + loss_I * I_hp
                    + loss_ode * ode_hp)
            loss.retain_grad()

            ep_S   += loss_S.item()
            ep_I   += loss_I.item()
            ep_ode += loss_ode.item()

            loss.backward()
            optimizer.step()
            optimizer_ode.step()

        params = list(ode_model.parameters())
        vert_vals.append(
            params[0].cpu().detach().numpy().flatten())
        amp1_vals.append(
            params[1].cpu().detach().numpy().flatten())
        amp2_vals.append(
            params[2].cpu().detach().numpy().flatten())

        with torch.no_grad():
            tp = model(test_data.t.to(device),
                       test_data.X.to(device)).cpu()
            S_test_vals.append(
                loss_fn(tp[:,0:1]/S_scale.cpu(),
                        test_data.S/S_scale.cpu()).item())
            I_test_vals.append(
                loss_fn(tp[:,1:2]/I_scale.cpu(),
                        test_data.I/I_scale.cpu()).item())

        ode_loss_vals.append(ep_ode)
        S_loss_vals.append(ep_S)
        I_loss_vals.append(ep_I)

        if (epoch+1) % 250 == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} | "
                  f"S={ep_S:.4f} I={ep_I:.4f} "
                  f"ODE={ep_ode:.4f} | "
                  f"test_I={I_test_vals[-1]:.4f}")

    # ── SAVE ──────────────────────────────────────────────────
    city_safe = args.city.replace(" ", "_")
    stem = (args.write_loc
            + f"tsirpinn_constrained_v3_k{args.k}"
            + f"_tlag{args.tlag}"
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

    model.eval()
    with torch.no_grad():
        tr_pred = model(train_data.t.to(device),
                        train_data.X.to(device)).cpu().numpy()
        te_pred = model(test_data.t.to(device),
                        test_data.X.to(device)).cpu().numpy()

    for pred_arr, t_orig, S_arr, I_arr, suffix in [
        (tr_pred, time_original_train,
         S_train, I_train, "train"),
        (te_pred, time_original_test,
         S_test,  I_test,  "test"),
    ]:
        pd.DataFrame({
            'S_pred':        pred_arr[:,0],
            'I_pred':        pred_arr[:,1],
            'R_pred':        pred_arr[:,2],
            'time_original': t_orig,
            'S':             S_arr.flatten(),
            'I':             I_arr.flatten(),
        }).to_parquet(stem + f"_{suffix}_predictions.parquet")

    print(f"\nSaved: {stem}")
    I_pred_test = te_pred[:,1]
    print(f"Test I_pred: mean={I_pred_test.mean():.2f} "
          f"range=[{I_pred_test.min():.2f}, "
          f"{I_pred_test.max():.2f}]")
    print(f"Test I obs:  mean={I_test.mean():.2f}")

if __name__ == "__main__":
    main()
