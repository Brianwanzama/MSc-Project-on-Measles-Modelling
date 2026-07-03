# ============================================================
# wb_naivepinn.py
# Naive PINN — West Bengal measles extension
# S_latent is a free learnable parameter (not from TSIR)
#
# CHANGES FROM naivepinn.py (London):
#   1. parquet path: prefit_cases1/ (not prefit_cases/)
#   2. year_cutoff: float 2017.0 (not integer 61)
#   3. train/test split: uses 'split' column
#   4. time transform: decimal year -> biweek index
#      t_biweek = round((time - 2008.0) * 26) + 1
#   5. births: NOT divided by 26 (already done in loader)
#   6. S_latent num_t: max biweek index (288 total, 182 train)
#   7. default city: South Twenty Four Parganas
#   8. default write_loc: wb_pinn/
#   9. default k, tlag, epochs aligned with WB pipeline
# ============================================================

import argparse
import os
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

BASE     = "/home/brain/Msc_project/"
PREFIT   = BASE + "output/data/prefit_cases1/"
WRITE    = BASE + "output/models/pinn_experiments/wb_pinn/"

# ── DATA LOADING ──────────────────────────────────────────────
def readin_data(k=1, tlag=52, year_test_cutoff=2017.0):
    path     = PREFIT + f"k{k}_tlag{tlag}.gzip"
    full_df  = pd.read_parquet(path)

    # Use split column if available, otherwise use time threshold
    if 'split' in full_df.columns:
        train_df = full_df[full_df['split'] == 'train'].copy()
        test_df  = full_df[full_df['split'] == 'test'].copy()
    else:
        train_df = full_df[full_df['time'] < year_test_cutoff].copy()
        test_df  = full_df[full_df['time'] >= year_test_cutoff].copy()

    return train_df, test_df

def get_cities(data, cities):
    return data[data['city'].isin(cities)].copy()

def process(data):
    """Convert decimal year to biweek index.
    Original: (time - 49) * 26 + 1  [London-specific]
    WB:       round((time - 2008.0) * 26) + 1
    Births already /26 in our parquet — do NOT divide again.
    """
    data = data.copy()
    data['time_original'] = data['time']
    data['time'] = np.round((data['time'] - 2008.0) * 26).astype(int) + 1
    # births already biweekly in prefit_cases1 parquet
    return data

def get_data(data):
    case_cols = [col for col in data.columns if "cases_lag_" in col]
    # naivepinn does NOT use susc_lag_ (S is latent free parameter)
    return data[["time", "susc", "cases", "births", "pop"] + case_cols]

def get_X_y(data):
    S  = data['susc'].to_numpy().reshape(-1, 1)
    I  = data['cases'].to_numpy().reshape(-1, 1)
    t  = data['time'].to_numpy().reshape(-1, 1)
    N  = data['pop'].to_numpy().reshape(-1, 1)
    Bi = data['births'].to_numpy().reshape(-1, 1)
    X  = data.drop(columns=['cases','time','susc','pop','births']).to_numpy()
    return S, I, t, N, Bi, X

# ── DATASET ───────────────────────────────────────────────────
class Data(Dataset):
    def __init__(self, t, S, I, Bi, N, X):
        self.t   = torch.from_numpy(t).float().reshape(1,-1).t()
        self.t_ode = torch.from_numpy(t).float().reshape(1,-1).t()
        self.S   = torch.from_numpy(S).float().reshape(1,-1).t()
        self.I   = torch.from_numpy(I).float().reshape(1,-1).t()
        self.Bi  = torch.from_numpy(Bi).float().reshape(1,-1).t()
        self.N   = torch.from_numpy(N).float().reshape(1,-1).t()
        self.X   = torch.from_numpy(X).float()
        self.len = self.t.shape[0]

    def __getitem__(self, index):
        return (self.t[index], self.t_ode[index],
                self.S[index], self.I[index],
                self.Bi[index], self.N[index],
                self.X[index])

    def __len__(self):
        return self.len

# ── FOURIER MAP ───────────────────────────────────────────────
class fourier_map(nn.Module):
    def __init__(self, size_in, B):
        super().__init__()
        self.size_B  = B.shape[1]
        self.B       = B
        self.size_in = size_in * self.size_B * 2
        self.size_out= self.size_in
        weights      = torch.Tensor(self.size_out, self.size_in)
        self.weights = nn.Parameter(weights)
        bias         = torch.Tensor(self.size_out)
        self.bias    = nn.Parameter(bias)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _    = nn.init._calculate_fan_in_and_fan_out(
                           self.weights)
        bound        = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x_sin = torch.sin(torch.mm(x, self.B))
        x_cos = torch.cos(torch.mm(x, self.B))
        x     = torch.cat((x_sin, x_cos), 1)
        return torch.add(torch.mm(x, self.weights.t()), self.bias)

# ── SEASONAL BETA ─────────────────────────────────────────────
def seasonal_beta_torch(t, vert, amp1, amp2, T):
    return (vert
            + amp1 * torch.sin(2 * torch.pi * (t / T))
            + amp2 * torch.cos(2 * torch.pi * (t / T)))

# ── ODE LAYER — naive (S_latent free parameter) ───────────────
class derivative_layer(nn.Module):
    def __init__(self, T, num_t):
        super().__init__()
        self.T       = T
        self.vert    = nn.Parameter(torch.randn(1))
        self.amp1    = nn.Parameter(torch.randn(1))
        self.amp2    = nn.Parameter(torch.randn(1))
        # S_latent has one entry per biweek in training period
        # num_t = max biweek index seen in training
        self.S_latent = nn.Parameter(torch.randn(num_t))

    def forward(self, t, SI, Bi, N):
        # Index S_latent by biweek: t is 1-indexed biweek
        # Clamp to valid range to avoid index out of bounds
        t_idx  = t.long().squeeze() - 1
        t_idx  = t_idx.clamp(0, self.S_latent.shape[0] - 1)
        S_lat  = torch.exp(self.S_latent[t_idx].unsqueeze(1) + 4.5) * 1e3
        I      = SI[:, 1:2]
        beta   = seasonal_beta_torch(
                     t, self.vert, self.amp1, self.amp2, self.T)
        der_S  = Bi - (beta * S_lat * I) / N
        der_I  = (beta * S_lat * I) / N - I
        return torch.cat((der_S, der_I), 1)

class ode_nn(nn.Module):
    def __init__(self, T, num_t):
        super().__init__()
        self.der = derivative_layer(T, num_t)

    def forward(self, t, SI, Bi, N):
        return self.der(t, SI, Bi, N)

# ── MAIN NETWORK ──────────────────────────────────────────────
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim_t, input_dim_X,
                 hidden_dim, output_dim, B):
        super().__init__()
        self.gelu     = nn.GELU()
        self.softplus = nn.Softplus()
        self.fm = fourier_map(input_dim_t, B)
        self.f1 = nn.Linear(B.shape[1]*2 + input_dim_X, hidden_dim)
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
    B = torch.FloatTensor(np.random.randn(1, num_features) * scale)
    return B

# ── MAIN ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-num",          type=int,   default=1)
    parser.add_argument("--k",                type=int,   default=1)
    parser.add_argument("--tlag",             type=int,   default=52)
    parser.add_argument("--year-test-cutoff", type=float, default=2017.0)
    parser.add_argument("--city",             type=str,
                        default="South Twenty Four Parganas")
    parser.add_argument("--num-epochs",       type=int,   default=2500)
    parser.add_argument("--wd-fnn",           type=float, default=0.025)
    parser.add_argument("--write-loc",        type=str,   default=WRITE)
    parser.add_argument("--no-cuda",          action="store_true",
                        default=False)
    args = parser.parse_args()

    os.makedirs(args.write_loc, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device} | k={args.k} | "
          f"run={args.run_num} | epochs={args.num_epochs}")

    np.random.seed(42 + args.run_num)
    torch.manual_seed(42 + args.run_num)

    train_df, test_df = readin_data(
        k=args.k, tlag=args.tlag,
        year_test_cutoff=args.year_test_cutoff)

    train_city = get_cities(train_df, [args.city])
    test_city  = get_cities(test_df,  [args.city])

    # process() adds time_original — save before get_data() strips it
    processed_train      = process(train_city)
    processed_test       = process(test_city)
    time_original_train  = processed_train['time_original'].to_numpy()
    time_original_test   = processed_test['time_original'].to_numpy()
    cases_train = get_data(processed_train)
    cases_test  = get_data(processed_test)

    print(f"Train rows: {len(cases_train)} | "
          f"Test rows: {len(cases_test)}")
    print(f"Time range train: {cases_train['time'].min()} - "
          f"{cases_train['time'].max()}")

    S_train, I_train, t_train, N_train, Bi_train, X_train = \
        get_X_y(cases_train)
    S_test, I_test, t_test, N_test, Bi_test, X_test = \
        get_X_y(cases_test)

    train_data = Data(t_train, S_train, I_train,
                      Bi_train, N_train, X_train)
    test_data  = Data(t_test,  S_test,  I_test,
                      Bi_test,  N_test,  X_test)

    train_loader = DataLoader(train_data, batch_size=64,
                              shuffle=True)

    # num_t = max biweek index across full dataset
    # (train + test) so S_latent covers all time points
    all_city = get_data(process(
        pd.concat([train_city, test_city])))
    num_t = int(all_city['time'].max())
    print(f"S_latent size: {num_t}")

    input_dim_t = 1
    input_dim_X = train_data.X.shape[1]
    hidden_dim  = 128
    output_dim  = 3
    B           = get_B(scale=0.1).to(device)

    model = NeuralNetwork(
        input_dim_t, input_dim_X,
        hidden_dim, output_dim, B).to(device)

    ode_model = ode_nn(T=26, num_t=num_t).to(device)

    loss_fn      = nn.L1Loss()
    optimizer    = torch.optim.Adam(
        model.parameters(), lr=args.wd_fnn)
    optimizer_ode = torch.optim.Adam(
        ode_model.parameters(), lr=0.1)

    S_hp  = 1/10
    I_hp  = 10
    ode_hp= 1

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
            t   = t.to(device);   t.requires_grad  = True
            S   = S.to(device);   S.requires_grad  = True
            I   = I.to(device);   I.requires_grad  = True
            Bi  = Bi.to(device)
            N   = N.to(device)
            X   = X.to(device)

            optimizer.zero_grad()
            optimizer_ode.zero_grad()

            pred   = model(t, X)
            S_pred = pred[:, 0:1]
            I_pred = pred[:, 1:2]

            # Jacobian for ODE residual
            u_x    = torch.autograd.functional.jacobian(
                model, (t, X), create_graph=True)
            u_t    = u_x[0]
            u_s    = torch.diagonal(
                torch.squeeze(u_t[:, 0:1], 1)).reshape(-1, 1)
            u_i    = torch.diagonal(
                torch.squeeze(u_t[:, 1:2], 1)).reshape(-1, 1)

            der    = ode_model(t=t, SI=pred, Bi=Bi, N=N)
            der_S  = der[:, 0:1]
            der_I  = der[:, 1:2]

            # S_latent for this batch
            t_idx  = t.long().squeeze().clamp(
                0, num_t-1)
            S_latent_batch = list(ode_model.parameters())[3][t_idx]

            loss_S    = loss_fn(S_pred, S_latent_batch.unsqueeze(1))
            loss_I    = loss_fn(I_pred, I)
            loss_ode  = loss_fn(der_S, u_s) + loss_fn(der_I, u_i)
            loss      = (loss_S * S_hp
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
        vert_vals.append(params[0].cpu().detach().numpy().flatten())
        amp1_vals.append(params[1].cpu().detach().numpy().flatten())
        amp2_vals.append(params[2].cpu().detach().numpy().flatten())

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

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} | "
                  f"S={ep_S:.4f} I={ep_I:.4f} "
                  f"ODE={ep_ode:.4f} | "
                  f"test_I={I_test_vals[-1]:.4f}")

    # ── SAVE ──────────────────────────────────────────────────
    city_safe = args.city.replace(" ", "_")
    stem      = (args.write_loc
                 + f"naivepinn_k{args.k}_tlag{args.tlag}"
                 + f"_city{city_safe}_run_{args.run_num}")

    torch.save(model.state_dict(),
               stem + "_feature_model.pt")
    torch.save(ode_model.state_dict(),
               stem + "_ode_model.pt")

    df = pd.DataFrame({
        'ode_loss':    ode_loss_vals,
        'S_loss':      S_loss_vals,
        'I_loss':      I_loss_vals,
        'S_test_loss': S_test_vals,
        'I_test_loss': I_test_vals,
        'vert':        vert_vals,
        'amp1':        amp1_vals,
        'amp2':        amp2_vals,
    })
    df.to_parquet(stem + "_fit_info.parquet")

    # Train predictions
    model.eval()
    with torch.no_grad():
        tr_pred = model(train_data.t.to(device),
                        train_data.X.to(device)).cpu().numpy()
    tr_df = pd.DataFrame(tr_pred,
                         columns=['S_pred','I_pred','R_pred'])
    tr_df['time']          = train_data.t.numpy().flatten()
    tr_df['time_original'] = time_original_train
    tr_df['S']             = S_train.flatten()
    tr_df['I']             = I_train.flatten()
    tr_df.to_parquet(stem + "_train_predictions.parquet")

    # Test predictions
    with torch.no_grad():
        te_pred = model(test_data.t.to(device),
                        test_data.X.to(device)).cpu().numpy()
    te_df = pd.DataFrame(te_pred,
                         columns=['S_pred','I_pred','R_pred'])
    te_df['time']          = test_data.t.numpy().flatten()
    te_df['time_original'] = time_original_test
    te_df['S']             = S_test.flatten()
    te_df['I']             = I_test.flatten()
    te_df.to_parquet(stem + "_test_predictions.parquet")

    print(f"\nSaved: {stem}")

if __name__ == "__main__":
    main()
