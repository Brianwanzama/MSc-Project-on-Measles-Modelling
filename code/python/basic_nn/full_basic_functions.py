import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# ---------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------

class Data(Dataset):

    def __init__(self, X, y):
        self.X   = torch.tensor(X.values).float()
        self.y   = torch.from_numpy(y).float()
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


# ---------------------------------------------------
# NEURAL NETWORK
# Architecture: Linear->ReLU->[Linear->ReLU]×n->Linear
# No activation on output — regression task
# Matches original paper (Madden et al. 2024) SFNN design
# ---------------------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_hidden_layers=3):

        super(NeuralNetwork, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


# ---------------------------------------------------
# DATA PROCESSING
# ---------------------------------------------------

def process_data(cases, year_test_cutoff):

    # ── TRAIN / TEST SPLIT ────────────────────────────────────
    # Use pre-computed 'split' column from data loader
    # More explicit than recomputing year from time
    if 'split' in cases.columns:
        cases_train = cases[cases['split'] == 'train'].copy()
        cases_test  = cases[cases['split'] == 'test'].copy()
    else:
        # Fallback: compute from time if split column absent
        cases['year'] = cases['time'].astype(int)
        cases_train   = cases[cases['year'] < year_test_cutoff].copy()
        cases_test    = cases[cases['year'] >= year_test_cutoff].copy()

    # ── IDs — kept for alignment with TSIR predictions ────────
    id_train = cases_train[['time', 'city']].reset_index(drop=True)
    id_test  = cases_test[['time', 'city']].reset_index(drop=True)

    # ── TARGETS ───────────────────────────────────────────────
    y_train = cases_train['cases_trans'].to_numpy().reshape((-1, 1))
    y_test  = cases_test['cases_trans'].to_numpy().reshape((-1, 1))

    # ── FEATURES ──────────────────────────────────────────────
    # Regex captures all engineered lag and distance features:
    #   cases_lag_*       — local incidence history
    #   susc_lag_*        — susceptible dynamics (TSIR-reconstructed)
    #   births_lag_*      — birth-driven susceptible recruitment
    #   v1_lag_*          — MCV1 vaccination coverage (YOUR EXTENSION)
    #   pop_lag_*         — population size
    #   dist_*            — distances to large districts
    #   nearest_*_city_dist — distances to nearest 10 districts
    #   cases_*_lag_*     — large district incidence lags
    #   cases_nc_*_lag_*  — nearest city incidence lags
    #   cases_nbc_lag_*   — nearest big city incidence lags
    X_train = cases_train.filter(regex="lag_|dist_")
    X_test  = cases_test.filter(regex="lag_|dist_")

    # ── VALIDATION: train and test must have identical features
    assert X_train.shape[1] == X_test.shape[1], (
        f"Feature count mismatch: "
        f"train={X_train.shape[1]}, test={X_test.shape[1]}"
    )
    assert list(X_train.columns) == list(X_test.columns), (
        "Column name or order mismatch between train and test"
    )

    # ── CLEAN NUMERICAL ISSUES ────────────────────────────────
    X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], 0).fillna(0)

    # ── DIAGNOSTICS ───────────────────────────────────────────
    print(f"  Train rows:      {X_train.shape[0]}")
    print(f"  Test rows:       {X_test.shape[0]}")
    print(f"  Feature count:   {X_train.shape[1]}")
    print(f"  NaNs in train:   {X_train.isna().sum().sum()}")
    print(f"  NaNs in test:    {X_test.isna().sum().sum()}")

    v1_cols = [c for c in X_train.columns if 'v1' in c]
    print(f"  V1 features:     {v1_cols}")

    susc_cols = [c for c in X_train.columns if 'susc' in c]
    print(f"  Susc features:   {len(susc_cols)} columns")

    # ── DATASETS ──────────────────────────────────────────────
    train_data = Data(X_train, y_train)
    test_data  = Data(X_test,  y_test)

    return train_data, test_data, X_train.shape[1], id_train, id_test


# ---------------------------------------------------
# DATALOADERS
# ---------------------------------------------------

def get_dataloaders(train_data, test_data, batch_size):

    train_loader = DataLoader(
        dataset    = train_data,
        batch_size = batch_size,
        shuffle    = True,       # randomise order each epoch
        pin_memory = True        # faster GPU transfer
    )

    test_loader = DataLoader(
        dataset    = test_data,
        batch_size = batch_size,
        shuffle    = False,      # preserve time order for evaluation
        pin_memory = True
    )

    return train_loader, test_loader


# ---------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------

def train(model, device, train_loader, optimizer,
          loss_fn, epoch, log_interval=10):

    model.train()
    train_loss = 0.0

    for batch_idx, (X, y) in enumerate(train_loader):

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(
                f"  Epoch {epoch} "
                f"[{batch_idx * len(X)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    train_loss /= len(train_loader)
    return train_loss


# ---------------------------------------------------
# TEST LOOP
# ---------------------------------------------------

def test(model, device, test_loader, loss_fn):

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:

            X, y  = X.to(device), y.to(device)
            pred  = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= len(test_loader)

    print(f"  Test loss: {test_loss:.6f}")

    return test_loss


# ---------------------------------------------------
# PREDICTION WITH IDs
# Returns predictions aligned to time and city
# Needed for RMSE computation per district per k
# ---------------------------------------------------

def predict(model, device, test_loader, id_test,
            cases_transform_output):

    model.eval()
    all_preds = []

    with torch.no_grad():
        for X, _ in test_loader:
            X    = X.to(device)
            pred = model(X).cpu().numpy()
            all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0)

    # Align predictions with time and city
    results = id_test.copy().reset_index(drop=True)
    results['pred_trans'] = preds.flatten()

    # Reverse standardisation to get predictions in log(cases+1) scale
    results = results.merge(
        cases_transform_output[['time', 'city',
                                 'cases_mean', 'cases_std']],
        on=['time', 'city'], how='left'
    )
    results['pred_log']  = (results['pred_trans'] *
                             results['cases_std'] +
                             results['cases_mean'])
    results['pred_cases'] = np.exp(results['pred_log']) - 1
    results['pred_cases'] = results['pred_cases'].clip(lower=0)

    return results