import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from ray import tune, train
import ray

# -----------------------------
# Setup Paths
# -----------------------------
data_processing_path = os.path.abspath(
    "/.../.../.../data_processing/"
)

original_sys_path = sys.path.copy()
sys.path.append(data_processing_path)
import prevac_measles_data_loader as mdl
sys.path = original_sys_path

import full_basic_functions as fbf

# Initialize Ray safely (append PYTHONPATH)
ray.init(runtime_env={
    "env_vars": {
        "PYTHONPATH": f"{data_processing_path}:{os.environ.get('PYTHONPATH','')}"
    }
})


# -----------------------------
# Training Function
# -----------------------------
def train_with_tuning(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load dataset
        full_cases_loc = (
            config["cases_data_loc"]
            + "/k"
            + str(config["k"])
            + "_tlag"
            + str(config["t_lag"])
            + ".gzip"
        )

        cases = pd.read_parquet(full_cases_loc)

        train_data, test_data, num_features, _, _ = fbf.process_data(
            cases, config["year_test_cutoff"]
        )

        # Model
        model = fbf.NeuralNetwork(
            num_features,
            config["hidden_dim"],
            1,
            num_hidden_layers=config["num_hidden_layers"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        loss_fn = nn.MSELoss()

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        # Training loop
        for epoch in range(config["num_epochs"]):
            fbf.train(
                model=model,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch=epoch,
                log_interval=100
            )
            fbf.test(model, device, test_loader, loss_fn)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            pred_train = model(train_data.X.to(device)).cpu().numpy()
            pred_test = model(test_data.X.to(device)).cpu().numpy()

        train_mse = np.mean((pred_train - train_data.y.numpy()) ** 2)
        test_mse = np.mean((pred_test - test_data.y.numpy()) ** 2)

        # Report ONCE
        train.report({
            "train_mse": train_mse,
            "test_mse": test_mse
        })

    except Exception as e:
        print("Trial crashed:", e)
        raise


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-num-epochs", type=int, default=10)
    parser.add_argument("--gpus-per-trial", type=float, default=0)
    args = parser.parse_args()

    # t_lag logic
    if args.k < 26:
        t_lag_options = [26, 52, 78]
    elif args.k < 52:
        t_lag_options = [52, 78, 104]
    else:
        t_lag_options = [78, 104, 130]

    config = {
        "k": args.k,
        "cases_data_loc": os.path.abspath(
            "/home/brain/deep_measles_dynamics-main/deep_measles_dynamics-main/output/data/prefit_cases"
        ),
        "year_test_cutoff": 61,
        "num_epochs": args.max_num_epochs,
        "lr": 0.001,
        "t_lag": tune.grid_search(t_lag_options),
        "hidden_dim": tune.grid_search([240, 721, 1201]),
        "weight_decay": tune.uniform(0.0001, 0.1),
        "num_hidden_layers": tune.grid_search([1, 2, 3]),
    }

    # Run tuning
    result = tune.run(
        train_with_tuning,
        config=config,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": 12, "gpu": args.gpus_per_trial},
    )

    # -----------------------------
    # Safe Result Processing
    # -----------------------------
    successful_trials = [
        trial for trial in result.trials
        if "test_mse" in trial.last_result
    ]

    if not successful_trials:
        raise RuntimeError(f"No successful trials for k={args.k}")

    best_trial = min(
        successful_trials,
        key=lambda t: t.last_result["test_mse"]
    )

    trials_data = []
    for trial in successful_trials:
        trials_data.append({
            "trial_id": trial.trial_id,
            "test_mse": trial.last_result["test_mse"],
            "is_best": trial.trial_id == best_trial.trial_id,
            **trial.config
        })

    df = pd.DataFrame(trials_data).sort_values("test_mse")

    save_dir = (
        "/home/brain/deep_measles_dynamics-main/"
        "deep_measles_dynamics-main/output/figures/basic_nn/raytune_hp_optim/"
    )

    os.makedirs(save_dir, exist_ok=True)

    save_path = save_dir + f"raytune_hp_optim_k_{args.k}.csv"
    df.to_csv(save_path, index=False)

    print(f"\n[DONE] Results saved to: {save_path}")
    print("Best config:", best_trial.config)
    print("Best test MSE:", best_trial.last_result["test_mse"])


if __name__ == "__main__":
    main()
