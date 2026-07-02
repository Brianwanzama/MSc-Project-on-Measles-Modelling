# Deep Neural Network Modelling of Endemic Measles Dynamics in Post-Vaccination West Bengal

**MSc Project — IIT Bombay, Department of Mathematics**
**Student:** Brian Wanzama (Roll No. 24N0269)
**Supervisor:** Prof. Siuli Mukhopadhyay
**Year:** 2026

---

## Overview

This project develops and compares neural network and physics-informed neural network (PINN) approaches to forecasting biweekly endemic measles incidence across 19 districts of West Bengal, India, over the period 2008–2019. The post-vaccination setting — where two-dose measles vaccination (MCV1 + MCV2) has substantially reduced the susceptible pool — creates a qualitatively different modelling challenge compared to pre-vaccination settings studied in the prior literature.

The central research questions are:
 1. Does the SFNN provide higher predictive accuracy than the TSIR model for long-horizon forecasting (k ∈ {1,4,12,20,34} biweeks) in West Bengal?
 2. How does the SFNN–TSIR performance gap vary with district population size and forecast horizon?
 3. Is the PINNframework, which integrates SIR physical laws into neural network training, transferable to the data-sparse environments characteristic of post vaccination endemicity?
 4. Can we quantitatively decompose the impact of MCV1 and MCV2 on measles case reduction at the district level?
---


## Repository Structure

```
Msc_project/
│
├── code/
│   ├── r/
│   │   ├── maps/
│   │   │   └── map_plot.R
│   │   ├── tsir/
│   │   │   ├── tsir_run_functions.R
│   │   │   ├── tsir_susceptibles_gen_V1V2.R
│   │   │   ├── tsir_wb_run_V1V2.R
│   │   │   └── tsir_wb_process_V1V2.R
│   │   ├── basic_nn/
│   │   │   ├── cases_process.R
│   │   │   ├── wb_optimal_V1V2.R
│   │   │   ├── optimal_basic_nn_process.R
│   │   │   ├── sfnn_explainability.R
│   │   │   ├── explain_plot.R
│   │   │   ├── wb_sfnn_feature_importance.R
│   │   │   ├── optimal_compare_plot.R
│   │   │   ├── wb_city_performance_plot.R
│   │   │   ├── population_plots_rmse.R
│   │   │   └── wb_city_rmse_table.R
│   │   ├── counterfactual/
│   │   │   ├── wb_tsir_counterfactual.R
│   │   │   ├── wb_counterfactual_all_cities.R
│   │   │   ├── counterfactual_plot_V1V2.R
│   │   │   └── wb_vaccination_impact_plot.R
│   │   └── wb_pinn_sweep_fig.R
│   │   └── wb_s_latent_fig.R
│   │   └── wb_pinn_final_analysis.R
│   │
│   └── python/
│       ├── basic_nn/
│       │   ├── sfnn_architectural_diagram.py
│       │   ├── raytune.py
│       │   ├── raytune.sh
│       │   ├── full_basic_functions.py
│       │   ├── full_basic_raytune.py
│       │   ├── full_basic_raytune.sh
│       │   ├── wb_full_basic.py
│       │   ├── wb_full_basic_optimal.sh
│       │   ├── wb_compare_model_V1V2.py
│       │   └── explain/
│       │       ├── data_process_explain.py
│       │       ├── data_process_explain.sh
│       │       ├── basic_nn_explain.py
│       │       └── basic_nn_explain.sh
│       ├── data_processing/
│       │   └── measles_prevac_loader_V1V2.py
│       └── pinn_experiments/
│           ├── wb_naivepinn_constrained_v3.py
│           ├── wb_tsirpinn_constrained_v3.py
│           ├── wb_tsirpinn_sweep.py
│           ├── wb_pinn_loss_weight_sweep.py
│           ├── wb_collect_sweep_results.py
│           └── wb_extract_s_latent.py
│
├── shell/
│   ├── wb_run_pinn_final.sh
│   └── wb_run_pinn_sweep.sh
│
├── output/                   # generated — not tracked in git
├── experiments/              # generated — not tracked in git
├── Makefile
└── README.md
```

---

## How to Run

All scripts must be run **in the order listed in the Makefile**. The full sequence is:

### Stage 1 — Maps and diagrams
```bash
make createfig1     # West Bengal district map
make createfig2     # SFNN architecture diagram
```

### Stage 2 — TSIR V1V2 reconstruction
```bash
make createtsir
```
Runs the tsiR model with two-dose vaccination (V1V2) for all 19 West Bengal districts. Produces the reconstructed susceptible series and processed case data used by all downstream models.

### Stage 3 — SFNN hyperparameter search and training
```bash
make createsfnn_hp  # raytune hyperparameter search
make createsfnn     # train optimal SFNN, 100 runs
```

### Stage 4 — SFNN explainability (SHAP)
```bash
make createexplain
make createfig3     # explainability plots
make createfig4     # performance comparison plots
```

### Stage 5 — Counterfactual vaccination impact
```bash
make createcounterfactual
make createfig5
```
Estimates measles cases averted by MCV1 + MCV2 across 17 West Bengal districts. Point estimates only — uncertainty analysis was not run.

### Stage 6 — Final PINN training (GPU required, ~5 days)
```bash
# Run inside tmux
tmux new-session -s pinn_final
conda activate finalmlenv
make createpinn
```
Trains NaivePINN v3 (GPU 0) and constrained TSIR-PINN v3 (GPU 1), 100 runs each, 2500 epochs per run.

### Stage 7 — Loss-weight sensitivity sweep (GPU required, ~5 days)
```bash
# Run inside tmux
tmux new-session -s pinn_sweep
conda activate finalmlenv
make createpinnsweep
```
Runs unconstrained TSIR-PINN at three loss weight ratios (10, 35, 10000) × 100 runs each to validate the gradient instability analysis.

### Stage 8 — Collect PINN results and generate figures
```bash
make collectpinn    # collect sweep results to CSV
make createfig6     # loss-weight sweep figure
make createfig7     # S_latent vs S_obs comparison
make createfig8     # PINN final analysis
```

---

## Requirements

### Python (conda environment: `finalmlenv`)
```bash
conda create -n finalmlenv python=3.9
conda activate finalmlenv
pip install torch pandas numpy pyarrow ray[tune]
```

### R (version >= 4.2)
```r
install.packages(c(
    "tidyverse", "ggplot2", "patchwork", "scales",
    "arrow", "readr", "dplyr", "sf", "viridis"
))
devtools::install_github("adbuckner/tsiR")
```

### Hardware
- CUDA GPU required for Stages 6 and 7
- Tested on NVIDIA RTX A5000 (24 GB VRAM)
- Stages 6 and 7 each take approximately 5 days on a single GPU

---

## Acknowledgements

This project extends the methodology of Madden et al. (2024) to a post-vaccination district-level setting. Supervision by Prof. Siuli Mukhopadhyay, Department of Mathematics, IIT Bombay. Computational resources provided by the IIT Bombay GPU server.
