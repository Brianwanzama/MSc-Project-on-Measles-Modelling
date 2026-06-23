# Deep Neural Network Modelling of Endemic Measles Dynamics in Post-Vaccination Settings

**MSc Thesis — IIT Bombay, Department of Mathematics**
**Student:** Brian Wanzama (Roll No. 24N0269)
**Supervisor:** Prof. Siuli Mukhopadhyay
**Year:** 2026

---

## About This Project

This project models endemic measles transmission in West Bengal, India, focusing on South Twenty-Four Parganas district using biweekly reported case data from 2008 to 2019. The post-vaccination period creates a challenging setting for mechanistic models: the susceptible pool is substantially depleted and seasonally flatter than in pre-vaccination settings which affects whether physics-informed neural networks can identify seasonal transmission parameters.

We compare four modelling approaches across a 34-biweek forecasting horizon:

- **TSIR V1V2** — Time-Series Susceptible-Infected-Recovered model with two-dose vaccination reconstruction
- **SFNN** — Standard feedforward neural network with lag features
- **NaivePINN v3** — Physics-informed neural network with a freely-learned latent susceptible
- **TSIR-PINN v3** — Physics-informed neural network soft-constrained onto the TSIR susceptible reconstruction

The central finding is that constraining the PINN to the TSIR susceptible reconstruction  which is seasonally flat in this post-vaccination setting prevents the model from identifying seasonal transmission parameters. The freely-learned NaivePINN recovers a seasonally-structured susceptible series and forecasts more accurately.

This extends the methodology of Madden et al. (2024) to a post-vaccination setting and identifies a setting in which the mechanistic susceptible constraint switches from helping to harming.

---

**Counterfactual (17 districts):** Two-dose vaccination averted an estimated 2,640 measles cases (15.4%) over the study period, ranging from 1.6% to 58.2% across districts.

---

## Repository Structure

```
Msc_project/
│
├── code/
│   ├── R/
│   │   ├── 01_data_prep/
│   │   │   ├── tsir_functions
│   │   │   ├── tsir_wb_run_V1V2.R
│   │   │   ├── tsir_susceptibles_gen_V1V2.R
|   |   |   ├── tsir_wb_process_V1V2.R
│   │   │   ├── optimal_basic_nn_process.R     # post-process SFNN runs
│   │   │   └── cases_process.R                # reported-cases processing
│   │   │
│   │   ├── 02_figures/
│   │   │   ├── optimal_compare_plots.R        # Fig 2: SFNN vs TSIR
│   │   │   ├── wb_city_performance_plot.R
│   │   │   ├── wb_population_rmse.R           # was "Population plots rmse.R"
│   │   │   ├── wb_sfnn_feature_importance.R
│   │   │   ├── wb_pinn_final_analysis.R
│   │   │   ├── wb_pinn_constrained_analysis.R
│   │   │   ├── wb_pinn_sweep_fig.R
│   │   │   └── wb_s_latent_fig.R
│   │   │
│   │   └── 03_counterfactual/
│   │       ├── counterfactual_plot_V1V2.R
│   │       ├── wb_tsir_counterfactual_V1V2.R
│   │       ├── wb_counterfactual_all_cities.R
│   │     
│   │
│   └── python/
│       ├── basic_nn/
│       │   ├── plotting/
│       │   │   └── gen_plot.sh  # Fig 1: network architecture
|       |   |--Meascles_Prevac_Loader           
│       │   ├── full_basic_raytune.sh         # SFNN hyperparameter search
│       │   └── full_basic_optimal.sh         # SFNN fit (optimal HPs)
│       │
│       └── pinn_experiments/
│           ├── wb_naivepinn_constrained_v3.py   # final NaivePINN
│           ├── wb_tsirpinn_constrained_v3.py    # final TSIR-PINN
│           ├── wb_tsirpinn.py                   # unconstrained (sweep base)
│           ├── wb_tsirpinn_sweep.py             # loss-weight sweep
│           ├── wb_collect_sweep_results.py      # collect sweep CSVs
│           └── wb_extract_s_latent.py           # extract S_latent
│
├── shell/
│   ├── wb_run_pinn_final.sh      # 100-run final PINN training
│   └── wb_run_pinn_sweep.sh      # 300-run loss-weight sweep
│
├── output/                       # generated outputs — not in git
│   ├── data/
│   ├── models/
│   └── figures/
├── experiments/                  # tables and figures — 
├── Makefile                      # pipeline execution order
└── README.md
```

---

## How to Run

See the `Makefile` for the full ordered pipeline. The non-GPU stages can be built at once with:

```bash
make all
```

### Pipeline critical information

**Stage 3 — SFNN (100 runs)**
```bash
bash shell/wb_full_basic_optimal_V1V2.sh
```

**Stage 4 — Final PINN training (100 runs each, ~5 days, GPU required)**
```bash
# Run inside tmux
tmux new-session -s pinn_final
conda activate finalmlenv
bash shell/wb_run_pinn_final.sh
```

**Stage 5 — Loss-weight sweep (300 runs, ~5 days, GPU required)**
```bash
# Run inside tmux
tmux new-session -s pinn_sweep
conda activate finalmlenv
bash shell/wb_run_pinn_sweep.sh
```

**Stage 6 — Collect results**
```bash
conda activate finalmlenv
python code/python/pinn_experiments/wb_collect_sweep_results.py
python code/python/pinn_experiments/wb_extract_s_latent.py
```
**Stage 7 — Figures** (see the figure targets below)
```bash
make createfig1 createfig2
Rscript code/R/02_figures/wb_pinn_sweep_fig.R
Rscript code/R/02_figures/wb_s_latent_fig.R
Rscript code/R/02_figures/wb_city_rmse_pub.R
Rscript code/R/03_counterfactual/wb_counterfactual_all_cities.R
```

---


### `createfig1` — network architecture

```makefile
# Fig 1: output/figures/feedforward_network_structure.png
createfig1:
	mkdir -p output/figures
	cd code/python/basic_nn/plotting && ./gen_plot.sh
```

### `createfig2` — SFNN vs TSIR

```makefile
# Fig 2: output/figures/rmse_reg_and_gain_nn_tsir_k_facet.png
createfig2:
	# 1. TSIR reconstruction
	mkdir -p output/data/tsir/wb/raw output/data/tsir/wb/processed output/data/tsir_susceptibles
	cd code/R/01_data_prep && Rscript tsir_wb_run_V1V2.R
	cd code/R/01_data_prep && Rscript tsir_wb_process_V1V2.R
	cd code/R/01_data_prep && Rscript tsir_susceptibles_gen_V1V2.R
	# 2. SFNN hyperparameter search + fit
	mkdir -p output/data/basic_nn_optimal/raytune_hp_optim output/models/basic_nn_optimal
	cd code/python/basic_nn && ./full_basic_raytune.sh
	cd code/python/basic_nn && ./full_basic_optimal.sh
	# 3. Post-process + plot
	cd code/R/01_data_prep && Rscript optimal_basic_nn_process.R
	cd code/R/01_data_prep && Rscript cases_process.R
	cd code/R/02_figures && Rscript optimal_compare_plots.R
```

---

## Requirements

**Python (conda environment: `finalmlenv`)**
```
python >= 3.9
torch >= 2.0
pandas
numpy
pyarrow
```

**R >= 4.2**
```r
install.packages(c("tidyverse", "ggplot2", "patchwork",
                   "scales", "arrow", "readr", "dplyr"))
devtools::install_github("adbuckner/tsiR")
```

**Hardware**
- CUDA GPU required for Stages 4 and 5
- Tested on NVIDIA RTX A5000 (24GB)
- Stages 4 and 5 each take approximately 5 days

---

## Key Design Decisions and Limitations

**Two susceptible reconstructions.** The TSIR model uses the raw tsiR output (S̄ = 132,952). The PINN feature file uses a separately preprocessed susceptible series (S̄ = 240,617). Both are documented in the respective pipeline scripts.

**Constrained model ceiling.** The constrained TSIR-PINN runs (RMSE = 52.3) used BETA_MAX = 31.6, which is 10.5% below the correct β_eq = 35.07 from the prefit data. This is disclosed in the code and in the thesis Limitations section.

**Single district.** All PINN results are for South Twenty-Four Parganas only. Whether the findings generalise across the S̄/Ī range is an open question.

**Association not causation.** The association between S_latent's seasonal structure and NaivePINN's accuracy is demonstrated (r = 0.51). Causation would require a progressive-relaxation experiment not run here.

---

## Acknowledgements

This project extends the methodology of:

> Madden, J.M. et al. (2024). Physics-informed neural networks for measles transmission dynamics. *[journal]*.

Supervision: Prof. Siuli Mukhopadhyay, Department of Mathematics, IIT Bombay.
Computational resources: IIT Bombay GPU server.
