# Deep Neural Network Modelling of Endemic Measles Dynamics in Post-Vaccination Settings

**MSc Thesis — IIT Bombay, Department of Mathematics**
**Student:** Einstein (Roll No. 24N0269)
**Supervisor:** Prof. Siuli Mukhopadhyay
**Year:** 2026

---

## About This Project

This project models endemic measles transmission in West Bengal, India, focusing on South Twenty-Four Parganas district, using biweekly reported case data from 2008 to 2019. The post-vaccination period creates a challenging setting for mechanistic models: the susceptible pool is substantially depleted and seasonally flatter than in pre-vaccination settings, which affects whether physics-informed neural networks can identify seasonal transmission parameters.

We compare four modelling approaches across a 34-biweek forecasting horizon:

- **TSIR V1V2** — Time-Series Susceptible-Infected-Recovered model with two-dose vaccination reconstruction
- **SFNN** — Standard feedforward neural network with lag features
- **NaivePINN v3** — Physics-informed neural network with a freely-learned latent susceptible
- **TSIR-PINN v3** — Physics-informed neural network soft-constrained onto the TSIR susceptible reconstruction

The central finding is that constraining the PINN to the TSIR susceptible reconstruction — which is seasonally flat in this post-vaccination setting — prevents the model from identifying seasonal transmission parameters. The freely-learned NaivePINN recovers a seasonally-structured susceptible series and forecasts more accurately.

This extends the methodology of Madden et al. (2024) to a post-vaccination setting and identifies a setting in which the mechanistic susceptible constraint switches from helping to harming.

---

## Results Summary

Test period: 2017–2019, k = 34 biweek forecast horizon, South Twenty-Four Parganas

| Model | Median RMSE | Beats TSIR V1V2 |
|-------|-------------|-----------------|
| SFNN | 15.7 | 100 / 100 runs |
| **NaivePINN v3** | **29.2** | **64 / 100 runs** |
| TSIR V1V2 | 33.0 | baseline |
| TSIR-PINN (unconstrained) | 38.8 | 0 / 100 runs |
| TSIR-PINN (constrained) | 52.3 | 12 / 100 runs |

**S_latent diagnostics (NaivePINN):**

| Statistic | Value |
|-----------|-------|
| S_latent mean | 271,774 |
| S_obs mean (TSIR) | 240,617 |
| S_pred / S_obs ratio (TSIR-PINN) | 0.987 |
| S_latent seasonal amplitude | 90.1% of mean |
| corr(S_latent seasonal, I seasonal) | 0.46 |
| corr(S_latent 26-wk cycle, TSIR-PINN residual) | 0.51 |

**Counterfactual (17 districts):** Two-dose vaccination averted an estimated 2,640 measles cases (15.4%) over the study period, ranging from 1.6% to 58.2% across districts.

---

## Repository Structure

```
Msc_project/
│
├── code/
│   ├── R/
│   │   ├── 01_data_prep/
│   │   │   ├── wb_data_prep.R
│   │   │   ├── tsir_wb_process_V1V2.R
│   │   │   ├── tsir_wb_run_V1V2.R
│   │   │   └── tsir_susceptibles_gen_V1V2.R
│   │   │
│   │   ├── 02_figures/
│   │   │   ├── wb_city_rmse_pub.R
│   │   │   ├── wb_city_performance_plot.R
│   │   │   ├── wb_pinn_final_analysis.R
│   │   │   ├── wb_pinn_constrained_analysis.R
│   │   │   ├── wb_pinn_sweep_fig.R
│   │   │   ├── wb_s_latent_fig.R
│   │   │   └── optimal_compare_plots.R
│   │   │
│   │   └── 03_counterfactual/
│   │       ├── counterfactual_plot_V1V2.R
│   │       ├── wb_tsir_counterfactual_V1V2.R
│   │       ├── wb_counterfactual_all_cities.R
│   │       └── wb_counterfactual_uncertainty.R
│   │
│   └── python/
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
├── output/                        # generated outputs — not in git
├── experiments/                   # tables and figures — not in git
├── Makefile                       # pipeline execution order
└── README.md
```

---

## How to Run

See the `Makefile` for the full ordered pipeline. The steps in order are:

**Stage 1 — Data preparation**
```bash
Rscript code/R/01_data_prep/wb_data_prep.R
```

**Stage 2 — TSIR V1V2 reconstruction**
```bash
Rscript code/R/01_data_prep/tsir_wb_process_V1V2.R
Rscript code/R/01_data_prep/tsir_wb_run_V1V2.R
Rscript code/R/01_data_prep/tsir_susceptibles_gen_V1V2.R
```

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

**Stage 7 — Figures**
```bash
Rscript code/R/02_figures/wb_pinn_sweep_fig.R
Rscript code/R/02_figures/wb_s_latent_fig.R
Rscript code/R/02_figures/wb_city_rmse_pub.R
Rscript code/R/03_counterfactual/wb_counterfactual_all_cities.R
```

Or run all non-GPU stages at once:
```bash
make all
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
