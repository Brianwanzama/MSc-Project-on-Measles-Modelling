# ============================================================
# Makefile
# Deep Neural Network Modelling of Endemic Measles Dynamics
# in Post-Vaccination West Bengal
#
# MSc Thesis — IIT Bombay
# Student: Einstein (Roll 24N0269)
# Supervisor: Prof. Siuli Mukhopadhyay
#
# Scripts must be run in the order listed below.
# GPU stages (PINN) require CUDA and run inside tmux.
# ============================================================

# ── FIGURE 1: Map of West Bengal districts ────────────────────
# Output: output/figures/wb_district_map.png
createfig1:
	mkdir -p output/figures/
	cd code/r/maps && Rscript map_plot.R

# ── FIGURE 2: SFNN architecture diagram ───────────────────────
# Output: output/figures/sfnn_architecture_diagram.png
createfig2:
	mkdir -p output/figures/
	cd code/python/basic_nn && python sfnn_architectural_diagram.py

# ── STAGE 1: TSIR V1V2 reconstruction ────────────────────────
# Runs tsiR model with two-dose vaccination (V1V2)
# Output: output/data/tsir/wb/
createtsir:
	mkdir -p output/data/tsir/wb/raw
	cd code/r/tsir && Rscript tsir_run_functions.R
	cd code/r/tsir && Rscript tsir_susceptibles_gen_V1V2.R
	cd code/r/tsir && Rscript tsir_wb_run_V1V2.R
	mkdir -p output/data/tsir/wb/processed
	cd code/r/tsir && Rscript tsir_wb_process_V1V2.R
	cd code/r/basic_nn && Rscript cases_process.R

# ── STAGE 2: SFNN hyperparameter optimisation ─────────────────
# Loads West Bengal data and runs raytune search
# Output: output/data/basic_nn_optimal/raytune_hp_optim/
createsfnn_hp:
	mkdir -p output/data/prefit_cases1
	cd code/python/data_processing && python measles_prevac_loader_V1V2.py
	mkdir -p output/data/basic_nn_optimal/raytune_hp_optim
	cd code/python/basic_nn && python raytune.py
	cd code/python/basic_nn && ./raytune.sh

# ── STAGE 3: SFNN training (100 runs) ─────────────────────────
# Trains optimal SFNN across all West Bengal districts
# Output: output/models/basic_nn_optimal/
createsfnn:
	cd code/python/basic_nn && python full_basic_functions.py
	cd code/python/basic_nn && python full_basic_raytune.py
	cd code/python/basic_nn && ./full_basic_raytune.sh
	mkdir -p output/models/basic_nn_optimal
	cd code/python/basic_nn && python wb_full_basic.py
	cd code/python/basic_nn && ./wb_full_basic_optimal.sh
	cd code/r/basic_nn && Rscript wb_optimal_V1V2.R
	cd code/python/basic_nn && python wb_compare_model_V1V2.py

# ── STAGE 4: SFNN explainability (SHAP) ──────────────────────
# Computes SHAP feature importance for SFNN
# Output: output/data/basic_nn_optimal/explain/
createexplain:
	mkdir -p output/data/basic_nn_optimal/prefit
	cd code/python/basic_nn/explain && python data_process_explain.py
	cd code/python/basic_nn/explain && ./data_process_explain.sh
	mkdir -p output/data/basic_nn_optimal/explain
	cd code/python/basic_nn/explain && python basic_nn_explain.py
	cd code/python/basic_nn/explain && ./basic_nn_explain.sh
	cd code/r/basic_nn && Rscript optimal_basic_nn_process.R

# ── FIGURE 3: SFNN explainability plots ──────────────────────
# Output: output/figures/sfnn_explainability*.png
createfig3:
	mkdir -p output/figures/
	cd code/r/basic_nn && Rscript sfnn_explainability.R
	cd code/r/basic_nn && Rscript explain_plot.R
	cd code/r/basic_nn && Rscript wb_sfnn_feature_importance.R

# ── FIGURE 4: SFNN performance comparison ────────────────────
# Output: output/figures/wb_sfnn_performance*.png
createfig4:
	mkdir -p output/figures/
	cd code/r/basic_nn && Rscript optimal_compare_plot.R
	cd code/r/basic_nn && Rscript wb_city_performance_plot.R
	cd code/r/basic_nn && Rscript population_plots_rmse.R
	mkdir -p output/tables/
	cd code/r/basic_nn && Rscript wb_city_rmse_table.R

# ── STAGE 5: Counterfactual vaccination impact ────────────────
# Estimates measles cases averted by MCV1 + MCV2 in WB
# Output: output/data/counterfactual/
# Note: uncertainty analysis not run — point estimates only
createcounterfactual:
	mkdir -p output/data/counterfactual
	cd code/r/counterfactual && Rscript wb_tsir_counterfactual.R
	cd code/r/counterfactual && Rscript wb_counterfactual_all_cities.R

# ── FIGURE 5: Counterfactual plots ───────────────────────────
# Output: output/figures/wb_counterfactual*.png
createfig5:
	mkdir -p output/figures/
	cd code/r/counterfactual && Rscript counterfactual_plot_V1V2.R
	cd code/r/counterfactual && Rscript wb_vaccination_impact_plot.R

# ── STAGE 6: Final PINN training (GPU, ~5 days) ───────────────
# NaivePINN v3 (GPU 0) and TSIR-PINN v3 (GPU 1)
# 100 runs each, 2500 epochs per run
# Run inside tmux: tmux new-session -s pinn_final
# Output: output/models/pinn_experiments/wb_pinn_final/
createpinn:
	mkdir -p output/models/pinn_experiments/wb_pinn_final
	mkdir -p output/logs/pinn_final
	cd code/python/pinn_experiments && \
	    echo "Training NaivePINN v3 and TSIR-PINN v3..." && \
	    bash $(HOME)/Msc_project/code/wb_run_pinn_final.sh

# ── STAGE 7: Loss-weight sweep (GPU, ~5 days) ─────────────────
# Unconstrained TSIR-PINN at 3 ratios x 100 runs
# Ratios: 10 (below beta_eq), 35 (at), 10000 (above)
# Run inside tmux: tmux new-session -s pinn_sweep
# Output: output/models/pinn_experiments/wb_pinn_sweep/
createpinnsweep:
	mkdir -p output/models/pinn_experiments/wb_pinn_sweep
	mkdir -p output/logs/pinn_sweep
	cd code/python/pinn_experiments && python wb_tsirpinn_sweep.py
	bash $(HOME)/Msc_project/code/wb_run_pinn_sweep.sh

# ── STAGE 8: Collect and extract sweep results ────────────────
# Output: experiments/tables/loss_weight_sweep/
#         experiments/tables/s_latent_comparison.csv
collectpinn:
	mkdir -p experiments/tables/loss_weight_sweep
	cd code/python/pinn_experiments && \
	    python wb_collect_sweep_results.py
	cd code/python/pinn_experiments && \
	    python wb_extract_s_latent.py
	cd code/python/pinn_experiments && \
	    python wb_pinn_loss_weight_sweep.py

# ── FIGURE 6: Loss-weight sweep figure ───────────────────────
# Three-panel: vert distribution, amp1 distribution, RMSE
# Output: experiments/figures/wb_pinn_sweep_main.pdf/.png
createfig6:
	mkdir -p experiments/figures
	cd code/r && Rscript wb_pinn_sweep_fig.R

# ── FIGURE 7: S_latent vs S_obs comparison ───────────────────
# Output: experiments/figures/wb_s_latent_comparison.pdf/.png
createfig7:
	mkdir -p experiments/figures
	cd code/r && Rscript wb_s_latent_fig.R

# ── FIGURE 8: PINN final analysis ────────────────────────────
# Parameter trajectories, prediction intervals, RMSE
# Output: experiments/figures/wb_pinn_final_analysis.pdf/.png
createfig8:
	mkdir -p experiments/figures
	cd code/r && Rscript wb_pinn_final_analysis.R

# ── ALL (non-GPU stages only) ─────────────────────────────────
all:
	make createfig1
	make createfig2
	make createtsir
	make createsfnn_hp
	make createsfnn
	make createexplain
	make createfig3
	make createfig4
	make createcounterfactual
	make createfig5
	make collectpinn
	make createfig6
	make createfig7
	make createfig8
	@echo "Done. GPU stages (createpinn, createpinnsweep) must be run manually in tmux."
