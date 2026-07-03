# ============================================================
# wb_pinn_sweep_predictions.R
#
# Predicted-vs-observed incidence figure + MAE/correlation
# table for the TSIR-PINN loss-weight sweep, in the style of
# the existing pinn_test_predictions / mae_test code.
#
# Reference code facets by MODEL (Naive vs TSIR). This sweep
# has a single model across loss-weight RATIOS, so we facet by
# ratio (10 / 35 / 10000) instead — the equivalent comparison.
#
# Predicted line = MEAN across the 100 runs (matches the
# reference's mean()). A light IQR ribbon is added so the
# spike-vs-zero spread across runs is visible.
#
# NOTE: parameter-evolution-over-epochs is intentionally NOT
# reproduced here — that requires per-epoch vert/amp1/amp2 from
# the _fit_info.parquet files, which are not in the CSV exports.
# ============================================================

library(tidyverse)
library(patchwork)

theme_set(theme_classic())

# -------------------------------
# PATHS
# -------------------------------
BASE         <- "/.../.../.../"
IN_DIR       <- paste0(BASE, "experiments/tables/loss_weight_sweep/")
save_fig_dir <- paste0(BASE, "experiments/figures/")
save_tab_dir <- paste0(BASE, "experiments/tables/loss_weight_sweep/")
dir.create(save_fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(save_tab_dir, showWarnings = FALSE, recursive = TRUE)

preds_path <- paste0(IN_DIR, "sweep_predictions_test.csv")
if (!file.exists(preds_path)) {
  stop(paste0("File not found: ", preds_path))
}

# Ratios kept consistent with the core sweep figures
KEEP_RATIOS <- c(10, 35, 10000)

# -------------------------------
# LOAD + LABEL
# -------------------------------
test_preds <- read_csv(preds_path, show_col_types = FALSE) %>%
  filter(ratio %in% KEEP_RATIOS) %>%
  mutate(
    ratio_f = factor(
      ratio,
      levels = c(10, 35, 10000),
      labels = c(
        "Ratio 10  (below \u03b2eq)",
        "Ratio 35  (at \u03b2eq)",
        "Ratio 10,000  (above \u03b2eq)"
      )
    )
  )

cat(sprintf("Loaded %d rows | ratios: %s | runs/ratio: %d | timesteps: %d\n",
            nrow(test_preds),
            paste(sort(unique(test_preds$ratio)), collapse = ", "),
            dplyr::n_distinct(test_preds$run),
            dplyr::n_distinct(test_preds$time)))

# -------------------------------
# AGGREGATE ACROSS RUNS
#   mean predicted line (+ IQR ribbon); observed is identical
#   across runs so first() is exact.
# -------------------------------
pred_summary <- test_preds %>%
  group_by(ratio_f, time) %>%
  summarize(
    pred_mean = mean(pred_raw, na.rm = TRUE),
    pred_lo   = quantile(pred_raw, 0.25, na.rm = TRUE),
    pred_hi   = quantile(pred_raw, 0.75, na.rm = TRUE),
    obs       = first(obs_raw),
    .groups   = "drop"
  )

# -------------------------------
# PLOT: TEST PREDICTIONS (reference style)
# -------------------------------
long_df <- pred_summary %>%
  select(ratio_f, time, pred_mean, obs) %>%
  pivot_longer(
    cols      = c("pred_mean", "obs"),
    names_to  = "pred_actual",
    values_to = "I"
  ) %>%
  mutate(pred_actual = recode(
    pred_actual,
    pred_mean = "Predicted Incidence",
    obs       = "Observed Incidence"
  ))

tf_I_p <- ggplot() +
  # IQR ribbon across runs (predicted)
  geom_ribbon(
    data = pred_summary,
    aes(x = time, ymin = pred_lo, ymax = pred_hi),
    fill = "blue3", alpha = 0.15
  ) +
  geom_line(
    data = long_df,
    aes(x = time, y = I,
        color = pred_actual, linetype = pred_actual),
    linewidth = 0.6
  ) +
  facet_wrap(~ratio_f, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(
    "Observed Incidence"  = "black",
    "Predicted Incidence" = "blue3"
  )) +
  scale_linetype_manual(values = c(
    "Observed Incidence"  = "solid",
    "Predicted Incidence" = "22"
  )) +
  theme(
    panel.border    = element_rect(colour = "black", fill = NA),
    legend.position = "bottom"
  ) +
  labs(
    x        = "Time",
    y        = "Incidence (cases / biweek)",
    color    = "", linetype = "",
    title    = "TSIR-PINN test predictions vs observed incidence",
    subtitle = paste0(
      "Mean prediction across 100 runs (blue) with IQR ribbon  ",
      "\u00b7  observed (black)  \u00b7  ",
      "predictions collapse toward zero at all ratios")
  )

ggsave(
  paste0(save_fig_dir, "sweep_test_predictions.png"),
  tf_I_p, width = 8, height = 7, dpi = 600, bg = "white"
)
ggsave(
  paste0(save_fig_dir, "sweep_test_predictions.pdf"),
  tf_I_p, width = 8, height = 7
)
cat("Saved: sweep_test_predictions.png / .pdf\n")

# -------------------------------
# METRICS TABLE (reference style: per-ratio MAE + correlation)
#   computed on the per-time mean prediction, like mae_table.
# -------------------------------
mae_table <- test_preds %>%
  group_by(ratio_f, time) %>%
  summarize(
    mean_pred = mean(pred_raw, na.rm = TRUE),
    I         = first(obs_raw),
    .groups   = "drop"
  ) %>%
  group_by(ratio_f) %>%
  summarize(
    mae_I = mean(abs(mean_pred - I), na.rm = TRUE),
    cor_I = suppressWarnings(cor(mean_pred, I, use = "complete.obs")),
    .groups = "drop"
  )

cat("\n=== MAE / correlation by ratio ===\n")
print(as.data.frame(mae_table))

write_csv(mae_table, paste0(save_tab_dir, "sweep_mae_test.csv"))
cat("\nSaved: sweep_mae_test.csv\n")

cat("\nAll done.\n")
