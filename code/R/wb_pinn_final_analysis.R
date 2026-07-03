# ============================================================
# wb_pinn_final_analysis.R
# PINN Final Analysis — West Bengal
# NaivePINN v3 + TSIRPINN v3 | k=34 | 100 runs
# Aligned with Madden et al. (2024) figure style
#
# FIGURES:
#   pinn_final_test_pred_param.png  — A/B combined
#   pinn_final_test_pred.png        — predictions only
#   pinn_final_param_evolution.png  — parameter only
#
# TABLES:
#   pinn_final_rmse_table.txt       — LaTeX RMSE table
# ============================================================

library(arrow)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(readr)

# ── PATHS ─────────────────────────────────────────────────────
BASE     <- "/.../.../.../"
PINN_DIR <- paste0(BASE,
  "output/models/pinn_experiments/wb_pinn_final/")
FIG_DIR  <- paste0(BASE, "experiments/figures/pinn/")
TAB_DIR  <- paste0(BASE, "experiments/tables/")

dir.create(FIG_DIR, showWarnings=FALSE, recursive=TRUE)
dir.create(TAB_DIR, showWarnings=FALSE, recursive=TRUE)

# ── SETTINGS ──────────────────────────────────────────────────
K       <- 34
TLAG    <- 52
CITY    <- "South Twenty Four Parganas"
N_RUNS  <- 100

# ── HELPERS ───────────────────────────────────────────────────
get_model_name <- function(text) {
  if (grepl("tsirpinn", text)) return("tsirpinn")
  if (grepl("naivepinn", text)) return("naivepinn")
  return("neither")
}

get_run <- function(x) {
  as.integer(gsub("^.*?_run_([0-9]+)_.*$", "\\1", x))
}

# ── LOAD FIT INFO (parameter evolution) ───────────────────────
cat("Loading fit_info files...\n")

dirs <- list.files(PINN_DIR, full.names=TRUE)

fit_info <- lapply(
  grep("_fit_info", dirs, value=TRUE),
  function(x) {
    dat <- tryCatch(
      arrow::read_parquet(x), error=function(e) NULL)
    if (is.null(dat)) return(NULL)
    req <- c("ode_loss","I_loss","I_test_loss",
             "vert","amp1","amp2")
    if (!all(req %in% colnames(dat))) return(NULL)
    dat <- dat[, req]
    dat$vert  <- unlist(dat$vert)
    dat$amp1  <- unlist(dat$amp1)
    dat$amp2  <- unlist(dat$amp2)
    dat$model <- get_model_name(x)
    dat$run   <- get_run(x)
    dat$iter  <- seq_len(nrow(dat))
    dat
  }) |>
  (\(x) Filter(Negate(is.null), x))() |>
  bind_rows() |>
  mutate(model = factor(model,
    levels = c("naivepinn","tsirpinn"),
    labels = c("Naive-PINN Model","TSIR-PINN Model")))

cat(sprintf("fit_info: %d rows | %d runs\n",
            nrow(fit_info),
            length(unique(fit_info$run))))

# ── LOAD TEST PREDICTIONS ─────────────────────────────────────
cat("Loading test_predictions files...\n")

test_preds <- lapply(
  grep("_test_predictions", dirs, value=TRUE),
  function(x) {
    dat <- tryCatch(
      arrow::read_parquet(x), error=function(e) NULL)
    if (is.null(dat)) return(NULL)
    dat$model <- get_model_name(x)
    dat$run   <- get_run(x)
    dat
  }) |>
  (\(x) Filter(Negate(is.null), x))() |>
  bind_rows() |>
  mutate(model = factor(model,
    levels = c("naivepinn","tsirpinn"),
    labels = c("Naive-PINN Model","TSIR-PINN Model")))

cat(sprintf("test_preds: %d rows\n", nrow(test_preds)))
cat(sprintf("Models:     %s\n",
            paste(levels(test_preds$model), collapse=", ")))
cat(sprintf("Runs:       %d\n",
            length(unique(test_preds$run))))

# ── DIAGNOSTICS ───────────────────────────────────────────────
cat("\nPrediction diagnostics:\n")
test_preds |>
  group_by(model) |>
  summarise(
    mean_pred = mean(I_pred, na.rm=TRUE),
    mean_obs  = mean(I,      na.rm=TRUE),
    min_pred  = min(I_pred,  na.rm=TRUE),
    max_pred  = max(I_pred,  na.rm=TRUE),
    .groups="drop"
  ) |> print()

# ── RMSE PER RUN ──────────────────────────────────────────────
rmse_per_run <- test_preds |>
  group_by(model, run) |>
  summarise(
    rmse = sqrt(mean((I_pred - I)^2, na.rm=TRUE)),
    .groups="drop"
  )

cat("\nRMSE summary:\n")
rmse_per_run |>
  group_by(model) |>
  summarise(
    median  = median(rmse),
    mean    = mean(rmse),
    std     = sd(rmse),
    min     = min(rmse),
    max     = max(rmse),
    beats33 = sum(rmse < 33.0),
    .groups ="drop"
  ) |> print()

# ── PUBLICATION THEME ─────────────────────────────────────────
theme_pub <- function(base_size=11) {
  theme_classic(base_size=base_size) +
    theme(
      panel.border     = element_rect(colour="black",
                                      fill=NA, linewidth=1),
      strip.background = element_rect(colour="black",
                                      fill="white",
                                      linewidth=0.8),
      strip.text       = element_text(face="bold", size=10),
      axis.text        = element_text(size=9, colour="black"),
      axis.title       = element_text(size=10),
      legend.position  = "bottom",
      legend.title     = element_blank(),
      legend.text      = element_text(size=9),
      legend.key.width = unit(1.5,"cm"),
      plot.title       = element_text(face="bold", size=11,
                                      hjust=0),
      plot.subtitle    = element_text(colour="grey40", size=8.5,
                                      hjust=0,
                                      margin=margin(b=6)),
      plot.caption     = element_text(colour="grey50", size=7.5,
                                      hjust=0),
      plot.margin      = margin(10,14,8,10)
    )
}

save_plot <- function(p, name, w=9, h=6) {
  ggsave(paste0(FIG_DIR, name, ".pdf"),
         p, width=w, height=h, device=cairo_pdf)
  ggsave(paste0(FIG_DIR, name, ".png"),
         p, width=w, height=h, dpi=300, bg="white")
  cat(sprintf("Saved: %s\n", name))
}

# ── FIGURE A: TEST PREDICTIONS ────────────────────────────────
tf_I_p <- test_preds |>
  group_by(time_original, model) |>
  summarise(
    I_pred = mean(I_pred, na.rm=TRUE),
    I      = first(I),
    .groups="drop"
  ) |>
  pivot_longer(
    cols      = c("I_pred","I"),
    names_to  = "pred_actual",
    values_to = "I_val"
  ) |>
  mutate(pred_actual = case_when(
    pred_actual == "I_pred" ~ "Predicted Incidence",
    pred_actual == "I"      ~ "Observed Incidence"
  )) |>
  ggplot(aes(x=time_original, y=I_val,
             color=pred_actual,
             linetype=pred_actual)) +
  geom_line(linewidth=0.8) +
  facet_wrap(~model, ncol=1) +
  scale_color_manual(
    values=c("Observed Incidence"  = "black",
             "Predicted Incidence" = "blue3")
  ) +
  scale_linetype_manual(
    values=c("Observed Incidence"  = "solid",
             "Predicted Incidence" = "dashed")
  ) +
  scale_x_continuous(
    breaks = seq(2017, 2019, by=0.5),
    labels = c("2017.0","2017.5","2018.0",
               "2018.5","2019.0")
  ) +
  scale_y_continuous(
    expand = expansion(mult=c(0.02, 0.08))
  ) +
  labs(
    x        = "Year",
    y        = "Incidence",
    color    = "",
    linetype = "",
    title    = paste0("Final PINN — Test Predictions | k=",K),
    subtitle = paste0(
      "South Twenty Four Parganas | ",
      "Mean across ",N_RUNS," runs | ")
  ) +
  theme_pub()

# ── FIGURE B: PARAMETER EVOLUTION ─────────────────────────────
fit_info_p2 <- fit_info |>
  group_by(iter, model) |>
  summarise(
    vert = mean(vert, na.rm=TRUE),
    amp1 = mean(amp1, na.rm=TRUE),
    amp2 = mean(amp2, na.rm=TRUE),
    .groups="drop"
  ) |>
  pivot_longer(
    cols      = c("vert","amp1","amp2"),
    names_to  = "param",
    values_to = "value"
  ) |>
  ggplot(aes(x=iter, y=value,
             color=param, linetype=param)) +
  geom_line(linewidth=0.8) +
  scale_color_manual(
    values=c("vert"="black",
             "amp1"="blue3",
             "amp2"="goldenrod3"),
    labels=c(
      "vert" = expression(nu),
      "amp1" = expression(alpha[1]),
      "amp2" = expression(alpha[2]))
  ) +
  scale_linetype_manual(
    values=c("vert"="solid",
             "amp1"="dashed",
             "amp2"="dotted"),
    labels=c(
      "vert" = expression(nu),
      "amp1" = expression(alpha[1]),
      "amp2" = expression(alpha[2]))
  ) +
  facet_wrap(~model, ncol=1, scales="free_y") +
  labs(
    x        = "Epoch",
    y        = "Parameter Value",
    color    = "",
    linetype = "",
    title    = paste0(
      "\u03b2(t) Parameter Evolution | v3 Final"),
    subtitle = paste0(
      "Mean across ",N_RUNS," runs")
  ) +
  theme_pub()

# ── COMBINED A/B FIGURE ───────────────────────────────────────
scale_factor <- 3

fig_combined <- tf_I_p + fit_info_p2 +
  plot_layout(widths=c(1,1), ncol=2, nrow=1) +
  plot_annotation(tag_levels="A")

save_plot(fig_combined,
          "pinn_final_combined",
          w = 3*scale_factor,
          h = 2*scale_factor)

# Individual figures
save_plot(tf_I_p,    "pinn_final_test_pred",    w=6, h=6)
save_plot(fit_info_p2,"pinn_final_param_evol",  w=6, h=6)

# ── RMSE TABLE — LaTeX ────────────────────────────────────────
rmse_table <- rmse_per_run |>
  group_by(model) |>
  summarise(
    `Median RMSE`  = round(median(rmse), 2),
    `Mean RMSE`    = round(mean(rmse), 2),
    `Std RMSE`     = round(sd(rmse), 2),
    `Min RMSE`     = round(min(rmse), 2),
    `Max RMSE`     = round(max(rmse), 2),
    `Beats TSIR`   = paste0(sum(rmse < 33.0),
                             "/", N_RUNS),
    .groups="drop"
  ) |>
  rename(Model = model)

# Add SFNN and TSIR reference rows
ref_rows <- tibble(
  Model          = c("SFNN (reference)",
                     "TSIR V1V2 (benchmark)"),
  `Median RMSE`  = c(15.70, 33.00),
  `Mean RMSE`    = c(NA_real_, NA_real_),
  `Std RMSE`     = c(NA_real_, NA_real_),
  `Min RMSE`     = c(NA_real_, NA_real_),
  `Max RMSE`     = c(NA_real_, NA_real_),
  `Beats TSIR`   = c("100/100", "baseline")
)

rmse_full <- bind_rows(rmse_table, ref_rows)

# Print to console
cat("\n=== FINAL RMSE TABLE ===\n")
print(rmse_full)

# Save as LaTeX — manual construction (no kableExtra)
fmt <- function(x) ifelse(is.na(x), "---", sprintf("%.2f", x))

lines <- c(
  "\\begin{table}[H]",
  "\\centering",
  "\\small",
  paste0("\\caption{PINN model RMSE comparison at $k=34$, ",
         "South Twenty Four Parganas, 2017--2019. ",
         "100 independent runs. TSIR V1V2 RMSE = 33.0 (benchmark).}"),
  "\\label{tab:pinn_rmse}",
  "\\begin{tabular}{lrrrrrr}",
  "\\toprule",
  paste0("\\textbf{Model} & \\textbf{Median} & ",
         "\\textbf{Mean} & \\textbf{Std} & ",
         "\\textbf{Min} & \\textbf{Max} & ",
         "\\textbf{Beats TSIR} \\\\"),
  "\\midrule"
)

for (i in seq_len(nrow(rmse_full))) {
  r <- rmse_full[i, ]
  row_str <- paste0(
    r$Model, " & ",
    fmt(r$`Median RMSE`), " & ",
    fmt(r$`Mean RMSE`),   " & ",
    fmt(r$`Std RMSE`),    " & ",
    fmt(r$`Min RMSE`),    " & ",
    fmt(r$`Max RMSE`),    " & ",
    r$`Beats TSIR`, " \\\\"
  )
  lines <- c(lines, row_str)
}

lines <- c(lines,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(lines,
  paste0(TAB_DIR, "pinn_final_rmse_table.txt"))

cat(sprintf("\nTable saved: %spinn_final_rmse_table.txt\n",
            TAB_DIR))

# ── MAE AND CORRELATION TABLE ─────────────────────────────────
# Aligned with Madden et al. (2024) Table 1 format
# Mean prediction per model then compute MAE and Cor
cat("\n=== MAE AND CORRELATION TABLE ===\n")

mae_cor_table <- test_preds |>
  group_by(time_original, model) |>
  summarise(
    mean_pred = mean(I_pred, na.rm=TRUE),
    I         = first(I),
    .groups   = "drop"
  ) |>
  group_by(model) |>
  summarise(
    `MAE`  = round(mean(abs(mean_pred - I),
                        na.rm=TRUE), 2),
    `Cor`  = round(cor(mean_pred, I,
                       use="complete.obs"), 3),
    .groups = "drop"
  ) |>
  rename(Model = model)

# Add SFNN and TSIR reference rows
mae_cor_ref <- tibble(
  Model = c("SFNN (reference)",
            "TSIR V1V2 (benchmark)"),
  MAE   = c(NA_real_, NA_real_),
  Cor   = c(NA_real_, NA_real_)
)

mae_cor_full <- bind_rows(mae_cor_table, mae_cor_ref)
print(mae_cor_full)

# Save as LaTeX
fmt2 <- function(x) ifelse(is.na(x), "---",
                             sprintf("%.3f", x))

lines2 <- c(
  "\\begin{table}[H]",
  "\\centering",
  "\\small",
  paste0("\\caption{Mean absolute error (MAE) and ",
         "Pearson correlation of ensemble mean predictions ",
         "vs.\\ observed incidence at $k=34$, ",
         "South Twenty Four Parganas, 2017--2019. ",
         "Ensemble = mean across 100 runs.}"),
  "\\label{tab:pinn_mae_cor}",
  "\\begin{tabular}{lrr}",
  "\\toprule",
  paste0("\\textbf{Model} & ",
         "\\textbf{$\\widehat{\\text{MAE}}(\\hat{I}, I)$} & ",
         "\\textbf{$\\widehat{\\text{Cor}}(\\hat{I}, I)$} \\\\"),
  "\\midrule"
)

for (i in seq_len(nrow(mae_cor_full))) {
  r <- mae_cor_full[i, ]
  row_str <- paste0(
    r$Model, " & ",
    fmt2(r$MAE), " & ",
    fmt2(r$Cor), " \\\\"
  )
  lines2 <- c(lines2, row_str)
}

lines2 <- c(lines2,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(lines2,
  paste0(TAB_DIR, "pinn_final_mae_cor_table.txt"))

cat(sprintf("Table saved: %spinn_final_mae_cor_table.txt\n",
            TAB_DIR))
cat(sprintf("\nAll figures saved to: %s\n", FIG_DIR))
cat("\nDone.\n")
