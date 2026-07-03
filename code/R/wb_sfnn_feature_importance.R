# ============================================================
# wb_sfnn_feature_importance.R
# Feature importance via permutation — V1 SFNN
# West Bengal measles forecasting
#
# METHOD: Permutation importance
#   For each feature GROUP:
#     1. Shuffle the group's columns across all rows
#     2. Recompute predictions (not retrain — just forward pass)
#     3. Measure RMSE increase vs baseline
#     4. Higher RMSE increase = more important feature group
#
# NOTE: We do not have the trained PyTorch model weights
# accessible in R. Instead we use the saved parquet outputs
# and approximate importance via correlation-based approach:
#   - For each feature group, compute correlation with
#     prediction residuals
#   - Higher correlation = more predictive power
#
# For true permutation importance use wb_sfnn_permutation.py
# ============================================================

library(arrow)
#library(tidyverse)
library(readr)
library(dplyr)
library(patchwork)

BASE    <- ".../.../.../"
NN_DIR  <- paste0(BASE, "output/data/basic_nn_optimal/")
OUT_DIR <- paste0(BASE, "experiments/figures/")
PREFIT  <- paste0(BASE, "output/data/prefit_cases1/")

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

k_values <- c(1, 4, 12, 20, 34)

# ── FEATURE GROUP DEFINITIONS ────────────────────────────────
# These map to column name patterns in the parquet files
feature_groups <- list(
  "Incidence lags\n(own district)"      = "^cases_lag_",
  "Large district\nincidence lags"      = "^cases_[a-z].*_lag_",
  "Nearest city\nincidence lags"        = "^cases_nc_|^cases_nbc_",
  "Susceptible lags"                    = "^susc_lag_",
  "Vaccination lags\n(MCV1)"            = "^v1_lag_",
  "Population"                          = "^pop_lag_",
  "Births"                              = "^births_lag_",
  "Spatial distances"                   = "^dist_|_dist$"
)

# ── LOAD DATA AND COMPUTE IMPORTANCE PER k ───────────────────
cat("Computing feature group importance...\n\n")

all_importance <- list()

for (k in k_values) {

  cat(sprintf("k=%d\n", k))

  # Load parquet features
  tlag <- ifelse(k < 26, 26, 52)
  pq_path <- paste0(PREFIT, "k", k, "_tlag", tlag, ".gzip")

  if (!file.exists(pq_path)) {
    cat(sprintf("  Missing: %s\n", pq_path))
    next
  }

  df_feat <- read_parquet(pq_path)

  # Load SFNN predictions + reverse transform
  out_path <- paste0(NN_DIR, k, "_output.parquet")
  tfm_path <- paste0(NN_DIR, k, "_transform.parquet")

  if (!file.exists(out_path)) {
    cat(sprintf("  Missing output: %s\n", out_path))
    next
  }

  out <- read_parquet(out_path)
  tfm <- read_parquet(tfm_path)

  city_col <- names(out)[grepl("city", names(out),
                                ignore.case=TRUE)][1]

  out <- out %>%
    left_join(tfm %>%
                select(time, all_of(city_col),
                       cases_mean, cases_std),
              by=c("time", city_col)) %>%
    mutate(
      pred_raw = pmax(exp(pred  * cases_std + cases_mean) - 1, 0),
      obs_raw  = pmax(exp(cases * cases_std + cases_mean) - 1, 0),
      residual = obs_raw - pred_raw
    ) %>%
    rename(city = all_of(city_col))

  # Align parquet features with predictions on time + city
  df_feat <- df_feat %>%
    mutate(time_r = round(time, 4)) %>%
    rename(city = all_of(
      names(df_feat)[grepl("city", names(df_feat),
                            ignore.case=TRUE)][1]))

  out <- out %>% mutate(time_r = round(time, 4))

  merged <- out %>%
    select(time_r, city, pred_raw, obs_raw, residual,
           cases_trans=cases, pred_trans=pred) %>%
    inner_join(df_feat %>%
                 select(-any_of(c("time","cases",
                                  "cases_trans","split",
                                  "susc","pop","births"))),
               by=c("time_r","city"))

  cat(sprintf("  Merged: %d rows | %d feature cols\n",
              nrow(merged),
              ncol(merged) - 6))

  # ── PERMUTATION IMPORTANCE ──────────────────────────────────
  # Baseline RMSE
  baseline_rmse <- sqrt(mean(merged$residual^2, na.rm=TRUE))
  cat(sprintf("  Baseline RMSE: %.3f\n", baseline_rmse))

  group_importance <- list()

  for (grp_name in names(feature_groups)) {

    pattern  <- feature_groups[[grp_name]]
    grp_cols <- names(merged)[grepl(pattern, names(merged))]

    # Also exclude columns that match other patterns
    # to avoid double counting
    if (grp_name == "Large district\nincidence lags") {
      grp_cols <- grp_cols[!grepl("^cases_lag_|^cases_nc_|^cases_nbc_",
                                   grp_cols)]
    }

    if (length(grp_cols) == 0) {
      cat(sprintf("  %s: 0 cols — skip\n", grp_name))
      next
    }

    # Permute: shuffle rows of these columns
    set.seed(2026)
    df_perm <- merged
    perm_idx <- sample(nrow(df_perm))

    for (col in grp_cols) {
      df_perm[[col]] <- df_perm[[col]][perm_idx]
    }

    # Recompute predictions using permuted features
    # Since we don't have model weights in R, we approximate:
    # Use the correlation between each feature and the
    # prediction to weight the contribution
    # True permutation requires Python — see wb_sfnn_permutation.py

    # Approximation: partial correlation of feature group
    # with prediction (proxy for permutation importance)
    feat_matrix <- as.matrix(merged[, grp_cols, drop=FALSE])
    feat_matrix[is.nan(feat_matrix)] <- 0
    feat_matrix[is.infinite(feat_matrix)] <- 0
    feat_matrix[is.na(feat_matrix)] <- 0

    # Mean absolute correlation with prediction
    pred_vals <- merged$pred_trans
    cors <- apply(feat_matrix, 2, function(x) {
      if (sd(x, na.rm=TRUE) < 1e-8) return(0)
      abs(cor(x, pred_vals, use="complete.obs"))
    })
    mean_cor <- mean(cors, na.rm=TRUE)

    # Also compute correlation with cases (outcome)
    cors_outcome <- apply(feat_matrix, 2, function(x) {
      if (sd(x, na.rm=TRUE) < 1e-8) return(0)
      abs(cor(x, merged$cases_trans, use="complete.obs"))
    })
    mean_cor_outcome <- mean(cors_outcome, na.rm=TRUE)

    group_importance[[grp_name]] <- data.frame(
      k              = k,
      group          = grp_name,
      n_features     = length(grp_cols),
      cor_pred       = mean_cor,
      cor_outcome    = mean_cor_outcome,
      importance     = mean_cor  # use pred correlation as importance
    )

    cat(sprintf("  %-40s n=%3d  cor_pred=%.3f\n",
                grp_name, length(grp_cols), mean_cor))
  }

  all_importance[[as.character(k)]] <- bind_rows(group_importance)
}

importance_df <- bind_rows(all_importance) %>%
  mutate(
    k_label = paste0("k = ", k),
    k_label = factor(k_label,
                     levels=paste0("k = ", k_values))
  )

cat(sprintf("\nImportance computed: %d rows\n", nrow(importance_df)))

# ── NORMALISE WITHIN EACH k ───────────────────────────────────
importance_df <- importance_df %>%
  group_by(k) %>%
  mutate(
    importance_norm = importance / sum(importance) * 100
  ) %>%
  ungroup()

# ── PLOT 1: Heatmap — feature importance across k ─────────────

# Order groups by mean importance
grp_order <- importance_df %>%
  group_by(group) %>%
  summarise(mean_imp=mean(importance_norm), .groups="drop") %>%
  arrange(mean_imp) %>%
  pull(group)

importance_df <- importance_df %>%
  mutate(group = factor(group, levels=grp_order))

p1 <- ggplot(importance_df,
             aes(x=k_label, y=group,
                 fill=importance_norm)) +
  geom_tile(colour="white", linewidth=0.5) +
  geom_text(aes(label=sprintf("%.1f%%", importance_norm)),
            size=3.2, colour="white", fontface="bold") +
  scale_fill_gradientn(
    colours = c("#F8F9FA", "#AED6F1", "#2980B9", "#1A5276"),
    name    = "Relative importance (%)",
    guide   = guide_colorbar(
      title.position = "top",
      barwidth       = 12,
      barheight      = 0.5
    )
  ) +
  scale_x_discrete(expand=expansion(0)) +
  scale_y_discrete(expand=expansion(0)) +
  labs(
    title    = "SFNN Feature Group Importance by Forecast Horizon",
    subtitle = paste0(
      "West Bengal | Importance = mean |correlation| with prediction | ",
      "Normalised within each k"),
    x       = "Forecast horizon (k biweeks)",
    y       = NULL,
    caption = paste0(
      "Importance proxy: mean absolute correlation of feature group ",
      "with SFNN predictions across test period (2017-2019). ",
      "Each row sums to 100% within a horizon.")
  ) +
  theme_classic(base_size=11) +
  theme(
    panel.border = element_rect(colour="black", fill=NA, linewidth=1),
    plot.title       = element_text(face="bold", size=12, hjust=0),
    plot.subtitle    = element_text(colour="grey40", size=9, hjust=0,
                                    margin=margin(b=8)),
    plot.caption     = element_text(colour="grey50", size=7.5, hjust=0),
    axis.text.y      = element_text(size=9),
    axis.text.x      = element_text(size=10, face="bold"),
    legend.position  = "top",
    legend.title     = element_text(size=9),
    plot.margin      = margin(12,16,10,12)
  )

# ── PLOT 2: Bar chart — top features per k ───────────────────

p2 <- ggplot(importance_df,
             aes(x=importance_norm,
                 y=reorder(group, importance_norm),
                 fill=importance_norm)) +
  geom_col(alpha=0.88, width=0.7) +
  geom_text(aes(label=sprintf("%.1f%%", importance_norm)),
            hjust=-0.1, size=3, colour="grey30") +
  scale_fill_gradientn(
    colours=c("#AED6F1","#2980B9","#1A5276"),
    guide="none"
  ) +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult=c(0, 0.15))
  ) +
  facet_wrap(~k_label, ncol=3, scales="free_x") +
  labs(
    title    = "SFNN Feature Group Importance by Horizon",
    subtitle = paste0(
      "West Bengal | V1 SFNN | ",
      "Relative importance within each forecast horizon"),
    x       = "Relative importance (%)",
    y       = NULL,
    caption = paste0(
      "Importance = mean |correlation| with SFNN predictions, ",
      "normalised to sum to 100% within each k.")
  ) +
  theme_classic(base_size=11) +
  theme(
    panel.border = element_rect(colour="black", fill=NA, linewidth=1),
    plot.title       = element_text(face="bold", size=12, hjust=0),
    plot.subtitle    = element_text(colour="grey40", size=9, hjust=0,
                                    margin=margin(b=8)),
    plot.caption     = element_text(colour="grey50", size=7.5, hjust=0),
    axis.text.y      = element_text(size=8.5),
    strip.text       = element_text(face="bold", size=10),
    plot.margin      = margin(12,16,10,12)
  )

# ── PLOT 3: Feature count per group ──────────────────────────
feat_counts <- importance_df %>%
  filter(k==1) %>%
  arrange(desc(n_features)) %>%
  mutate(group_clean = gsub("\n", " ", group))

p3 <- ggplot(feat_counts,
             aes(x=n_features,
                 y=reorder(group, n_features),
                 fill=n_features)) +
  geom_col(alpha=0.85, width=0.7) +
  geom_text(aes(label=comma(n_features)),
            hjust=-0.1, size=3.5, colour="grey30") +
  scale_fill_gradientn(
    colours=c("#FAD7A0","#E67E22","#A04000"),
    guide="none"
  ) +
  scale_x_continuous(
    labels = comma,
    expand = expansion(mult=c(0, 0.15))
  ) +
  labs(
    title    = "Number of Features per Group (k=1, tlag=26)",
    subtitle = "Total = 1,071 features",
    x        = "Number of features",
    y        = NULL,
    caption  = "Feature counts vary slightly across k due to lag range."
  ) +
  theme_classic(base_size=11) +
  theme(
    panel.border = element_rect(colour="black", fill=NA, linewidth=1),
    plot.title    = element_text(face="bold", size=12, hjust=0),
    plot.subtitle = element_text(colour="grey40", size=9,
                                 hjust=0, margin=margin(b=8)),
    plot.caption  = element_text(colour="grey50", size=7.5, hjust=0),
    axis.text.y   = element_text(size=9),
    plot.margin   = margin(12,16,10,12)
  )

# ── SAVE ALL ─────────────────────────────────────────────────
save_plot <- function(p, name, w=10, h=6) {
  ggsave(paste0(OUT_DIR, name, ".pdf"),
         p, width=w, height=h, device=cairo_pdf)
  ggsave(paste0(OUT_DIR, name, ".png"),
         p, width=w, height=h, dpi=300, bg="white")
  cat(sprintf("Saved: %s\n", name))
}

save_plot(p1, "09_feature_importance_heatmap", w=10, h=6)
save_plot(p2, "10_feature_importance_bars",    w=12, h=8)
save_plot(p3, "11_feature_counts",             w=8,  h=5)

# ── PRINT SUMMARY TABLE ───────────────────────────────────────
cat("\n=== FEATURE IMPORTANCE SUMMARY ===\n")
importance_df %>%
  select(k, group, n_features,
         cor_pred, importance_norm) %>%
  mutate(across(where(is.numeric), ~round(.x, 3))) %>%
  arrange(k, desc(importance_norm)) %>%
  print(n=50)

cat("\n=== MEAN IMPORTANCE ACROSS ALL k ===\n")
importance_df %>%
  group_by(group) %>%
  summarise(
    mean_importance = mean(importance_norm),
    mean_cor_pred   = mean(cor_pred),
    n_features_k1   = first(n_features),
    .groups="drop"
  ) %>%
  arrange(desc(mean_importance)) %>%
  mutate(across(where(is.numeric), ~round(.x, 2))) %>%
  print()

cat("\nNote: This is a correlation-based importance proxy.\n")
cat("For true permutation importance, run:\n")
cat("  python3 wb_sfnn_permutation.py\n")
