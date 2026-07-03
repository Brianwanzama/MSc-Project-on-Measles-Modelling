# ============================================================
# wb_city_rmse_pub.R
# PUBLICATION-QUALITY per-district RMSE comparison
# SFNN vs TSIR V1V2 — k=1 and k=34
#
# OUTPUT (all saved separately):
#   03a_rmse_scatter_k1.pdf/.png   — scatter k=1
#   03b_rmse_scatter_k34.pdf/.png  — scatter k=34
#   05a_rmse_bar_k1.pdf/.png       — bar chart k=1
#   05b_rmse_bar_k34.pdf/.png      — bar chart k=34
#
# Each file is a standalone publication-quality figure
# ============================================================

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(forcats)
library(stringr)
library(scales)

# ── PATHS ─────────────────────────────────────────────────────
BASE    <- "/.../.../.../"
COMP    <- paste0(BASE, "output/data/comparison/")
OUT_DIR <- paste0(BASE, "experiments/figures/")
DATA    <- paste0(BASE, "data/")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── COLOUR PALETTE (Wong 2011, colorblind-safe) ───────────────
COL_SFNN  <- "#0072B2"   # blue       — SFNN better
COL_TSIR  <- "#D55E00"   # vermillion — TSIR better
COL_ZERO  <- "#222222"   # dark reference line
COL_TREND <- "#444444"   # OLS trend line
COL_RIB   <- "#BBBBBB"   # OLS CI ribbon

# ── DISTRICT NAME SHORTENING ──────────────────────────────────
shorten_city <- function(x) {
  x |>
    str_replace("South Twenty Four Parganas", "S. 24 Pgs")    |>
    str_replace("North Twenty Four Parganas", "N. 24 Pgs")    |>
    str_replace("Paschim Medinipur",          "P. Medinipur")  |>
    str_replace("Purba Medinipur",            "Pu. Medinipur") |>
    str_replace("Dakshin Dinajpur",           "D. Dinajpur")   |>
    str_replace("Uttar Dinajpur",             "U. Dinajpur")
}

# ── PUBLICATION THEME ─────────────────────────────────────────
theme_pub <- function(base_size = 12) {
  theme_classic(base_size = base_size) +
    theme(
      panel.background    = element_rect(fill = "white",
                                         colour = NA),
      panel.border        = element_rect(colour = "#111111",
                                         fill = NA,
                                         linewidth = 0.8),
      panel.grid.major.y  = element_line(colour = "#E0E0E0",
                                         linewidth = 0.35),
      panel.grid.major.x  = element_blank(),
      panel.grid.minor    = element_blank(),
      axis.title          = element_text(size = rel(0.92),
                                         colour = "#222222",
                                         face = "bold"),
      axis.text           = element_text(size = rel(0.88),
                                         colour = "#222222"),
      axis.ticks          = element_line(colour = "#555555",
                                         linewidth = 0.45),
      axis.ticks.length   = unit(3, "pt"),
      axis.line           = element_blank(),
      strip.background    = element_blank(),
      strip.text          = element_text(size = rel(0.95),
                                         face = "bold",
                                         colour = "#111111"),
      legend.position     = "bottom",
      legend.title        = element_blank(),
      legend.text         = element_text(size = 11,
                                         colour = "#222222"),
      legend.key          = element_rect(fill = NA,
                                         colour = NA),
      legend.key.width    = unit(2.0, "cm"),
      legend.key.height   = unit(0.45, "cm"),
      legend.spacing.x    = unit(0.6, "cm"),
      legend.background   = element_rect(fill = NA,
                                         colour = NA),
      legend.margin       = margin(t = 6, b = 4),
      plot.title          = element_text(size = rel(1.10),
                                         face = "bold",
                                         colour = "#000000",
                                         hjust = 0,
                                         margin = margin(b = 4)),
      plot.title.position = "plot",
      plot.subtitle       = element_text(size = rel(0.86),
                                         colour = "#555555",
                                         hjust = 0,
                                         margin = margin(b = 8)),
      plot.caption        = element_text(size = rel(0.80),
                                         colour = "#777777",
                                         hjust = 0,
                                         margin = margin(t = 8)),
      plot.margin         = margin(10, 16, 8, 10)
    )
}

# ── SAVE HELPER ───────────────────────────────────────────────
save_plot <- function(p, name, w, h) {
  path <- file.path(OUT_DIR, name)
  ggsave(paste0(path, ".pdf"), p,
         width = w, height = h, device = cairo_pdf)
  ggsave(paste0(path, ".png"), p,
         width = w, height = h, dpi = 300, bg = "white")
  cat(sprintf("Saved: %s.pdf / .png\n", name))
}

# ── LOAD MERGED PREDICTIONS ───────────────────────────────────
cat("Loading merged predictions...\n")
merged <- read_csv(
  paste0(COMP, "merged_predictions.csv"),
  show_col_types = FALSE)

cat(sprintf(
  "Rows: %d | k values: %s | Districts: %d\n",
  nrow(merged),
  paste(sort(unique(merged$k)), collapse = ", "),
  length(unique(merged$city))))

# ── LOAD POPULATION DATA ──────────────────────────────────────
cat("Loading population data...\n")
pop_raw <- read_csv(
  paste0(DATA, "inferred_popn.csv"),
  show_col_types = FALSE) |>
  pivot_longer(-1,
               names_to  = "city",
               values_to = "pop") |>
  rename(time = 1) |>
  mutate(year = as.integer(floor(time))) |>
  filter(year >= 2017) |>
  group_by(city) |>
  summarise(mean_pop = mean(pop, na.rm = TRUE),
            .groups  = "drop") |>
  arrange(desc(mean_pop)) |>
  mutate(
    pop_rank   = row_number(),
    pop_M      = round(mean_pop / 1e6, 2),
    city_short = shorten_city(city),
    city_label = paste0(city_short, "  (", pop_M, "M)")
  )

cat("\nPopulation ranking:\n")
print(pop_raw |> select(pop_rank, city_short, pop_M))

# ── COMPUTE PER-DISTRICT RMSE ─────────────────────────────────
rmse_city <- merged |>
  filter(train_test == "test",
         k %in% c(1, 34)) |>
  group_by(city, k) |>
  summarise(
    rmse_sfnn = sqrt(mean((pred_cases - obs_cases)^2,
                          na.rm = TRUE)),
    rmse_tsir = sqrt(mean((tsir_cases - obs_cases)^2,
                          na.rm = TRUE)),
    mean_obs  = mean(obs_cases, na.rm = TRUE),
    n         = n(),
    .groups   = "drop"
  ) |>
  mutate(
    rmse_diff   = rmse_tsir - rmse_sfnn,
    sfnn_better = rmse_sfnn < rmse_tsir,
    winner      = ifelse(sfnn_better,
                         "SFNN better",
                         "TSIR better")
  ) |>
  left_join(
    pop_raw |> select(city, mean_pop, pop_rank,
                      pop_M, city_short, city_label),
    by = "city"
  ) |>
  arrange(k, pop_rank)

# ── SPEARMAN CORRELATIONS ─────────────────────────────────────
cat("\n=== SPEARMAN: Population vs RMSE difference ===\n")
spearman_results <- list()
for (k_val in c(1, 34)) {
  d  <- rmse_city |> filter(k == k_val)
  ct <- cor.test(d$mean_pop, d$rmse_diff,
                 method = "spearman")
  spearman_results[[as.character(k_val)]] <- ct
  cat(sprintf("k=%2d:  rho = %.3f   p = %.4f\n",
              k_val, ct$estimate, ct$p.value))
}

# ── SAVE RMSE TABLE ───────────────────────────────────────────
write_csv(rmse_city,
          paste0(COMP, "rmse_per_city_k1_k34.csv"))
cat("\nRMSE table saved.\n")

# ── PRINT SUMMARY TABLE ───────────────────────────────────────
cat("\n============================================================\n")
cat("FINAL RMSE TABLE — k=1 and k=34\n")
cat("============================================================\n")
rmse_city |>
  select(pop_rank, city_short, pop_M, k,
         rmse_sfnn, rmse_tsir, rmse_diff, winner) |>
  mutate(across(where(is.numeric), ~round(.x, 1))) |>
  arrange(k, pop_rank) |>
  print(n = 40)

# ── CLAIM TEST ────────────────────────────────────────────────
cat("\n=== CLAIM TEST: SFNN vs TSIR by city size ===\n")
for (k_val in c(1, 34)) {
  d     <- rmse_city |> filter(k == k_val)
  large <- d |> filter(pop_rank <= 5)
  small <- d |> filter(pop_rank > 14)
  cat(sprintf(
    "\nk=%d:\n  Large (top 5):    SFNN better in %d/5\n",
    k_val, sum(large$sfnn_better)))
  cat(sprintf(
    "  Small (bottom 5): SFNN better in %d/5\n",
    sum(small$sfnn_better)))
  cat(sprintf(
    "  Overall: SFNN better in %d/%d districts\n",
    sum(d$sfnn_better), nrow(d)))
}

# ══════════════════════════════════════════════════════════════
# SCATTER PLOTS — Population vs RMSE difference
# Saved as TWO SEPARATE FILES: k=1 and k=34
# ══════════════════════════════════════════════════════════════

make_scatter <- function(df, k_val, panel_letter,
                         subtitle_text, caption_text) {
  
  rho <- spearman_results[[as.character(k_val)]]$estimate
  pv  <- spearman_results[[as.character(k_val)]]$p.value
  
  pv_lab <- ifelse(
    pv < 0.001, "p < 0.001",
    sprintf("p = %.3f", pv))
  
  ann_lab <- sprintf(
    "Spearman \u03c1 = %.3f\n%s", rho, pv_lab)
  
  k_lab <- ifelse(
    k_val == 1,
    "k = 1  (\u22482 weeks ahead)",
    "k = 34  (\u224817 months ahead)")
  
  x_range <- range(df$pop_M,     na.rm = TRUE)
  y_range <- range(df$rmse_diff, na.rm = TRUE)
  y_span  <- diff(y_range)
  
  # Expand x slightly for label room
  x_lo <- x_range[1] - 0.05 * diff(x_range)
  x_hi <- x_range[2] + 0.10 * diff(x_range)
  
  # Expand y for annotations at top
  y_lo <- y_range[1] - 0.18 * y_span
  y_hi <- y_range[2] + 0.22 * y_span
  
  ggplot(df, aes(x      = pop_M,
                 y      = rmse_diff,
                 colour = winner,
                 label  = city_short)) +
    
    # ── Zero reference ────────────────────────────────────
    geom_hline(yintercept = 0,
               linetype   = "dashed",
               colour     = COL_ZERO,
               linewidth  = 0.65) +
    
    # ── OLS trend + 95% CI ────────────────────────────────
    geom_smooth(aes(x = pop_M, y = rmse_diff),
                method      = "lm",
                se          = TRUE,
                colour      = COL_TREND,
                fill        = COL_RIB,
                linewidth   = 1.1,
                linetype    = "solid",
                inherit.aes = FALSE,
                alpha       = 0.38) +
    
    # ── Points ────────────────────────────────────────────
    geom_point(size  = 5.5,
               alpha = 1.0,
               shape = 16) +
    
    # ── District labels ───────────────────────────────────
    geom_text(vjust    = -0.85,
              hjust    = 0.5,
              size     = 3.6,
              colour   = "#111111",
              fontface = "bold") +
    
    # ── Spearman annotation — top right ───────────────────
    annotate("text",
             x        = x_hi - 0.01 * diff(c(x_lo, x_hi)),
             y        = y_hi - 0.01 * diff(c(y_lo, y_hi)),
             label    = ann_lab,
             hjust    = 1,
             vjust    = 1,
             size     = 4.2,
             colour   = COL_SFNN,
             fontface = "bold") +
    
    # ── Panel letter — top left ───────────────────────────
    annotate("text",
             x        = x_lo + 0.01 * diff(c(x_lo, x_hi)),
             y        = y_hi - 0.01 * diff(c(y_lo, y_hi)),
             label    = panel_letter,
             hjust    = 0,
             vjust    = 1,
             size     = 6.0,
             fontface = "bold",
             colour   = "#000000") +
    
    # ── Scales ────────────────────────────────────────────
    scale_colour_manual(
      values = c("SFNN better" = COL_SFNN,
                 "TSIR better" = COL_TSIR),
      name   = NULL,
      guide  = guide_legend(
        override.aes = list(size = 5)
      )
    ) +
    
    scale_x_continuous(
      limits = c(x_lo, x_hi),
      labels = function(x) paste0(x, "M"),
      expand = expansion(0)
    ) +
    
    scale_y_continuous(
      limits = c(y_lo, y_hi),
      labels = label_number(accuracy = 1),
      expand = expansion(0)
    ) +
    
    labs(
      title    = k_lab,
      subtitle = subtitle_text,
      caption  = caption_text,
      x        = paste0(
        "District population",
        "  (millions, 2017\u20132019 mean)"),
      y        = paste0(
        "RMSE difference:  TSIR \u2212 SFNN",
        "  (cases / biweek)")
    ) +
    
    theme_pub()
}

# ── Build k=1 scatter ─────────────────────────────────────────
cat("\nBuilding scatter k=1...\n")

p_scat_k1 <- make_scatter(
  df            = rmse_city |> filter(k == 1),
  k_val         = 1,
  panel_letter  = "A",
  subtitle_text = paste0(
    "West Bengal, 2017\u20132019  \u00b7  19 districts  \u00b7  ",
    "Blue = SFNN achieves lower RMSE  \u00b7  ",
    "Red = TSIR achieves lower RMSE  \u00b7  ",
    "Grey band = 95% CI of OLS trend"),
  caption_text  = paste0(
    "Positive y-values indicate SFNN outperforms TSIR.  ",
    "Spearman rank correlation: \u03c1 and p-value annotated top right.  ",
    "OLS trend fitted across all 19 districts.")
)

save_plot(p_scat_k1,
          "03a_rmse_scatter_k1",
          w = 11, h = 9)

# ── Build k=34 scatter ────────────────────────────────────────
cat("Building scatter k=34...\n")

p_scat_k34 <- make_scatter(
  df            = rmse_city |> filter(k == 34),
  k_val         = 34,
  panel_letter  = "B",
  subtitle_text = paste0(
    "West Bengal, 2017\u20132019  \u00b7  19 districts  \u00b7  ",
    "Blue = SFNN achieves lower RMSE  \u00b7  ",
    "Red = TSIR achieves lower RMSE  \u00b7  ",
    "Grey band = 95% CI of OLS trend"),
  caption_text  = paste0(
    "Positive y-values indicate SFNN outperforms TSIR.  ",
    "Spearman rank correlation: \u03c1 and p-value annotated top right.  ",
    "OLS trend fitted across all 19 districts.")
)

save_plot(p_scat_k34,
          "03b_rmse_scatter_k34",
          w = 11, h = 9)

# ══════════════════════════════════════════════════════════════
# BAR CHARTS — RMSE difference by district
# Saved as TWO SEPARATE FILES: k=1 and k=34
# ══════════════════════════════════════════════════════════════

make_bar <- function(df, k_val, panel_letter,
                     subtitle_text, caption_text) {
  
  k_lab <- ifelse(
    k_val == 1,
    "k = 1  (\u22482 weeks ahead)",
    "k = 34  (\u224817 months ahead)")
  
  # Order districts by population — largest at top
  df <- df |>
    mutate(city_fac = fct_reorder(city_label, -pop_rank))
  
  x_lim <- max(abs(df$rmse_diff), na.rm = TRUE) * 1.30
  
  ggplot(df, aes(x    = rmse_diff,
                 y    = city_fac,
                 fill = winner)) +
    
    # ── Bars ──────────────────────────────────────────────
    geom_col(alpha = 0.90,
             width = 0.72) +
    
    # ── Zero reference ────────────────────────────────────
    geom_vline(xintercept = 0,
               colour     = COL_ZERO,
               linewidth  = 0.70) +
    
    # ── Value labels ──────────────────────────────────────
    geom_text(aes(
      label = sprintf("%+.1f", rmse_diff),
      hjust = ifelse(rmse_diff >= 0, -0.12, 1.12)
    ),
    size     = 3.4,
    colour   = "#111111",
    fontface = "bold") +
    
    # ── Panel letter — top left ───────────────────────────
    annotate("text",
             x        = -x_lim * 0.97,
             y        = nlevels(df$city_fac) + 0.60,
             label    = panel_letter,
             hjust    = 0,
             vjust    = 1,
             size     = 6.0,
             fontface = "bold",
             colour   = "#000000") +
    
    # ── Scales ────────────────────────────────────────────
    scale_fill_manual(
      values = c("SFNN better" = COL_SFNN,
                 "TSIR better" = COL_TSIR),
      name   = NULL,
      guide  = guide_legend(
        override.aes = list(size = 5)
      )
    ) +
    
    scale_x_continuous(
      limits = c(-x_lim, x_lim),
      expand = expansion(mult = c(0, 0)),
      labels = label_number(accuracy = 1)
    ) +
    
    labs(
      title    = k_lab,
      subtitle = subtitle_text,
      caption  = caption_text,
      x        = paste0(
        "RMSE difference:  TSIR \u2212 SFNN",
        "  (cases / biweek)"),
      y        = NULL
    ) +
    
    theme_pub() +
    theme(
      axis.text.y = element_text(
        size   = 10,
        colour = "#111111")
    )
}

# ── Build k=1 bar ─────────────────────────────────────────────
cat("Building bar chart k=1...\n")

p_bar_k1 <- make_bar(
  df            = rmse_city |> filter(k == 1),
  k_val         = 1,
  panel_letter  = "C",
  subtitle_text = paste0(
    "West Bengal, 2017\u20132019  \u00b7  ",
    "Districts ordered by population (largest at top)  \u00b7  ",
    "Blue = SFNN achieves lower RMSE  \u00b7  ",
    "Red = TSIR achieves lower RMSE"),
  caption_text  = paste0(
    "RMSE in raw cases per biweek.  ",
    "Positive values indicate SFNN outperforms TSIR.  ",
    "Population in parentheses = 2017\u20132019 mean district population.")
)

save_plot(p_bar_k1,
          "05a_rmse_bar_k1",
          w = 12, h = 9)

# ── Build k=34 bar ────────────────────────────────────────────
cat("Building bar chart k=34...\n")

p_bar_k34 <- make_bar(
  df            = rmse_city |> filter(k == 34),
  k_val         = 34,
  panel_letter  = "D",
  subtitle_text = paste0(
    "West Bengal, 2017\u20132019  \u00b7  ",
    "Districts ordered by population (largest at top)  \u00b7  ",
    "Blue = SFNN achieves lower RMSE  \u00b7  ",
    "Red = TSIR achieves lower RMSE"),
  caption_text  = paste0(
    "RMSE in raw cases per biweek.  ",
    "Positive values indicate SFNN outperforms TSIR.  ",
    "Population in parentheses = 2017\u20132019 mean district population.")
)

save_plot(p_bar_k34,
          "05b_rmse_bar_k34",
          w = 12, h = 9)

cat("\n============================================================\n")
cat("All four figures saved to:\n")
cat(paste0("  ", OUT_DIR, "\n"))
cat("  03a_rmse_scatter_k1.pdf/.png\n")
cat("  03b_rmse_scatter_k34.pdf/.png\n")
cat("  05a_rmse_bar_k1.pdf/.png\n")
cat("  05b_rmse_bar_k34.pdf/.png\n")
cat("============================================================\n")
cat("\nDone.\n")
