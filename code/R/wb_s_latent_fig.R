suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

# ── OPTIONS ───────────────────────────────────────────────────
SHOW_PANEL_TITLES <- FALSE   # TRUE -> short title above each panel
FIG_W <- 7.1                 # inches (~180 mm, journal double column)
FIG_H <- 6.9                 # inches

# ── PATHS ─────────────────────────────────────────────────────
BASE    <- "/.../.../.../"
IN_DIR  <- paste0(BASE, "experiments/tables/")
OUT_DIR <- paste0(BASE, "experiments/figures/")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── LOAD ──────────────────────────────────────────────────────
cat("Loading data...\n")
path <- paste0(IN_DIR, "s_latent_comparison.csv")
if (!file.exists(path))
  stop("Not found: ", path, "\nRun wb_extract_s_latent.py first.")

df <- read_csv(path, show_col_types = FALSE)
cat(sprintf("Loaded %d training biweeks\n", nrow(df)))

has_latent <- "S_latent_naive" %in% names(df)
has_tsir   <- "S_pred_tsir"    %in% names(df)

# ── SUMMARIES (console only) ──────────────────────────────────
s_obs_mean    <- mean(df$S_obs, na.rm = TRUE)
s_latent_mean <- if (has_latent) mean(df$S_latent_naive, na.rm = TRUE) else NA
s_pred_mean   <- if (has_tsir)   mean(df$S_pred_tsir,    na.rm = TRUE) else NA

cat(sprintf("S_obs mean:    %9.0f\n", s_obs_mean))
if (!is.na(s_latent_mean))
  cat(sprintf("S_latent mean: %9.0f  (ratio %.3f)\n",
              s_latent_mean, s_latent_mean / s_obs_mean))
if (!is.na(s_pred_mean))
  cat(sprintf("S_pred mean:   %9.0f  (ratio %.3f)\n",
              s_pred_mean, s_pred_mean / s_obs_mean))

# ── PALETTE (Okabe–Ito, colour-blind safe) ────────────────────
COL_OBS   <- "#555555"   # dark grey  — S_obs
COL_NAIVE <- "#009E73"   # green      — S_latent NaivePINN
COL_TSIR  <- "#D55E00"   # vermillion — S_pred  TSIR-PINN
COL_REF   <- "#9E9E9E"   # mid grey   — reference lines

# Short legend labels — quantitative detail goes in the caption
LAB_OBS   <- "S_obs (TSIR reconstruction)"
LAB_NAIVE <- "S_latent (NaivePINN)"
LAB_TSIR  <- "S_pred (TSIR-PINN)"

# ── THEME ─────────────────────────────────────────────────────
theme_pub <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(
      panel.border       = element_rect(colour = "#222222",
                                        fill = NA, linewidth = 0.6),
      panel.grid.major.y = element_line(colour = "#ECECEC",
                                        linewidth = 0.3),
      panel.grid.minor   = element_blank(),
      axis.line          = element_blank(),
      axis.title         = element_text(face = "bold"),
      axis.ticks         = element_line(linewidth = 0.35),
      legend.position    = "bottom",
      legend.title       = element_blank(),
      legend.text        = element_text(size = rel(0.95)),
      legend.key.width   = unit(1.1, "cm"),
      legend.margin      = margin(t = 2),
      plot.tag           = element_text(face = "bold", size = rel(1.25)),
      plot.tag.position  = c(0.013, 0.985),
      plot.title         = element_text(face = "bold", hjust = 0,
                                        margin = margin(b = 3)),
      plot.margin        = margin(4, 8, 2, 6)
    )
}

ttl <- function(s) if (SHOW_PANEL_TITLES) s else NULL

# ── PANEL A — TIME SERIES ─────────────────────────────────────
cat("Building Panel A...\n")

lvls <- c(LAB_OBS, LAB_NAIVE, LAB_TSIR)

df_long <- df %>%
  select(time, S_obs, any_of(c("S_latent_naive", "S_pred_tsir"))) %>%
  pivot_longer(-time, names_to = "series", values_to = "susceptible") %>%
  filter(!is.na(susceptible)) %>%
  mutate(series_label = factor(recode(series,
                                      S_obs          = LAB_OBS,
                                      S_latent_naive = LAB_NAIVE,
                                      S_pred_tsir    = LAB_TSIR),
                               levels = lvls))

series_cols <- setNames(c(COL_OBS, COL_NAIVE, COL_TSIR), lvls)
series_lty  <- setNames(c("solid", "solid", "dashed"),   lvls)

# x breaks derived from the data range (robust if span != 2008–2017)
yr        <- range(df$time, na.rm = TRUE)
yr_breaks <- seq(floor(yr[1]), ceiling(yr[2]), by = 2)

panel_a <- ggplot(df_long,
                  aes(time, susceptible,
                      colour = series_label, linetype = series_label)) +
  geom_hline(yintercept = s_obs_mean, colour = COL_REF,
             linewidth = 0.4, linetype = "dotted") +
  geom_line(linewidth = 0.7, alpha = 0.9) +
  scale_colour_manual(values = series_cols, drop = TRUE) +
  scale_linetype_manual(values = series_lty, drop = TRUE) +
  scale_y_continuous(labels = label_comma()) +
  scale_x_continuous(breaks = yr_breaks) +
  guides(colour   = guide_legend(nrow = 1),
         linetype = guide_legend(nrow = 1)) +
  labs(title = ttl("Susceptible trajectories"),
       x = "Year", y = "Susceptible individuals") +
  theme_pub()

# ── PANEL B — SCATTER vs S_obs ────────────────────────────────
cat("Building Panel B...\n")

scatter_rows <- list()
if (has_latent)
  scatter_rows[["naive"]] <- data.frame(
    S_obs = df$S_obs, S_model = df$S_latent_naive, series = LAB_NAIVE)
if (has_tsir)
  scatter_rows[["tsir"]] <- data.frame(
    S_obs = df$S_obs, S_model = df$S_pred_tsir, series = LAB_TSIR)

if (length(scatter_rows) > 0) {
  df_sc <- do.call(rbind, scatter_rows)
  df_sc$series <- factor(df_sc$series, levels = c(LAB_NAIVE, LAB_TSIR))
  
  xy_lo <- min(c(df_sc$S_obs, df_sc$S_model), na.rm = TRUE) * 0.95
  xy_hi <- max(c(df_sc$S_obs, df_sc$S_model), na.rm = TRUE) * 1.03
  
  panel_b <- ggplot(df_sc, aes(S_obs, S_model, colour = series)) +
    geom_abline(slope = 1, intercept = 0, colour = COL_REF,
                linewidth = 0.5, linetype = "dashed") +
    annotate("text", x = xy_hi, y = xy_hi,
             label = "S_model = S_obs", hjust = 1, vjust = 1.5,
             size = 2.7, colour = COL_REF, fontface = "italic") +
    geom_point(alpha = 0.6, size = 1.5) +
    scale_colour_manual(values = setNames(c(COL_NAIVE, COL_TSIR),
                                          c(LAB_NAIVE, LAB_TSIR))) +
    scale_x_continuous(labels = label_comma(), limits = c(xy_lo, xy_hi)) +
    scale_y_continuous(labels = label_comma(), limits = c(xy_lo, xy_hi)) +
    guides(colour = "none") +   # colours already defined in Panel A legend
    labs(title = ttl("Inferred S vs TSIR-reconstructed S_obs"),
         x = "S_obs (TSIR reconstruction)", y = "Model-inferred S") +
    theme_pub()
  
} else {
  panel_b <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "Data not available",
             size = 4, colour = "#888888") +
    theme_void()
}

# ── COMPOSE ───────────────────────────────────────────────────
cat("Composing figure...\n")

# A over B: equal width, equal height.
fig <- panel_a / panel_b +
  plot_layout(heights = c(1, 1), guides = "collect") +
  plot_annotation(tag_levels = "A") &
  theme(legend.position = "bottom")

# ── SAVE ──────────────────────────────────────────────────────
out_base <- paste0(OUT_DIR, "wb_s_latent_comparison")

ggsave(paste0(out_base, ".pdf"), fig,
       width = FIG_W, height = FIG_H, device = cairo_pdf)
cat(sprintf("Saved: %s.pdf\n", out_base))

ggsave(paste0(out_base, ".png"), fig,
       width = FIG_W, height = FIG_H, dpi = 600, bg = "white")
cat(sprintf("Saved: %s.png\n", out_base))

cat("\nDone.\n")