
library(arrow)
library(dplyr)
library(ggplot2)
library(patchwork)
library(scales)

# ── PATHS ─────────────────────────────────────────────────────
BASE      <- "/.../.../.../output/data/basic_nn_optimal/"
FIG_DIR   <- "/.../.../.../experiments/figures/"
CITY      <- "South Twenty Four Parganas"
ks        <- c(1, 4, 12, 20, 34)
CUTOFF    <- 2017.0
TRAIN_END <- 2017.0
TEST_END  <- 2020.0

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ── COLOUR PALETTE (Wong 2011, colorblind-safe) ───────────────
# Black, orange, sky blue, bluish green, yellow,
# blue, vermillion, reddish purple
COL_OBS    <- "#000000"   # black — observed
COL_PRED   <- "#D55E00"   # vermillion — predicted
COL_TRAIN  <- "#F5F5F5"   # near-white grey — train region
COL_TEST   <- "#EBF4FB"   # pale blue — test region
COL_SPLIT  <- "#999999"   # mid-grey — split line
COL_ANNOT  <- "#0072B2"   # blue — RMSE annotation text

# ── HORIZON LABELS (panel titles) ─────────────────────────────
# Concise: just k and approximate calendar length
horizon_labels <- c(
  "1"  = "k = 1  (2 weeks)",
  "4"  = "k = 4  (2 months)",
  "12" = "k = 12  (6 months)",
  "20" = "k = 20  (10 months)",
  "34" = "k = 34  (17 months)"
)

# Panel letter labels A–E
panel_letters <- c("1"="A", "4"="B", "12"="C", "20"="D", "34"="E")

# ── PUBLICATION THEME ─────────────────────────────────────────
# Clean white background, minimal grid, journal-standard fonts
theme_pub <- function(base_size = 10.5) {
  theme_classic(base_size = base_size) +
    theme(
      # Panel
      panel.background  = element_rect(fill = "white", colour = NA),
      panel.border      = element_rect(
        colour = "#333333", fill = NA,
        linewidth = 0.55),
      panel.grid.major.y = element_line(
        colour = "#EBEBEB", linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      
      # Axes
      axis.title.x = element_text(
        size = rel(0.88), colour = "#333333",
        margin = margin(t = 5)),
      axis.title.y = element_text(
        size = rel(0.88), colour = "#333333",
        margin = margin(r = 5)),
      axis.text    = element_text(
        size = rel(0.82), colour = "#444444"),
      axis.ticks   = element_line(
        colour = "#888888", linewidth = 0.35),
      axis.ticks.length = unit(2.5, "pt"),
      axis.line    = element_blank(),   # border replaces axis lines
      
      # Strip (facet label)
      strip.background = element_rect(
        fill = "#F8F8F8", colour = "#CCCCCC",
        linewidth = 0.4),
      strip.text       = element_text(
        size = rel(0.88), face = "bold",
        colour = "#222222",
        margin = margin(t=3, b=3)),
      
      # Legend
      legend.position    = "bottom",
      legend.title       = element_blank(),
      legend.text        = element_text(
        size = rel(0.88), colour = "#333333"),
      legend.key         = element_rect(fill = NA, colour = NA),
      legend.key.width   = unit(1.8, "cm"),
      legend.key.height  = unit(0.4, "cm"),
      legend.spacing.x   = unit(0.6, "cm"),
      legend.background  = element_rect(fill = NA, colour = NA),
      legend.margin      = margin(t = 4, b = 2),
      
      # Titles
      plot.title          = element_text(
        size = rel(0.95), face = "bold",
        colour = "#111111", hjust = 0,
        margin = margin(b = 2)),
      plot.title.position = "plot",
      plot.subtitle       = element_text(
        size = rel(0.82), colour = "#666666",
        hjust = 0, margin = margin(b = 6)),
      plot.caption        = element_text(
        size = rel(0.78), colour = "#888888",
        hjust = 0, margin = margin(t = 6)),
      plot.margin         = margin(6, 10, 4, 6)
    )
}

# ── LOAD DATA ─────────────────────────────────────────────────
load_k_data <- function(k) {
  output_path    <- paste0(BASE, k, "_output.parquet")
  transform_path <- paste0(BASE, k, "_transform.parquet")
  
  if (!file.exists(output_path)) {
    warning(sprintf("Missing: %s", output_path))
    return(NULL)
  }
  
  df  <- arrow::read_parquet(output_path)
  tfm <- arrow::read_parquet(transform_path)
  
  city_col <- names(df)[grepl(
    "city|district", names(df), ignore.case = TRUE)][1]
  
  df <- df |>
    filter(.data[[city_col]] == CITY) |>
    left_join(
      tfm |>
        filter(.data[[city_col]] == CITY) |>
        select(time, cases_mean, cases_std),
      by = "time"
    ) |>
    arrange(time) |>
    mutate(
      pred_raw  = pmax(exp(pred  * cases_std + cases_mean) - 1, 0),
      cases_raw = pmax(exp(cases * cases_std + cases_mean) - 1, 0),
      period    = ifelse(time >= CUTOFF, "Test", "Train"),
      k_label   = horizon_labels[as.character(k)],
      k_val     = k
    )
  
  df
}

cat("Loading data...\n")
data_list <- lapply(ks, load_k_data)
names(data_list) <- as.character(ks)

# ── COMPUTE RMSE PER PANEL ────────────────────────────────────
# Only on test period — what gets reported in the thesis
compute_rmse <- function(df) {
  if (is.null(df)) return(NA_real_)
  df |>
    filter(period == "Test") |>
    summarise(rmse = sqrt(mean((pred_raw - cases_raw)^2,
                               na.rm = TRUE))) |>
    pull(rmse)
}

rmse_vals <- sapply(data_list, compute_rmse)
cat("\nTest-period RMSE per horizon:\n")
for (k in ks) {
  cat(sprintf("  k=%2d:  RMSE = %.1f\n", k, rmse_vals[as.character(k)]))
}

# ── PLOT FUNCTION ─────────────────────────────────────────────
plot_k <- function(df, k) {
  if (is.null(df)) return(NULL)
  
  rmse  <- rmse_vals[as.character(k)]
  label <- panel_letters[as.character(k)]
  
  # RMSE annotation string
  rmse_label <- sprintf("RMSE = %.1f", rmse)
  
  # y range for annotation placement
  y_max <- max(df$cases_raw, df$pred_raw, na.rm = TRUE)
  y_ann <- y_max * 0.93
  
  ggplot(df, aes(x = time)) +
    
    # ── Train region shading ─────────────────────────────────
    annotate("rect",
             xmin = min(df$time), xmax = CUTOFF,
             ymin = -Inf, ymax = Inf,
             fill = COL_TRAIN, alpha = 0.55) +
    
    # ── Test region shading ──────────────────────────────────
    annotate("rect",
             xmin = CUTOFF, xmax = max(df$time),
             ymin = -Inf, ymax = Inf,
             fill = COL_TEST, alpha = 0.55) +
    
    # ── Train / test split line ──────────────────────────────
    geom_vline(xintercept = CUTOFF,
               colour = COL_SPLIT,
               linewidth = 0.45,
               linetype = "dashed") +
    
    # ── Observed series ──────────────────────────────────────
    geom_line(aes(y = cases_raw, colour = "Observed"),
              linewidth = 0.80,
              lineend   = "round",
              linejoin  = "round") +
    
    # ── Predicted series ─────────────────────────────────────
    geom_line(aes(y = pred_raw, colour = "Predicted"),
              linewidth = 0.65,
              linetype  = "longdash",
              lineend   = "round",
              linejoin  = "round") +
    
    # ── RMSE annotation ──────────────────────────────────────
    annotate("text",
             x     = CUTOFF + 0.15,
             y     = y_ann,
             label = rmse_label,
             hjust = 0,
             vjust = 1,
             size  = 3.0,
             colour = COL_ANNOT,
             fontface = "bold") +
    
    # ── Train / Test region labels ───────────────────────────
    annotate("text",
             x = (min(df$time) + CUTOFF) / 2,
             y = y_max * 0.05,
             label = "Train",
             hjust = 0.5, vjust = 0,
             size = 2.8,
             colour = "#AAAAAA",
             fontface = "italic") +
    
    annotate("text",
             x = (CUTOFF + max(df$time)) / 2,
             y = y_max * 0.05,
             label = "Test",
             hjust = 0.5, vjust = 0,
             size = 2.8,
             colour = "#AAAAAA",
             fontface = "italic") +
    
    # ── Panel label (A, B, C ...) ─────────────────────────────
    annotate("text",
             x = min(df$time) + 0.1,
             y = y_max * 0.97,
             label = label,
             hjust = 0, vjust = 1,
             size  = 4.2,
             colour = "#111111",
             fontface = "bold") +
    
    # ── Scales ───────────────────────────────────────────────
    scale_colour_manual(
      values = c("Observed"  = COL_OBS,
                 "Predicted" = COL_PRED),
      guide  = guide_legend(
        override.aes = list(
          linewidth = c(0.85, 0.65),
          linetype  = c("solid", "longdash")
        )
      )
    ) +
    
    scale_x_continuous(
      limits = c(min(df$time), max(df$time)),
      expand = expansion(mult = c(0.005, 0.005)),
      breaks = seq(2008, 2019, by = 3),
      labels = function(x) as.integer(x)
    ) +
    
    scale_y_continuous(
      expand = expansion(mult = c(0.02, 0.10)),
      labels = label_comma(accuracy = 1)
    ) +
    
    labs(
      title = unique(df$k_label),
      x     = NULL,
      y     = "Cases per biweek"
    ) +
    
    theme_pub()
}

# ── BUILD ALL PANELS ──────────────────────────────────────────
cat("\nBuilding panels...\n")
plots <- lapply(
  setNames(ks, as.character(ks)),
  function(k) plot_k(data_list[[as.character(k)]], k)
)

# Strip legend from individual panels
plots_nl <- lapply(plots, function(p) {
  if (is.null(p)) return(NULL)
  p + theme(legend.position = "none")
})

# ── COMBINE ───────────────────────────────────────────────────
# 2 rows × 3 cols — k=34 bottom right empty replaced by
# a clean empty spacer so the legend has breathing room
combined <-
  (plots_nl[["1"]]  | plots_nl[["4"]]  | plots_nl[["12"]]) /
  (plots_nl[["20"]] | plots_nl[["34"]] | plot_spacer()) +
  plot_layout(guides = "collect") &
  plot_annotation(
    title   = paste0(
      "SFNN Forecast Evaluation \u2014 ",
      "South Twenty Four Parganas"),
    subtitle = paste0(
      "V1V2 SFNN \u00b7 Train: 2008\u20132016 ",
      "\u00b7 Test: 2017\u20132019 ",
      "\u00b7 Dashed vertical = train/test split ",
      "\u00b7 Blue annotation = test-period RMSE"),
    theme = theme(
      plot.title    = element_text(
        size = 13, face = "bold",
        colour = "#111111", hjust = 0,
        margin = margin(b = 3)),
      plot.subtitle = element_text(
        size = 9, colour = "#666666",
        hjust = 0, margin = margin(b = 10)),
      legend.position  = "bottom",
      legend.text      = element_text(
        size = 10, colour = "#333333"),
      legend.key.width = unit(2.0, "cm"),
      legend.key.height = unit(0.45, "cm")
    )
  )

# Apply clean margins to all panels
combined <- combined & theme(plot.margin = margin(5, 8, 5, 6))

# ── SAVE ──────────────────────────────────────────────────────
out_base <- file.path(FIG_DIR, "04_sfnn_city_performance")

# PDF — vector for publication submission
ggsave(
  filename = paste0(out_base, ".pdf"),
  plot     = combined,
  width    = 14, height = 8.5,
  device   = cairo_pdf
)

# PNG — 300 DPI for thesis document
ggsave(
  filename = paste0(out_base, ".png"),
  plot     = combined,
  width    = 14, height = 8.5,
  dpi      = 300,
  bg       = "white"
)

cat(sprintf(
  "\nSaved:\n  %s.pdf\n  %s.png\n",
  out_base, out_base))

cat("\nDone.\n")
