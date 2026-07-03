
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(patchwork)
library(scales)

# ── PATHS ─────────────────────────────────────────────────────
BASE    <- "/.../.../.../"
IN_DIR  <- paste0(BASE, "experiments/tables/loss_weight_sweep/")
OUT_DIR <- paste0(BASE, "experiments/figures/")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── CONSTANTS ─────────────────────────────────────────────────
BETA_EQ     <- 35.07
LOG_BETA_EQ <- log(BETA_EQ)     # 3.5573
TSIR_RMSE   <- 33.0
NAIVE_RMSE  <- 29.2

# ── LOAD DATA ─────────────────────────────────────────────────
cat("Loading sweep data...\n")

runs_path <- paste0(IN_DIR, "sweep_all_runs.csv")
if (!file.exists(runs_path)) {
  stop(paste0("File not found: ", runs_path,
              "\nRun wb_collect_sweep_results.py first."))
}

df <- read_csv(runs_path, show_col_types = FALSE) %>%
  mutate(
    ratio   = as.integer(round(ratio)),
    ratio_f = factor(
      ratio,
      levels = c(10, 35, 10000),
      labels = c(
        "10\n(below \u03b2eq)",
        "35\n(at \u03b2eq)",
        "10000\n(above \u03b2eq)"
      )
    ),
    regime = factor(regime, levels = c("BELOW","AT","ABOVE"))
  )

cat(sprintf("Loaded %d runs across %d ratios\n",
            nrow(df), length(unique(df$ratio))))

# factor level count — used for annotation x positions
N_LEVELS <- nlevels(df$ratio_f)

# Exact factor labels — annotate() must receive the discrete
# level *string* (not a numeric index) to avoid the
# "Discrete value supplied to a continuous scale" error.
LVL_FIRST <- levels(df$ratio_f)[1]          # leftmost box
LVL_LAST  <- levels(df$ratio_f)[N_LEVELS]   # rightmost box

# ── COLOURS (Wong 2011 colour-blind safe) ─────────────────────
COL_BELOW <- "#D55E00"
COL_AT    <- "#CC79A7"
COL_ABOVE <- "#0072B2"
COL_EQ    <- "#009E73"
COL_TSIR  <- "#999999"
COL_NAIVE <- "#56B4E9"

ratio_cols <- c(
  "10\n(below \u03b2eq)"    = COL_BELOW,
  "35\n(at \u03b2eq)"       = COL_AT,
  "10000\n(above \u03b2eq)" = COL_ABOVE
)

# ── THEME ─────────────────────────────────────────────────────
theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      panel.border       = element_rect(colour = "#222222",
                                        fill = NA,
                                        linewidth = 0.7),
      panel.grid.major.y = element_line(colour = "#EBEBEB",
                                        linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      axis.line          = element_blank(),
      axis.title         = element_text(size = rel(0.88),
                                        face = "bold"),
      axis.text          = element_text(size = rel(0.82)),
      axis.ticks         = element_line(linewidth = 0.4),
      legend.position    = "none",
      plot.title         = element_text(size = rel(0.92),
                                        face = "bold",
                                        hjust = 0,
                                        margin = margin(b=2)),
      plot.title.position = "plot",
      plot.margin        = margin(5, 10, 4, 6)
    )
}

# ── PANEL A — vert distribution ───────────────────────────────
cat("Building Panel A: vert distribution...\n")

# y range for annotation placement
y_max_a <- max(df$final_vert, na.rm = TRUE)
y_min_a <- min(df$final_vert, na.rm = TRUE)

panel_a <- ggplot(df,
                  aes(x = ratio_f,
                      y = final_vert,
                      fill = ratio_f,
                      colour = ratio_f)) +
  
  # log(beta_eq) reference line
  geom_hline(yintercept = LOG_BETA_EQ,
             colour = COL_EQ, linewidth = 1.0,
             linetype = "solid") +
  
  # Label sits just above the green line, right-aligned.
  # x must be an integer factor-level index on a discrete scale.
  annotate("text",
           x     = LVL_LAST,
           y     = LOG_BETA_EQ + (y_max_a - y_min_a) * 0.04,
           label = paste0("log(\u03b2eq) = ",
                          round(LOG_BETA_EQ, 2)),
           hjust = 1, size = 3.2,
           colour = COL_EQ, fontface = "bold") +
  
  geom_boxplot(alpha = 0.25, outlier.shape = 16,
               outlier.size = 1.2, outlier.alpha = 0.5,
               linewidth = 0.6, width = 0.55) +
  geom_jitter(width = 0.12, alpha = 0.25, size = 0.9) +
  
  # Panel letter — top left (anchored on first level, nudged out)
  annotate("text",
           x = LVL_FIRST, y = y_max_a * 1.0,
           label = "A", hjust = 1.8, vjust = 0,
           size = 5.5, fontface = "bold") +
  
  scale_fill_manual(values = ratio_cols) +
  scale_colour_manual(values = ratio_cols) +
  coord_cartesian(clip = "off") +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +
  
  labs(
    title = "Inferred \u03bd (vert) at convergence",
    x     = NULL,
    y     = "Final \u03bd  (100 independent runs)"
  ) +
  theme_pub()

# ── PANEL B — amp1 distribution ───────────────────────────────
cat("Building Panel B: amp1 distribution...\n")

y_max_b <- max(df$final_amp1, na.rm = TRUE)
y_min_b <- min(df$final_amp1, na.rm = TRUE)

panel_b <- ggplot(df,
                  aes(x = ratio_f,
                      y = final_amp1,
                      fill = ratio_f,
                      colour = ratio_f)) +
  
  geom_hline(yintercept = 0,
             colour = "#888888", linewidth = 0.5,
             linetype = "dashed") +
  
  geom_boxplot(alpha = 0.25, outlier.shape = 16,
               outlier.size = 1.2, outlier.alpha = 0.5,
               linewidth = 0.6, width = 0.55) +
  geom_jitter(width = 0.12, alpha = 0.25, size = 0.9) +
  
  annotate("text",
           x = LVL_FIRST, y = y_max_b,
           label = "B", hjust = 1.8, vjust = 0,
           size = 5.5, fontface = "bold") +
  
  scale_fill_manual(values = ratio_cols) +
  scale_colour_manual(values = ratio_cols) +
  coord_cartesian(clip = "off") +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +
  
  labs(
    title = "Seasonal amplitude \u03b11 at convergence",
    x     = NULL,
    y     = "Final \u03b11  (100 independent runs)"
  ) +
  theme_pub()

# ── PANEL C — RMSE distribution ───────────────────────────────
cat("Building Panel C: RMSE distribution...\n")

# FIX: expand limits to include both benchmark lines so they
# (and their labels) are never clipped below the axis floor.
data_max_c <- max(df$test_rmse, na.rm = TRUE)
data_min_c <- min(df$test_rmse, na.rm = TRUE)

y_max_c <- max(data_max_c, TSIR_RMSE, NAIVE_RMSE) * 1.05
y_min_c <- min(data_min_c, TSIR_RMSE, NAIVE_RMSE) * 0.95

panel_c <- ggplot(df,
                  aes(x = ratio_f,
                      y = test_rmse,
                      fill = ratio_f,
                      colour = ratio_f)) +
  
  # TSIR benchmark
  geom_hline(yintercept = TSIR_RMSE,
             colour = COL_TSIR, linewidth = 0.7,
             linetype = "dashed") +
  annotate("text",
           x     = LVL_LAST,
           y     = TSIR_RMSE + (y_max_c - y_min_c) * 0.04,
           label = paste0("TSIR V1V2 (", TSIR_RMSE, ")"),
           hjust = 1, size = 3.0,
           colour = COL_TSIR, fontface = "italic") +
  
  # NaivePINN benchmark
  geom_hline(yintercept = NAIVE_RMSE,
             colour = COL_NAIVE, linewidth = 0.7,
             linetype = "dotted") +
  annotate("text",
           x     = LVL_LAST,
           y     = NAIVE_RMSE - (y_max_c - y_min_c) * 0.04,
           label = paste0("NaivePINN (", NAIVE_RMSE, ")"),
           hjust = 1, size = 3.0,
           colour = COL_NAIVE, fontface = "italic") +
  
  geom_boxplot(alpha = 0.25, outlier.shape = 16,
               outlier.size = 1.2, outlier.alpha = 0.5,
               linewidth = 0.6, width = 0.55) +
  geom_jitter(width = 0.12, alpha = 0.25, size = 0.9) +
  
  annotate("text",
           x = LVL_FIRST, y = y_max_c,
           label = "C", hjust = 1.8, vjust = 0,
           size = 5.5, fontface = "bold") +
  
  scale_fill_manual(values = ratio_cols) +
  scale_colour_manual(values = ratio_cols) +
  scale_y_continuous(labels = label_number(accuracy = 1)) +
  # FIX: use coord_cartesian for limits (does not drop geoms
  # the way scale limits do) and keep clip off for labels.
  coord_cartesian(ylim = c(y_min_c, y_max_c), clip = "off") +
  
  labs(
    title = "Test RMSE at convergence",
    x     = "\u03bbI / \u03bbODE  (loss weight ratio)",
    y     = "Test RMSE  (cases / biweek)"
  ) +
  theme_pub()

# ── PANEL D — Dead zone epoch ─────────────────────────────────
cat("Building Panel D: dead zone epoch...\n")

has_dead_zone <- "dead_zone_epoch" %in% names(df) &&
  sum(!is.na(df$dead_zone_epoch)) > 10

if (has_dead_zone) {
  df_dz <- df %>% filter(!is.na(dead_zone_epoch))
  y_max_d <- max(df_dz$dead_zone_epoch, na.rm = TRUE)
  
  panel_d <- ggplot(df_dz,
                    aes(x = ratio_f,
                        y = dead_zone_epoch,
                        fill = ratio_f,
                        colour = ratio_f)) +
    
    geom_boxplot(alpha = 0.25, outlier.shape = 16,
                 outlier.size = 1.0, outlier.alpha = 0.5,
                 linewidth = 0.5, width = 0.55) +
    geom_jitter(width = 0.10, alpha = 0.20, size = 0.7) +
    
    annotate("text",
             x = LVL_FIRST, y = y_max_d,
             label = "D", hjust = 1.8, vjust = 0,
             size = 4.5, fontface = "bold") +
    
    scale_fill_manual(values = ratio_cols) +
    scale_colour_manual(values = ratio_cols) +
    coord_cartesian(clip = "off") +
    scale_y_continuous(labels = label_number(accuracy = 1)) +
    
    labs(
      title = "Gradient dead zone onset (epoch)",
      x     = NULL,
      y     = "First frozen epoch"
    ) +
    theme_pub(base_size = 9)
}

# ── COMBINE ───────────────────────────────────────────────────
cat("Combining panels...\n")

if (has_dead_zone) {
  fig <- (panel_a | panel_b) /
    (panel_c | panel_d) +
    plot_layout(heights = c(1, 1))
} else {
  fig <- panel_a / panel_b / panel_c +
    plot_layout(heights = c(1, 1, 1))
}

fig <- fig &
  plot_annotation(
    theme = theme(
      plot.title    = element_text(size = 12, face = "bold",
                                   hjust = 0,
                                   margin = margin(b = 3)),
      plot.subtitle = element_text(size = 8.5,
                                   colour = "#555555",
                                   hjust = 0,
                                   margin = margin(b = 6)),
      plot.caption  = element_text(size = 7.5,
                                   colour = "#666666",
                                   hjust = 0,
                                   margin = margin(t = 6))
    )
  )

# ── SAVE ──────────────────────────────────────────────────────
# FIX: 3-panel fallback uses a saner aspect ratio (was 8x14).
w <- if (has_dead_zone) 12 else 8
h <- if (has_dead_zone) 10 else 12

out_base <- paste0(OUT_DIR, "wb_pinn_sweep_main")

ggsave(paste0(out_base, ".pdf"), fig,
       width = w, height = h,
       device = cairo_pdf)
cat(sprintf("Saved: %s.pdf\n", out_base))

ggsave(paste0(out_base, ".png"), fig,
       width = w, height = h,
       dpi = 300, bg = "white")
cat(sprintf("Saved: %s.png\n", out_base))

# ── TRAJECTORY FIGURE ─────────────────────────────────────────
cat("\nBuilding trajectory figure (run 1 comparison)...\n")

read_trajectory <- function(ratio) {
  city_safe <- "South_Twenty_Four_Parganas"
  rl        <- paste0("ratio_", ratio)
  path      <- file.path(
    BASE,
    "output/models/pinn_experiments/wb_pinn_sweep",
    "tsirpinn_sweep",
    paste0("ratio_", ratio),
    paste0("tsirpinn_sweep_", rl,
           "_k34_tlag52_city", city_safe,
           "_run_1_fit_info.parquet"))
  if (!file.exists(path)) {
    cat(sprintf("  File not found: %s\n", path))
    return(NULL)
  }
  library(arrow)
  df_fi <- arrow::read_parquet(path) %>%
    mutate(
      epoch = row_number(),
      ratio = as.character(ratio),
      vert  = sapply(vert, function(x)
        if (is.list(x)) x[[1]][1]
        else if (length(x) > 1) x[1]
        else as.numeric(x)),
      amp1  = sapply(amp1, function(x)
        if (is.list(x)) x[[1]][1]
        else if (length(x) > 1) x[1]
        else as.numeric(x))
    )
  df_fi
}

tryCatch({
  df10    <- read_trajectory(10)
  df10000 <- read_trajectory(10000)
  
  if (!is.null(df10) && !is.null(df10000)) {
    
    df_traj <- bind_rows(df10, df10000) %>%
      mutate(
        ratio_label = case_when(
          ratio == "10"    ~
            "Ratio 10  (BELOW \u03b2eq)",
          ratio == "10000" ~
            "Ratio 10,000  (ABOVE \u03b2eq)",
          TRUE ~ ratio
        ),
        ratio_label = factor(ratio_label,
                             levels = c(
                               "Ratio 10  (BELOW \u03b2eq)",
                               "Ratio 10,000  (ABOVE \u03b2eq)"
                             ))
      )
    
    traj_cols <- c(
      "Ratio 10  (BELOW \u03b2eq)"    = COL_BELOW,
      "Ratio 10,000  (ABOVE \u03b2eq)" = COL_ABOVE
    )
    
    y_max_v <- max(df_traj$vert, na.rm = TRUE)
    y_max_a1 <- max(df_traj$amp1, na.rm = TRUE)
    
    p_vert <- ggplot(df_traj,
                     aes(x = epoch, y = vert,
                         colour = ratio_label)) +
      
      geom_hline(yintercept = LOG_BETA_EQ,
                 colour = COL_EQ, linewidth = 0.8,
                 linetype = "solid") +
      annotate("text",
               x     = max(df_traj$epoch) * 0.98,
               y     = LOG_BETA_EQ + y_max_v * 0.03,
               label = paste0("log(\u03b2eq) = ",
                              round(LOG_BETA_EQ, 2)),
               hjust = 1, size = 3.0, colour = COL_EQ) +
      
      geom_line(linewidth = 0.8, alpha = 0.9) +
      
      annotate("text",
               x = 0, y = y_max_v,
               label = "A", hjust = -0.3, vjust = 0,
               size = 5, fontface = "bold") +
      
      scale_colour_manual(values = traj_cols, name = NULL) +
      coord_cartesian(clip = "off") +
      labs(
        title = "\u03bd trajectory across 2,500 epochs (run 1)",
        x     = "Epoch",
        y     = "\u03bd (vert)"
      ) +
      theme_pub() +
      theme(legend.position  = "bottom",
            legend.text      = element_text(size = 8),
            legend.key.width = unit(1.5, "cm"))
    
    p_amp1 <- ggplot(df_traj,
                     aes(x = epoch, y = amp1,
                         colour = ratio_label)) +
      
      geom_hline(yintercept = 0,
                 colour = "#888888", linewidth = 0.4,
                 linetype = "dashed") +
      
      geom_line(linewidth = 0.8, alpha = 0.9) +
      
      annotate("text",
               x = 0, y = y_max_a1,
               label = "B", hjust = -0.3, vjust = 0,
               size = 5, fontface = "bold") +
      
      scale_colour_manual(values = traj_cols, name = NULL) +
      coord_cartesian(clip = "off") +
      labs(
        title = "\u03b11 trajectory across 2,500 epochs (run 1)",
        x     = "Epoch",
        y     = "\u03b11 (amp1)"
      ) +
      theme_pub() +
      theme(legend.position  = "bottom",
            legend.text      = element_text(size = 8),
            legend.key.width = unit(1.5, "cm"))
    
    fig_traj <- (p_vert / p_amp1) &
      plot_annotation(
        title    = paste0(
          "Parameter trajectories: ratio controls gradient ",
          "direction but cannot prevent structural collapse"),
        theme = theme(
          plot.title    = element_text(size = 11,
                                       face = "bold"),
          plot.subtitle = element_text(size = 8.5,
                                       colour = "#555555")
        )
      )
    
    out_traj <- paste0(OUT_DIR, "wb_pinn_sweep_trajectories")
    ggsave(paste0(out_traj, ".pdf"), fig_traj,
           width = 9, height = 8,
           device = cairo_pdf)
    ggsave(paste0(out_traj, ".png"), fig_traj,
           width = 9, height = 8,
           dpi = 300, bg = "white")
    cat(sprintf("Saved: %s.pdf\n", out_traj))
    cat(sprintf("Saved: %s.png\n", out_traj))
  }
  
}, error = function(e) {
  cat("  Trajectory figure error:", conditionMessage(e), "\n")
})

cat("\nAll done.\n")
