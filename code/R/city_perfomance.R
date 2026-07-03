# -----------------------------------------------------------------------------
# LIBRARIES
# -----------------------------------------------------------------------------
library(arrow)
library(dplyr)
library(ggplot2)
library(patchwork)

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
BASE <- ".../.../.../output/models/basic_nn_optimal/"
ks   <- c(1, 4, 12, 20, 34, 52)
CITY <- "South Twenty Four Parganas"

# -----------------------------------------------------------------------------
# SHARED THEME
# -----------------------------------------------------------------------------
theme_publication <- function(base_size = 11) {
  theme_bw(base_size = base_size) %+replace%
    theme(
      # Panel
      panel.background  = element_rect(fill = "#FAFAFA", colour = NA),
      panel.border      = element_rect(colour = "#CCCCCC", fill = NA, linewidth = 0.5),
      panel.grid.major  = element_line(colour = "#E5E5E5", linewidth = 0.35),
      panel.grid.minor  = element_blank(),
      
      # Axes
      axis.title        = element_text(size = rel(0.82), colour = "#444444",
                                       margin = margin(t = 4, r = 4)),
      axis.text         = element_text(size = rel(0.74), colour = "#666666"),
      axis.ticks        = element_line(colour = "#BBBBBB", linewidth = 0.35),
      axis.ticks.length = unit(2.5, "pt"),
      
      # Strip (not used here but good to have)
      strip.background  = element_rect(fill = "#F0F0F0", colour = "#CCCCCC",
                                       linewidth = 0.4),
      strip.text        = element_text(size = rel(0.82), face = "bold",
                                       colour = "#333333"),
      
      # Legend
      legend.position      = "bottom",
      legend.title         = element_blank(),
      legend.text          = element_text(size = rel(0.82), colour = "#444444"),
      legend.key           = element_rect(fill = NA, colour = NA),
      legend.key.width     = unit(2.2, "cm"),
      legend.key.height    = unit(0.45, "cm"),
      legend.spacing.x     = unit(0.4, "cm"),
      legend.background    = element_rect(fill = NA, colour = NA),
      legend.margin        = margin(t = 2, b = 0),
      
      # Plot title
      plot.title           = element_text(size = rel(0.95), face = "bold",
                                          colour = "#222222", hjust = 0,
                                          margin = margin(b = 5)),
      plot.title.position  = "plot",
      
      # Margins
      plot.margin          = margin(6, 8, 6, 6)
    )
}

# -----------------------------------------------------------------------------
# COLOUR / LINE PALETTE
# -----------------------------------------------------------------------------
OBS_COL  <- "#1A1A2E"   # near-black navy
PRED_COL <- "#C0392B"   # vivid crimson

# -----------------------------------------------------------------------------
# LOAD FUNCTION
# -----------------------------------------------------------------------------
load_k_data <- function(k) {
  df <- read_parquet(paste0(BASE, k, "_output.parquet"))
  
  city_col <- names(df)[grepl("city|district", names(df), ignore.case = TRUE)][1]
  
  df %>%
    filter(.data[[city_col]] == CITY) %>%
    arrange(time) %>%
    mutate(k_label = paste0("k = ", k))
}

# -----------------------------------------------------------------------------
# LOAD ALL DATA
# -----------------------------------------------------------------------------
data_list <- lapply(ks, load_k_data)

# -----------------------------------------------------------------------------
# PLOT FUNCTION
# -----------------------------------------------------------------------------
plot_k <- function(df) {
  
  k_val <- unique(df$k_label)
  
  ggplot(df, aes(x = time)) +
    
    # Observed — solid, slightly thicker
    geom_line(aes(y = cases, colour = "Observed"),
              linewidth = 0.75, lineend = "round") +
    
    # Predicted — dashed, slightly thinner so it reads under the observed
    geom_line(aes(y = pred, colour = "Predicted"),
              linewidth = 0.65, linetype = "longdash", lineend = "round") +
    
    scale_colour_manual(
      values = c("Observed" = OBS_COL, "Predicted" = PRED_COL),
      guide  = guide_legend(
        override.aes = list(
          linewidth = c(0.85, 0.65),
          linetype  = c("solid", "longdash")
        )
      )
    ) +
    
    scale_x_continuous(expand = expansion(mult = c(0.01, 0.01))) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.08)),
                       labels = scales::comma) +
    
    labs(
      title = k_val,
      x     = NULL,         # suppress per-panel x-label; add once via patchwork
      y     = "Weekly cases"
    ) +
    
    theme_publication()
}

# -----------------------------------------------------------------------------
# BUILD INDIVIDUAL PANELS
# -----------------------------------------------------------------------------
plots <- lapply(data_list, plot_k)

# Remove legend from all but the last panel — collect via patchwork instead
plots_no_leg <- lapply(plots, function(p)
  p + theme(legend.position = "none"))

# -----------------------------------------------------------------------------
# COMBINE  (3 × 2)  with patchwork
# -----------------------------------------------------------------------------
final_plot <-
  (plots_no_leg[[1]] | plots_no_leg[[2]] | plots_no_leg[[3]]) /
  (plots_no_leg[[4]] | plots_no_leg[[5]] | plots_no_leg[[6]]) +
  
  plot_layout(guides = "collect") &   # single shared legend
  
  plot_annotation(
    title    = "Basic Neural Network — Epidemic Forecast Evaluation",
    subtitle = paste0("District: ", CITY,
                      "  ·  Horizons: k = 1, 4, 12, 20, 34, 52 weeks"),
    caption  = "Dashed line = model predictions  ·  Solid line = observed cases",
    theme    = theme(
      plot.title    = element_text(size = 13, face = "bold",
                                   colour = "#111111", hjust = 0,
                                   margin = margin(b = 3)),
      plot.subtitle = element_text(size = 10, colour = "#555555",
                                   hjust = 0, margin = margin(b = 10)),
      plot.caption  = element_text(size = 8,  colour = "#888888",
                                   hjust = 0, margin = margin(t = 8)),
      legend.position  = "bottom",
      legend.text      = element_text(size = 10, colour = "#333333"),
      legend.key.width = unit(2.2, "cm")
    )
  )

# Add a shared x-axis label via an invisible bottom annotation
final_plot <- final_plot &
  theme(plot.margin = margin(5, 8, 5, 5))

# -----------------------------------------------------------------------------
# SAVE — PDF (vector) + PNG (raster) for journal submission
# -----------------------------------------------------------------------------
out_base <- "/home/brain/Msc_project/output/figures/city_comparison_plot"

# PDF (preferred for publication — lossless vector)
ggsave(
  filename = paste0(out_base, ".pdf"),
  plot     = final_plot,
  width    = 14, height = 8,
  device   = cairo_pdf   # better font rendering than default pdf()
)

# High-res PNG (for presentations / supplementary material)
ggsave(
  filename = paste0(out_base, ".png"),
  plot     = final_plot,
  width    = 14, height = 8,
  dpi      = 600,
  bg       = "white"
)

final_plot
