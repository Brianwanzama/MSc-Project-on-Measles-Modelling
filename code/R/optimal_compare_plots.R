
#library(tidyverse)
library(readr)
library(dplyr)
library(patchwork)
library(ggplot2)

set.seed(42)

BASE     <- "/.../.../.../"
SAVE_DIR <- paste0(BASE, "experiments/figures/")
DATA_DIR <- paste0(BASE, "data/")
NN_DIR   <- paste0(BASE, "output/data/basic_nn_optimal/")
TSIR_CSV <- paste0(BASE, "output/data/basic_nn_optimal/",
                   "tsir_preds_processed_V1V2.csv")

dir.create(SAVE_DIR, recursive=TRUE, showWarnings=FALSE)

k_values <- c(1, 4, 12, 20, 34)

# ── THEME ─────────────────────────────────────────────────────
theme_pub <- function(base_size=10) {
  theme_classic(base_size=base_size) +
    theme(
      text             = element_text(colour="#1a1a1a"),
      axis.title       = element_text(size=base_size, face="plain"),
      axis.text        = element_text(size=base_size*0.8,
                                      colour="#1a1a1a"),
      axis.ticks       = element_line(colour="#1a1a1a",
                                      linewidth=0.3),
      axis.line        = element_line(colour="#1a1a1a",
                                      linewidth=0.3),
      panel.border     = element_rect(colour="#1a1a1a", fill=NA,
                                      linewidth=0.4),
      panel.grid.major = element_line(colour="#e8e8e8",
                                      linewidth=0.25),
      panel.grid.minor = element_blank(),
      panel.spacing    = unit(0.6, "lines"),
      strip.background = element_rect(fill="#f0f0f0",
                                      colour="#1a1a1a",
                                      linewidth=0.4),
      strip.text       = element_text(size=base_size*0.85,
                                      face="bold",
                                      margin=margin(3,3,3,3)),
      legend.position  = "bottom",
      legend.title     = element_text(size=base_size*0.85,
                                      face="plain"),
      legend.text      = element_text(size=base_size*0.75),
      legend.key.width = unit(1.8, "cm"),
      legend.key.height= unit(0.3, "cm"),
      legend.margin    = margin(t=4, b=2),
      plot.margin      = margin(6, 8, 4, 6),
      plot.caption     = element_text(size=base_size*0.7,
                                      hjust=0, colour="#444444",
                                      margin=margin(t=6)),
      plot.tag         = element_text(size=base_size*1.2,
                                      face="bold")
    )
}

# ── LOAD POPULATION ───────────────────────────────────────────
pop_raw <- read_csv(paste0(DATA_DIR, "inferred_popn.csv"),
                    show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="pop") %>%
  rename(time=`...1`) %>%
  mutate(year=as.integer(floor(time))) %>%
  group_by(city, year) %>%
  summarise(pop=mean(pop, na.rm=TRUE), .groups="drop")

pop_min <- pop_raw %>%
  group_by(city) %>%
  summarise(min_pop=min(pop, na.rm=TRUE), .groups="drop")

# ── LOAD TSIR PREDICTIONS ─────────────────────────────────────
tsir_dat <- read_csv(TSIR_CSV, show_col_types=FALSE) %>%
  mutate(
    tsir = pmax(tsir, 0),
    time = round(time, 3),
    k    = as.integer(k)
  ) %>%
  filter(!is.na(time), k %in% k_values)

# ── LOAD SFNN PREDICTIONS ─────────────────────────────────────
sfnn_all <- list()

for (k in k_values) {
  out_path <- paste0(NN_DIR, k, "_output.parquet")
  tfm_path <- paste0(NN_DIR, k, "_transform.parquet")

  if (!file.exists(out_path)) next

  out <- arrow::read_parquet(out_path)
  tfm <- arrow::read_parquet(tfm_path)

  city_col <- names(out)[grepl("city|district",
                                names(out),
                                ignore.case=TRUE)][1]

  out <- out %>%
    left_join(tfm %>%
                select(time, all_of(city_col),
                       cases_mean, cases_std),
              by=c("time", city_col)) %>%
    mutate(
      pred_raw = pmax(exp(pred  * cases_std + cases_mean) - 1, 0),
      obs_raw  = pmax(exp(cases * cases_std + cases_mean) - 1, 0),
      k        = as.integer(k),
      time     = round(time, 3)
    ) %>%
    filter(train_test == "test") %>%
    rename(city = all_of(city_col)) %>%
    select(time, city, k, pred_raw, obs_raw)

  sfnn_all[[as.character(k)]] <- out
}

sfnn_dat <- bind_rows(sfnn_all)

# ── MERGE ─────────────────────────────────────────────────────
full_dat <- sfnn_dat %>%
  inner_join(
    tsir_dat %>% select(time, city, k, tsir),
    by = c("time", "city", "k")
  ) %>%
  left_join(pop_min, by="city")

# ── COMPUTE RMSE ──────────────────────────────────────────────
rmse_summary <- full_dat %>%
  group_by(city, k) %>%
  summarise(
    min_pop   = unique(min_pop),
    tsir_rmse = sqrt(mean((obs_raw - tsir)^2,     na.rm=TRUE)),
    nn_rmse   = sqrt(mean((obs_raw - pred_raw)^2, na.rm=TRUE)),
    .groups   = "drop"
  ) %>%
  mutate(
    k_label = paste0("k = ", k),
    k_label = factor(k_label, levels=paste0("k = ", k_values))
  )

cat("Mean RMSE by k:\n")
rmse_summary %>%
  group_by(k) %>%
  summarise(
    mean_tsir    = mean(tsir_rmse),
    mean_sfnn    = mean(nn_rmse),
    improvement  = (mean_tsir - mean_sfnn) / mean_tsir * 100,
    pct_sfnn_wins = mean(nn_rmse < tsir_rmse) * 100,
    .groups="drop"
  ) %>%
  print()

# ── AXIS LIMITS — 90th percentile, no outlier markers ─────────
rmse_limit <- quantile(
  c(rmse_summary$tsir_rmse, rmse_summary$nn_rmse),
  0.90, na.rm=TRUE
) %>% ceiling()

gain_limit <- quantile(
  abs(rmse_summary$tsir_rmse - rmse_summary$nn_rmse),
  0.90, na.rm=TRUE
) %>% ceiling()

cat(sprintf("\nAxis limits: rmse_limit=%.0f | gain_limit=%.0f\n",
            rmse_limit, gain_limit))

# ── RMSE SCATTER ──────────────────────────────────────────────
prmse <- ggplot(rmse_summary,
                aes(x=tsir_rmse, y=nn_rmse,
                    colour=log(min_pop))) +
  geom_abline(colour="#888888", linewidth=0.4,
              linetype="dashed") +
  geom_point(size=1.2, alpha=0.85, shape=16) +
  scale_color_gradientn(
    colours = c("#002A66","#1a6aa8","#f0c400","#FBE045"),
    name    = "Log(Population)",
    guide   = guide_colorbar(
      title.position  = "top",
      title.hjust     = 0.5,
      barwidth        = 10,
      barheight       = 0.5,
      ticks.linewidth = 0.5
    )
  ) +
  scale_x_continuous(
    limits = c(0, rmse_limit),
    oob    = scales::squish   # squish values beyond limits to edge
  ) +
  scale_y_continuous(
    limits = c(0, rmse_limit),
    oob    = scales::squish
  ) +
  facet_wrap(~k_label, ncol=3) +
  labs(
    x       = expression(RMSE[TSIR]),
    y       = expression(RMSE[SFNN]),
    caption = paste0(
      "Points below diagonal = SFNN outperforms TSIR. ",
      "TSIR: Ferrari et al. (2012) Eq. 3 (V1+V2).")
  ) +
  theme_pub()

# ── RMSE GAIN PLOT ────────────────────────────────────────────
prmsegain_dat <- rmse_summary %>%
  mutate(rmse_gain = tsir_rmse - nn_rmse)

prmsegain <- ggplot(prmsegain_dat,
                    aes(x=log(min_pop), y=rmse_gain)) +
  geom_hline(yintercept=0, linetype="dashed",
             colour="#888888", linewidth=0.4) +
  geom_point(shape=21, size=1.1, alpha=0.45,
             fill="#1a1a1a", colour="transparent") +
  geom_smooth(method="loess", colour="#002A66",
              fill="#002A66", alpha=0.15,
              linewidth=0.6, se=TRUE) +
  scale_y_continuous(
    limits = c(-gain_limit, gain_limit),
    oob    = scales::squish
  ) +
  facet_wrap(~k_label, ncol=3) +
  labs(
    x       = "Log(Population)",
    y       = expression(RMSE[TSIR] - RMSE[SFNN]),
    caption = paste0(
      "Positive values = SFNN outperforms TSIR. ",
      "Loess trend with 95% CI.")
  ) +
  theme_pub()

# ── COMBINE ───────────────────────────────────────────────────
figrmseall <- prmse / guide_area() / prmsegain +
  plot_layout(
    heights = c(3, 0.4, 3),
    guides  = "collect"
  ) +
  plot_annotation(
    tag_levels = "A",
    title      = "SFNN vs TSIR (V1+V2): RMSE Comparison Across Districts",
    subtitle   = paste0(
      "West Bengal 2017-2019 | ",
      "19 districts | k = 1, 4, 12, 20, 34 biweeks"),
    theme = theme(
      plot.title    = element_text(size=12, face="bold",
                                   colour="#111111", hjust=0),
      plot.subtitle = element_text(size=9.5, colour="#555555",
                                   hjust=0, margin=margin(b=8)),
      plot.tag      = element_text(size=11, face="bold")
    )
  )

# ── SAVE ──────────────────────────────────────────────────────
scale_factor <- 2

ggsave(
  paste0(SAVE_DIR, "rmse_scatter_sfnn_tsir.pdf"),
  figrmseall,
  width=3*scale_factor, height=4*scale_factor,
  device=cairo_pdf
)

ggsave(
  paste0(SAVE_DIR, "rmse_scatter_sfnn_tsir.png"),
  figrmseall,
  width=3*scale_factor, height=4*scale_factor,
  dpi=300, bg="white"
)

cat(sprintf("\nSaved to: %s\n", SAVE_DIR))
