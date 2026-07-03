# ============================================================
# wb_city_rmse_table.R
# Per-city RMSE comparison: SFNN vs TSIR V1V2
# Focus on k=1 and k=34
# Tests claim: TSIR better in large cities,
#              SFNN better in small cities at long horizons
# ============================================================

library(forcats)
library(stringr)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(scales)

BASE    <- "/.../.../.../"
COMP    <- paste0(BASE, "output/data/comparison/")
OUT_DIR <- paste0(BASE, "experiments/figures/")
DATA    <- paste0(BASE, "data/")

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# ── LOAD DATA ─────────────────────────────────────────────────
merged <- read_csv(paste0(COMP, "merged_predictions.csv"),
                   show_col_types=FALSE)

cat("Merged predictions:", nrow(merged), "rows\n")
cat("k values:", paste(sort(unique(merged$k)), collapse=", "), "\n")
cat("Cities:", length(unique(merged$city)), "\n")

# ── LOAD POPULATION — rank cities by size ─────────────────────
pop_raw <- read_csv(paste0(DATA, "inferred_popn.csv"),
                    show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="pop") %>%
  rename(time=`...1`) %>%
  mutate(year=as.integer(floor(time))) %>%
  filter(year >= 2017) %>%
  group_by(city) %>%
  summarise(mean_pop=mean(pop, na.rm=TRUE), .groups="drop") %>%
  arrange(desc(mean_pop)) %>%
  mutate(
    pop_rank   = row_number(),
    pop_M      = round(mean_pop / 1e6, 2),
    city_label = paste0(city, "\n(", pop_M, "M)")
  )

cat("\nPopulation ranking (2017-2019 mean):\n")
print(pop_raw %>% select(pop_rank, city, pop_M))

# ── COMPUTE PER-CITY RMSE FOR k=1 AND k=34 ───────────────────
rmse_city <- merged %>%
  filter(train_test == "test",
         k %in% c(1, 34)) %>%
  group_by(city, k) %>%
  summarise(
    rmse_sfnn = sqrt(mean((pred_cases - obs_cases)^2,
                           na.rm=TRUE)),
    rmse_tsir = sqrt(mean((tsir_cases - obs_cases)^2,
                           na.rm=TRUE)),
    n         = n(),
    mean_obs  = mean(obs_cases, na.rm=TRUE),
    .groups   = "drop"
  ) %>%
  mutate(
    rmse_diff   = rmse_tsir - rmse_sfnn,  # + = SFNN better
    sfnn_better = rmse_sfnn < rmse_tsir,
    pct_diff    = (rmse_tsir - rmse_sfnn) / rmse_tsir * 100
  ) %>%
  left_join(pop_raw %>%
              select(city, mean_pop, pop_rank, pop_M),
            by="city") %>%
  arrange(k, pop_rank)

cat("\n=== PER-CITY RMSE (k=1) ===\n")
rmse_city %>%
  filter(k==1) %>%
  select(pop_rank, city, pop_M,
         rmse_sfnn, rmse_tsir, rmse_diff, sfnn_better) %>%
  mutate(across(where(is.numeric), ~round(.x, 2))) %>%
  print(n=20)

cat("\n=== PER-CITY RMSE (k=34) ===\n")
rmse_city %>%
  filter(k==34) %>%
  select(pop_rank, city, pop_M,
         rmse_sfnn, rmse_tsir, rmse_diff, sfnn_better) %>%
  mutate(across(where(is.numeric), ~round(.x, 2))) %>%
  print(n=20)

# ── SUMMARY: does TSIR beat SFNN in large cities? ─────────────
cat("\n=== CLAIM TEST: TSIR better in large cities? ===\n")
for (k_val in c(1, 34)) {
  d     <- rmse_city %>% filter(k == k_val)
  large <- d %>% filter(pop_rank <= 5)
  small <- d %>% filter(pop_rank > 14)
  cat(sprintf("\nk=%d:\n", k_val))
  cat(sprintf("  Large cities (top 5):    SFNN better in %d/5\n",
              sum(large$sfnn_better)))
  cat(sprintf("  Small cities (bottom 5): SFNN better in %d/5\n",
              sum(small$sfnn_better)))
  cat(sprintf("  Overall: SFNN better in %d/%d cities\n",
              sum(d$sfnn_better), nrow(d)))
}

# ── SAVE RMSE TABLE ───────────────────────────────────────────
write_csv(rmse_city,
          paste0(COMP, "rmse_per_city_k1_k34.csv"))

# ── SHARED: diff_df with short city names ─────────────────────
diff_df <- rmse_city %>%
  mutate(
    k_label    = paste0("k = ", k,
                        ifelse(k==1," (2 weeks)"," (17 months)")),
    city_order = fct_reorder(city, -pop_rank),
    winner     = ifelse(sfnn_better, "SFNN better", "TSIR better"),
    city_short = str_replace_all(city, c(
      "Twenty Four Parganas" = "24 Pgs",
      "Twenty Four Pargana"  = "24 Pgs",
      "Paschim Medinipur"    = "P. Medinipur",
      "Purba Medinipur"      = "Pu. Medinipur",
      "Dakshin Dinajpur"     = "D. Dinajpur",
      "Uttar Dinajpur"       = "U. Dinajpur",
      "Koch Bihar"           = "Koch Bihar"
    ))
  )

# ── FIGURE 1: Heatmap RMSE by city × model × k ───────────────
plot_df <- rmse_city %>%
  select(city, k, pop_rank, pop_M, rmse_sfnn, rmse_tsir) %>%
  pivot_longer(cols=c(rmse_sfnn, rmse_tsir),
               names_to="model", values_to="rmse") %>%
  mutate(
    model        = recode(model,
                          rmse_sfnn="SFNN",
                          rmse_tsir="TSIR (V1V2)"),
    k_label      = paste0("k = ", k,
                          ifelse(k==1," (2 weeks)"," (17 months)")),
    city_ordered = fct_reorder(city, -pop_rank)
  )

p1 <- ggplot(plot_df,
             aes(x=model, y=city_ordered, fill=rmse)) +
  geom_tile(colour="white", linewidth=0.4) +
  geom_text(aes(label=round(rmse, 1)),
            size=2.8, colour="white", fontface="bold") +
  scale_fill_gradientn(
    colours=c("#D4EDDA","#F9C74F","#F94144"),
    name="RMSE\n(cases/biweek)",
    guide=guide_colorbar(title.position="top",
                         barwidth=8, barheight=0.5)
  ) +
  facet_wrap(~k_label, ncol=2) +
  scale_x_discrete(expand=expansion(0)) +
  scale_y_discrete(expand=expansion(0)) +
  labs(
    title    = "Per-District RMSE: SFNN vs TSIR (V1V2)",
    subtitle = paste0(
      "West Bengal 2017-2019 | ",
      "Districts ordered by population (largest at top) | ",
      "k=1 (2 weeks) and k=34 (17 months)"),
    x=NULL, y=NULL,
    caption  = paste0(
      "RMSE in raw cases per biweek. ",
      "Green = low error, Red = high error. ",
      "Ferrari et al. (2012) Eq.3 susceptible reconstruction.")
  ) +
  theme_classic(base_size=10) +
  theme(
    plot.title      = element_text(face="bold", size=11, hjust=0),
    plot.subtitle   = element_text(colour="grey40", size=8,
                                   hjust=0, margin=margin(b=6)),
    plot.caption    = element_text(colour="grey50", size=7, hjust=0),
    strip.text      = element_text(face="bold", size=10),
    axis.text.y     = element_text(size=8),
    axis.text.x     = element_text(size=9, face="bold"),
    legend.position = "top",
    plot.margin     = margin(10,14,8,10)
  )

ggsave(paste0(OUT_DIR, "12_city_rmse_heatmap.pdf"),
       p1, width=10, height=8, device=cairo_pdf)
ggsave(paste0(OUT_DIR, "12_city_rmse_heatmap.png"),
       p1, width=10, height=8, dpi=300, bg="white")
cat("\nSaved: 12_city_rmse_heatmap\n")

# ── FIGURE 2: Bar chart — RMSE difference by city ─────────────
p2 <- ggplot(diff_df,
             aes(x=rmse_diff, y=city_order, fill=winner)) +
  geom_col(alpha=0.85, width=0.7) +
  geom_vline(xintercept=0, linewidth=0.6, colour="grey30") +
  geom_text(aes(
    label = sprintf("%+.1f", rmse_diff),
    hjust = ifelse(rmse_diff >= 0, -0.1, 1.1)
  ), size=2.8, colour="grey20") +
  scale_fill_manual(
    values=c("SFNN better"="#2196F3",
             "TSIR better"="#E53935"),
    name=NULL
  ) +
  scale_x_continuous(expand=expansion(mult=c(0.15, 0.15))) +
  facet_wrap(~k_label, ncol=2, scales="free_x") +
  labs(
    title    = "RMSE Difference (TSIR − SFNN) by District",
    subtitle = paste0(
      "West Bengal 2017-2019 | ",
      "Districts ordered by population (largest at top) | ",
      "Blue = SFNN better | Red = TSIR better"),
    x       = "RMSE difference (TSIR − SFNN, cases/biweek)",
    y       = NULL,
    caption = paste0(
      "Positive values indicate SFNN achieves lower RMSE. ",
      "Negative values indicate TSIR achieves lower RMSE.")
  ) +
  theme_classic(base_size=10) +
  theme(
    plot.title      = element_text(face="bold", size=11, hjust=0),
    plot.subtitle   = element_text(colour="grey40", size=8,
                                   hjust=0, margin=margin(b=6)),
    plot.caption    = element_text(colour="grey50", size=7, hjust=0),
    strip.text      = element_text(face="bold", size=10),
    axis.text.y     = element_text(size=8),
    legend.position = "top",
    plot.margin     = margin(10,14,8,10)
  )

ggsave(paste0(OUT_DIR, "13_city_rmse_diff.pdf"),
       p2, width=11, height=8, device=cairo_pdf)
ggsave(paste0(OUT_DIR, "13_city_rmse_diff.png"),
       p2, width=11, height=8, dpi=300, bg="white")
cat("Saved: 13_city_rmse_diff\n")

# ── FIGURE 3: Scatter — population vs RMSE diff (ggrepel) ─────
p3 <- ggplot(diff_df,
             aes(x=pop_M, y=rmse_diff,
                 colour=winner, label=city_short)) +
  geom_hline(yintercept=0, linetype="dashed",
             colour="grey50", linewidth=0.5) +
  geom_smooth(aes(x=pop_M, y=rmse_diff),
              method="lm", se=TRUE,
              colour="grey40", linewidth=0.7,
              linetype="dashed", inherit.aes=FALSE) +
  geom_point(size=3, alpha=0.9) +
  geom_text(size=2.2, colour="grey20",
            vjust=-0.8, hjust=0.5) +
  scale_colour_manual(
    values=c("SFNN better"="#2196F3",
             "TSIR better"="#E53935"),
    name=NULL
  ) +
  scale_x_continuous(labels=function(x) paste0(x, "M")) +
  facet_wrap(~k_label, ncol=2, scales="free_y") +
  labs(
    title    = "Population vs RMSE Difference (TSIR − SFNN)",
    subtitle = paste0(
      "West Bengal 2017-2019 | ",
      "Blue = SFNN better | Red = TSIR better | ",
      "Dashed = OLS trend"),
    x       = "District population (millions, 2017-2019 mean)",
    y       = "RMSE difference (TSIR − SFNN, cases/biweek)",
    caption = paste0(
      "Positive y = SFNN achieves lower RMSE. ",
      "Negative y = TSIR achieves lower RMSE.")
  ) +
  theme_classic(base_size=10) +
  theme(
    plot.title      = element_text(face="bold", size=11, hjust=0),
    plot.subtitle   = element_text(colour="grey40", size=8,
                                   hjust=0, margin=margin(b=6)),
    plot.caption    = element_text(colour="grey50", size=7, hjust=0),
    strip.text      = element_text(face="bold", size=10),
    legend.position = "top",
    plot.margin     = margin(10,14,8,10)
  )

ggsave(paste0(OUT_DIR, "14_population_vs_rmse_diff.pdf"),
       p3, width=12, height=6, device=cairo_pdf)
ggsave(paste0(OUT_DIR, "14_population_vs_rmse_diff.png"),
       p3, width=12, height=6, dpi=300, bg="white")
cat("Saved: 14_population_vs_rmse_diff\n")

# ── FINAL SUMMARY TABLE ───────────────────────────────────────
cat("\n============================================================\n")
cat("FINAL COMPARISON TABLE — k=1 and k=34\n")
cat("============================================================\n")

rmse_city %>%
  select(pop_rank, city, pop_M, k,
         rmse_sfnn, rmse_tsir, rmse_diff,
         sfnn_better, pct_diff) %>%
  mutate(
    rmse_sfnn = round(rmse_sfnn, 1),
    rmse_tsir = round(rmse_tsir, 1),
    rmse_diff = round(rmse_diff, 1),
    pct_diff  = round(pct_diff,  1),
    winner    = ifelse(sfnn_better, "SFNN", "TSIR")
  ) %>%
  arrange(k, pop_rank) %>%
  print(n=40)

# ── CORRELATION TEST ──────────────────────────────────────────
cat("\n=== CORRELATION: Population vs RMSE difference ===\n")
for (k_val in c(1, 34)) {
  d  <- rmse_city %>% filter(k == k_val)
  ct <- cor.test(d$mean_pop, d$rmse_diff, method="spearman")
  cat(sprintf("k=%d: Spearman rho=%.3f, p=%.4f\n",
              k_val, ct$estimate, ct$p.value))
}
