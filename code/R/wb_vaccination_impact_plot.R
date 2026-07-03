# ============================================================
# wb_vaccination_impact_plot.R
# Combined vaccination impact — all valid districts
# Excludes North 24 Parganas and Maldah (anomalous outbreaks)
#
# FIGURES:
#   1. Stacked bar — observed + averted (MCV1+MCV2) per district
#   2. % averted dot plot — MCV1 vs MCV2 split
#   3. MCV2 coverage vs additional cases averted scatter
#   4. West Bengal total summary bar
#   5. All districts — biweekly trajectories with shaded impact
#   5b. West Bengal annual totals — observed vs averted
# ============================================================

#library(tidyverse)
library(readr)
library(dplyr)
library(patchwork)
library(scales)

BASE     <- "/.../.../.../"
OUT_DIR  <- paste0(BASE, "experiments/figures/")
CSV_PATH <- paste0(BASE, "experiments/",
                   "counterfactual_all_cities_summary.csv")
RAW_PATH <- paste0(BASE, "experiments/",
                   "counterfactual_all_cities.csv")
V2_PATH  <- paste0(BASE, "data/V2.csv")

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# ── LOAD ──────────────────────────────────────────────────────
df_all      <- read_csv(CSV_PATH, show_col_types=FALSE)
all_results <- read_csv(RAW_PATH, show_col_types=FALSE) %>%
  mutate(averted = pmax(no_vacc - obs, 0),
         year    = floor(time))

# ── EXCLUDE ANOMALOUS DISTRICTS ───────────────────────────────
# North 24 Parganas: 2017 outbreak (3,948 cases vs ~850 baseline)
#   Observed > counterfactual — surge dynamics outside model scope
# Maldah: elevated 2017-2018 cases above training baseline
anomalous <- c("North Twenty Four Parganas", "Maldah")

df          <- df_all  %>% filter(!city %in% anomalous)
all_results <- all_results %>% filter(!city %in% anomalous)

cat(sprintf("Valid districts: %d\n", nrow(df)))
cat(sprintf("Excluded: %s\n\n", paste(anomalous, collapse=", ")))

cat("=== ANOMALOUS DISTRICTS ===\n")
df_all %>%
  filter(city %in% anomalous) %>%
  select(city, obs_total, no_vacc_total,
         averted_total, pct_averted) %>%
  mutate(across(where(is.numeric), \(x) round(x, 1))) %>%
  print()

cat("\n=== VALID 17 DISTRICTS TOTAL ===\n")
cat(sprintf("  Observed:       %d\n",
            round(sum(df$obs_total))))
cat(sprintf("  No vaccination: %d\n",
            round(sum(df$no_vacc_total))))
cat(sprintf("  Total averted:  %d (%.1f%%)\n",
            round(sum(df$averted_total)),
            sum(df$averted_total)/sum(df$no_vacc_total)*100))
cat(sprintf("  By MCV1:        %d (%.1f%% of averted)\n",
            round(sum(df$averted_mcv1)),
            sum(df$averted_mcv1)/sum(df$averted_total)*100))
cat(sprintf("  By MCV2:        %d (%.1f%% of averted)\n",
            round(sum(df$averted_mcv2)),
            sum(df$averted_mcv2)/sum(df$averted_total)*100))

# ── SHORT DISTRICT NAMES ──────────────────────────────────────
df <- df %>%
  mutate(city_short = case_when(
    city == "North Twenty Four Parganas" ~ "North 24 Pgs",
    city == "South Twenty Four Parganas" ~ "South 24 Pgs",
    city == "Dakshin Dinajpur"           ~ "D. Dinajpur",
    city == "Uttar Dinajpur"             ~ "U. Dinajpur",
    city == "Paschim Medinipur"          ~ "P. Medinipur",
    city == "Purba Medinipur"            ~ "Purba Med.",
    TRUE ~ city
  ))

all_results <- all_results %>%
  left_join(df %>% select(city, city_short), by="city")

# ── THEME ─────────────────────────────────────────────────────
base_theme <- theme_classic(base_size=11) +
  theme(
    plot.title         = element_text(face="bold", size=12,
                                      hjust=0),
    plot.subtitle      = element_text(colour="grey40", size=9,
                                      hjust=0,
                                      margin=margin(b=8)),
    plot.caption       = element_text(colour="grey50", size=7.5,
                                      hjust=0),
    axis.title         = element_text(size=10),
    axis.text          = element_text(size=9),
    panel.grid.major.x = element_line(colour="grey92",
                                      linewidth=0.4),
    panel.grid.minor   = element_blank(),
    plot.margin        = margin(12,16,10,12),
    legend.position    = "top",
    legend.title       = element_blank(),
    legend.text        = element_text(size=9)
  )

save_plot <- function(p, name, w=10, h=6) {
  ggsave(paste0(OUT_DIR, name, ".pdf"),
         p, width=w, height=h, device=cairo_pdf)
  ggsave(paste0(OUT_DIR, name, ".png"),
         p, width=w, height=h, dpi=300, bg="white")
  cat(sprintf("Saved: %s\n", name))
}

# =============================================================================
# FIGURE 1: Stacked bar — burden decomposition per district
# =============================================================================

df_long <- df %>%
  arrange(desc(no_vacc_total)) %>%
  mutate(city_short = factor(city_short, levels=rev(city_short))) %>%
  select(city_short, obs_total, averted_mcv1, averted_mcv2) %>%
  pivot_longer(cols=c(obs_total, averted_mcv1, averted_mcv2),
               names_to="component", values_to="cases") %>%
  mutate(
    component = recode(component,
      "obs_total"    = "Observed burden",
      "averted_mcv1" = "Averted by MCV1",
      "averted_mcv2" = "Additional averted by MCV2"
    ),
    component = factor(component, levels=c(
      "Observed burden",
      "Averted by MCV1",
      "Additional averted by MCV2"))
  )

df_labels <- df %>%
  arrange(desc(no_vacc_total)) %>%
  mutate(
    city_short = factor(city_short, levels=rev(city_short)),
    label_pct  = sprintf("%.0f%% averted", pct_averted),
    label_obs  = comma(round(obs_total))
  )

p1 <- ggplot(df_long,
             aes(x=city_short, y=cases, fill=component)) +
  geom_col(width=0.72, alpha=0.88) +
  geom_text(data=df_labels,
            aes(x=city_short, y=no_vacc_total,
                label=label_pct),
            inherit.aes=FALSE,
            hjust=-0.08, size=2.9, colour="grey30") +
  coord_flip(clip="off") +
  scale_fill_manual(values=c(
    "Observed burden"            = "#CF222E",
    "Averted by MCV1"            = "#1F6FEB",
    "Additional averted by MCV2" = "#1A7F37"
  )) +
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult=c(0, 0.20))
  ) +
  labs(
    title    = "Measles Burden and Vaccination Impact by District (2017-2019)",
    subtitle = paste0(
      "West Bengal (17 districts) | ",
      "Red = observed cases | Blue = averted by MCV1 | ",
      "Green = additional averted by MCV2"),
    x       = NULL,
    y       = "Measles cases",
    caption = paste0(
      "Cases averted = TSIR counterfactual (Ferrari et al. 2012 Eq.3). ",
      "Bars ordered by counterfactual total. ",
      "North 24 Parganas and Maldah excluded (anomalous outbreaks).")
  ) +
  base_theme +
  theme(
    panel.grid.major.x = element_line(colour="grey92",
                                      linewidth=0.4),
    panel.grid.major.y = element_blank()
  )

save_plot(p1, "01_vaccination_burden_all_districts", w=11, h=8)

# =============================================================================
# FIGURE 2: % averted dot plot
# =============================================================================

df_pct <- df %>%
  arrange(pct_averted) %>%
  mutate(city_short = factor(city_short, levels=city_short)) %>%
  select(city_short, pct_mcv1, pct_mcv2, pct_averted) %>%
  pivot_longer(cols=c(pct_mcv1, pct_mcv2),
               names_to="vaccine", values_to="pct") %>%
  mutate(vaccine = recode(vaccine,
    "pct_mcv1" = "MCV1 contribution",
    "pct_mcv2" = "MCV2 contribution"
  ))

df_total_pct <- df %>%
  arrange(pct_averted) %>%
  mutate(city_short = factor(city_short, levels=city_short))

p2 <- ggplot() +
  geom_segment(
    data=df_total_pct,
    aes(x=0, xend=pct_averted,
        y=city_short, yend=city_short),
    colour="grey80", linewidth=0.8
  ) +
  geom_point(
    data=df_pct,
    aes(x=pct, y=city_short, colour=vaccine),
    size=3, alpha=0.9
  ) +
  geom_point(
    data=df_total_pct,
    aes(x=pct_averted, y=city_short),
    shape=21, size=3.5, fill="white",
    colour="grey40", stroke=0.8
  ) +
  geom_text(
    data=df_total_pct,
    aes(x=pct_averted, y=city_short,
        label=sprintf("%.0f%%", pct_averted)),
    hjust=-0.4, size=2.9, colour="grey30"
  ) +
  scale_colour_manual(values=c(
    "MCV1 contribution" = "#1F6FEB",
    "MCV2 contribution" = "#1A7F37"
  )) +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult=c(0.02, 0.15)),
    limits = c(0, NA)
  ) +
  labs(
    title    = "Percentage of Measles Cases Averted by Vaccination (2017-2019)",
    subtitle = paste0(
      "West Bengal (17 districts) | ",
      "Circle = total % averted | ",
      "Blue = MCV1 | Green = MCV2 contribution"),
    x       = "Cases averted (%)",
    y       = NULL,
    caption = paste0(
      "Total % averted = (no_vacc - observed) / no_vacc × 100. ",
      "Ordered by total % averted. ",
      "North 24 Parganas and Maldah excluded.")
  ) +
  base_theme +
  theme(
    panel.grid.major.y = element_line(colour="grey92",
                                      linewidth=0.4),
    panel.grid.major.x = element_blank()
  )

save_plot(p2, "02_pct_averted_all_districts", w=10, h=8)

# =============================================================================
# FIGURE 3: MCV2 coverage vs % averted by MCV2
# =============================================================================

v2_raw <- read_csv(V2_PATH, show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="v2") %>%
  mutate(year=as.integer(Year)) %>%
  filter(year %in% 2017:2019) %>%
  group_by(city) %>%
  summarise(mean_v2=mean(v2, na.rm=TRUE), .groups="drop")

df_scatter <- df %>%
  left_join(v2_raw, by="city") %>%
  filter(!is.na(mean_v2))

p3 <- ggplot(df_scatter,
             aes(x=mean_v2*100, y=pct_mcv2)) +
  geom_smooth(method="lm", se=TRUE,
              colour="#1A7F37", fill="#B3E6C0",
              linewidth=0.9, alpha=0.3) +
  geom_point(aes(size=obs_total),
             colour="#1A7F37", alpha=0.75) +
  geom_text(aes(label=city_short),
            vjust=-0.8, size=2.8, colour="grey40") +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    limits = c(40, 100)
  ) +
  scale_y_continuous(
    labels = function(x) paste0(x, "%")
  ) +
  scale_size_continuous(
    name   = "Observed cases",
    labels = comma,
    range  = c(2, 8)
  ) +
  labs(
    title    = "MCV2 Coverage vs Additional Cases Averted by MCV2",
    subtitle = paste0(
      "West Bengal districts (17) | 2017-2019 | ",
      "Point size = observed case burden"),
    x       = "Mean MCV2 coverage 2017-2019 (%)",
    y       = "Cases averted by MCV2 (% of counterfactual)",
    caption = "Linear trend with 95% CI. Descriptive association only."
  ) +
  base_theme

save_plot(p3, "03_mcv2_coverage_vs_averted", w=9, h=7)

# =============================================================================
# FIGURE 4: West Bengal total summary
# =============================================================================

wb_total <- tibble(
  component = c("Observed burden",
                "Averted by MCV1",
                "Averted by MCV2"),
  cases = c(
    sum(df$obs_total),
    sum(df$averted_mcv1),
    sum(df$averted_mcv2)
  )
) %>%
  mutate(
    component = factor(component, levels=c(
      "Averted by MCV2",
      "Averted by MCV1",
      "Observed burden")),
    pct = cases / sum(cases) * 100
  )

p4 <- ggplot(wb_total,
             aes(x=1, y=cases, fill=component)) +
  geom_col(width=0.5, alpha=0.88) +
  geom_text(aes(label=sprintf("%d\n(%.0f%%)",
                               round(cases), pct)),
            position=position_stack(vjust=0.5),
            size=3.5, colour="white", fontface="bold") +
  scale_fill_manual(values=c(
    "Observed burden"  = "#CF222E",
    "Averted by MCV1"  = "#1F6FEB",
    "Averted by MCV2"  = "#1A7F37"
  )) +
  scale_y_continuous(labels=comma) +
  labs(
    title    = "West Bengal Total: Vaccination Programme Impact 2017-2019",
    subtitle = sprintf(
      "17 districts | Counterfactual total: %s | Averted: %s (%.0f%%)",
      comma(round(sum(df$no_vacc_total))),
      comma(round(sum(df$averted_total))),
      sum(df$averted_total)/sum(df$no_vacc_total)*100),
    x       = NULL,
    y       = "Measles cases",
    caption = paste0(
      "TSIR counterfactual with Ferrari et al. (2012) Eq. 3 (V1+V2). ",
      "North 24 Parganas and Maldah excluded.")
  ) +
  base_theme +
  theme(
    axis.text.x  = element_blank(),
    axis.ticks.x = element_blank(),
    axis.line.x  = element_blank()
  )

save_plot(p4, "04_wb_total_summary", w=6, h=7)

# =============================================================================
# FIGURE 5: All districts — biweekly trajectories with shaded impact
# =============================================================================

all_results <- all_results %>%
  mutate(city_short = factor(
    city_short,
    levels = df %>%
      arrange(desc(pct_averted)) %>%
      pull(city_short)
  ))

p5 <- ggplot(all_results, aes(x=time)) +
  geom_ribbon(aes(ymin=obs, ymax=no_vacc),
              fill="#1F6FEB", alpha=0.25) +
  geom_line(aes(y=no_vacc, colour="Without vaccination"),
            linewidth=0.5, alpha=0.8) +
  geom_line(aes(y=obs, colour="Observed"),
            linewidth=0.6) +
  scale_colour_manual(values=c(
    "Without vaccination" = "#CF222E",
    "Observed"            = "#1A1A2E"
  ), name=NULL) +
  scale_x_continuous(
    breaks = c(2017, 2018, 2019),
    labels = c("2017","2018","2019"),
    expand = expansion(mult=0.02)
  ) +
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult=c(0.02, 0.1))
  ) +
  facet_wrap(~city_short, ncol=4, scales="free_y") +
  labs(
    title    = "Combined Vaccination Impact by District (MCV1+MCV2), 2017-2019",
    subtitle = paste0(
      "West Bengal (17 districts) | ",
      "Blue shaded = cases averted by combined programme | ",
      "Free y-axis scale"),
    x       = "Year",
    y       = "Measles cases per biweek",
    caption = paste0(
      "Red = TSIR counterfactual (V1=0, V2=0). ",
      "Navy = observed. ",
      "Shaded = vaccination impact. ",
      "Ferrari et al. (2012) Eq. 3. ",
      "North 24 Parganas and Maldah excluded.")
  ) +
  base_theme +
  theme(
    strip.text      = element_text(size=7.5, face="bold"),
    axis.text       = element_text(size=7),
    axis.text.x     = element_text(angle=45, hjust=1),
    legend.position = "top",
    panel.spacing   = unit(0.5, "lines")
  )

save_plot(p5, "05_combined_vaccination_impact_all_districts",
          w=14, h=10)

# =============================================================================
# FIGURE 5b: West Bengal annual totals
# =============================================================================

wb_annual <- all_results %>%
  group_by(year) %>%
  summarise(
    obs     = sum(obs,     na.rm=TRUE),
    no_vacc = sum(no_vacc, na.rm=TRUE),
    averted = pmax(sum(no_vacc - obs, na.rm=TRUE), 0),
    .groups = "drop"
  ) %>%
  pivot_longer(cols=c(obs, averted),
               names_to="component", values_to="cases") %>%
  mutate(component = recode(component,
    "obs"     = "Observed burden",
    "averted" = "Cases averted (MCV1+MCV2)"
  ))

wb_annual_pct <- all_results %>%
  group_by(year) %>%
  summarise(
    total   = sum(no_vacc, na.rm=TRUE),
    averted = pmax(sum(no_vacc - obs, na.rm=TRUE), 0),
    .groups = "drop"
  ) %>%
  mutate(pct = averted / total * 100)

p5b <- ggplot(wb_annual,
              aes(x=factor(year), y=cases,
                  fill=component)) +
  geom_col(width=0.6, alpha=0.88) +
  geom_text(
    data=wb_annual_pct,
    aes(x=factor(year), y=total,
        label=sprintf("%.0f%% averted", pct)),
    inherit.aes=FALSE,
    vjust=-0.4, size=3.5, colour="grey30"
  ) +
  scale_fill_manual(values=c(
    "Observed burden"           = "#CF222E",
    "Cases averted (MCV1+MCV2)" = "#1F6FEB"
  )) +
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult=c(0, 0.12))
  ) +
  labs(
    title    = "West Bengal Annual Measles: Observed vs Averted (2017-2019)",
    subtitle = sprintf(
      "17 districts | Combined MCV1+MCV2 impact | Counterfactual total: %s",
      comma(round(sum(df$no_vacc_total)))),
    x       = "Year",
    y       = "Measles cases (17 districts)",
    caption = paste0(
      "TSIR counterfactual (Ferrari et al. 2012 Eq.3). ",
      "North 24 Parganas and Maldah excluded.")
  ) +
  base_theme

save_plot(p5b, "05b_wb_annual_combined_impact", w=7, h=6)

# =============================================================================
# FINAL QUANTIFIED TABLE
# =============================================================================

cat("\n=== FINAL QUANTIFIED IMPACT TABLE (17 districts) ===\n")
df %>%
  select(city, obs_total, no_vacc_total,
         averted_total, pct_averted,
         averted_mcv1, averted_mcv2) %>%
  mutate(across(where(is.numeric), \(x) round(x, 1))) %>%
  arrange(desc(averted_total)) %>%
  print(n=20)

cat(sprintf("\nAll figures saved to: %s\n", OUT_DIR))
