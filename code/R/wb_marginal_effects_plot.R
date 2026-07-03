# ============================================================
# wb_marginal_effects_plot.R
# Marginal effects of MCV1 and MCV2 with uncertainty bands
# South Twenty Four Parganas | 2008-2019
#
# FIGURES:
#   1. Three marginal effects — total, MCV1, MCV2
#      with 5th-95th percentile bands
#   2. Scenario trajectories — observed, no_vacc, v1_only
#      with uncertainty bands
# ============================================================

#library(tidyverse)
library(patchwork)
library(readr)
library(dplyr)
library(scales)

BASE    <- ".../.../.../"
OUT_DIR <- paste0(BASE, "experiments/figures/")
EXP_DIR <- paste0(BASE, "experiments/")
CITY    <- "South Twenty Four Parganas"

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

df_effects   <- read_csv(
  paste0(EXP_DIR, "counterfactual_effects_uncertainty.csv"),
  show_col_types=FALSE)
df_scenarios <- read_csv(
  paste0(EXP_DIR, "counterfactual_scenarios_uncertainty.csv"),
  show_col_types=FALSE)

# Also load actual observed cases for reference
cases_obs <- read_csv(
  paste0(BASE, "data/cases_biweekly.csv"),
  show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="cases") %>%
  rename(time=`...1`) %>%
  filter(city == CITY) %>%
  mutate(time=round(time, 4))

cat(sprintf("Effects: %d rows | Scenarios: %d rows\n",
            nrow(df_effects), nrow(df_scenarios)))

save_plot <- function(p, name, w=11, h=5.5) {
  ggsave(paste0(OUT_DIR, name, ".pdf"),
         p, width=w, height=h, device=cairo_pdf)
  ggsave(paste0(OUT_DIR, name, ".png"),
         p, width=w, height=h, dpi=300, bg="white")
  cat(sprintf("Saved: %s\n", name))
}

base_theme <- theme_classic(base_size=11) +
  theme(
    panel.border       = element_rect(colour="black",
                                      fill=NA, linewidth=1),
    strip.background   = element_rect(colour="black",
                                      fill="white",
                                      linewidth=0.8),
    strip.text         = element_text(face="bold", size=10),
    plot.title         = element_text(face="bold", size=12,
                                      hjust=0),
    plot.subtitle      = element_text(colour="grey40", size=9,
                                      hjust=0,
                                      margin=margin(b=8)),
    plot.caption       = element_text(colour="grey50", size=7.5,
                                      hjust=0),
    axis.title         = element_text(size=10),
    axis.text          = element_text(size=9, colour="black"),
    panel.grid.major   = element_line(colour="grey92",
                                      linewidth=0.35),
    panel.grid.minor   = element_blank(),
    plot.margin        = margin(12,16,10,12),
    legend.position    = "top",
    legend.title       = element_blank(),
    legend.text        = element_text(size=9),
    legend.key.width   = unit(1.5, "cm")
  )

year_breaks <- seq(floor(min(df_effects$time)),
                   ceiling(max(df_effects$time)), by=1)

# ── COLOUR PALETTE ────────────────────────────────────────────
pal <- c(
  "Total effect\n(no vacc - observed)"    = "#8250DF",
  "Effect of MCV1\n(no vacc - MCV1 only)" = "#1F6FEB",
  "Effect of MCV2\n(MCV1 only - observed)"= "#1A7F37"
)

# =============================================================================
# FIGURE 1: Three marginal effects with uncertainty bands
# =============================================================================

# Separate effects for cleaner plotting
df_total <- df_effects %>%
  filter(grepl("Total", effect))
df_mcv1  <- df_effects %>%
  filter(grepl("MCV1", effect))
df_mcv2  <- df_effects %>%
  filter(grepl("MCV2", effect))

p1 <- ggplot() +

  geom_hline(yintercept=0, linetype="dashed",
             colour="#888888", linewidth=0.5) +

  # Total effect band + line
  geom_ribbon(data=df_total,
              aes(x=time, ymin=lo, ymax=hi),
              fill="#8250DF", alpha=0.15) +
  geom_line(data=df_total,
            aes(x=time, y=mean,
                colour="Total effect\n(no vacc - observed)"),
            linewidth=0.75) +

  # MCV1 effect band + line
  geom_ribbon(data=df_mcv1,
              aes(x=time, ymin=lo, ymax=hi),
              fill="#1F6FEB", alpha=0.15) +
  geom_line(data=df_mcv1,
            aes(x=time, y=mean,
                colour="Effect of MCV1\n(no vacc - MCV1 only)"),
            linewidth=0.75) +

  # MCV2 effect band + line
  geom_ribbon(data=df_mcv2,
              aes(x=time, ymin=lo, ymax=hi),
              fill="#1A7F37", alpha=0.15) +
  geom_line(data=df_mcv2,
            aes(x=time, y=mean,
                colour="Effect of MCV2\n(MCV1 only - observed)"),
            linewidth=0.75) +

  scale_colour_manual(values=pal, name=NULL) +
  scale_x_continuous(
    breaks = year_breaks,
    expand = expansion(mult=0.01)
  ) +
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult=c(0.05, 0.05))
  ) +
  labs(
    title    = "Marginal Effects of MCV1 and MCV2 on Measles Cases",
    subtitle = paste0(CITY,
                      " | 2008-2019 | ",
                      "Positive = cases averted by vaccination"),
    x        = "Year",
    y        = "Case difference",
    caption  = paste0(
      "Shaded bands: 5th-95th percentile across ",
      "100 stochastic TSIR simulations. ",
      "Dashed line = zero effect. ",
      "Ferrari et al. (2012) Eq. 3.")
  ) +
  base_theme +
  theme(legend.key.width=unit(1.8, "cm"))

save_plot(p1, "06_marginal_effects_uncertainty", w=12, h=6)

# =============================================================================
# FIGURE 2: Scenario trajectories with uncertainty
# =============================================================================

# Join actual observed cases
df_scen_plot <- df_scenarios %>%
  left_join(cases_obs %>%
              select(time, actual_cases=cases) %>%
              mutate(time=round(time,4)),
            by="time")

p2 <- ggplot(df_scen_plot, aes(x=time)) +

  # No vaccination band
  geom_ribbon(aes(ymin=novacc_lo, ymax=novacc_hi),
              fill="#CF222E", alpha=0.12) +
  geom_line(aes(y=no_vacc,
                colour="No vaccination (V1=0, V2=0)"),
            linewidth=0.7, linetype="dashed") +

  # MCV1 only band
  geom_ribbon(aes(ymin=v1only_lo, ymax=v1only_hi),
              fill="#1F6FEB", alpha=0.12) +
  geom_line(aes(y=v1_only,
                colour="MCV1 only (V2=0)"),
            linewidth=0.7, linetype="dashed") +

  # Observed actual cases (solid)
  geom_line(aes(y=actual_cases,
                colour="Observed (MCV1+MCV2)"),
            linewidth=0.85) +

  scale_colour_manual(
    values=c(
      "No vaccination (V1=0, V2=0)" = "#CF222E",
      "MCV1 only (V2=0)"            = "#1F6FEB",
      "Observed (MCV1+MCV2)"        = "#1A1A2E"
    ),
    name=NULL
  ) +
  scale_x_continuous(
    breaks = year_breaks,
    expand = expansion(mult=0.01)
  ) +
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult=c(0.02, 0.08))
  ) +
  labs(
    title    = "Measles Cases Under Three Vaccination Scenarios",
    subtitle = paste0(CITY,
                      " | 2008-2019 | ",
                      "Dashed = counterfactual TSIR | ",
                      "Solid = actual observed"),
    x        = "Year",
    y        = "Measles cases per biweek",
    caption  = paste0(
      "Shaded bands: 5th-95th percentile uncertainty. ",
      "Ferrari et al. (2012) Eq. 3. ",
      "Solid navy line = actual reported cases (not TSIR fitted).")
  ) +
  base_theme

save_plot(p2, "07_scenario_trajectories_uncertainty", w=12, h=6)

# =============================================================================
# COMBINED FIGURE (p2 over p1)
# =============================================================================

p_combined <- p2 / p1 +
  plot_annotation(
    title    = paste0("Vaccination Impact Analysis — ", CITY),
    subtitle = "2008-2019 | TSIR counterfactual with stochastic uncertainty",
    theme    = theme(
      plot.title    = element_text(size=13, face="bold",
                                   hjust=0),
      plot.subtitle = element_text(size=10, colour="#555555",
                                   hjust=0, margin=margin(b=8))
    )
  )

save_plot(p_combined, "08_combined_marginal_effects",
          w=12, h=11)

cat("\nDone. Figures saved to:", OUT_DIR, "\n")
