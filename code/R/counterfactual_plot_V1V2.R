# ============================================================
# counterfactual_plot_V1V2.R
# Counterfactual figures with MCV1+MCV2 decomposition
# THREE SCENARIOS: observed, v1_only, no_vacc
#
# Figures:
#   1. Trajectory — three scenario lines
#   2. Annual cases averted — MCV1 vs MCV2 stacked bars
#   3. Coverage vs cases scatter
#   4. Cumulative averted — MCV1 and MCV2 decomposed
#   5. Biweekly marginal effects
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

BASE     <- "/.../.../.../"
CSV_PATH <- paste0(BASE,"experiments/counterfactual_results_V1V2.csv")
OUT_DIR  <- paste0(BASE,"experiments/figures/V1V2/")
dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

save_plot <- function(p, name, w=10, h=5.5) {
  ggsave(paste0(OUT_DIR, name, ".pdf"),
         plot=p, width=w, height=h, device="pdf")
  ggsave(paste0(OUT_DIR, name, ".png"),
         plot=p, width=w, height=h, dpi=300)
  cat("Saved:", name, "\n")
}

df <- read.csv(CSV_PATH)

cat(sprintf("Loaded %d rows\n", nrow(df)))
cat(sprintf("Scenarios: %s\n",
            paste(unique(df$scenario), collapse=", ")))

df$scenario <- factor(df$scenario,
  levels = c("no_vacc", "v1_only", "observed"),
  labels = c("No vaccination",
             "MCV1 only (no MCV2)",
             "Observed (MCV1 + MCV2)")
)
df$year <- floor(df$time)

# In counterfactual_plot_V1V2.R, find and replace the pal_line definition

# Colour palette:
pal_line <- c(
  "No vaccination"          = "#CF222E",
  "MCV1 only (no MCV2)"     = "#1A7F37",
  "Observed (MCV1 + MCV2)"  = "#1F6FEB"
)
pal_fill <- c(
  "No vaccination"          = "#FFBDBD",
  "MCV1 only (no MCV2)"     = "#B3E6C0",
  "Observed (MCV1 + MCV2)"  = "#BDD7FF"
)
base_theme <- theme_classic(base_size=11) +
  theme(
    plot.title         = element_text(face="bold", size=12),
    plot.subtitle      = element_text(colour="grey40", size=9,
                                      margin=margin(b=8)),
    plot.caption       = element_text(colour="grey50", size=7.5,
                                      hjust=0),
    legend.position    = "top",
    legend.key.width   = unit(1.4,"cm"),
    legend.text        = element_text(size=9),
    axis.title         = element_text(size=10),
    axis.text          = element_text(size=9),
    panel.grid.major.y = element_line(colour="grey92",
                                      linewidth=0.4),
    panel.grid.minor   = element_blank(),
    plot.margin        = margin(12,16,10,12)
  )

year_breaks <- seq(floor(min(df$time)),
                   ceiling(max(df$time)), by=1)

# Wide format
df_wide <- df %>%
  select(time, year, scenario, mean) %>%
  pivot_wider(names_from=scenario, values_from=mean)

names(df_wide) <- gsub(" ", "_", names(df_wide))
names(df_wide) <- gsub("[()+-]", "", names(df_wide))
names(df_wide) <- gsub("__+", "_", names(df_wide))

# Rename to safe keys
df_wide <- df_wide %>%
  rename_with(~ gsub("No_vaccination", "no_vacc", .x)) %>%
  rename_with(~ gsub("MCV1_only_no_MCV2", "v1_only", .x)) %>%
  rename_with(~ gsub("Observed_MCV1_MCV2", "observed", .x))

# =============================================================================
# FIGURE 1: Three scenario trajectories
# =============================================================================
p1 <- ggplot(df, aes(x=time, colour=scenario, fill=scenario)) +
  geom_ribbon(aes(ymin=lower, ymax=upper),
              alpha=0.15, colour=NA) +
  geom_line(aes(y=mean), linewidth=0.9) +
  scale_colour_manual(values=pal_line, name=NULL) +
  scale_fill_manual(values=pal_fill,   name=NULL) +
  scale_x_continuous(breaks=year_breaks,
                     expand=expansion(mult=0.01)) +
  scale_y_continuous(labels=comma,
                     expand=expansion(mult=c(0.02,0.08))) +
  labs(
    title    = "Measles Cases Under MCV1+MCV2 Vaccination Scenarios",
    subtitle = paste0(
      "South Twenty Four Parganas | 2017-2019 | ",
      "Ferrari et al. (2012) Eq. 3"),
    x       = "Year",
    y       = "Measles cases",
    caption = paste0(
      "Observed line = actual reported cases. ",
      "Counterfactual lines = TSIR model with V1=0,V2=0 ",
      "and V1=observed,V2=0.")
  ) +
  base_theme

save_plot(p1, "01_counterfactual_trajectories_V1V2")

# =============================================================================
# FIGURE 2: Annual cases averted — stacked bars (MCV1 + MCV2)
# =============================================================================
df_annual <- df_wide %>%
  group_by(year) %>%
  summarise(
    averted_mcv1 = sum(no_vacc - v1_only,  na.rm=TRUE),
    averted_mcv2 = sum(v1_only - observed, na.rm=TRUE),
    .groups="drop"
  ) %>%
  pivot_longer(cols=c(averted_mcv1, averted_mcv2),
               names_to="vaccine", values_to="cases") %>%
  mutate(vaccine=recode(vaccine,
    "averted_mcv1"="Cases averted by MCV1",
    "averted_mcv2"="Additional cases averted by MCV2"
  ))

p2 <- ggplot(df_annual,
             aes(x=factor(year), y=cases, fill=vaccine)) +
  geom_col(position="stack", width=0.65, alpha=0.88) +
  geom_hline(yintercept=0, colour="grey40",
             linewidth=0.5, linetype="dashed") +
  scale_fill_manual(
    values=c("Cases averted by MCV1"                  = "#1F6FEB",
             "Additional cases averted by MCV2"        = "#1A7F37"),
    name=NULL) +
  scale_y_continuous(labels=comma,
                     expand=expansion(mult=c(0.05,0.12))) +
  labs(
    title    = "Annual Measles Cases Averted: MCV1 vs MCV2 Contributions",
    subtitle = paste0(
      "South Twenty Four Parganas | ",
      "Blue=MCV1 impact | Green=additional MCV2 impact"),
    x       = "Year",
    y       = "Cases averted",
    caption = paste0(
      "MCV1 impact = no_vacc - v1_only. ",
      "MCV2 impact = v1_only - observed. ",
      "Ferrari et al. (2012) Eq. 3.")
  ) +
  base_theme

save_plot(p2, "02_cases_averted_decomposed_V1V2")

# =============================================================================
# FIGURE 3: Coverage vs observed cases
# =============================================================================
v1_df <- data.frame(
  year = 2008:2019,
  mcv1 = c(0.86,0.87,0.89,0.90,0.91,
            0.92,0.94,0.94,0.94,0.93,0.93,0.93)
)
v2_df <- data.frame(
  year = 2008:2019,
  mcv2 = c(0,0,0,0.27,0.36,0.51,
            0.60,0.69,0.76,0.95,0.93,0.93)
)

df_obs_annual <- df %>%
  filter(scenario=="Observed (MCV1 + MCV2)") %>%
  group_by(year) %>%
  summarise(total_cases=sum(mean, na.rm=TRUE), .groups="drop") %>%
  left_join(v1_df, by="year") %>%
  left_join(v2_df, by="year") %>%
  filter(!is.na(mcv1)) %>%
  mutate(combined=mcv1 + (1-mcv1)*mcv2)

p3 <- ggplot(df_obs_annual,
             aes(x=combined*100, y=total_cases)) +
  geom_smooth(method="loess", se=TRUE,
              colour="#1F6FEB", fill="#BDD7FF",
              linewidth=1, alpha=0.3) +
  geom_point(size=3.5, colour="#1F6FEB", alpha=0.85) +
  geom_text(aes(label=year), vjust=-0.9,
            size=3, colour="grey40") +
  scale_x_continuous(labels=function(x) paste0(x,"%"),
                     breaks=seq(86,100,by=2)) +
  scale_y_continuous(labels=comma) +
  labs(
    title    = "Combined Two-Dose Coverage vs Annual Observed Cases",
    subtitle = paste0(
      "South Twenty Four Parganas | ",
      "x = V1 + (1-V1)*V2 (Ferrari et al. 2012)"),
    x       = "Combined two-dose coverage (%)",
    y       = "Annual observed cases",
    caption = "Each point = one year. Loess trend with 95% CI. Descriptive."
  ) +
  base_theme

save_plot(p3, "03_coverage_vs_cases_V1V2", w=8, h=5.5)

# =============================================================================
# FIGURE 4: Cumulative averted — MCV1 and MCV2 decomposed
# =============================================================================
df_cumul <- df_wide %>%
  arrange(time) %>%
  mutate(
    mcv1_step  = no_vacc  - v1_only,
    mcv2_step  = v1_only  - observed,
    cumul_mcv1 = cumsum(replace_na(mcv1_step, 0)),
    cumul_mcv2 = cumsum(replace_na(mcv2_step, 0)),
    cumul_total= cumul_mcv1 + cumul_mcv2
  ) %>%
  select(time, cumul_mcv1, cumul_mcv2, cumul_total) %>%
  pivot_longer(-time, names_to="type", values_to="cases") %>%
  mutate(type=recode(type,
    "cumul_mcv1"  = "Cumulative averted by MCV1",
    "cumul_mcv2"  = "Additional averted by MCV2",
    "cumul_total" = "Total averted (MCV1 + MCV2)"
  ))

p4 <- ggplot(df_cumul,
             aes(x=time, y=cases,
                 colour=type, linetype=type)) +
  geom_line(linewidth=0.9) +
  geom_hline(yintercept=0, colour="grey60",
             linewidth=0.4, linetype="dashed") +
  scale_colour_manual(
    values=c("Cumulative averted by MCV1"     = "#1F6FEB",
             "Additional averted by MCV2"      = "#1A7F37",
             "Total averted (MCV1 + MCV2)"     = "#8250DF"),
    name=NULL) +
  scale_linetype_manual(
    values=c("Cumulative averted by MCV1"     = "solid",
             "Additional averted by MCV2"      = "solid",
             "Total averted (MCV1 + MCV2)"     = "dashed"),
    name=NULL) +
  scale_x_continuous(breaks=year_breaks,
                     expand=expansion(mult=0.01)) +
  scale_y_continuous(labels=comma,
                     expand=expansion(mult=c(0.02,0.08))) +
  labs(
    title    = "Cumulative Cases Averted: MCV1 and MCV2 Contributions",
    subtitle = "South Twenty Four Parganas | Running totals by vaccine dose",
    x       = "Year",
    y       = "Cumulative cases averted",
    caption = "Purple dashed = total. Blue = MCV1 contribution. Green = MCV2 contribution."
  ) +
  base_theme

save_plot(p4, "04_cumulative_averted_V1V2")

# =============================================================================
# FIGURE 5: Biweekly marginal effects
# =============================================================================
df_marginal <- df_wide %>%
  arrange(time) %>%
  mutate(
    averted_total = no_vacc - observed,
    positive      = averted_total >= 0
  ) %>%
  filter(!is.na(averted_total))

p5 <- ggplot(df_marginal,
             aes(x=time, y=averted_total,
                 fill=positive)) +
  geom_col(width=0.038, alpha=0.85) +
  geom_hline(yintercept=0, colour="grey40",
             linewidth=0.5, linetype="dashed") +
  geom_smooth(method="loess", se=FALSE,
              colour="#8250DF", linewidth=0.9,
              aes(group=1)) +
  scale_fill_manual(
    values=c("TRUE"="#1F6FEB","FALSE"="#CF222E"),
    labels=c("TRUE"="Averted","FALSE"="Excess"),
    name=NULL) +
  scale_x_continuous(breaks=year_breaks,
                     expand=expansion(mult=0.01)) +
  scale_y_continuous(labels=comma,
                     expand=expansion(mult=c(0.08,0.08))) +
  labs(
    title    = "Biweekly Cases Averted by Combined MCV1+MCV2 Programme",
    subtitle = paste0(
      "South Twenty Four Parganas | ",
      "Each bar = one biweek | Total vaccination impact"),
    x       = "Year",
    y       = "Cases averted per biweek",
    caption = paste0(
      "Cases averted = TSIR(V1=0,V2=0) - observed. ",
      "Purple line = loess trend.")
  ) +
  base_theme +
  theme(legend.position="top")

save_plot(p5, "05_marginal_effects_V1V2")

# =============================================================================
# SUMMARY
# =============================================================================
total_obs     <- sum(df_wide$observed, na.rm=TRUE)
total_no_vacc <- sum(df_wide$no_vacc,  na.rm=TRUE)
total_v1_only <- sum(df_wide$v1_only,  na.rm=TRUE)

averted_total <- total_no_vacc - total_obs
averted_mcv1  <- total_no_vacc - total_v1_only
averted_mcv2  <- total_v1_only - total_obs

cat(sprintf("\n=== Summary ===\n"))
cat(sprintf("  Observed cases:          %d\n", round(total_obs)))
cat(sprintf("  No vaccination:          %d\n", round(total_no_vacc)))
cat(sprintf("  MCV1 only:               %d\n", round(total_v1_only)))
cat(sprintf("  Total averted:           %d (%.1f%%)\n",
            round(averted_total), averted_total/total_no_vacc*100))
cat(sprintf("  Averted by MCV1:         %d (%.1f%%)\n",
            round(averted_mcv1), averted_mcv1/averted_total*100))
cat(sprintf("  Averted by MCV2:         %d (%.1f%%)\n",
            round(averted_mcv2), averted_mcv2/averted_total*100))

cat(sprintf("\n=== Figures saved to: %s ===\n", OUT_DIR))
cat("  01_counterfactual_trajectories_V1V2.pdf/.png\n")
cat("  02_cases_averted_decomposed_V1V2.pdf/.png\n")
cat("  03_coverage_vs_cases_V1V2.pdf/.png\n")
cat("  04_cumulative_averted_V1V2.pdf/.png\n")
cat("  05_marginal_effects_V1V2.pdf/.png\n")
