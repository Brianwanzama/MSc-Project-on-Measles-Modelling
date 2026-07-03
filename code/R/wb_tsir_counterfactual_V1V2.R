# ============================================================
# wb_tsir_counterfactual_V1V2.R
# TSIR counterfactual — Two-dose vaccination impact
# South Twenty Four Parganas | West Bengal measles
#
# THREE SCENARIOS using Ferrari et al. (2012) Equation 3:
#   observed  — actual observed cases (ground truth)
#   no_vacc   — TSIR with V1=0, V2=0 (no vaccination)
#   v1_only   — TSIR with V1=observed, V2=0 (MCV1 only)
#
# This allows decomposition of vaccination impact:
#   MCV1 contribution:  no_vacc  - v1_only
#   MCV2 contribution:  v1_only  - observed
#   Total averted:      no_vacc  - observed
#
# OUTPUT: counterfactual_results_V1V2.csv
# ============================================================

source("/.../.../.../code/R/tsir_run_functions.R")

library(dplyr)
library(readr)
library(tidyr)

BASE    <- "/.../.../.../"
CITY    <- "South Twenty Four Parganas"
OUT_DIR <- paste0(BASE, "experiments/")
dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# ── LOAD DATA ─────────────────────────────────────────────────
births <- read_csv(paste0(BASE,"data/Births.csv"),
                   show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="births") %>%
  mutate(year=as.integer(...1), births=births/26) %>%
  select(year, city, births)

inf_pop <- read_csv(paste0(BASE,"data/inferred_popn.csv"),
                    show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="pop") %>%
  rename(time_orig=`...1`) %>%
  mutate(year=as.integer(floor(time_orig))) %>%
  group_by(city, year) %>%
  summarise(pop=mean(pop, na.rm=TRUE), .groups="drop")

cases_raw <- read_csv(paste0(BASE,"data/cases_biweekly.csv"),
                      show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="cases") %>%
  rename(time=`...1`) %>%
  mutate(year=as.integer(floor(time))) %>%
  select(time, year, city, cases)

v1 <- read_csv(paste0(BASE,"data/V1.csv"),
               show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="v1") %>%
  mutate(year=as.integer(Year)) %>%
  select(year, city, v1)

v2 <- read_csv(paste0(BASE,"data/V2.csv"),
               show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="v2") %>%
  mutate(year=as.integer(Year)) %>%
  select(year, city, v2)

# ── MERGE ─────────────────────────────────────────────────────
all_data <- cases_raw %>%
  left_join(births,  by=c("year","city")) %>%
  left_join(inf_pop, by=c("year","city")) %>%
  left_join(v1,      by=c("year","city")) %>%
  left_join(v2,      by=c("year","city")) %>%
  arrange(city, time)

all_data$v1[is.na(all_data$v1)] <- 0
all_data$v2[is.na(all_data$v2)] <- 0

if (max(all_data$v1, na.rm=TRUE) > 1)
  all_data$v1 <- all_data$v1 / 100
if (max(all_data$v2, na.rm=TRUE) > 1)
  all_data$v2 <- all_data$v2 / 100

# ── FILTER TO CITY ────────────────────────────────────────────
city_data <- all_data %>%
  filter(city == CITY) %>%
  arrange(time)

cat(sprintf("City: %s | Rows: %d\n", CITY, nrow(city_data)))
cat(sprintf("V1 range: %.3f - %.3f\n",
            min(city_data$v1), max(city_data$v1)))
cat(sprintf("V2 range: %.3f - %.3f\n",
            min(city_data$v2), max(city_data$v2)))

# ── HELPER: BUILD TSIR DATA WITH OVERRIDES ────────────────────
# v1_override: -1 = use observed, 0 = zero, other = fixed value
# v2_override: -1 = use observed, 0 = zero, other = fixed value

build_tsir_data <- function(city_df, v1_override, v2_override) {

  df <- city_df %>% arrange(time)

  if (v1_override >= 0) df$v1 <- v1_override
  if (v2_override >= 0) df$v2 <- v2_override

  df <- df %>%
    group_by(city) %>%
    mutate(
      v1_bw = {
        yrs <- unique(year)
        v1s <- v1[match(yrs, year)]
        approx(x=yrs, y=v1s, xout=time, method="linear", rule=2)$y
      },
      v2_bw = {
        yrs <- unique(year)
        v2s <- v2[match(yrs, year)]
        approx(x=yrs, y=v2s, xout=time, method="linear", rule=2)$y
      },
      v1_lag = lag(v1_bw, n=5, default=first(v1_bw)),
      v2_lag = lag(v2_bw, n=5, default=first(v2_bw)),
      # Ferrari et al. (2012) Equation 3
      births_adjusted = births * (
        1
        - 0.85 * v1_lag * (1 - v2_lag)
        - 0.99 * v1_lag * v2_lag
      )
    ) %>%
    ungroup()

  df %>% select(time, cases, births=births_adjusted, pop)
}

# ── HELPER: RUN TSIR SCENARIO ─────────────────────────────────
run_tsir_scenario <- function(city_df, v1_override, v2_override,
                               scenario_name, tlag=52) {

  cat(sprintf("\nRunning: %s (V1=%s, V2=%s)\n",
              scenario_name,
              ifelse(v1_override<0,"obs",as.character(v1_override)),
              ifelse(v2_override<0,"obs",as.character(v2_override))))

  df_tsir     <- build_tsir_data(city_df, v1_override, v2_override)
  train_index <- which(floor(df_tsir$time) >= 2017)[1]

  cat(sprintf("  train_index=%d | tlag=%d\n", train_index, tlag))

  result <- tryCatch({
    get_preds_one_city(
      dat         = df_tsir,
      train_index = train_index,
      k           = 1,
      t_lag       = tlag
    )
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
    NULL
  })

  if (is.null(result)) return(NULL)

  result %>%
    filter(!is.na(time)) %>%
    mutate(
      scenario = scenario_name,
      mean     = pmax(cases, 0),
      lower    = pmax(cases, 0),
      upper    = pmax(cases, 0)
    ) %>%
    select(time, scenario, mean, lower, upper)
}

# ── RUN THREE SCENARIOS ───────────────────────────────────────
cat("\n=== Running three TSIR scenarios ===\n")

# Scenario 1: No vaccination (V1=0, V2=0)
s_no_vacc <- run_tsir_scenario(
  city_data, v1_override=0, v2_override=0,
  scenario_name="no_vacc")

# Scenario 2: MCV1 only (V1=observed, V2=0)
s_v1_only <- run_tsir_scenario(
  city_data, v1_override=-1, v2_override=0,
  scenario_name="v1_only")

# Scenario 3: Observed (V1=observed, V2=observed)
# Use actual observed cases — no TSIR needed
if (!is.null(s_no_vacc)) {
  time_range <- range(s_no_vacc$time, na.rm=TRUE)
} else {
  time_range <- c(2017, 2019)
}

s_observed <- city_data %>%
  filter(time >= time_range[1], time <= time_range[2]) %>%
  mutate(
    scenario = "observed",
    mean     = pmax(cases, 0),
    lower    = pmax(cases, 0),
    upper    = pmax(cases, 0)
  ) %>%
  select(time, scenario, mean, lower, upper)

cat(sprintf("\nObserved rows: %d\n", nrow(s_observed)))
cat(sprintf("No vacc rows:  %d\n",
            ifelse(is.null(s_no_vacc), 0, nrow(s_no_vacc))))
cat(sprintf("V1 only rows:  %d\n",
            ifelse(is.null(s_v1_only), 0, nrow(s_v1_only))))

# ── COMBINE ───────────────────────────────────────────────────
scenarios_list <- Filter(Negate(is.null),
                         list(s_no_vacc, s_v1_only, s_observed))

cf_results <- bind_rows(scenarios_list) %>%
  arrange(scenario, time)

# ── VALIDATION ────────────────────────────────────────────────
cat("\n=== Validation ===\n")
cf_results %>%
  group_by(scenario) %>%
  summarise(
    n           = n(),
    total_cases = sum(mean, na.rm=TRUE),
    .groups     = "drop"
  ) %>%
  print()

# ── SAVE ──────────────────────────────────────────────────────
out_path <- paste0(OUT_DIR, "counterfactual_results_V1V2.csv")
write_csv(cf_results, out_path)
cat(sprintf("\nSaved: %s\n", out_path))

# ── IMPACT SUMMARY ────────────────────────────────────────────
cat("\n=== Vaccination Impact Decomposition ===\n")

if (!is.null(s_no_vacc) && !is.null(s_v1_only)) {

  impact <- s_observed %>%
    rename(obs=mean) %>% select(time, obs) %>%
    inner_join(s_no_vacc %>%
                 rename(no_vacc=mean) %>% select(time, no_vacc),
               by="time") %>%
    inner_join(s_v1_only %>%
                 rename(v1_only=mean) %>% select(time, v1_only),
               by="time")

  total_obs     <- sum(impact$obs,     na.rm=TRUE)
  total_no_vacc <- sum(impact$no_vacc, na.rm=TRUE)
  total_v1_only <- sum(impact$v1_only, na.rm=TRUE)

  averted_total <- total_no_vacc - total_obs
  averted_mcv1  <- total_no_vacc - total_v1_only
  averted_mcv2  <- total_v1_only - total_obs

  cat(sprintf("  Period:                    2017-2019\n"))
  cat(sprintf("  Observed cases:            %d\n", round(total_obs)))
  cat(sprintf("  No vaccination (V1=V2=0):  %d\n", round(total_no_vacc)))
  cat(sprintf("  MCV1 only (V2=0):          %d\n", round(total_v1_only)))
  cat(sprintf("\n"))
  cat(sprintf("  Total cases averted:       %d (%.1f%%)\n",
              round(averted_total),
              averted_total/total_no_vacc*100))
  cat(sprintf("  Averted by MCV1:           %d (%.1f%%)\n",
              round(averted_mcv1),
              averted_mcv1/total_no_vacc*100))
  cat(sprintf("  Averted by MCV2:           %d (%.1f%%)\n",
              round(averted_mcv2),
              averted_mcv2/total_no_vacc*100))
  cat(sprintf("  MCV2 share of impact:      %.1f%%\n",
              averted_mcv2/averted_total*100))
}

cat("\nDone. Run counterfactuall_plot_V1V2.R for figures.\n")
