# ============================================================
# wb_counterfactual_all_cities.R
# Run TSIR counterfactual for ALL 19 districts
# Quantify cases averted by MCV1+MCV2 programme 2017-2019
#
# For each district:
#   observed  — actual reported cases
#   no_vacc   — TSIR with V1=0, V2=0
#   v1_only   — TSIR with V1=observed, V2=0
#
# Output: counterfactual_all_cities.csv
#         counterfactual_all_cities_summary.csv
# ============================================================

source("/.../.../.../code/R/tsir_run_functions.R")

library(dplyr)
library(readr)
library(tidyr)
library(parallel)

BASE    <- "/.../.../.../"
DATA    <- paste0(BASE, "data/")
OUT_DIR <- paste0(BASE, "experiments/")
dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# ── LOAD DATA ─────────────────────────────────────────────────
births <- read_csv(paste0(DATA,"Births.csv"),
                   show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city",
               values_to="births") %>%
  mutate(year=as.integer(...1), births=births/26) %>%
  select(year, city, births)

inf_pop <- read_csv(paste0(DATA,"inferred_popn.csv"),
                    show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city",
               values_to="pop") %>%
  rename(time_orig=`...1`) %>%
  mutate(year=as.integer(floor(time_orig))) %>%
  group_by(city, year) %>%
  summarise(pop=mean(pop, na.rm=TRUE), .groups="drop")

cases_raw <- read_csv(paste0(DATA,"cases_biweekly.csv"),
                      show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city",
               values_to="cases") %>%
  rename(time=`...1`) %>%
  mutate(year=as.integer(floor(time))) %>%
  select(time, year, city, cases)

v1 <- read_csv(paste0(DATA,"V1.csv"),
               show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city",
               values_to="v1") %>%
  mutate(year=as.integer(Year)) %>%
  select(year, city, v1)

v2 <- read_csv(paste0(DATA,"V2.csv"),
               show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.), names_to="city",
               values_to="v2") %>%
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

cities <- sort(unique(all_data$city))
cat(sprintf("Districts: %d\n", length(cities)))

# ── HELPER: BUILD TSIR INPUT ───────────────────────────────────
build_tsir_data <- function(city_df, v1_override, v2_override) {
  df <- city_df %>% arrange(time)
  if (v1_override >= 0) df$v1 <- v1_override
  if (v2_override >= 0) df$v2 <- v2_override

  df %>%
    group_by(city) %>%
    mutate(
      v1_bw  = approx(x=unique(year),
                      y=v1[match(unique(year), year)],
                      xout=time, method="linear", rule=2)$y,
      v2_bw  = approx(x=unique(year),
                      y=v2[match(unique(year), year)],
                      xout=time, method="linear", rule=2)$y,
      v1_lag = lag(v1_bw, n=5, default=first(v1_bw)),
      v2_lag = lag(v2_bw, n=5, default=first(v2_bw)),
      births_adj = births * (
        1
        - 0.85 * v1_lag * (1 - v2_lag)
        - 0.99 * v1_lag * v2_lag
      )
    ) %>%
    ungroup() %>%
    select(time, cases, births=births_adj, pop)
}

# ── HELPER: RUN ONE SCENARIO ───────────────────────────────────
run_scenario <- function(city_df, v1_override, v2_override,
                         tlag=52) {
  df          <- build_tsir_data(city_df, v1_override, v2_override)
  train_index <- which(floor(df$time) >= 2017)[1]

  result <- tryCatch({
    get_preds_one_city(
      dat         = df,
      train_index = train_index,
      k           = 1,
      t_lag       = tlag
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", conditionMessage(e)))
    NULL
  })

  if (is.null(result)) return(NULL)
  result %>%
    filter(!is.na(time)) %>%
    mutate(cases = pmax(cases, 0)) %>%
    select(time, cases)
}

# ── RUN ALL CITIES ─────────────────────────────────────────────
cat("\nRunning counterfactual for all districts...\n")

run_one_city <- function(city_name) {
  cat(sprintf("  %s\n", city_name))
  city_df <- all_data %>% filter(city == city_name)

  # Observed cases in test period
  obs <- city_df %>%
    filter(year >= 2017) %>%
    mutate(cases = pmax(cases, 0)) %>%
    select(time, obs = cases)

  # No vaccination
  s_no_vacc <- run_scenario(city_df, 0, 0)
  # MCV1 only
  s_v1_only <- run_scenario(city_df, -1, 0)

  if (is.null(s_no_vacc) || is.null(s_v1_only)) {
    cat(sprintf("    SKIPPING %s — scenario failed\n", city_name))
    return(NULL)
  }

  # Align time ranges
  time_range <- range(s_no_vacc$time, na.rm=TRUE)
  obs <- obs %>%
    filter(time >= time_range[1],
           time <= time_range[2])

  result <- obs %>%
    inner_join(s_no_vacc %>% rename(no_vacc=cases),
               by="time") %>%
    inner_join(s_v1_only %>% rename(v1_only=cases),
               by="time") %>%
    mutate(city = city_name)

  return(result)
}

# Run in parallel — one core per city
n_cores <- min(length(cities), 19L)
cat(sprintf("Using %d cores\n\n", n_cores))

results_list <- mclapply(
  cities,
  run_one_city,
  mc.cores = n_cores
)
names(results_list) <- cities

# ── COMBINE ───────────────────────────────────────────────────
all_results <- bind_rows(Filter(Negate(is.null), results_list))

cat(sprintf("\nResults: %d rows | %d districts\n",
            nrow(all_results),
            n_distinct(all_results$city)))

# ── SUMMARY TABLE ─────────────────────────────────────────────
summary_df <- all_results %>%
  group_by(city) %>%
  summarise(
    obs_total     = sum(obs,     na.rm=TRUE),
    no_vacc_total = sum(no_vacc, na.rm=TRUE),
    v1_only_total = sum(v1_only, na.rm=TRUE),
    .groups       = "drop"
  ) %>%
  mutate(
    averted_total = no_vacc_total - obs_total,
    averted_mcv1  = no_vacc_total - v1_only_total,
    averted_mcv2  = v1_only_total - obs_total,
    pct_averted   = averted_total / no_vacc_total * 100,
    pct_mcv1      = averted_mcv1  / no_vacc_total * 100,
    pct_mcv2      = averted_mcv2  / no_vacc_total * 100
  ) %>%
  arrange(desc(averted_total))

cat("\n=== VACCINATION IMPACT SUMMARY ===\n")
print(summary_df %>%
        select(city, obs_total, no_vacc_total,
               averted_total, pct_averted,
               averted_mcv1, averted_mcv2) %>%
        mutate(across(where(is.numeric), \(x) round(x, 0))),
      n=20)

cat(sprintf("\n=== WEST BENGAL TOTAL ===\n"))
cat(sprintf("  Observed cases:          %d\n",
            round(sum(summary_df$obs_total))))
cat(sprintf("  No vaccination:          %d\n",
            round(sum(summary_df$no_vacc_total))))
cat(sprintf("  Total averted:           %d (%.1f%%)\n",
            round(sum(summary_df$averted_total)),
            sum(summary_df$averted_total) /
              sum(summary_df$no_vacc_total) * 100))
cat(sprintf("  Averted by MCV1:         %d (%.1f%%)\n",
            round(sum(summary_df$averted_mcv1)),
            sum(summary_df$averted_mcv1) /
              sum(summary_df$averted_total) * 100))
cat(sprintf("  Averted by MCV2:         %d (%.1f%%)\n",
            round(sum(summary_df$averted_mcv2)),
            sum(summary_df$averted_mcv2) /
              sum(summary_df$averted_total) * 100))

# ── SAVE ──────────────────────────────────────────────────────
write_csv(all_results,
          paste0(OUT_DIR, "counterfactual_all_cities.csv"))
write_csv(summary_df,
          paste0(OUT_DIR, "counterfactual_all_cities_summary.csv"))

cat(sprintf("\nSaved:\n  %scounterfactual_all_cities.csv\n",
            OUT_DIR))
cat(sprintf("  %scounterfactual_all_cities_summary.csv\n",
            OUT_DIR))
cat("\nRun wb_vaccination_impact_plot.R for figures.\n")
