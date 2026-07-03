# ============================================================
# tsir_wb_run_V1V2.R
# TSIR k-step ahead forecasting for West Bengal measles
# Two-dose vaccination — Ferrari et al. (2012) Equation 3
#
# KEY DIFFERENCES FROM tsir_wb_run.R:
#   1. Loads V2.csv and applies Ferrari Eq.3 (V1+V2)
#   2. tlag = 52 (not 130)
#   3. train_index = 235 (first row of 2017)
#   4. k_values = c(1,4,12,20,34) (not 52)
#   5. Output folder: V1V2_raw/
#   6. Parallel: mclapply across cities (64 cores)
# ============================================================

source(".../.../.../code/R/tsir_run_functions.R")

library(tsiR)
#library(tidyverse)
library(kernlab)
library(parallel)

set.seed(2026)

BASE    <- ".../.../.../"
DATA    <- paste0(BASE, "data/")
RAW_OUT <- paste0(BASE, "output/data/tsir/wb/V1V2_raw/")
dir.create(RAW_OUT, showWarnings=FALSE, recursive=TRUE)

# ============================================================
# SECTION 1: LOAD DATA
# ============================================================

births <- read_csv(
  paste0(DATA, "Births.csv"),
  show_col_types=FALSE
) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="births") %>%
  mutate(year=as.integer(...1), births=births/26) %>%
  select(year, city, births)

inf_pop_urb <- read_csv(
  paste0(DATA, "inferred_popn.csv"),
  show_col_types=FALSE
) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="pop") %>%
  rename(time_orig=`...1`) %>%
  mutate(year=as.integer(floor(time_orig))) %>%
  group_by(city, year) %>%
  summarise(pop=mean(pop, na.rm=TRUE), .groups="drop")

cases <- read_csv(
  paste0(DATA, "cases_biweekly.csv"),
  show_col_types=FALSE
) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="cases") %>%
  rename(time=`...1`) %>%
  mutate(year=as.integer(floor(time))) %>%
  select(time, year, city, cases)

# MCV1
v1 <- read_csv(
  paste0(DATA, "V1.csv"),
  show_col_types=FALSE
) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="v1") %>%
  mutate(year=as.integer(Year)) %>%
  select(year, city, v1)

# MCV2
v2 <- read_csv(
  paste0(DATA, "V2.csv"),
  show_col_types=FALSE
) %>%
  pivot_longer(2:ncol(.), names_to="city", values_to="v2") %>%
  mutate(year=as.integer(Year)) %>%
  select(year, city, v2)

# ============================================================
# SECTION 2: MERGE
# ============================================================

all_cities <- cases %>%
  left_join(births,      by=c("year","city")) %>%
  left_join(inf_pop_urb, by=c("year","city")) %>%
  left_join(v1,          by=c("year","city")) %>%
  left_join(v2,          by=c("year","city")) %>%
  arrange(city, time)

expected <- n_distinct(cases$time) * n_distinct(cases$city)
cat(sprintf("Rows after merge: %d (expected %d) — %s\n",
            nrow(all_cities), expected,
            ifelse(nrow(all_cities)==expected, "OK", "ERROR")))

# ============================================================
# SECTION 3: VACCINATION ADJUSTMENT — FERRARI EQ. 3
#
# X_t = B_t * {1 - 0.85*V1_t*(1-V2_t) - 0.99*V1_t*V2_t}
# ============================================================

all_cities$v1[is.na(all_cities$v1)] <- 0
all_cities$v2[is.na(all_cities$v2)] <- 0

if (max(all_cities$v1, na.rm=TRUE) > 1)
  all_cities$v1 <- all_cities$v1 / 100
if (max(all_cities$v2, na.rm=TRUE) > 1)
  all_cities$v2 <- all_cities$v2 / 100

all_cities <- all_cities %>%
  arrange(city, time) %>%
  group_by(city) %>%
  mutate(
    # Interpolate annual -> biweekly
    v1_biweekly = {
      yrs <- unique(year); v1s <- v1[match(yrs, year)]
      approx(x=yrs, y=v1s, xout=time, method="linear", rule=2)$y
    },
    v2_biweekly = {
      yrs <- unique(year); v2s <- v2[match(yrs, year)]
      approx(x=yrs, y=v2s, xout=time, method="linear", rule=2)$y
    },
    # Lag 5 biweeks
    v1_lagged       = lag(v1_biweekly, n=5, default=first(v1_biweekly)),
    v2_lagged       = lag(v2_biweekly, n=5, default=first(v2_biweekly)),
    births_original = births,
    # Ferrari et al. (2012) Equation 3
    births          = births * (
      1
      - 0.85 * v1_lagged * (1 - v2_lagged)
      - 0.99 * v1_lagged * v2_lagged
    )
  ) %>%
  ungroup()

cat(sprintf("Mean births_original: %.2f\n",
            mean(all_cities$births_original, na.rm=TRUE)))
cat(sprintf("Mean births_adjusted: %.2f\n",
            mean(all_cities$births, na.rm=TRUE)))
cat(sprintf("Mean reduction:       %.1f%%\n",
            (1 - mean(all_cities$births) /
               mean(all_cities$births_original)) * 100))

# ============================================================
# SECTION 4: RUN TSIR FORECASTS — PARALLEL ACROSS CITIES
#
# tlag = 52, train_index = 235 (first row of 2017)
# k_values = c(1, 4, 12, 20, 34)
# ============================================================

cities_to_fit <- sort(unique(all_cities$city))
k_values      <- c(1, 4, 12, 20, 34)
tlag          <- 52
train_index   <- 235   # first biweek of 2017, precomputed

cat(sprintf("\nRunning TSIR: %d cities, k=%s, tlag=%d, train_index=%d\n",
            length(cities_to_fit),
            paste(k_values, collapse=","),
            tlag, train_index))

# Parallel function — one city, all k values
run_one_city <- function(city_name) {

  df_to_fit <- all_cities %>%
    filter(city == city_name) %>%
    select(time, cases, births, pop)

  for (k in k_values) {

    cat(sprintf("Running: %s | k=%d\n", city_name, k))

    result <- tryCatch({
      get_preds_one_city(
        dat         = df_to_fit,
        train_index = train_index,
        k           = k,
        t_lag       = tlag
      )
    }, error = function(e) {
      cat(sprintf("  ERROR %s k=%d: %s\n",
                  city_name, k, conditionMessage(e)))
      NULL
    })

    if (!is.null(result)) {
      saveRDS(
        result,
        paste0(RAW_OUT, "tsir_", city_name, "_k", k,
               "_test_fit.rds")
      )
      cat(sprintf("  Saved: tsir_%s_k%d_test_fit.rds\n",
                  city_name, k))
    }
  }
  return(city_name)
}

# Use 19 cores — one per city
n_cores <- min(length(cities_to_fit), detectCores() - 1)
cat(sprintf("Parallel: %d cores\n\n", n_cores))

results <- mclapply(
  cities_to_fit,
  run_one_city,
  mc.cores = n_cores
)

cat("\n=== ALL CITIES COMPLETE ===\n")
cat(sprintf("RDS files saved to: %s\n", RAW_OUT))
cat(sprintf("Files written: %d\n",
            length(list.files(RAW_OUT, pattern="\\.rds$"))))
cat("\nRun tsir_wb_process_V1V2.R to combine into CSV.\n")

