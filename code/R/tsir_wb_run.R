# ============================================================
# tsir_wb_run.R
# TSIR k-step ahead forecasting — West Bengal measles
# Vaccination-adjusted following Ferrari et al. (2012)
# Parallel: 19 cores via mclapply (Linux, no install needed)
# Server:   /home/brain/Msc_project/
#
# DATA:     288 rows per district (2008 - early 2019)
# tlag:     52 biweeks (2 years)
# CUTOFF:   train_index = first row of 2017 (row 235)
#           predictions run from 2017 to end of data
# ============================================================

# ============================================================
# PATHS — Linux server, case-sensitive
# ============================================================

ROOT_DIR   <- "/.../.../.../"
DATA_DIR   <- file.path(ROOT_DIR, "data")
CODE_DIR   <- file.path(ROOT_DIR, "code/R")
OUTPUT_DIR <- file.path(ROOT_DIR, "output/data/tsir/wb/V1_raw")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

input_files <- c(
  Births     = file.path(DATA_DIR, "Births.csv"),
  Population = file.path(DATA_DIR, "inferred_popn.csv"),
  Cases      = file.path(DATA_DIR, "cases_biweekly.csv"),
  V1         = file.path(DATA_DIR, "V1.csv"),
  Functions  = file.path(CODE_DIR, "tsir_run_functions.R")
)

# Validate all files exist
cat("=== FILE VALIDATION ===\n")
all_exist <- TRUE
for (name in names(input_files)) {
  exists <- file.exists(input_files[[name]])
  cat(sprintf("  %-12s %s\n", paste0(name, ":"),
              ifelse(exists, "OK", "MISSING")))
  if (!exists) all_exist <- FALSE
}
if (!all_exist) stop("Missing files. Fix paths before proceeding.")

# ============================================================
# LOAD LIBRARIES AND FUNCTIONS
# ============================================================

source(input_files[["Functions"]])

library(tsiR)
library(dplyr)
library(readr)
library(tidyr)
library(kernlab)
library(parallel)   # base R — no install needed on Linux

set.seed(2026)

# ============================================================
# SECTION 1: LOAD DATA
# ============================================================

# Births — annual, unnamed first column -> '...1'
births <- read_csv(
  input_files[["Births"]],
  show_col_types = FALSE
) %>%
  pivot_longer(2:ncol(.), names_to = "city", values_to = "births") %>%
  mutate(
    year   = as.integer(...1),
    births = births / 26          # annual -> biweekly
  ) %>%
  select(year, city, births)

# Population — converted to annual mean to avoid time join mismatch
inf_pop_urb <- read_csv(
  input_files[["Population"]],
  show_col_types = FALSE
) %>%
  pivot_longer(2:ncol(.), names_to = "city", values_to = "pop") %>%
  rename(time_orig = `...1`) %>%
  mutate(year = as.integer(floor(time_orig))) %>%
  group_by(city, year) %>%
  summarise(pop = mean(pop, na.rm = TRUE), .groups = "drop")

# Cases — biweekly
cases <- read_csv(
  input_files[["Cases"]],
  show_col_types = FALSE
) %>%
  pivot_longer(2:ncol(.), names_to = "city", values_to = "cases") %>%
  rename(time = `...1`) %>%
  mutate(year = as.integer(floor(time))) %>%
  select(time, year, city, cases)

# MCV1 — annual, first column named 'Year' in V1.csv
v1 <- read_csv(
  input_files[["V1"]],
  show_col_types = FALSE
) %>%
  pivot_longer(2:ncol(.), names_to = "city", values_to = "v1") %>%
  mutate(year = as.integer(Year)) %>%
  select(year, city, v1)

# ============================================================
# SECTION 2: MERGE
# All joins on year+city — avoids decimal time precision mismatch
# ============================================================

all_cities <- cases %>%
  left_join(births,      by = c("year", "city")) %>%
  left_join(inf_pop_urb, by = c("year", "city")) %>%
  left_join(v1,          by = c("year", "city")) %>%
  arrange(city, time)

n_time   <- n_distinct(cases$time)
n_city   <- n_distinct(cases$city)
expected <- n_time * n_city

cat("\n=== MERGE VALIDATION ===\n")
cat(sprintf("Time points per district: %d\n",  n_time))
cat(sprintf("Districts:                %d\n",  n_city))
cat(sprintf("Rows after merge:         %d\n",  nrow(all_cities)))
cat(sprintf("Expected rows:            %d\n",  expected))
cat(sprintf("Status:                   %s\n",
            ifelse(nrow(all_cities) == expected, "OK", "ERROR")))
cat(sprintf("NA in births:             %d\n",  sum(is.na(all_cities$births))))
cat(sprintf("NA in pop:                %d\n",  sum(is.na(all_cities$pop))))
cat(sprintf("NA in v1:                 %d\n",  sum(is.na(all_cities$v1))))

if (nrow(all_cities) != expected) {
  stop("Row count mismatch. Check raw data before proceeding.")
}


# SECTION 3: VACCINATION ADJUSTMENT


all_cities$v1[is.na(all_cities$v1)] <- 0

if (max(all_cities$v1, na.rm = TRUE) > 1) {
  all_cities$v1 <- all_cities$v1 / 100
}

all_cities <- all_cities %>%
  arrange(city, time) %>%
  group_by(city) %>%
  mutate(
    # Interpolate annual MCV1 to biweekly — removes step jumps
    v1_biweekly = {
      yrs <- unique(year)
      v1s <- v1[match(yrs, year)]
      approx(x=yrs, y=v1s, xout=time, method="linear", rule=2)$y
    },
    # Lag 5 biweeks — vaccination at 9-12 months, not at birth
    v1_lagged       = lag(v1_biweekly, n=5, default=first(v1_biweekly)),
    births_original = births,
    births          = births * (1 - 0.85 * v1_lagged)
  ) %>%
  ungroup()

cat("\n=== VACCINATION ADJUSTMENT ===\n")
cat(sprintf("Mean births_original: %.1f biweekly\n",
            mean(all_cities$births_original, na.rm=TRUE)))
cat(sprintf("Mean births_adjusted: %.1f biweekly\n",
            mean(all_cities$births, na.rm=TRUE)))
cat(sprintf("Mean reduction:       %.1f%%\n",
            (1 - mean(all_cities$births, na.rm=TRUE) /
               mean(all_cities$births_original, na.rm=TRUE)) * 100))

# ============================================================
# SECTION 4: FORECAST PARAMETERS
# train_index: first row of 2017 — computed from data directly
#              so it works regardless of exact row count
# tlag:        52 biweeks (2-year rolling training window)
#              must match t_lag inside get_preds_one_city
# ============================================================

cities_to_fit <- sort(unique(all_cities$city))
k_values      <- c(1, 4, 12, 20, 34, 52)
tlag          <- 52L

# Compute train_index from data — first row where year >= 2017
# More robust than hardcoding row 235
one_city_data <- all_cities %>% filter(city == cities_to_fit[1])
train_index   <- which(floor(one_city_data$time) >= 2017)[1]

cat("\n=== FORECAST PARAMETERS ===\n")
cat(sprintf("Districts:            %d\n",   length(cities_to_fit)))
cat(sprintf("k values:             %s\n",   paste(k_values, collapse=", ")))
cat(sprintf("tlag:                 %d biweeks (%d years)\n",
            tlag, tlag/26L))
cat(sprintf("train_index:          %d (first row of 2017)\n", train_index))
cat(sprintf("Time at train_index:  %.4f\n",
            one_city_data$time[train_index]))
cat(sprintf("Test rows:            %d biweeks\n",
            nrow(one_city_data) - train_index + 1))
cat(sprintf("Total models to fit:  %d\n",
            length(cities_to_fit) * length(k_values)))

# ============================================================
# SECTION 5: WORKER FUNCTION
# Each parallel worker receives one district and runs all k.
# Self-contained: sets lib path, loads packages, sources functions.
# ============================================================

run_one_city <- function(city_name,
                         city_data_list,
                         k_values,
                         tlag,
                         train_index,
                         output_dir,
                         functions_path,
                         lib_paths) {
  
  # Set library path — required for forked workers on Linux
  .libPaths(lib_paths)
  
  # Load individual packages — tidyverse not available on server
  library(tsiR)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(kernlab)
  
  # Source TSIR functions after packages loaded
  source(functions_path)
  
  df_city <- city_data_list[[city_name]] %>%
    select(time, cases, births, pop)
  
  city_results <- list()
  
  for (k in k_values) {
    
    result <- tryCatch({
      
      all_dat_df <- get_preds_one_city(
        dat         = df_city,
        train_index = train_index,   # first row of 2017
        k           = k,
        t_lag       = tlag           # 52 biweeks
      )
      
      saveRDS(
        all_dat_df,
        file.path(
          output_dir,
          sprintf("tsir_%s_k%d_test_fit.rds", city_name, k)
        )
      )
      
      list(
        status = "success",
        city   = city_name,
        k      = k,
        nrows  = nrow(all_dat_df)
      )
      
    }, error = function(e) {
      list(
        status  = "error",
        city    = city_name,
        k       = k,
        message = conditionMessage(e)
      )
    })
    
    city_results[[as.character(k)]] <- result
  }
  
  return(city_results)
}

# ============================================================
# SECTION 6: SEQUENTIAL TEST — run one city before parallel
# Confirms everything works before committing 19 cores
# ============================================================

cat("\n=== SEQUENTIAL TEST (Bankura, k=1) ===\n")

test_result <- tryCatch({
  run_one_city(
    city_name      = cities_to_fit[1],
    city_data_list = split(
      all_cities %>% arrange(city, time),
      all_cities %>% arrange(city, time) %>% pull(city)
    ),
    k_values       = c(1),
    tlag           = tlag,
    train_index    = train_index,
    output_dir     = OUTPUT_DIR,
    functions_path = input_files[["Functions"]],
    lib_paths      = .libPaths()
  )
}, error = function(e) {
  cat("SEQUENTIAL TEST ERROR:", conditionMessage(e), "\n")
  NULL
})

if (!is.null(test_result) &&
    test_result[["1"]]$status == "success") {
  cat(sprintf("Sequential test PASSED: %d rows\n",
              test_result[["1"]]$nrows))
} else {
  cat("Sequential test FAILED — fix error before parallel run\n")
  print(test_result)
  stop("Aborting parallel run.")
}

# ============================================================
# SECTION 7: PARALLEL EXECUTION
# mclapply — fork-based, native Linux, no package install
# One core per district
# ============================================================

# Split data into named list — one element per city
city_data_list <- all_cities %>%
  arrange(city, time) %>%
  group_by(city) %>%
  group_split() %>%
  setNames(cities_to_fit)

n_cores <- 19L
cat(sprintf("\nCores available: %d\n", detectCores()))
cat(sprintf("Cores to use:    %d\n",  n_cores))
cat("Starting parallel TSIR forecasting...\n\n")

start_time <- proc.time()

results <- mclapply(
  X              = cities_to_fit,
  FUN            = run_one_city,
  city_data_list = city_data_list,
  k_values       = k_values,
  tlag           = tlag,
  train_index    = train_index,
  output_dir     = OUTPUT_DIR,
  functions_path = input_files[["Functions"]],
  lib_paths      = .libPaths(),
  mc.cores       = n_cores,
  mc.set.seed    = TRUE,
  mc.silent      = FALSE
)

names(results) <- cities_to_fit

elapsed <- proc.time() - start_time
cat(sprintf("\nCompleted in %.1f seconds (%.1f minutes)\n",
            elapsed["elapsed"], elapsed["elapsed"] / 60))

# ============================================================
# SECTION 8: RESULTS REPORT
# ============================================================

cat("\n=== RUN SUMMARY ===\n")

success_count <- 0
error_count   <- 0
error_log     <- list()

for (city_name in cities_to_fit) {
  
  city_result <- results[[city_name]]
  
  if (inherits(city_result, "try-error") ||
      inherits(city_result, "error")) {
    cat(sprintf("CITY ERROR: %s\n", city_name))
    error_count <- error_count + length(k_values)
    next
  }
  
  for (k_result in city_result) {
    if (k_result$status == "success") {
      success_count <- success_count + 1
      cat(sprintf("  OK  | %-35s | k=%2d | %d rows\n",
                  k_result$city, k_result$k, k_result$nrows))
    } else {
      error_count <- error_count + 1
      cat(sprintf("  ERR | %-35s | k=%2d | %s\n",
                  k_result$city, k_result$k, k_result$message))
      error_log[[length(error_log) + 1]] <- k_result
    }
  }
}

total_expected <- length(cities_to_fit) * length(k_values)
cat(sprintf("\nSuccessful: %d / %d\n", success_count, total_expected))
cat(sprintf("Errors:     %d\n",       error_count))

# Save error log if any failures
if (length(error_log) > 0) {
  error_df <- do.call(rbind, lapply(error_log, function(x) {
    data.frame(city    = x$city,
               k       = x$k,
               message = x$message,
               stringsAsFactors = FALSE)
  }))
  write.csv(error_df,
            file.path(OUTPUT_DIR, "../tsir_error_log.csv"),
            row.names = FALSE)
  cat("Error log saved: tsir_error_log.csv\n")
}

# Verify RDS files
rds_files <- list.files(OUTPUT_DIR,
                        pattern    = "_test_fit\\.rds$",
                        full.names = FALSE)
cat(sprintf("\nRDS files created: %d (expected %d)\n",
            length(rds_files), total_expected))

if (length(rds_files) == total_expected) {
  cat("All output files created successfully.\n")
} else {
  expected_files <- sprintf(
    "tsir_%s_k%d_test_fit.rds",
    rep(cities_to_fit, each  = length(k_values)),
    rep(k_values,      times = length(cities_to_fit))
  )
  missing <- setdiff(expected_files, rds_files)
  cat(sprintf("Missing %d files:\n", length(missing)))
  cat(paste(" -", missing), sep = "\n")
}