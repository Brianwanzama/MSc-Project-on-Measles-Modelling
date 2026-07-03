# ============================================================
# tsir_susceptibles_gen_V1V2.R
# Susceptible Reconstruction for West Bengal Measles 2008-2019
# Two-dose vaccination following Ferrari et al. (2012) Eq. 3
#
# FORMULA (Ferrari et al. 2012, Equation 3):
#   X_t = B_t * {1 - 0.85*V1_t*(1-V2_t) - 0.99*V1_t*V2_t}
#
# Where:
#   V1_t = MCV1 coverage (proportion)
#   V2_t = MCV2 coverage (proportion)
#   0.85 = single-dose efficacy
#   0.99 = two-dose efficacy (Uzicanin & Zimmerman 2011)
#
# V2 DATA:
#   2008-2010: V2 = 0 (MCV2 not introduced)
#   2011-2016: National WUENIC estimates (uniform across districts)
#   2017-2019: District-level West Bengal data
#
# OUTPUT: tsir_susceptibles_V1V2.csv
# ============================================================

library(tsiR)
library(dplyr)
library(readr)
library(kernlab)

source("/.../.../.../code/R/tsir_run_functions.R")

BASE    <- "/.../.../.../"
DATA    <- paste0(BASE, "data/")
OUT_DIR <- paste0(BASE, "output/data/tsir/")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# SECTION 1: LOAD RAW DATA
# ============================================================

births_raw <- read_csv(
  paste0(DATA, "Births.csv"),
  show_col_types = FALSE
) %>%
  pivot_longer(cols = 2:ncol(.), names_to = "city",
               values_to = "births_annual") %>%
  mutate(year = as.integer(...1)) %>%
  select(year, city, births_annual)

inf_pop_urb <- read_csv(
  paste0(DATA, "inferred_popn.csv"),
  show_col_types = FALSE
) %>%
  pivot_longer(cols = 2:ncol(.), names_to = "city",
               values_to = "pop") %>%
  rename(time = `...1`) %>%
  mutate(year = as.integer(floor(time))) %>%
  group_by(city, year) %>%
  summarise(pop = mean(pop, na.rm = TRUE), .groups = "drop")

cases <- read_csv(
  paste0(DATA, "cases_biweekly.csv"),
  show_col_types = FALSE
) %>%
  pivot_longer(cols = 2:ncol(.), names_to = "city",
               values_to = "cases") %>%
  rename(time = `...1`) %>%
  mutate(year = as.integer(floor(time))) %>%
  select(time, year, city, cases)

# MCV1 — district level, 2008-2019
v1_raw <- read_csv(
  paste0(DATA, "V1.csv"),
  show_col_types = FALSE
) %>%
  pivot_longer(cols = 2:ncol(.), names_to = "city",
               values_to = "v1_annual") %>%
  mutate(year = as.integer(Year)) %>%
  select(year, city, v1_annual)

# MCV2 — district level 2017-2019, national 2011-2016, 0 before 2011
v2_raw <- read_csv(
  paste0(DATA, "V2.csv"),
  show_col_types = FALSE
) %>%
  pivot_longer(cols = 2:ncol(.), names_to = "city",
               values_to = "v2_annual") %>%
  mutate(year = as.integer(Year)) %>%
  select(year, city, v2_annual)

# ============================================================
# SECTION 2: MERGE ALL DATA
# ============================================================

all_cities <- cases %>%
  left_join(births_raw,  by = c("year", "city")) %>%
  left_join(inf_pop_urb, by = c("year", "city")) %>%
  left_join(v1_raw,      by = c("year", "city")) %>%
  left_join(v2_raw,      by = c("year", "city")) %>%
  arrange(city, time)

n_districts  <- n_distinct(all_cities$city)
n_timepoints <- n_distinct(all_cities$time)
expected_rows <- n_districts * n_timepoints

cat("=== MERGE VALIDATION ===\n")
cat(sprintf("Districts:         %d\n", n_districts))
cat(sprintf("Time points:       %d\n", n_timepoints))
cat(sprintf("Rows after merge:  %d\n", nrow(all_cities)))
cat(sprintf("Expected rows:     %d\n", expected_rows))
cat(sprintf("Row count correct: %s\n",
            ifelse(nrow(all_cities) == expected_rows,
                   "YES", "NO — check data")))
cat(sprintf("NA in births:      %d\n", sum(is.na(all_cities$births_annual))))
cat(sprintf("NA in pop:         %d\n", sum(is.na(all_cities$pop))))
cat(sprintf("NA in v1:          %d\n", sum(is.na(all_cities$v1_annual))))
cat(sprintf("NA in v2:          %d\n", sum(is.na(all_cities$v2_annual))))

if (nrow(all_cities) != expected_rows) {
  stop("Row count mismatch after merge. Check raw data files.")
}

# ============================================================
# SECTION 3: DATA CLEANING
# ============================================================

all_cities <- all_cities %>%
  mutate(
    v1_annual = replace_na(v1_annual, 0),
    v2_annual = replace_na(v2_annual, 0)
  )

# Safety: ensure proportions (0-1)
if (max(all_cities$v1_annual, na.rm = TRUE) > 1) {
  all_cities <- all_cities %>% mutate(v1_annual = v1_annual / 100)
  cat("Note: V1 converted from percentage to proportion\n")
}
if (max(all_cities$v2_annual, na.rm = TRUE) > 1) {
  all_cities <- all_cities %>% mutate(v2_annual = v2_annual / 100)
  cat("Note: V2 converted from percentage to proportion\n")
}

cat("\n=== V2 SUMMARY ===\n")
cat(sprintf("V2 range: %.3f - %.3f\n",
            min(all_cities$v2_annual), max(all_cities$v2_annual)))
cat("V2 by year (mean across districts):\n")
all_cities %>%
  group_by(year) %>%
  summarise(mean_v2 = mean(v2_annual), .groups = "drop") %>%
  print()

# ============================================================
# SECTION 4: BIWEEKLY BIRTH CONVERSION
# ============================================================

all_cities <- all_cities %>%
  mutate(births_original = births_annual / 26)

# ============================================================
# SECTION 5: INTERPOLATE V1 AND V2 FROM ANNUAL TO BIWEEKLY
# ============================================================

all_cities <- all_cities %>%
  arrange(city, time) %>%
  group_by(city) %>%
  mutate(
    # V1: interpolate annual -> biweekly
    v1_biweekly = {
      yrs <- unique(year)
      v1s <- v1_annual[match(yrs, year)]
      approx(x=yrs, y=v1s, xout=time,
             method="linear", rule=2)$y
    },
    # V2: interpolate annual -> biweekly
    v2_biweekly = {
      yrs <- unique(year)
      v2s <- v2_annual[match(yrs, year)]
      approx(x=yrs, y=v2s, xout=time,
             method="linear", rule=2)$y
    }
  ) %>%
  ungroup()

# ============================================================
# SECTION 6: LAG V1 AND V2 BY 5 BIWEEKS
# MCV given at 9-12 months (~5 biweeks after birth cohort)
# ============================================================

all_cities <- all_cities %>%
  arrange(city, time) %>%
  group_by(city) %>%
  mutate(
    v1_lagged = lag(v1_biweekly, n=5,
                    default=first(v1_biweekly)),
    v2_lagged = lag(v2_biweekly, n=5,
                    default=first(v2_biweekly))
  ) %>%
  ungroup()

# ============================================================
# SECTION 7: FERRARI ET AL. (2012) EQUATION 3 — TWO-DOSE
#
# X_t = B_t * {1 - 0.85*V1_t*(1-V2_t) - 0.99*V1_t*V2_t}
#
# Decomposition:
#   V1 only (not V2): proportion = V1*(1-V2), efficacy = 0.85
#   Both V1 and V2:   proportion = V1*V2,     efficacy = 0.99
#   Unvaccinated:     proportion = (1-V1),    efficacy = 0
#
# Using lagged V1 and V2 (5 biweek lag)
# ============================================================

all_cities <- all_cities %>%
  mutate(
    births_adjusted = births_original * (
      1
      - 0.85 * v1_lagged * (1 - v2_lagged)   # V1 only
      - 0.99 * v1_lagged * v2_lagged           # V1 + V2
    )
  )

# Validate — births_adjusted must be non-negative
stopifnot(all(all_cities$births_adjusted >= 0, na.rm = TRUE))
stopifnot(all(
  all_cities$births_adjusted <= all_cities$births_original + 1e-9,
  na.rm = TRUE
))

cat("\n=== VACCINATION ADJUSTMENT SUMMARY (Ferrari Eq. 3) ===\n")
cat(sprintf("Mean births_original (biweekly): %.2f\n",
            mean(all_cities$births_original, na.rm=TRUE)))
cat(sprintf("Mean births_adjusted (biweekly): %.2f\n",
            mean(all_cities$births_adjusted, na.rm=TRUE)))
cat(sprintf("Mean reduction:                  %.1f%%\n",
            (1 - mean(all_cities$births_adjusted) /
               mean(all_cities$births_original)) * 100))
cat(sprintf("V1 range (biweekly): %.3f - %.3f\n",
            min(all_cities$v1_biweekly), max(all_cities$v1_biweekly)))
cat(sprintf("V2 range (biweekly): %.3f - %.3f\n",
            min(all_cities$v2_biweekly), max(all_cities$v2_biweekly)))

# Show formula impact by year
cat("\nNon-immune fraction by year (mean across districts):\n")
all_cities %>%
  group_by(year) %>%
  summarise(
    v1      = mean(v1_annual),
    v2      = mean(v2_annual),
    old_f   = mean(1 - 0.85 * v1_annual),
    new_f   = mean(1 - 0.85*v1_lagged*(1-v2_lagged)
                     - 0.99*v1_lagged*v2_lagged),
    .groups = "drop"
  ) %>%
  mutate(diff_pp = (old_f - new_f) * 100) %>%
  print()

# ============================================================
# SECTION 8: TSIR FITTING — ONE MODEL PER DISTRICT
# ============================================================

tsir_s_dfs <- list()
cities     <- sort(unique(all_cities$city))

fit_log <- data.frame(
  city      = character(),
  regtype   = character(),
  attempt   = integer(),
  alpha     = numeric(),
  mean_beta = numeric(),
  sbar      = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_along(cities)) {

  fit_data <- all_cities %>%
    filter(city == cities[i]) %>%
    select(time, cases, births = births_adjusted, pop)

  zero_prop <- sum(fit_data$cases == 0) / nrow(fit_data)
  epidemics  <- ifelse(zero_prop > 0.3, "break", "cont")

  cat(sprintf(
    "\n[%d/%d] %s | zero_prop=%.1f%% | epidemics='%s'\n",
    i, length(cities), cities[i], zero_prop * 100, epidemics
  ))

  result <- tryCatch({
    fit <- runtsir_updated(
      data=fit_data, IP=2, xreg='cumcases',
      regtype='gaussian', alpha=NULL, sbar=NULL,
      family='gaussian', link='identity',
      method='negbin', nsim=100,
      epidemics=epidemics, sigmamax=3
    )
    list(fit=fit, attempt=1L, regtype="gaussian")

  }, error = function(e) {
    cat(sprintf("  gaussian failed: %s\n", conditionMessage(e)))
    cat("  Retrying with regtype='lm'...\n")
    tryCatch({
      fit <- runtsir_updated(
        data=fit_data, IP=2, xreg='cumcases',
        regtype='lm', alpha=0.97, sbar=NULL,
        family='gaussian', link='identity',
        method='negbin', nsim=100, epidemics=epidemics
      )
      list(fit=fit, attempt=2L, regtype="lm")
    }, error = function(e2) {
      cat(sprintf("  lm failed: %s\n", conditionMessage(e2)))
      cat("  Retrying with lm + fixed sbar=0.05...\n")
      fit <- runtsir_updated(
        data=fit_data, IP=2, xreg='cumcases',
        regtype='lm', alpha=0.97, sbar=0.05,
        family='gaussian', link='identity',
        method='negbin', nsim=100, epidemics=epidemics
      )
      list(fit=fit, attempt=3L, regtype="lm_fixed")
    })
  })

  tsir_s_dfs[[i]] <- result$fit

  cat(sprintf(
    "  Done [attempt=%d | regtype=%s | alpha=%.3f | mean_beta=%.2e | sbar=%.0f]\n",
    result$attempt, result$regtype,
    result$fit$alpha, mean(result$fit$beta), result$fit$sbar
  ))

  fit_log <- bind_rows(fit_log, data.frame(
    city=cities[i], regtype=result$regtype,
    attempt=result$attempt, alpha=result$fit$alpha,
    mean_beta=mean(result$fit$beta), sbar=result$fit$sbar,
    stringsAsFactors=FALSE
  ))
}

# ============================================================
# SECTION 9: EXTRACT SUSCEPTIBLES
# ============================================================

tsir_dfs <- list()

for (i in seq_along(cities)) {
  city_times <- all_cities %>%
    filter(city == cities[i]) %>%
    arrange(time) %>%
    pull(time)

  n_susc <- length(tsir_s_dfs[[i]]$simS$mean)
  n_time <- length(city_times)

  if (n_susc != n_time) {
    warning(sprintf(
      "Row mismatch for %s: simS=%d rows, data=%d rows",
      cities[i], n_susc, n_time))
  }

  tsir_dfs[[i]] <- data.frame(
    time = city_times,
    city = cities[i],
    susc = tsir_s_dfs[[i]]$simS$mean,
    stringsAsFactors = FALSE
  )
}

tsir_susc <- Reduce(rbind, tsir_dfs) %>%
  left_join(
    all_cities %>%
      select(time, city, cases,
             births_original, births_adjusted, pop,
             v1_annual, v1_biweekly, v1_lagged,
             v2_annual, v2_biweekly, v2_lagged),
    by = c("time", "city")
  ) %>%
  arrange(city, time)

# ============================================================
# SECTION 10: VALIDATION AND SAVE
# ============================================================

cat("\n=== SUSCEPTIBLE RECONSTRUCTION VALIDATION ===\n")
cat(sprintf("Rows:      %d\n", nrow(tsir_susc)))
cat(sprintf("Districts: %d\n", n_distinct(tsir_susc$city)))
cat(sprintf("NA susc:   %d\n", sum(is.na(tsir_susc$susc))))
cat(sprintf("susc range: %.0f - %.0f\n",
            min(tsir_susc$susc, na.rm=TRUE),
            max(tsir_susc$susc, na.rm=TRUE)))

# Save susceptibles
write.csv(
  tsir_susc,
  paste0(OUT_DIR, "tsir_susceptibles_V1V2.csv"),
  row.names = FALSE
)
cat(sprintf("\nSaved: %s\n",
            paste0(OUT_DIR, "tsir_susceptibles_V1V2.csv")))

# Save fit log
write.csv(
  fit_log,
  paste0(OUT_DIR, "tsir_fit_log_V1V2.csv"),
  row.names = FALSE
)
cat(sprintf("Saved: %s\n",
            paste0(OUT_DIR, "tsir_fit_log_V1V2.csv")))

cat("\nDone. Run tsir_wb_run_V1V2.R next.\n")
