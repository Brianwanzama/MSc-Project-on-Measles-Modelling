# ============================================================
# tsir_wb_process_V1V2.R
# Combine TSIR k-step ahead forecast outputs — V1+V2 version
# ============================================================

library(tidyverse)

BASE      <- "/.../.../.../"
RAW_DIR   <- paste0(BASE, "output/data/tsir/wb/V1V2_raw/")
PROC_DIR  <- paste0(BASE, "output/data/tsir/wb/V1V2_processed/")
FINAL_OUT <- paste0(BASE, "output/data/basic_nn_optimal/",
                    "tsir_preds_processed_V1V2.csv")

dir.create(PROC_DIR, showWarnings=FALSE, recursive=TRUE)

k_select <- c(1, 4, 12, 20, 34)

# ============================================================
# SECTION 1: PROCESS INDIVIDUAL k FILES
# ============================================================

process_data <- function(loc, k) {
  dat      <- readRDS(loc)
  filename <- basename(loc)
  dat$city <- gsub("^tsir_(.+)_k\\d+_test_fit\\.rds$",
                   "\\1", filename)
  dat <- dat[dat$k == k, ]
  return(dat)
}

process_all_k <- function(k) {

  rds_files <- list.files(
    RAW_DIR,
    pattern    = paste0("_k", k, "_test_fit\\.rds$"),
    full.names = TRUE
  )

  if (length(rds_files) == 0) {
    warning(sprintf("No RDS files found for k=%d", k))
    return(NULL)
  }

  cat(sprintf("k=%d: %d RDS files found\n", k, length(rds_files)))

  city_data_list <- lapply(rds_files,
                           function(x) process_data(loc=x, k=k))

  tsir_dat <- Reduce(rbind, city_data_list) %>%
    rename(tsir = cases) %>%
    filter(k == k)

  out_path <- paste0(PROC_DIR, "tsir_", k, ".csv")
  write.csv(tsir_dat, out_path, row.names=FALSE)
  cat(sprintf("  Saved: tsir_%d.csv (%d rows)\n", k, nrow(tsir_dat)))

  return(tsir_dat)
}

all_k_data <- lapply(k_select, process_all_k)

# ============================================================
# SECTION 2: COMBINE ALL k INTO SINGLE FILE
# ============================================================

tsir_dat <- list.files(
  PROC_DIR,
  pattern    = "tsir_\\d+\\.csv$",
  full.names = TRUE
) %>%
  lapply(function(x) {
    k_val <- as.integer(
      gsub(".*tsir_(\\d+)\\.csv$", "\\1", basename(x)))
    read_csv(x, show_col_types=FALSE) %>%
      mutate(k = k_val)
  }) %>%
  {Reduce(rbind, .)}

cat(sprintf("\nCombined file: %d rows, %d columns\n",
            nrow(tsir_dat), ncol(tsir_dat)))
cat(sprintf("k values:   %s\n",
            paste(sort(unique(tsir_dat$k)), collapse=", ")))
cat(sprintf("Districts:  %d\n", n_distinct(tsir_dat$city)))

# Remove NA time rows (forecasts beyond end of dataset)
n_before <- nrow(tsir_dat)
tsir_dat <- tsir_dat %>% filter(!is.na(time))
cat(sprintf("Rows after NA removal: %d (removed %d)\n",
            nrow(tsir_dat), n_before - nrow(tsir_dat)))

write_csv(tsir_dat, FINAL_OUT)
cat(sprintf("\nFinal output saved: %s\n", FINAL_OUT))

# Row count check
cat("\nRows per k:\n")
tsir_dat %>%
  group_by(k) %>%
  summarise(n=n(), districts=n_distinct(city), .groups="drop") %>%
  print()

