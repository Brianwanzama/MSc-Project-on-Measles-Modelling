# ============================================================
# tsir_wb_process.R
# Combine TSIR k-step ahead RDS outputs into single CSV
# Server: /home/brain/Msc_project/
# ============================================================

library(dplyr)
library(readr)

# ============================================================
# PATHS — Linux server, case-sensitive
# ============================================================

ROOT_DIR      <- "/.../.../.../"
RAW_DIR       <- file.path(ROOT_DIR, "output/data/tsir/wb/V1_raw")
PROCESSED_DIR <- file.path(ROOT_DIR, "output/data/tsir/wb/V1_processed")
FINAL_DIR     <- file.path(ROOT_DIR, "output/data/basic_nn_optimal")

dir.create(PROCESSED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FINAL_DIR,     recursive = TRUE, showWarnings = FALSE)

cat("=== PATH VALIDATION ===\n")
cat(sprintf("Raw RDS dir:      %s — %s\n", RAW_DIR,
            ifelse(dir.exists(RAW_DIR), "EXISTS", "MISSING")))
cat(sprintf("Processed dir:    %s — %s\n", PROCESSED_DIR,
            ifelse(dir.exists(PROCESSED_DIR), "EXISTS", "CREATED")))
cat(sprintf("Final output dir: %s — %s\n", FINAL_DIR,
            ifelse(dir.exists(FINAL_DIR), "EXISTS", "CREATED")))

# Confirm total RDS files before starting
all_rds <- list.files(RAW_DIR, pattern = "_test_fit\\.rds$")
cat(sprintf("\nRDS files found: %d (expected 114)\n", length(all_rds)))

if (length(all_rds) == 0) {
  stop("No RDS files found in ", RAW_DIR, ". Check path and run tsir_wb_run.R first.")
}

# ============================================================
# SECTION 1: HELPER FUNCTION
# Reads one RDS file and extracts city name from filename
# ============================================================

process_data <- function(loc, k) {
  
  dat      <- readRDS(loc)
  filename <- basename(loc)
  
  # Extract city name robustly from filename
  # Format: tsir_CITYNAME_k52_test_fit.rds
  dat$city <- gsub(
    sprintf("^tsir_(.+)_k%d_test_fit\\.rds$", k),
    "\\1",
    filename
  )
  
  # Filter to this k only — each file should already be one k
  dat <- dat[dat$k == k, ]
  
  return(dat)
}

# ============================================================
# SECTION 2: PROCESS EACH k VALUE SEPARATELY
# Reads all RDS files for a given k, combines into one CSV
# ============================================================

process_all_k <- function(k) {
  
  # Find RDS files for this specific k value
  rds_files <- list.files(
    RAW_DIR,
    pattern    = paste0("_k", k, "_test_fit\\.rds$"),
    full.names = TRUE
  )
  
  if (length(rds_files) == 0) {
    warning(sprintf("No RDS files found for k=%d", k))
    return(NULL)
  }
  
  cat(sprintf("\nProcessing k=%d: %d files found\n", k, length(rds_files)))
  
  # Read and combine all cities for this k
  city_data_list <- lapply(rds_files, function(x) {
    process_data(loc = x, k = k)
  })
  
  tsir_dat <- Reduce(rbind, city_data_list)
  
  # Rename cases -> tsir for clarity downstream
  if ("cases" %in% names(tsir_dat)) {
    tsir_dat <- tsir_dat %>% rename(tsir = cases)
  }
  
  # Save processed CSV for this k
  out_path <- file.path(PROCESSED_DIR, sprintf("tsir_%d.csv", k))
  write.csv(tsir_dat, out_path, row.names = FALSE)
  
  cat(sprintf("  Saved: tsir_%d.csv | %d rows | %d districts\n",
              k,
              nrow(tsir_dat),
              length(unique(tsir_dat$city))))
  
  return(tsir_dat)
}

# Run for all k values
k_select    <- c(1, 4, 12, 20, 34, 52)
all_k_data  <- lapply(k_select, process_all_k)
names(all_k_data) <- as.character(k_select)

# ============================================================
# SECTION 3: COMBINE ALL k INTO SINGLE FILE
# ============================================================

cat("\n=== COMBINING ALL k VALUES ===\n")

# Remove any NULL results
all_k_data <- Filter(Negate(is.null), all_k_data)

if (length(all_k_data) == 0) {
  stop("No data to combine. Check Section 2 for errors.")
}

tsir_final <- Reduce(rbind, all_k_data)

# ============================================================
# SECTION 4: VALIDATION
# ============================================================

cat("\n=== VALIDATION ===\n")
cat(sprintf("Total rows:       %d\n",  nrow(tsir_final)))
cat(sprintf("Columns:          %s\n",  paste(names(tsir_final), collapse=", ")))
cat(sprintf("k values present: %s\n",  paste(sort(unique(tsir_final$k)),
                                             collapse=", ")))
cat(sprintf("Districts:        %d\n",  length(unique(tsir_final$city))))
cat(sprintf("NA in tsir:       %d\n",  sum(is.na(tsir_final$tsir))))
cat(sprintf("NA in time:       %d\n",  sum(is.na(tsir_final$time))))

cat("\nRows per k:\n")
tsir_final %>%
  group_by(k) %>%
  summarise(rows = n(), districts = n_distinct(city), .groups = "drop") %>%
  arrange(k) %>%
  as.data.frame() %>%
  print()

cat("\nRows per district (k=1 only):\n")
tsir_final %>%
  filter(k == 1) %>%
  group_by(city) %>%
  summarise(rows = n(), .groups = "drop") %>%
  arrange(city) %>%
  as.data.frame() %>%
  print()


# Check how many rows have NA time
cat(sprintf("\nRows with NA time: %d\n", sum(is.na(tsir_final$time))))

# Remove rows where time is NA
# These are forecast targets beyond the end of the dataset
tsir_final <- tsir_final %>%
  filter(!is.na(time))

cat(sprintf("Rows after removing NA time: %d\n", nrow(tsir_final)))

# Re-validate
cat(sprintf("k values still present: %s\n",
            paste(sort(unique(tsir_final$k)), collapse=", ")))
cat(sprintf("Districts still present: %d\n",
            length(unique(tsir_final$city))))

final_path <- file.path(FINAL_DIR, "tsir_preds_processed.csv")
write_csv(tsir_final, final_path)


