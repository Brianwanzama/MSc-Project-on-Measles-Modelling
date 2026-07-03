
library(dplyr)
library(tidyr)
library(readr)

BASE     <- "/.../.../.../"
DATA_DIR <- paste0(BASE, "data/")
OUT_DIR  <- paste0(BASE, "output/data/basic_nn_optimal/")

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# ── BIRTHS ────────────────────────────────────────────────────
births <- read_csv(paste0(DATA_DIR, "Births.csv"),
                   show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.),
               names_to  = "city",
               values_to = "births") %>%
  rename(year = `...1`) %>%
  mutate(
    year   = as.integer(year),
    births = births / 26      # annual -> biweekly
  ) %>%
  select(year, city, births)

cat(sprintf("Births: %d rows | years %d-%d | %d cities\n",
            nrow(births),
            min(births$year), max(births$year),
            length(unique(births$city))))

# ── POPULATION ────────────────────────────────────────────────
inf_pop <- read_csv(paste0(DATA_DIR, "inferred_popn.csv"),
                    show_col_types=FALSE) %>%
  rename(time = `...1`) %>%
  pivot_longer(2:ncol(.),
               names_to  = "city",
               values_to = "pop")

cat(sprintf("Population: %d rows | %d cities\n",
            nrow(inf_pop),
            length(unique(inf_pop$city))))

# ── CASES ─────────────────────────────────────────────────────
cases <- read_csv(paste0(DATA_DIR, "cases_biweekly.csv"),
                  show_col_types=FALSE) %>%
  rename(time = `...1`) %>%
  pivot_longer(2:ncol(.),
               names_to  = "city",
               values_to = "cases") %>%
  mutate(year = as.integer(floor(time)))

cat(sprintf("Cases: %d rows | time %.2f-%.2f | %d cities\n",
            nrow(cases),
            min(cases$time), max(cases$time),
            length(unique(cases$city))))

# ── COORDINATES ───────────────────────────────────────────────
coords_raw <- read_csv(paste0(DATA_DIR, "coordinates.csv"),
                       show_col_types=FALSE)

# Handle transposed format same as original paper
coords <- tryCatch({
  # Try standard format first: city column + lat/lon columns
  if ("city" %in% names(coords_raw)) {
    coords_raw %>%
      rename_with(tolower) %>%
      select(city, any_of(c("lat","lon","latitude",
                            "longitude","x","y")))
  } else {
    # Transposed format — same as original
    ct <- t(coords_raw)
    colnames(ct) <- ct[1, ]
    ct <- ct[2:nrow(ct), ] %>%
      as_tibble() %>%
      rename_with(tolower) %>%
      mutate(across(everything(), as.double)) %>%
      mutate(city = rownames(ct)[2:nrow(ct)])
    ct
  }
}, error = function(e) {
  message("Coordinates load error: ", e$message)
  NULL
})

cat(sprintf("Coordinates: %d cities\n",
            if (!is.null(coords)) nrow(coords) else 0))

# ── MCV1 COVERAGE ─────────────────────────────────────────────
v1 <- read_csv(paste0(DATA_DIR, "V1.csv"),
               show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.),
               names_to  = "city",
               values_to = "v1") %>%
  mutate(year = as.integer(Year)) %>%
  select(year, city, v1)

# Convert percentage to proportion if needed
if (max(v1$v1, na.rm=TRUE) > 1)
  v1 <- v1 %>% mutate(v1 = v1 / 100)

cat(sprintf("MCV1: %d rows | V1 range [%.3f, %.3f]\n",
            nrow(v1),
            min(v1$v1, na.rm=TRUE),
            max(v1$v1, na.rm=TRUE)))

# ── MCV2 COVERAGE ─────────────────────────────────────────────
v2 <- read_csv(paste0(DATA_DIR, "V2.csv"),
               show_col_types=FALSE) %>%
  pivot_longer(2:ncol(.),
               names_to  = "city",
               values_to = "v2") %>%
  mutate(year = as.integer(Year)) %>%
  select(year, city, v2)

if (max(v2$v2, na.rm=TRUE) > 1)
  v2 <- v2 %>% mutate(v2 = v2 / 100)

v2$v2[is.na(v2$v2)] <- 0  # pre-2010: MCV2 = 0

cat(sprintf("MCV2: %d rows | V2 range [%.3f, %.3f]\n",
            nrow(v2),
            min(v2$v2, na.rm=TRUE),
            max(v2$v2, na.rm=TRUE)))

# ── MERGE ALL ─────────────────────────────────────────────────
all_cities <- cases %>%
  left_join(births,  by=c("year","city")) %>%
  left_join(inf_pop, by=c("time","city")) %>%
  left_join(v1,      by=c("year","city")) %>%
  left_join(v2,      by=c("year","city")) %>%
  # NOTE: no + 1900 offset — WB years are actual calendar years
  select(-year) %>%
  group_by(city) %>%
  mutate(min_pop=min(pop, na.rm=TRUE)) %>%
  ungroup()

# Join coordinates if available
if (!is.null(coords)) {
  all_cities <- all_cities %>%
    left_join(coords, by="city")
}

cat(sprintf("\nMerged: %d rows | %d cities | time %.2f-%.2f\n",
            nrow(all_cities),
            length(unique(all_cities$city)),
            min(all_cities$time, na.rm=TRUE),
            max(all_cities$time, na.rm=TRUE)))

# ── DIAGNOSTICS ───────────────────────────────────────────────
cat("\nNA counts:\n")
all_cities %>%
  summarise(across(everything(),
                   ~sum(is.na(.)))) %>%
  pivot_longer(everything(),
               names_to="col",
               values_to="n_na") %>%
  filter(n_na > 0) %>%
  print()

cat(sprintf("\nCities (%d):\n",
            length(unique(all_cities$city))))
cat(paste(sort(unique(all_cities$city)),
          collapse="\n"), "\n")

# ── SAVE ──────────────────────────────────────────────────────
out_path <- paste0(OUT_DIR, "all_cases_wb.csv")
write.csv(all_cities, out_path, row.names=FALSE)
cat(sprintf("\nSaved: %s\n", out_path))
cat(sprintf("Rows: %d | Cols: %s\n",
            nrow(all_cities),
            paste(names(all_cities), collapse=", ")))
