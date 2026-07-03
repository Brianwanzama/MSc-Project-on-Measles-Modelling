

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

set.seed(42)

# ── PATHS ─────────────────────────────────────────────────────
BASE     <- "/.../.../.../"
read_dir <- paste0(BASE,
                   "output/data/basic_nn_optimal/explain/")
save_dir <- paste0(BASE, "experiments/figures/")
data_dir <- paste0(BASE, "data/")
nn_dir   <- paste0(BASE,
                   "output/data/basic_nn_optimal/")

dir.create(save_dir, showWarnings=FALSE,
           recursive=TRUE)

# ── THEME ─────────────────────────────────────────────────────
theme_pub <- function() {
  theme_classic() +
    theme(
      panel.border     = element_rect(fill=NA,
                                      colour="black",
                                      linewidth=1),
      strip.background = element_rect(fill="white",
                                      colour="black",
                                      linewidth=0.8),
      strip.text       = element_text(face="bold",
                                      size=10),
      axis.text        = element_text(colour="black",
                                      size=9),
      axis.title       = element_text(size=10),
      plot.title       = element_text(face="bold",
                                      size=11, hjust=0),
      plot.subtitle    = element_text(colour="grey40",
                                      size=9, hjust=0,
                                      margin=margin(b=6)),
      plot.caption     = element_text(colour="grey50",
                                      size=7.5, hjust=0),
      plot.margin      = margin(10,14,8,10)
    )
}
theme_set(theme_pub())

# ── ANCILLARY DATA ────────────────────────────────────────────
all_cities <- read_csv(
  paste0(nn_dir, "all_cases_wb.csv"),
  show_col_types=FALSE)

nbc <- read_csv(
  paste0(data_dir, "nearest_big_city.csv"),
  show_col_types=FALSE) |>
  select(city, nearest_big_city) |>
  distinct()

# ── READ AND CLEAN SHAP CSV ───────────────────────────────────
# Aligned with original read_in_cap() but for WB column names
read_in_cap <- function(x) {
  
  # Drop row-index column if present
  if (names(x)[1] %in% c("...1","X","X1"))
    x <- x[, 2:ncol(x)]
  
  # Strip lag numbers and _lag_ from column names
  # to get group-level columns
  x <- x |>
    rename_with(~gsub("_lag_[0-9]+", "", .)) |>
    rename_with(~gsub("[[:digit:]]+", "", .))
  
  # Handle cases_nc_ -> cases_nc rename
  # (may not exist in all WB k outputs)
  if ("cases_nc_" %in% names(x))
    x <- x |> rename(cases_nc = cases_nc_)
  
  # Aggregate duplicate columns by summing
  # (result of stripping lag numbers)
  x <- x |>
    pivot_longer(
      -any_of(c("time","city")),
      names_to="feat", values_to="val") |>
    group_by(time, city, feat) |>
    summarise(val=sum(val, na.rm=TRUE),
              .groups="drop") |>
    pivot_wider(names_from="feat",
                values_from="val")
  
  # Join nearest big city and population
  x <- x |>
    left_join(nbc, by="city") |>
    # NOTE: no + 1900 — WB years are actual
    mutate(time_temp=round(time, 2)) |>
    left_join(
      all_cities |>
        select(city, time, min_pop) |>
        mutate(time=round(time, 2)) |>
        rename(city_pop=min_pop),
      by=c("city","time_temp"="time")
    ) |>
    select(-time_temp)
  
  # Rename to readable labels
  rename_map <- c(
    "cases"    = "Incidence Lags",
    "pop"      = "Population",
    "births"   = "Births",
    "cases_nc" = "Nearby City Incidence Lags",
    "susc"     = "Susceptible Lags",
    "v"        = "MCV1 Lags",
    "dist"     = "Distances"
  )
  for (old in names(rename_map)) {
    if (old %in% names(x))
      x <- x |>
        rename(!!rename_map[old] := all_of(old))
  }
  
  return(x)
}

# ── PROCESS FUNCTION ──────────────────────────────────────────
# Compute relative importance within each observation
process <- function(x) {
  id_cols <- c("time","city",
               "nearest_big_city","city_pop")
  feat_cols <- setdiff(names(x), id_cols)
  
  rel_within_obs <- x |>
    pivot_longer(
      cols      = all_of(feat_cols),
      names_to  = "name",
      values_to = "value"
    ) |>
    group_by(time, city) |>
    mutate(rel_value=abs(value) /
             sum(abs(value), na.rm=TRUE)) |>
    ungroup() |>
    group_by(city, nearest_big_city,
             city_pop, name) |>
    summarise(rel_value=mean(rel_value,
                             na.rm=TRUE),
              .groups="drop")
  
  return(list(rel_within_obs=rel_within_obs))
}

# ── MAKE BOXPLOT ──────────────────────────────────────────────
make_shap_boxplot <- function(k_val) {
  
  csv_path <- paste0(read_dir, k_val,
                     "_svs_explain.csv")
  if (!file.exists(csv_path)) {
    message(sprintf("Skipping k=%s: file not found",
                    k_val))
    return(NULL)
  }
  
  cat(sprintf("\nProcessing k=%s...\n", k_val))
  
  svs_exp <- read_csv(csv_path,
                      show_col_types=FALSE) |>
    read_in_cap()
  
  svs_processed    <- process(svs_exp)
  svs_rel_within_obs <- svs_processed$rel_within_obs
  
  # Filter to incidence-related features
  # (any column containing "Incidence")
  big_city_sep_high <- svs_rel_within_obs |>
    filter(grepl("Incidence", name,
                 ignore.case=TRUE))
  
  if (nrow(big_city_sep_high)==0) {
    message(sprintf(
      "  k=%s: no incidence features found",
      k_val))
    return(NULL)
  }
  
  # Top 4 cities by population
  big_cities <- big_city_sep_high |>
    group_by(city) |>
    summarise(city_pop=max(city_pop,
                           na.rm=TRUE),
              .groups="drop") |>
    arrange(desc(city_pop)) |>
    slice_head(n=4) |>
    pull(city)
  
  cat(sprintf("  Top 4 cities: %s\n",
              paste(big_cities, collapse=", ")))
  
  # Build plot data
  plot_data <- big_city_sep_high |>
    filter(name %in%
             c("Incidence Lags",
               "Nearby City Incidence Lags")) |>
    filter(!is.na(city_pop)) |>
    group_by(name) |>
    mutate(
      rel_value = (rel_value -
                     min(rel_value, na.rm=TRUE)) /
        (max(rel_value, na.rm=TRUE) -
           min(rel_value, na.rm=TRUE) + 1e-10)
    ) |>
    ungroup() |>
    mutate(
      pop_group = cut_number(
        log(city_pop+1), n=10,
        labels=FALSE)
    ) |>
    mutate(
      pop_group = factor(
        pop_group,
        labels=paste0("Q",1:10))
    )
  
  # For Madden-style: filter to top 4 big cities
  # and show their incidence lags only
  madden_data <- big_city_sep_high |>
    filter(nearest_big_city %in% big_cities,
           name=="Nearby City Incidence Lags") |>
    mutate(
      city_label = paste0(nearest_big_city,
                          "\nIncidence Lags"),
      city_label = factor(
        city_label,
        levels=paste0(big_cities,
                      "\nIncidence Lags"))
    ) |>
    filter(!is.na(city_pop)) |>
    group_by(city_label) |>
    mutate(
      rel_value = (rel_value -
                     min(rel_value, na.rm=TRUE)) /
        (max(rel_value, na.rm=TRUE) -
           min(rel_value, na.rm=TRUE) + 1e-10),
      pop_group = cut_number(
        log(city_pop+1), n=10)
    ) |>
    ungroup()
  
  if (nrow(madden_data)==0) {
    message(sprintf(
      "  k=%s: no data for Madden-style plot",
      k_val))
    return(NULL)
  }
  
  p <- ggplot(madden_data,
              aes(x=pop_group,
                  y=rel_value)) +
    geom_boxplot(outlier.size=0.3,
                 fill="grey92",
                 colour="black") +
    facet_wrap(~city_label, nrow=1) +
    theme(
      axis.text.x=element_text(
        angle=90, hjust=1, size=8)
    ) +
    labs(
      title    = paste0(
        "Core City Contribution to Local ",
        "Transmission | k=",k_val),
      subtitle = paste0(
        "South Twenty Four Parganas | ",
        "Normalised relative SHAP | ",
        "West Bengal 2017-2019"),
      y       = paste0(
        "Relative Contribution of Core Cities\n",
        "to Local Transmission"),
      x       = "Log Population Size (10-Quantile)",
      caption = paste0(
        "SHAP values from ShapleyValueSampling ",
        "(captum). ",
        "Aligned with Madden et al. (2024) Fig. 3.")
    )
  
  # Save
  name_stem <- paste0(
    "shap_10quantile_boxplots_k", k_val)
  
  ggsave(paste0(save_dir, name_stem, ".pdf"),
         p, width=9, height=5,
         device=cairo_pdf)
  ggsave(paste0(save_dir, name_stem, ".png"),
         p, width=9, height=5,
         dpi=300, bg="white")
  cat(sprintf("  Saved: %s\n", name_stem))
  
  return(p)
}

# ── RUN ALL k VALUES ──────────────────────────────────────────
k_values <- c("1","4","12","20","34")

plots <- lapply(k_values, make_shap_boxplot)
names(plots) <- paste0("k", k_values)

# ── ALSO SAVE k=34 STANDALONE (primary result) ────────────────
if (!is.null(plots[["k34"]])) {
  ggsave(
    paste0(save_dir,
           "shap_10quantile_boxplots.png"),
    plots[["k34"]],
    width=9, height=5, dpi=300, bg="white")
  ggsave(
    paste0(save_dir,
           "shap_10quantile_boxplots.pdf"),
    plots[["k34"]],
    width=9, height=5, device=cairo_pdf)
  cat("\nPrimary figure saved: ",
      "shap_10quantile_boxplots\n")
}

cat(sprintf("\nAll figures saved to: %s\n",
            save_dir))

