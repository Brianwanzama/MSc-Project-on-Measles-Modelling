
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(patchwork)
library(arrow)
library(purrr)

set.seed(42)

# ── PUBLICATION THEME ─────────────────────────────────────────
theme_pub <- function(base_size=14) {
  theme_classic(base_size=base_size) +
    theme(
      panel.border      = element_rect(fill=NA, colour="black",
                                       linewidth=0.8),
      strip.background  = element_rect(fill="grey92",
                                       colour="black",
                                       linewidth=0.6),
      strip.text        = element_text(face="bold",
                                       size=base_size-1),
      axis.text         = element_text(colour="black"),
      axis.title        = element_text(face="bold"),
      legend.background = element_rect(fill=NA),
      legend.key        = element_rect(fill=NA),
      plot.title        = element_text(face="bold",
                                       size=base_size+1),
      plot.subtitle     = element_text(size=base_size-1,
                                       colour="grey30")
    )
}
theme_set(theme_pub())

# ── COLOUR PALETTES ───────────────────────────────────────────
pal_k <- c(
  "k = 1"  = "#E69F00",
  "k = 4"  = "#56B4E9",
  "k = 12" = "#009E73",
  "k = 20" = "#CC79A7",
  "k = 34" = "#D55E00"
)
pal_type <- c(
  "Local Incidence"        = "#D55E00",
  "Nearby City Incidence"  = "#0072B2",
  "Susceptibles"           = "#009E73",
  "Births"                 = "#56B4E9",
  "Population"             = "#CC79A7",
  "MCV1/MCV2 Vaccination"  = "#E69F00",
  "Distances"              = "#999999",
  "Other"                  = "#444444"
)

# ── DIRECTORIES ───────────────────────────────────────────────
BASE      <- "/.../.../.../"
read_dir  <- paste0(BASE,
                    "output/data/basic_nn_optimal/explain/")
save_dir  <- paste0(BASE,
                    "experiments/figures/explain/")
data_dir  <- paste0(BASE, "data/")
nn_dir    <- paste0(BASE,
                    "output/data/basic_nn_optimal/")

dir.create(save_dir, recursive=TRUE,
           showWarnings=FALSE)

k_values <- c("1","4","12","20","34")

# ── ANCILLARY DATA ────────────────────────────────────────────
all_cities <- read_csv(
  paste0(nn_dir, "all_cases_wb.csv"),
  show_col_types=FALSE
)

# nearest_big_city lookup
# Build from coordinates if file does not exist
nbc_path <- paste0(data_dir, "nearest_big_city.csv")

if (file.exists(nbc_path)) {
  nbc <- read_csv(nbc_path, show_col_types=FALSE) |>
    select(city, nearest_big_city) |>
    distinct()
} else {
  # Derive from population: top 7 cities are "big cities"
  big_cities <- all_cities |>
    group_by(city) |>
    summarise(min_pop=max(pop, na.rm=TRUE),
              .groups="drop") |>
    arrange(desc(min_pop)) |>
    slice_head(n=7) |>
    pull(city)
  
  # Assign each city its nearest big city
  # Using coordinates if available, else use first big city
  coords_path <- paste0(data_dir, "coordinates.csv")
  if (file.exists(coords_path)) {
    coords_raw <- read_csv(coords_path,
                           show_col_types=FALSE)
    if ("city" %in% names(coords_raw)) {
      coords <- coords_raw |>
        rename_with(tolower) |>
        select(city, any_of(c("lat","lon",
                              "latitude","longitude")))
    } else {
      ct <- t(coords_raw)
      colnames(ct) <- ct[1,]
      coords <- ct[2:nrow(ct),] |>
        as_tibble() |>
        rename_with(tolower) |>
        mutate(across(everything(), as.double)) |>
        mutate(city=rownames(ct)[2:nrow(ct)])
    }
    lat_col <- names(coords)[grepl("lat",
                                   names(coords))][1]
    lon_col <- names(coords)[grepl("lon",
                                   names(coords))][1]
    all_c <- unique(all_cities$city)
    nbc <- lapply(all_c, function(city_i) {
      ci <- coords |> filter(city==city_i)
      if (nrow(ci)==0) return(
        tibble(city=city_i,
               nearest_big_city=big_cities[1]))
      dists <- sapply(big_cities, function(bc) {
        cbc <- coords |> filter(city==bc)
        if (nrow(cbc)==0) return(Inf)
        sqrt((ci[[lat_col]]-cbc[[lat_col]])^2 +
               (ci[[lon_col]]-cbc[[lon_col]])^2)
      })
      tibble(city=city_i,
             nearest_big_city=big_cities[which.min(dists)])
    }) |> bind_rows()
  } else {
    nbc <- tibble(
      city=unique(all_cities$city),
      nearest_big_city=big_cities[1])
  }
  write_csv(nbc, nbc_path)
  cat("Created nearest_big_city.csv\n")
}

# ── SAVE HELPER ───────────────────────────────────────────────
save_plot <- function(p, name, w=10, h=6) {
  ggsave(paste0(save_dir, name, ".pdf"),
         p, width=w, height=h, device=cairo_pdf)
  ggsave(paste0(save_dir, name, ".png"),
         p, width=w, height=h, dpi=300, bg="white")
  message("  Saved: ", name)
}

# ============================================================================
# 1. HELPER FUNCTIONS
# ============================================================================

# ── Read SHAP CSV ─────────────────────────────────────────────
read_shap_raw <- function(k) {
  # Our output filename is {k}_svs_explain.csv
  path <- paste0(read_dir, k, "_svs_explain.csv")
  if (!file.exists(path)) {
    warning("File not found: ", path); return(NULL)
  }
  dat <- read_csv(path, show_col_types=FALSE)
  # Drop row-index column if present
  if (names(dat)[1] %in% c("...1","X","X1"))
    dat <- dat[, 2:ncol(dat)]
  
  dat |>
    left_join(nbc, by="city") |>
    mutate(time_temp=round(time, 2)) |>
    # NOTE: no + 1900 — WB years are actual
    left_join(
      all_cities |>
        select(city, time, min_pop) |>
        mutate(time=round(time, 2)) |>
        rename(city_pop=min_pop),
      by=c("city","time_temp"="time")
    ) |>
    select(-time_temp) |>
    mutate(k=k)
}

# ── Read feature parquets ─────────────────────────────────────
read_features <- function(k, split="test") {
  path <- paste0(read_dir, k, "_X_", split, ".parquet")
  if (!file.exists(path)) {
    warning("Not found: ", path); return(NULL)
  }
  read_parquet(path) |>
    mutate(k=k, split=split)
}

read_ids <- function(k, split="test") {
  path <- paste0(read_dir, k, "_id_", split, ".parquet")
  if (!file.exists(path)) {
    warning("Not found: ", path); return(NULL)
  }
  read_parquet(path) |>
    mutate(k=k, split=split)
}

# ── Feature classification ────────────────────────────────────
classify_feature <- function(col_name) {
  dplyr::case_when(
    stringr::str_detect(col_name, "cases_nc|nbc_lag|nc_.*_lag") ~
      "Nearby City Incidence",
    stringr::str_detect(col_name, "cases_lag") ~
      "Local Incidence",
    stringr::str_detect(col_name, "susc_lag") ~
      "Susceptibles",
    stringr::str_detect(col_name, "pop_lag")  ~
      "Population",
    stringr::str_detect(col_name, "births_lag") ~
      "Births",
    stringr::str_detect(col_name, "v1_lag|v2_lag") ~
      "MCV1/MCV2 Vaccination",
    stringr::str_detect(col_name, "dist") ~
      "Distances",
    TRUE ~ "Other"
  )
}

# ============================================================================
# 2. LOAD ALL SHAP DATA
# ============================================================================

message("\n── Loading SHAP values for all k ───────────────────────────────────")

id_cols <- c("time","city","nearest_big_city",
             "city_pop","k")

shap_all <- map(k_values, read_shap_raw) |>
  compact() |>
  bind_rows()

if (nrow(shap_all)==0) stop("No SHAP data loaded.")

cat(sprintf("Loaded: %d rows | %d cities\n",
            nrow(shap_all),
            length(unique(shap_all$city))))

# ── Pivot to long format ──────────────────────────────────────
shap_long <- shap_all |>
  pivot_longer(
    cols      = -all_of(id_cols),
    names_to  = "feature",
    values_to = "shap"
  ) |>
  mutate(
    lag          = as.integer(
      stringr::str_extract(feature, "[0-9]+$")),
    feature_type = classify_feature(feature),
    k_num        = as.integer(k),
    k_label      = factor(paste0("k = ",k),
                          levels=paste0("k = ",k_values))
  ) |>
  filter(!is.na(shap))

# ── Relative contribution within each (time, city) ───────────
shap_long <- shap_long |>
  group_by(k, time, city) |>
  mutate(rel=abs(shap)/sum(abs(shap), na.rm=TRUE)) |>
  ungroup()

message(sprintf("shap_long: %d rows", nrow(shap_long)))

# ── City-level summary ────────────────────────────────────────
shap_city <- shap_long |>
  group_by(k, k_label, city, nearest_big_city,
           city_pop, feature, feature_type, lag) |>
  summarise(
    mean_abs_shap = mean(abs(shap), na.rm=TRUE),
    mean_rel      = mean(rel,       na.rm=TRUE),
    .groups="drop"
  )

# ── Top 5 cities by population ────────────────────────────────
top_cities <- all_cities |>
  group_by(city) |>
  summarise(city_pop=max(pop, na.rm=TRUE),
            .groups="drop") |>
  arrange(desc(city_pop)) |>
  slice_head(n=5) |>
  pull(city)

message("Top 5 cities: ",
        paste(top_cities, collapse=", "))

# ── Lag importance ────────────────────────────────────────────
lag_importance <- shap_long |>
  filter(!is.na(lag)) |>
  group_by(k_label, k_num, feature_type, lag) |>
  summarise(
    mean_abs = mean(abs(shap), na.rm=TRUE),
    mean_rel = mean(rel,       na.rm=TRUE),
    .groups  = "drop"
  )

# ── Spatial vs local ─────────────────────────────────────────
# Build spatial_local before Figure C
spatial_local <- shap_long |>
  filter(feature_type %in% c("Local Incidence",
                             "Nearby City Incidence")) |>
  group_by(k_label, city, city_pop, time,
           feature_type) |>
  summarise(sum_abs=sum(abs(shap), na.rm=TRUE),
            .groups="drop") |>
  group_by(k_label, city, city_pop, time) |>
  mutate(share=sum_abs/sum(sum_abs, na.rm=TRUE)) |>
  ungroup() |>
  group_by(k_label, city, city_pop, feature_type) |>
  summarise(share=mean(share, na.rm=TRUE),
            .groups="drop") |>
  mutate(
    pop_decile = cut_number(
      log(city_pop+1), n=5,
      labels=paste0("Q",1:5))
  )

# ============================================================================
# 3. FIGURE A — Population-Stratified SHAP
# ============================================================================

message("\n── Figure A: Population-stratified SHAP ────────────────────────────")

fig_a_madden_data <- shap_city |>
  filter(k=="12",
         stringr::str_detect(feature,
                             "cases_nc|nbc_lag")) |>
  group_by(city, city_pop,
           nearest_big_city=nearest_big_city) |>
  summarise(mean_rel=mean(mean_rel, na.rm=TRUE),
            .groups="drop") |>
  filter(nearest_big_city %in% top_cities[1:4]) |>
  group_by(nearest_big_city) |>
  mutate(
    norm_rel  = (mean_rel-min(mean_rel)) /
      (max(mean_rel)-min(mean_rel)+1e-10),
    pop_group = cut_number(
      log(city_pop+1),
      n=min(8, n_distinct(city_pop)))
  ) |>
  ungroup() |>
  mutate(nearest_big_city=factor(
    nearest_big_city, levels=top_cities[1:4]))

if (nrow(fig_a_madden_data) > 0) {
  fig_a <- ggplot(fig_a_madden_data,
                  aes(x=pop_group, y=norm_rel)) +
    geom_boxplot(outlier.size=0.4,
                 fill="grey90", colour="black") +
    stat_summary(aes(group=1),
                 fun=median, geom="line",
                 colour="#D55E00", linewidth=0.9) +
    facet_wrap(~nearest_big_city, nrow=1) +
    scale_y_continuous(
      labels=function(x) paste0(round(x*100,1),"%")) +
    labs(
      title    = paste0(
        "Population-Stratified Contribution ",
        "of Core Cities (k = 12)"),
      subtitle = paste0(
        "Normalised relative SHAP contribution; ",
        "orange line = median trend"),
      x = "Log Population Size (Quantile)",
      y = "Relative Contribution to\nLocal Transmission"
    ) +
    theme(axis.text.x=element_text(
      angle=90, hjust=1, size=9))
  
  save_plot(fig_a,
            "figA_pop_stratified_k12", w=10, h=5)
}

# ── All k version ─────────────────────────────────────────────
fig_a_allk_data <- shap_city |>
  filter(stringr::str_detect(feature,
                             "cases_nc|nbc_lag")) |>
  group_by(k_label, city, city_pop,
           nearest_big_city) |>
  summarise(mean_rel=mean(mean_rel, na.rm=TRUE),
            .groups="drop") |>
  filter(nearest_big_city %in% top_cities[1:4]) |>
  group_by(k_label, nearest_big_city) |>
  mutate(
    norm_rel  = (mean_rel-min(mean_rel)) /
      (max(mean_rel)-min(mean_rel)+1e-10),
    pop_group = cut_number(
      log(city_pop+1),
      n=min(8, n_distinct(city_pop)))
  ) |>
  ungroup() |>
  mutate(nearest_big_city=factor(
    nearest_big_city, levels=top_cities[1:4]))

if (nrow(fig_a_allk_data) > 0) {
  fig_a_allk <- ggplot(fig_a_allk_data,
                       aes(x=pop_group, y=norm_rel)) +
    geom_boxplot(outlier.size=0.3, fill="grey92") +
    stat_summary(aes(group=1),
                 fun=median, geom="line",
                 colour="#D55E00", linewidth=0.8) +
    facet_grid(k_label~nearest_big_city) +
    scale_y_continuous(
      labels=function(x) paste0(round(x*100,1),"%")) +
    labs(
      title    = paste0(
        "Population-Stratified Nearby ",
        "City Contribution — All k"),
      subtitle = "Rows = k; columns = core city",
      x = "Log Population Size (Quantile)",
      y = "Normalised Relative SHAP"
    ) +
    theme(axis.text.x=element_text(
      angle=90, hjust=1, size=8))
  
  save_plot(fig_a_allk,
            "figA_pop_stratified_allk", w=14, h=16)
}

# ============================================================================
# 4. FIGURE B — Temporal Lag Importance
# ============================================================================

message("\n── Figure B: Temporal lag importance ───────────────────────────────")

fig_b <- ggplot(
  lag_importance |>
    filter(feature_type %in% c("Local Incidence",
                               "Nearby City Incidence")),
  aes(x=lag, y=mean_rel,
      colour=feature_type,
      fill=feature_type)
) +
  geom_area(alpha=0.25, position="identity") +
  geom_line(linewidth=0.9) +
  geom_point(size=1.5, shape=21,
             colour="white") +
  scale_colour_manual(values=pal_type,
                      name="Feature Type") +
  scale_fill_manual(values=pal_type,
                    name="Feature Type") +
  scale_x_continuous(
    breaks=seq(2008, 2019, by=2)) +
  scale_y_continuous(
    labels=function(x) paste0(round(x*100,1),"%")) +
  facet_wrap(~k_label, scales="free_x", nrow=2) +
  labs(
    title    = "Lag-Resolved SHAP Importance",
    subtitle = "Mean relative contribution per lag step",
    x        = "Lag (biweeks)",
    y        = "Mean Relative SHAP Contribution"
  )

save_plot(fig_b, "figB_lag_importance", w=14, h=8)

# ── Decay plot ────────────────────────────────────────────────
lag_decay <- lag_importance |>
  filter(feature_type=="Local Incidence") |>
  group_by(k_label) |>
  mutate(norm_imp=mean_rel/max(mean_rel,
                               na.rm=TRUE)) |>
  ungroup()

fig_b_decay <- ggplot(lag_decay,
                      aes(x=lag, y=norm_imp,
                          colour=k_label, group=k_label)) +
  geom_line(linewidth=1) +
  geom_point(size=2) +
  scale_colour_manual(values=pal_k,
                      name="Lag window k") +
  scale_y_continuous(
    labels=function(x) paste0(round(x*100,1),"%")) +
  labs(
    title    = "Importance Decay Across Lag Windows",
    subtitle = paste0(
      "Normalised local incidence SHAP ",
      "vs. lag — coloured by k"),
    x = "Lag (biweeks)",
    y = "Normalised SHAP Importance"
  )

save_plot(fig_b_decay, "figB_lag_decay", w=9, h=5)

# ============================================================================
# 5. FIGURE C — Spatial vs Local Contribution
# ============================================================================

message("\n── Figure C: Spatial vs Local ───────────────────────────────────────")

fig_c <- ggplot(spatial_local,
                aes(x=pop_decile, y=share,
                    fill=feature_type)) +
  geom_boxplot(position=position_dodge(0.8),
               outlier.size=0.4, alpha=0.85) +
  scale_fill_manual(values=pal_type,
                    name="Feature Type") +
  scale_y_continuous(
    labels=function(x) paste0(round(x*100,1),"%")) +
  facet_wrap(~k_label, nrow=2) +
  labs(
    title    = paste0(
      "Local vs Spatial SHAP Contribution ",
      "by Population"),
    subtitle = paste0(
      "Share of total SHAP attributed to ",
      "own vs. nearby-city lags"),
    x = "Population Quintile (log scale)",
    y = "Share of Total SHAP"
  )

save_plot(fig_c, "figC_spatial_local", w=14, h=8)

# ── Scatter: city_pop vs spatial share ───────────────────────
spatial_local_k12 <- spatial_local |>
  filter(k_label=="k = 12",
         feature_type=="Nearby City Incidence")

if (nrow(spatial_local_k12) > 0) {
  fig_c_scatter <- ggplot(spatial_local_k12,
                          aes(x=log(city_pop+1), y=share)) +
    geom_point(alpha=0.5, size=1.8,
               colour="#0072B2") +
    geom_smooth(method="loess", se=TRUE,
                colour="#D55E00",
                fill="#D55E00", alpha=0.15) +
    geom_text(
      data=spatial_local_k12 |>
        filter(city %in% top_cities),
      aes(label=city),
      size=3, vjust=-0.8, hjust=0.5,
      check_overlap=TRUE
    ) +
    scale_y_continuous(
      labels=function(x) paste0(round(x*100,1),"%")) +
    labs(
      title    = paste0(
        "Spatial Contribution vs ",
        "Population Size (k = 12)"),
      subtitle = paste0(
        "Each point = one city; ",
        "LOESS trend in orange"),
      x = "Log Population Size",
      y = "Share of SHAP from Nearby City Lags"
    )
  
  save_plot(fig_c_scatter,
            "figC_spatial_pop_scatter", w=8, h=6)
}

# ============================================================================
# 6. FIGURE D — Feature Importance Heatmap
# ============================================================================

message("\n── Figure D: Feature importance heatmap ────────────────────────────")

heatmap_data <- shap_long |>
  group_by(k_label, feature_type) |>
  summarise(mean_abs=mean(abs(shap), na.rm=TRUE),
            .groups="drop") |>
  group_by(k_label) |>
  mutate(norm_abs=mean_abs/sum(mean_abs)) |>
  ungroup() |>
  mutate(feature_type=factor(feature_type,
                             levels=c("Local Incidence",
                                      "Nearby City Incidence",
                                      "Susceptibles",
                                      "Births","Population",
                                      "MCV1/MCV2 Vaccination",
                                      "Distances","Other")))

fig_d <- ggplot(heatmap_data,
                aes(x=k_label, y=feature_type,
                    fill=norm_abs)) +
  geom_tile(colour="white", linewidth=0.5) +
  geom_text(
    aes(label=sprintf("%.1f%%", norm_abs*100)),
    size=4, colour="white", fontface="bold") +
  scale_fill_gradient2(
    low="#F8F9FA", mid="#2980B9", high="#1A5276",
    midpoint=0.15,
    name="Normalised\nMean |SHAP|") +
  scale_x_discrete(expand=c(0,0)) +
  scale_y_discrete(expand=c(0,0)) +
  labs(
    title    = "Feature Importance Heatmap",
    subtitle = "Rows = feature group; columns = k",
    x        = "Lag Window (k)",
    y        = "Feature Group"
  ) +
  theme(axis.text.x=element_text(
    angle=45, hjust=1))

save_plot(fig_d,
          "figD_heatmap_feature_k", w=8, h=5)

# ── Lag × k heatmap ──────────────────────────────────────────
lag_heat <- shap_long |>
  filter(feature_type=="Local Incidence",
         !is.na(lag)) |>
  group_by(k_label, lag) |>
  summarise(mean_abs=mean(abs(shap), na.rm=TRUE),
            .groups="drop") |>
  group_by(k_label) |>
  mutate(norm_abs=mean_abs/max(mean_abs,
                               na.rm=TRUE)) |>
  ungroup()

fig_d_lag <- ggplot(lag_heat,
                    aes(x=factor(lag), y=k_label,
                        fill=norm_abs)) +
  geom_tile(colour="grey80", linewidth=0.3) +
  scale_fill_gradient2(
    low="#F8F9FA", mid="#CC79A7", high="#6D0072",
    midpoint=0.5,
    name="Normalised\n|SHAP|") +
  scale_x_discrete(
    guide=guide_axis(check.overlap=TRUE)) +
  labs(
    title    = "Lag × k Importance Heatmap (Local Incidence)",
    subtitle = paste0(
      "Colour = importance normalised ",
      "to maximum within each k"),
    x = "Lag Step",
    y = "Lag Window (k)"
  ) +
  theme(axis.text.x=element_text(
    angle=90, hjust=1, size=8))

save_plot(fig_d_lag,
          "figD_heatmap_lag_k", w=14, h=5)

# ============================================================================
# 7. FIGURE E — Dependence Plots
# ============================================================================

message("\n── Figure E: Dependence plots ───────────────────────────────────────")

load_merged <- function(k_val) {
  ids  <- read_ids(k_val,      split="test")
  feat <- read_features(k_val, split="test")
  shp  <- read_shap_raw(k_val)
  if (is.null(ids)||is.null(feat)||is.null(shp))
    return(NULL)
  n <- min(nrow(ids), nrow(feat), nrow(shp))
  bind_cols(
    ids[1:n,]  |>
      select(any_of(c("city","time"))),
    feat[1:n,] |>
      select(-any_of(c("city","time","k","split"))),
    shp[1:n,]  |>
      select(-any_of(c("city","time",
                       "nearest_big_city",
                       "city_pop","k"))) |>
      rename_with(~paste0("shap_",.))
  ) |> mutate(k=k_val)
}

merged_all <- map(k_values, load_merged) |>
  compact() |>
  bind_rows()

if (nrow(merged_all) > 0) {
  
  # E1: Local incidence lag-1
  local_feat <- merged_all |>
    select(starts_with("cases_lag"),
           -starts_with("cases_nc")) |>
    colnames() |> head(1)
  
  shap_local <- merged_all |>
    select(starts_with("shap_cases_lag"),
           -starts_with("shap_cases_nc")) |>
    colnames() |> head(1)
  
  if (length(local_feat)==1 &&
      length(shap_local)==1) {
    dep_local <- merged_all |>
      select(k,
             feat_val=all_of(local_feat),
             shap_val=all_of(shap_local)) |>
      filter(!is.na(feat_val),!is.na(shap_val)) |>
      mutate(k_label=factor(paste0("k = ",k),
                            levels=paste0("k = ",k_values)))
    
    fig_e_local <- ggplot(dep_local,
                          aes(x=feat_val, y=shap_val)) +
      geom_point(alpha=0.15, size=0.8,
                 colour="#0072B2") +
      geom_smooth(method="loess", se=TRUE,
                  colour="#D55E00",
                  fill="#D55E00",
                  alpha=0.20, linewidth=0.9) +
      geom_hline(yintercept=0,
                 linetype="dashed",
                 colour="grey50") +
      scale_x_log10(labels=function(x) format(x, big.mark=",", scientific=FALSE)) +
      facet_wrap(~k_label,
                 scales="free", nrow=2) +
      labs(
        title    = "Dependence — Local Incidence Lag 1",
        subtitle = "Feature value (log) vs SHAP; LOESS in orange",
        x        = "Incidence (lag 1, log scale)",
        y        = "SHAP Value"
      )
    
    save_plot(fig_e_local,
              "figE_dep_local_incidence",
              w=14, h=8)
  }
  
  # E2: Births
  births_feat <- merged_all |>
    select(starts_with("births_lag")) |>
    colnames() |> head(1)
  shap_births <- merged_all |>
    select(starts_with("shap_births")) |>
    colnames() |> head(1)
  
  if (length(births_feat)==1 &&
      length(shap_births)==1) {
    dep_births <- merged_all |>
      select(k,
             feat_val=all_of(births_feat),
             shap_val=all_of(shap_births)) |>
      filter(!is.na(feat_val),!is.na(shap_val)) |>
      mutate(k_label=factor(paste0("k = ",k),
                            levels=paste0("k = ",k_values)))
    
    fig_e_births <- ggplot(dep_births,
                           aes(x=feat_val, y=shap_val)) +
      geom_point(alpha=0.15, size=0.8,
                 colour="#009E73") +
      geom_smooth(method="loess", se=TRUE,
                  colour="#D55E00",
                  fill="#D55E00",
                  alpha=0.2, linewidth=0.9) +
      geom_hline(yintercept=0,
                 linetype="dashed",
                 colour="grey50") +
      facet_wrap(~k_label,
                 scales="free", nrow=2) +
      labs(
        title    = "Dependence — Births",
        subtitle = "Feature value vs SHAP; LOESS in orange",
        x        = "Births",
        y        = "SHAP Value"
      )
    
    save_plot(fig_e_births,
              "figE_dep_births", w=14, h=8)
  }
  
  # E3: Population
  pop_feat <- merged_all |>
    select(starts_with("pop_lag")) |>
    colnames() |> head(1)
  shap_pop <- merged_all |>
    select(starts_with("shap_pop")) |>
    colnames() |> head(1)
  
  if (length(pop_feat)==1 &&
      length(shap_pop)==1) {
    dep_pop <- merged_all |>
      select(k,
             feat_val=all_of(pop_feat),
             shap_val=all_of(shap_pop)) |>
      filter(!is.na(feat_val),!is.na(shap_val)) |>
      mutate(k_label=factor(paste0("k = ",k),
                            levels=paste0("k = ",k_values)))
    
    fig_e_pop <- ggplot(dep_pop,
                        aes(x=feat_val, y=shap_val)) +
      geom_point(alpha=0.15, size=0.8,
                 colour="#CC79A7") +
      geom_smooth(method="loess", se=TRUE,
                  colour="#D55E00",
                  fill="#D55E00",
                  alpha=0.2, linewidth=0.9) +
      geom_hline(yintercept=0,
                 linetype="dashed",
                 colour="grey50") +
      scale_x_log10(labels=function(x) format(x, big.mark=",", scientific=FALSE)) +
      facet_wrap(~k_label,
                 scales="free", nrow=2) +
      labs(
        title    = "Dependence — Population",
        subtitle = "Population (log) vs SHAP; LOESS in orange",
        x        = "Population (log scale)",
        y        = "SHAP Value"
      )
    
    save_plot(fig_e_pop,
              "figE_dep_population", w=14, h=8)
  }
}

# ============================================================================
# 8. FIGURE F — City-Specific Dynamics Over Time
# ============================================================================

message("\n── Figure F: City-specific dynamics ────────────────────────────────")

city_dynamics <- shap_long |>
  filter(city %in% top_cities[1:5]) |>
  # NOTE: no + 1900 — WB years are actual
  mutate(year=round(time, 2)) |>
  group_by(k_label, city, year,
           feature_type) |>
  summarise(mean_abs=mean(abs(shap),
                          na.rm=TRUE),
            .groups="drop") |>
  group_by(k_label, city, year) |>
  mutate(share=mean_abs/sum(mean_abs)) |>
  ungroup() |>
  filter(feature_type %in%
           c("Local Incidence",
             "Nearby City Incidence",
             "Susceptibles",
             "Births","Population",
             "MCV1/MCV2 Vaccination")) |>
  mutate(
    city    = factor(city,
                     levels=top_cities[1:5]),
    k_label = factor(k_label,
                     levels=paste0("k = ",
                                   k_values))
  )

for (k_val in k_values) {
  cd_k <- city_dynamics |>
    filter(k_label==paste0("k = ",k_val))
  if (nrow(cd_k)==0) next
  
  p <- ggplot(cd_k,
              aes(x=year, y=share,
                  fill=feature_type)) +
    geom_area(position="stack", alpha=0.85) +
    scale_fill_manual(values=pal_type,
                      name="Feature Group") +
    scale_y_continuous(
      labels=function(x) paste0(round(x*100,1),"%")) +
    scale_x_continuous(
      breaks=seq(2008, 2019, by=2)) +
    facet_wrap(~city, nrow=1) +
    labs(
      title    = paste0(
        "Feature Contribution Dynamics — k = ",
        k_val),
      subtitle = paste0(
        "Stacked area = share of total ",
        "SHAP per time point"),
      x = "Year",
      y = "Share of SHAP Contribution"
    ) +
    theme(legend.position="bottom",
          axis.text.x=element_text(
            angle=45, hjust=1))
  
  save_plot(p,
            paste0("figF_city_dynamics_k", k_val),
            w=14, h=5)
}

fig_f_allk <- ggplot(city_dynamics,
                     aes(x=year, y=share,
                         fill=feature_type)) +
  geom_area(position="stack", alpha=0.85) +
  scale_fill_manual(values=pal_type,
                    name="Feature Group") +
  scale_y_continuous(
    labels=function(x) paste0(round(x*100,1),"%")) +
  facet_grid(k_label~city) +
  labs(
    title    = "City-Specific SHAP Dynamics — All k",
    subtitle = "Rows = k; columns = city",
    x        = "Year",
    y        = "Share of SHAP"
  ) +
  theme(
    axis.text.x=element_text(
      angle=45, hjust=1, size=7),
    legend.position="bottom"
  )

save_plot(fig_f_allk,
          "figF_city_dynamics_allk",
          w=18, h=14)

# ============================================================================
# 9. SUPPLEMENTARY — Mean |SHAP| Ranking
# ============================================================================

message("\n── Supplementary: Feature ranking ──────────────────────────────────")

feat_rank <- shap_long |>
  group_by(k_label, feature_type) |>
  summarise(mean_abs=mean(abs(shap),
                          na.rm=TRUE),
            .groups="drop") |>
  group_by(k_label) |>
  mutate(k_label=factor(k_label,
                        levels=paste0("k = ",k_values)))

fig_s_rank <- ggplot(feat_rank,
                     aes(x=reorder(feature_type,-mean_abs),
                         y=mean_abs,
                         fill=feature_type)) +
  geom_col(show.legend=FALSE) +
  scale_fill_manual(values=c(
    "Local Incidence"       = "#D55E00",
    "Nearby City Incidence" = "#0072B2",
    "Susceptibles"          = "#009E73",
    "Births"                = "#56B4E9",
    "Population"            = "#CC79A7",
    "MCV1"                  = "#E69F00",
    "Distances"             = "#999999",
    "Other"                 = "#444444")) +
  scale_y_continuous(
    expand=expansion(mult=c(0,0.1))) +
  facet_wrap(~k_label,
             scales="free_y", nrow=2) +
  labs(
    title = "Mean Absolute SHAP by Feature Group",
    x     = "Feature Group",
    y     = "Mean |SHAP|"
  ) +
  theme(axis.text.x=element_text(
    angle=30, hjust=1))

save_plot(fig_s_rank,
          "figS_feature_ranking",
          w=14, h=8)

# ============================================================================
# 10. SUMMARY
# ============================================================================

message("\n", strrep("=",60))
message("All figures saved to: ", save_dir)
message(strrep("=",60))
message("
  figA_pop_stratified_k12.*       Madden replication
  figA_pop_stratified_allk.*      Extended all k
  figB_lag_importance.*           Temporal lag importance
  figB_lag_decay.*                Importance decay by k
  figC_spatial_local.*            Spatial vs local
  figC_spatial_pop_scatter.*      Spatial share vs population
  figD_heatmap_feature_k.*        Feature x k heatmap
  figD_heatmap_lag_k.*            Lag x k heatmap
  figE_dep_local_incidence.*      Dependence: incidence
  figE_dep_births.*               Dependence: births
  figE_dep_population.*           Dependence: population
  figF_city_dynamics_k{k}.*       City dynamics per k
  figF_city_dynamics_allk.*       City dynamics all k
  figS_feature_ranking.*          Mean SHAP ranking
")