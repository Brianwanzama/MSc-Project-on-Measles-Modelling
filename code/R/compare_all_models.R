library(tidyverse)
library(patchwork)
library(dplyr)
library(stringr)
library(viridis)
theme_set(theme_classic())
set.seed(42)
save_dir <- ".../.../.../output/figures/basic_nn_optimal/"
read_dir <- ".../.../.../data/"

#all original cases/birth/etc data from max's github
all_cities <- read_csv("output/data/basic_nn_optimal/all_cases_from_max_gh.csv")

#tsir data predictions
tsir_dat <- read_csv("output/data/basic_nn_optimal/tsir_preds_processed.csv") |>
    mutate(tsir = ifelse(tsir < 0, 0, tsir)) |>
    filter(!is.na(time))

#neural net predictions
nn_dat <- read_csv("output/data/basic_nn_optimal/basic_nn_preds.csv")

#rf predictions
rf100_dat <- read_csv("output/data/random_forest_100/random_forest_preds.csv")%>%
  mutate(k = as.integer(str_extract(k, "\\d+$")))

#gbboost predictions
gb_dat <- read_csv("output/data/gradientboost/gradientboost_preds.csv") %>%
  mutate(k = as.integer(str_extract(k, "\\d+")))

#xgboost predictions
xg_dat <- read_csv("output/data/xgboost_100/xg_preds.csv")%>%
  mutate(k = as.integer(str_extract(k, "\\d+$")))

#auto.arima predictions
arima_dat <- read_csv("output/data/autoarima/autoarima_uk_processed.csv")

pr_dat <- list.files("output/data/prophet/", pattern = "city", full.names = T) |>
    lapply(read_csv) |>
    bind_rows() |>
    filter(!is.na(time)) |>
    mutate(pr = prophet_pred,
           time = time + 1900)

full_dat_temp <- nn_dat[, c("time", "nn", "city", "k", "cases_mean", "cases_std", "nn_orig", "train_test")] %>% 
    mutate(k = substr(k, nchar(k) - 1, nchar(k)),
           k = as.integer(k)) |>
    filter(train_test == "test") |>
#     left_join(nn_dat_no_susc[, c("time", "nn_no_susc", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(tsir_dat[, c("time", "tsir", "city", "k")], by = c("time", "city", "k")) %>% 
    left_join(all_cities, by = c("time", "city")) %>%
    left_join(rf100_dat[, c("time", "rf100", "city", "k")], by = c("time", "city", "k")) |>
    left_join(gb_dat[, c("time", "gb", "city", "k")], by = c("time", "city", "k")) |>
    left_join(xg_dat[, c("time", "xg", "city", "k")], by = c("time", "city", "k")) |>
    left_join(arima_dat[, c("time", "auto_arima", "city", "k")], by = c("time", "city", "k")) |>
    left_join(pr_dat[, c("time", "pr", "city", "k")], by = c("time", "city", "k"))

full_dat <- full_dat_temp |>
  mutate(k = as.integer(k)) |>
  filter(k %in% c(1, 4, 12)) |>
  mutate(k = factor(k, levels = c(1, 4, 12)))


k_corrmse <- c(1, 4, 12)


prmse <- full_dat %>% 
    filter(k %in% k_corrmse) %>%
    group_by(city, k) %>% 
    mutate(nn_standardized = (log(nn + 1) - mean(log(cases + 1))) / sd(log(cases + 1)), 
           rf100_standardized = (log(rf100 + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           gb_standardized = (log(gb + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           xg_standardized = (log(xg + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           aa_standardized = (log(auto_arima + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           pr_standardized = (log(pr + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           tsir_standardized = (log(tsir + 1) - mean(log(cases + 1))) / sd(log(cases + 1)),
           cases_standardized = (log(cases + 1) - mean(log(cases + 1))) / sd(log(cases + 1)))
    
k_by_mod_table <- prmse |>
    ungroup() |>
    group_by(k) |>
    summarize(tsir_rmse = sqrt(mean((cases_standardized - tsir_standardized) ^ 2, na.rm = T)), 
              rf100_rmse = sqrt(mean((cases_standardized - rf100_standardized) ^ 2, na.rm = T)), 
              gb_rmse = sqrt(mean((cases_standardized - gb_standardized) ^ 2, na.rm = T)), 
              xg_rmse = sqrt(mean((cases_standardized - xg_standardized) ^ 2, na.rm = T)), 
              aa_rmse = sqrt(mean((cases_standardized - aa_standardized) ^ 2, na.rm = T)), 
              pr_rmse = sqrt(mean((cases_standardized - pr_standardized) ^ 2, na.rm = T)), 
              nn_rmse = sqrt(mean((cases_standardized - nn_standardized) ^ 2, na.rm = T)))


prmse_summ <- prmse |>
    summarize(min_pop = unique(min_pop),
              tsir_rmse = sqrt(mean((cases_standardized - tsir_standardized) ^ 2, na.rm = T)), 
              rf100_rmse = sqrt(mean((cases_standardized - rf100_standardized) ^ 2, na.rm = T)), 
              gb_rmse = sqrt(mean((cases_standardized - gb_standardized) ^ 2, na.rm = T)), 
              xg_rmse = sqrt(mean((cases_standardized - xg_standardized) ^ 2, na.rm = T)), 
              aa_rmse = sqrt(mean((cases_standardized - aa_standardized) ^ 2, na.rm = T)), 
              pr_rmse = sqrt(mean((cases_standardized - pr_standardized) ^ 2, na.rm = T)), 
              nn_rmse = sqrt(mean((cases_standardized - nn_standardized) ^ 2, na.rm = T)),
              case_sum = sum(cases))
write_csv(prmse_summ, paste0(save_dir, "rmse_over_models_by_k.csv"))




# ---- ALL MODELS (ALL k) ----
all_mod_rmse <- prmse_summ |>
  pivot_longer(cols = c(tsir_rmse, rf100_rmse, gb_rmse, xg_rmse, aa_rmse, pr_rmse), 
               names_to = "model", 
               values_to = "rmse") |>
  mutate(k = paste0("k = ", k),
         k = factor(k, levels = paste0("k = ", k_corrmse))) |>
  ggplot() +
  geom_point(aes(x = rmse, y = nn_rmse, colour = log(min_pop)),
             size = 0.1) +
  geom_abline(color = "black") +
  scale_color_viridis(option = "viridis", direction = -1) +
  facet_grid(model ~ k) +
  labs(x = expression(RMSE[ComparisonModel]),
       y = expression(RMSE[SFNN]),
       colour = "Log(Population)") +
  scale_x_continuous(breaks = seq(0.0, 1.5, 0.5), limits = c(0, 2)) +
  scale_y_continuous(breaks = seq(0, 1.5, 0.5), limits = c(0, 2)) +
  theme(panel.border = element_rect(colour = "black", fill = NA, size = 1),
        legend.position = "bottom")

ggsave(paste0(save_dir, "rmse_all_models.png"),
       all_mod_rmse, width = 8, height = 8, dpi = 600)


# ---- ALL MODELS (k = 12 only) ----
all_mod_k12_rmse <- prmse_summ |>
  pivot_longer(cols = c(rf100_rmse, gb_rmse, aa_rmse, pr_rmse), 
               names_to = "model", 
               values_to = "rmse") |>
  filter(k == 12) |>
  mutate(model = case_when(
    model == "rf100_rmse" ~ "Random Forest",
    model == "gb_rmse" ~ "Gradient Boost",
    model == "aa_rmse" ~ "ARIMA",
    model == "pr_rmse" ~ "Prophet"
  ),
  k = factor("k = 12", levels = paste0("k = ", k_corrmse))) |>
  ggplot() +
  geom_point(aes(x = rmse, y = nn_rmse, colour = log(min_pop)),
             size = 0.1) +
  geom_abline(color = "black") +
  scale_color_viridis(option = "viridis", direction = -1) +
  facet_wrap(~model, ncol = 2) +
  labs(x = expression(RMSE[ComparisonModel]),
       y = expression(RMSE[SFNN]),
       colour = "Log(Population)") +
  scale_x_continuous(breaks = seq(0.0, 1.5, 0.5), limits = c(0, 2)) +
  scale_y_continuous(breaks = seq(0, 1.5, 0.5), limits = c(0, 2)) +
  theme(panel.border = element_rect(colour = "black", fill = NA, size = 1),
        legend.position = "bottom")

ggsave(paste0(save_dir, "rmse_all_models_k12.png"),
       all_mod_k12_rmse, width = 5, height = 5, dpi = 600)




save_dir <- "~/resubmission_nn_temp/"
