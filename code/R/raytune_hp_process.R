# Specify the path to your CSV files
path_to_csvs <- "/.../.../.../output/data/basic_nn_optimal/raytune_hp_optim"

# List all csv files following the naming pattern
file_list <- list.files(path = path_to_csvs, pattern = "raytune_hp_optim_k_\\d+\\.csv", full.names = TRUE)

# Initialize an empty list to store data frames
data_list <- list()

# Loop through each file
for (file_name in file_list) {
  # Read the first row of the csv
  data <- read.csv(file_name, nrows = 1)
  
  # Extract the XX value from the filename
  # This gets everything between 'k_' and '.csv'
  k_part <- sub(".*/raytune_hp_optim_k_(\\d+)\\.csv", "\\1", file_name)
  k_value <- as.numeric(k_part)
  
  # Add the k_value as a column
  data$k <- k_value
  
  # Append the data frame to the list
  data_list[[length(data_list) + 1]] <- data
}

# Combine all data frames into one
final_data <- do.call(rbind, data_list)
final_small <- final_data[, c("k", "t_lag", "hidden_dim", "num_hidden_layers", "weight_decay")]
final_small <- final_small[order(final_small$k),]
final_small$weight_decay <- round(final_small$weight_decay, 4)

# Write the final data frame to a CSV file
write.csv(final_small, file = "/.../.../.../output/tables/raytune_hp_optim_all.csv", row.names = FALSE)

