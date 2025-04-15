import os
import re
import pandas as pd

# Define the folder containing the best_F_episodes CSV files
folder = "fidelity_history_files"

# Regex pattern to extract the wg_dist part from filenames like "best_F_episodes_wg_dist_M.csv"
pattern = r"best_F_episodes_(wg_dist_[^\.]+)\.csv"

# List to collect average info for each wg_dist group
results = []

# Iterate over each file in the folder
for filename in os.listdir(folder):
    # We want files that start with "best_F_episodes_" and end with ".csv"
    if filename.startswith("best_F_episodes_") and filename.endswith(".csv"):
        match = re.search(pattern, filename)
        if match:
            wg_dist = match.group(1)  # e.g., "wg_dist_0p25um"
            file_path = os.path.join(folder, filename)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                # Calculate average final_episode and average final_best_fidelity
                avg_episode = df["final_episode"].mean()
                avg_best_F = df["final_best_fidelity"].mean()
                
                # Remove the "wg_dist_" prefix and "um" suffix, then convert the remaining value to a float.
                wg_value_str = wg_dist.replace("wg_dist_", "").replace("um", "")
                wg_value_numeric = float(wg_value_str.replace("p", "."))
                
                # Append the result to our list. Now we use the numeric value directly.
                results.append({
                    "wg_dist [um]": wg_value_numeric,
                    "avg_final_episode": avg_episode,
                    "avg_final_best_F": avg_best_F,
                })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

# Create a DataFrame from the collected results
result_df = pd.DataFrame(results)

# Sort by the numerical wg distance
result_df = result_df.sort_values("wg_dist [um]")

# Define the output CSV file path (it will be saved in the same folder)
output_path = os.path.join(folder, "avg_F_episodes.csv")

# Save the result to CSV
result_df.to_csv(output_path, index=False)
print(f"Created file: {output_path}")
