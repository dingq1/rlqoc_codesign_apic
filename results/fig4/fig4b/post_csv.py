import os
import re
import pandas as pd

# Define the folder where the collected CSV files are stored
folder = "fidelity_history_files"

# This regex pattern will extract the key number and the wg_dist part from filenames like:
# "fidelity_history_key_number_X_wg_dist_M.csv"
pattern = r"fidelity_history_key_number_(\d+)_((wg_dist_[^\.]+))\.csv"

# Dictionary to store final data for each wg_dist type
# Keys will be e.g., "wg_dist_0p25um", "wg_dist_0p5um", etc.
# Values will be a list of dictionaries with keys: key_number, final_episode, final_best_fidelity
groups = {}

# Iterate over each file in the folder
for filename in os.listdir(folder):
    # We only want CSV files that match the naming pattern we expect.
    if filename.endswith(".csv") and filename.startswith("fidelity_history_key_number_"):
        match = re.search(pattern, filename)
        if match:
            key_number = match.group(1)         # e.g., "46", "938", etc.
            wg_dist = match.group(2)            # e.g., "wg_dist_0p25um"
            
            file_path = os.path.join(folder, filename)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                # Get the last row (final episode info)
                last_row = df.iloc[-1]
                final_episode = last_row["Episodes"]
                final_best_fidelity = last_row["Best Fidelity"]
                
                # Initialize group list if not already present
                if wg_dist not in groups:
                    groups[wg_dist] = []
                
                # Append a record for this file
                groups[wg_dist].append({
                    "key_number": key_number,
                    "final_episode": final_episode,
                    "final_best_fidelity": final_best_fidelity
                })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

# For each wg_dist group, write a new CSV file collecting the final episode and fidelity info
for wg_dist, records in groups.items():
    # Optionally, sort records by key_number (converted to int for proper numerical order)
    records_sorted = sorted(records, key=lambda x: int(x["key_number"]))
    df_group = pd.DataFrame(records_sorted)
    
    # Create the output filename. Example: "best_F_episodes_wg_dist_0p25um.csv"
    output_filename = f"best_F_episodes_{wg_dist}.csv"
    output_path = os.path.join(folder, output_filename)
    
    # Write the collected records to the new CSV file
    df_group.to_csv(output_path, index=False)
    print(f"Created file: {output_path}")
