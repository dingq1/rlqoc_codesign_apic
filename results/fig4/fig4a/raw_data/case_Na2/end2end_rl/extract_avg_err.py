import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of CSV files for the 5 different target gates (Sets) for end‑2‑end RL
csv_files = [
    "fidelity_history_seed141.csv",
    "fidelity_history_seed521.csv",
    "fidelity_history_seed249.csv",
    "fidelity_history_seed36.csv",
    "fidelity_history_seed782.csv"
]

# Read each CSV file into a DataFrame
dfs = [pd.read_csv(file) for file in csv_files]

# Compute the maximum episode from each run (for episodes statistics)
episode_max_list = [df["Episodes"].max() for df in dfs]
avg_episodes_per_run = np.mean(episode_max_list)
max_episodes_per_run = np.max(episode_max_list)
min_episodes_per_run = np.min(episode_max_list)

# Determine the common episode range using the maximum of the maximum episodes across all files
max_max_episode = max(episode_max_list)
common_episodes = np.arange(0, int(max_max_episode) + 1)  # integer steps from 0 to max_max_episode

# Interpolate fidelity values from each DataFrame onto the common episode grid,
# then compute (1-F) for each run.
all_one_minus_f = []
for df in dfs:
    df_sorted = df.sort_values("Episodes")
    x = df_sorted["Episodes"].values
    F = df_sorted["Best Fidelity"].values
    # Interpolate fidelity onto the common grid
    F_interp = np.interp(common_episodes, x, F)
    one_minus_F = 1.0 - F_interp
    # Optionally, clamp to a small value if needed (to avoid log issues later)
    # one_minus_F = np.clip(one_minus_F, 1e-12, None)
    all_one_minus_f.append(one_minus_F)

# Convert list to a NumPy array: shape (number of files, number of common episodes)
all_one_minus_f = np.array(all_one_minus_f)

# Compute the mean and standard deviation in the (1-F) space
mean_1mF = np.mean(all_one_minus_f, axis=0)
std_1mF  = np.std(all_one_minus_f, axis=0)

# Save the dataset used for the plot into a CSV file
plot_data = pd.DataFrame({
    "Episodes": common_episodes,
    "Mean (1-F)": mean_1mF,
    "Std (1-F)": std_1mF
})
plot_data.to_csv("plot_data_end2end_Na2_1minusF.csv", index=False)
print("Plot dataset saved as 'plot_data_end2end_Na2_1minusF.csv'")
