import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of CSV files for case N_a=3, end-to-end RL (5 different target gates/sets)
csv_files = [
    "fidelity_history_seed105.csv",
    "fidelity_history_seed108.csv",
    "fidelity_history_seed109.csv",
    "fidelity_history_seed110.csv",
    "fidelity_history_seed113.csv"
]

# Read each CSV file into a DataFrame
dfs = [pd.read_csv(file) for file in csv_files]

# Compute the maximum episode from each run (for episodes statistics)
episode_max_list = [df["Episodes"].max() for df in dfs]
avg_episodes_per_run = np.mean(episode_max_list)
max_episodes_per_run = np.max(episode_max_list)
min_episodes_per_run = np.min(episode_max_list)

# Determine the common episode range using the maximum of the maximum episodes
max_max_episode = max(episode_max_list)
common_episodes = np.arange(0, int(max_max_episode) + 1)  # Common grid from 0 to max_max_episode

# For each run, interpolate fidelity values onto the common grid, then compute (1-F)
all_one_minus_f = []
for df in dfs:
    df_sorted = df.sort_values("Episodes")
    x = df_sorted["Episodes"].values
    F = df_sorted["Best Fidelity"].values
    F_interp = np.interp(common_episodes, x, F)
    one_minus_F = 1.0 - F_interp
    all_one_minus_f.append(one_minus_F)

# Convert list to a NumPy array: shape (num_runs, num_common_episodes)
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
plot_data.to_csv("plot_data_end2end_Na3_1minusF.csv", index=False)
print("Plot dataset saved as 'plot_data_end2end_Na3_1minusF.csv'")