import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files
csv_files = [
    "fidelity_history_seed46.csv",
    "fidelity_history_seed938.csv",
    "fidelity_history_seed12.csv",
    "fidelity_history_seed122.csv",
    "fidelity_history_seed888.csv",
]

# Initialize figure
plt.figure(figsize=(7, 3))

# Loop over files and plot them
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)  # Read CSV file

    # Compute 1 - Fidelity
    df["1 - Fidelity"] = 1 - df["Best Fidelity"]

    # Plot with log-log scale
    plt.plot(df["Episodes"], df["1 - Fidelity"], label=f"Set {i+1}")

# Formatting the plot
plt.xscale("log")  # Log scale for x-axis
plt.yscale("log")  # Log scale for y-axis
plt.xlabel("Training Episodes", fontsize=16)
plt.ylabel("Gate Error", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)

# Save the figure
plt.savefig("error_vs_episodes_all_sets_loglog.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'error_vs_episodes_all_sets_loglog.png'")
