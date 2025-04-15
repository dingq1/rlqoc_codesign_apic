import os
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
folder_name = "../ct_3sQ_XII_manual_decay_adam_set1"
file_v0 = os.path.join(folder_name, "V0_optimal_multi_qubit_withCT_3sQ_dwg_0p6um.csv")
file_v1 = os.path.join(folder_name, "V1_optimal_multi_qubit_withCT_3sQ_dwg_0p6um.csv")

# Read CSV files
df_v0 = pd.read_csv(file_v0, index_col=0)
df_v1 = pd.read_csv(file_v1, index_col=0)

# Convert column names to integers for correct time steps
df_v0.columns = df_v0.columns.astype(int)
df_v1.columns = df_v1.columns.astype(int)

# Initialize subplots (3 rows, 1 column)
fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

# Define line styles
line_styles = ['-', '--']  # Solid for V0, Dashed for V1
colors = ['b', 'r', 'g']   # Colors for channels 0, 1, 2

# Loop over channels 0-2, plotting each in a separate subplot
for i, ch in enumerate([0, 1, 2]):
    ax = axes[i]
    ax.step(df_v0.columns, df_v0.loc[ch], linestyle=line_styles[0], color=colors[i], where='post', label=f"V0 - Channel {ch}")
    ax.step(df_v1.columns, df_v1.loc[ch], linestyle=line_styles[1], color=colors[i], where='post', label=f"V1 - Channel {ch}")

    # Formatting each subplot
    ax.set_ylabel("Voltage (V)", fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(False)  # Remove grid for clarity

# Set common x-axis label
axes[-1].set_xlabel("Time Step", fontsize=14)

# Set overall title
fig.suptitle("Piecewise Constant V(t) for Channels 0-2", fontsize=16)

# Adjust layout for clarity
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
plt.savefig("voltage_vs_time_piecewise_subplot.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'voltage_vs_time_piecewise_subplot.png'")
