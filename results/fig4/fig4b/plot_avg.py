import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Read the CSV file containing the averaged data
df = pd.read_csv("fidelity_history_files/avg_F_episodes.csv")

# Extract the data columns
x = df["wg_dist [um]"]              # channel pitch in um
y1 = 1 - df["avg_final_best_F"]       # avg. gate error (left y-axis)
y2 = df["avg_final_episode"]          # avg. training episodes (right y-axis)

# Define individual annotation offsets for left y-axis (gate error)
# Keys are indices of data points, values are (x_offset, y_offset)
left_offsets = {
    0: (12, -22),
    1: (16, 12),
    2: (16, -22),
    3: (2, 8),
    4: (-10, -22),
}
default_left_offset = (2, 8)

# Define individual annotation offsets for right y-axis (training episodes)
right_offsets = {
    0: (14, 20),
    1: (0, -22),
    2: (14, 16),
    3: (2, 12),
    4: (-10, 12),
}
default_right_offset = (0, 8)

# Create a figure and primary axis
fig, ax1 = plt.subplots(figsize=(9, 4))

# Plot left y-axis data with blue circle markers only (no connecting line)
line1 = ax1.plot(x, y1, 'o', color='blue', markersize=8, label='avg. gate error')[0]
ax1.set_xlabel("Channel Pitch [um]", fontsize=22)
ax1.set_ylabel("Gate Error", fontsize=22, color='blue')
ax1.set_yscale("log")
ax1.tick_params(axis='x', labelsize=22)
ax1.tick_params(axis='y', labelcolor='blue', labelsize=22)

# Set x-axis range
ax1.set_xlim(0, 4.25)
# Remove major ticks and set minor ticks at intervals of 0.25
# ax1.xaxis.set_ticks([])  
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax1.tick_params(axis='x', which='major', labelsize=22)

# Set the y-axis limits for the left axis: lower bound 1e-4 and upper bound 1e-2.
ax1.set_ylim(1e-4, 1e-2)

# Annotate each left y-axis data point in scientific notation with individual offsets
for i, (xi, yi) in enumerate(zip(x, y1)):
    offset = left_offsets.get(i, default_left_offset)
    ax1.annotate(f"{yi:.2e}", (xi, yi), textcoords="offset points", xytext=offset,
                 ha='center', color='blue', fontsize=15)

# Create a secondary axis for the right y-axis data and plot red cross markers only
ax2 = ax1.twinx()
line2 = ax2.plot(x, y2, 'x', color='red', markersize=8, label='avg. training episodes')[0]
ax2.set_ylabel("Training Episodes", fontsize=22, color='red')
ax2.set_yscale("log")
ax2.tick_params(axis='y', labelcolor='red', labelsize=22)

# Set the y-axis limits for the right axis: lower bound 1e2 and upper bound 1e4.
ax2.set_ylim(1e2, 1e4)

# Annotate each right y-axis data point in scientific notation with individual offsets
for i, (xi, yi) in enumerate(zip(x, y2)):
    offset = right_offsets.get(i, default_right_offset)
    ax2.annotate(f"{yi:.2e}", (xi, yi), textcoords="offset points", xytext=offset,
                 ha='center', color='red', fontsize=15)

# Combine legends from both axes
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=16)

# Set a title and adjust layout
plt.tight_layout()

# Save the figure to a file instead of displaying it
plt.savefig("fidelity_history_files/plot_avg_gate_error_training_episodes.png", dpi=300)
print("Plot saved to fidelity_history_files/plot_avg_gate_error_training_episodes.png")