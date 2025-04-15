import os
import pandas as pd
import matplotlib.pyplot as plt

# Initialize figure
plt.figure(figsize=(15, 4))

# Loop over N = 1 to 5
for N in range(1, 6):
    folder_name = f"../sade_adam_targetXII_set{N}"
    file_path = os.path.join(folder_name, "fidelity_over_generations_multi_qubit_withCT_3sQ_XII.csv")

    # Check if file exists
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Plot the fidelity vs step
        plt.plot(df["Step"], 1-df["Fidelity"], label=f"Set {N}")

    else:
        print(f"Warning: {file_path} not found.")

# Formatting the plot
plt.xlabel("Optimization Steps", fontsize=30)
plt.ylabel("Gate Error", fontsize=30)
#plt.title("Fidelity vs Step for Different Sets", fontsize=16)

# Increase tick font size
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.xscale("log")  # Log scale for x-axis
plt.yscale("log")  # Log scale for y-axis

# Increase legend font size
plt.legend(fontsize=20)

# Remove grid
plt.grid(False)

# Save the figure
plt.savefig("fidelity_vs_steps.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'fidelity_vs_steps.png'")
