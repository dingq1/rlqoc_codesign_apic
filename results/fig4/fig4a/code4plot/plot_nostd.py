import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load a dataset and ensure the x-axis column is named "Episodes"
def load_plot_data(filename):
    df = pd.read_csv(filename)
    if "Step" in df.columns:
        df.rename(columns={"Step": "Episodes"}, inplace=True)
    return df

# Number of runs (assumed to be 5 for each dataset)
num_runs = 5

# ---------------------
# Load datasets for N_a=2
# ---------------------
df_end2end_Na2   = load_plot_data("plot_data_end2end_Na2_1minusF.csv")
df_ppoRL_Na2     = load_plot_data("plot_data_ppoRL_Na2_1minusF.csv")
df_sade_adam_Na2 = load_plot_data("plot_data_sade_adam_Na2_1minusF.csv")

# ---------------------
# Load datasets for N_a=3
# ---------------------
df_end2end_Na3   = load_plot_data("plot_data_end2end_Na3_1minusF.csv")
df_ppoRL_Na3     = load_plot_data("plot_data_ppoRL_Na3_1minusF.csv")
df_sade_adam_Na3 = load_plot_data("plot_data_sade_adam_Na3_1minusF.csv")

# Compute SEM from the stored standard deviation (for (1-F) space)
def compute_sem(df):
    return df["Std (1-F)"] / np.sqrt(num_runs)

# Create a figure with 2 subplots (side-by-side) sharing the y-axis
fig, axs = plt.subplots(1, 2, figsize=(25, 7), sharey=True)

# ===================================================================================
# Left subplot: N_a = 2
# ===================================================================================
ax = axs[0]

# Plot in desired order: first SADE-Adam, then PPO-RL, then End-to-End-RL

# SADE-Adam curve for Na=2
# sem_sade = compute_sem(df_sade_adam_Na2)
ax.plot(df_sade_adam_Na2["Episodes"], df_sade_adam_Na2["Mean (1-F)"],
        label="SADE-Adam", color="red")
# ax.fill_between(df_sade_adam_Na2["Episodes"],
#                 df_sade_adam_Na2["Mean (1-F)"] - sem_sade,
#                 df_sade_adam_Na2["Mean (1-F)"] + sem_sade,
#                 color="red", alpha=0.3)

# PPO-RL curve for Na=2
# sem_ppoRL = compute_sem(df_ppoRL_Na2)
ax.plot(df_ppoRL_Na2["Episodes"], df_ppoRL_Na2["Mean (1-F)"],
        label="PPO-RL", color="green")
# ax.fill_between(df_ppoRL_Na2["Episodes"],
#                 df_ppoRL_Na2["Mean (1-F)"] - sem_ppoRL,
#                 df_ppoRL_Na2["Mean (1-F)"] + sem_ppoRL,
#                 color="green", alpha=0.3)

# End-to-End-RL curve for Na=2
# sem_end2end = compute_sem(df_end2end_Na2)
ax.plot(df_end2end_Na2["Episodes"], df_end2end_Na2["Mean (1-F)"],
        label="End-to-End-RL", color="blue")
# ax.fill_between(df_end2end_Na2["Episodes"],
#                 df_end2end_Na2["Mean (1-F)"] - sem_end2end,
#                 df_end2end_Na2["Mean (1-F)"] + sem_end2end,
#                 color="blue", alpha=0.3)

# Annotate final values for each curve (display mean ± SEM)
# SADE-Adam final annotation
final_x = df_sade_adam_Na2["Episodes"].iloc[-1]
final_mean = df_sade_adam_Na2["Mean (1-F)"].iloc[-1]
# final_sem = sem_sade.iloc[-1]
ax.annotate(f"{final_mean:.3e}", xy=(final_x, final_mean), 
            xycoords='data', xytext=(-50, -50), textcoords='offset points', 
            fontsize=23, color="red")
# PPO-RL final annotation
final_x = df_ppoRL_Na2["Episodes"].iloc[-1]
final_mean = df_ppoRL_Na2["Mean (1-F)"].iloc[-1]
# final_sem = sem_ppoRL.iloc[-1]
ax.annotate(f"{final_mean:.3e}", xy=(final_x, final_mean), 
            xycoords='data', xytext=(-240, -40), textcoords='offset points', 
            fontsize=23, color="green")
# End-to-End-RL final annotation
final_x = df_end2end_Na2["Episodes"].iloc[-1]
final_mean = df_end2end_Na2["Mean (1-F)"].iloc[-1]
# final_sem = sem_end2end.iloc[-1]
ax.annotate(f"{final_mean:.3e}", xy=(final_x, final_mean), 
            xycoords='data', xytext=(-20, -30), textcoords='offset points', 
            fontsize=23, color="blue")

ax.set_title("$N_g = 2$", fontsize=30)
ax.set_xlabel("Training Episodes", fontsize=30)
ax.set_ylabel("Gate Error", fontsize=30)
ax.tick_params(axis="both", which="major", labelsize=30, width=2, length=6)
ax.tick_params(axis="both", which="minor", labelsize=24, width=1.5, length=4)
ax.legend(fontsize=26, loc='lower left')
ax.grid(False)
ax.set_xscale("log")   # X-axis log scale
ax.set_yscale("log")   # Y-axis log scale
ax.set_ylim(1e-4, 1)    # Y-axis range: from 10^-4 to 10^0

# ===================================================================================
# Right subplot: N_a = 3
# ===================================================================================
ax = axs[1]

# Plot in desired order: first SADE-Adam, then PPO-RL, then End-to-End-RL

# SADE-Adam curve for Na=3
# sem_sade = compute_sem(df_sade_adam_Na3)
ax.plot(df_sade_adam_Na3["Episodes"], df_sade_adam_Na3["Mean (1-F)"],
        label="SADE-Adam", color="red")
# ax.fill_between(df_sade_adam_Na3["Episodes"],
#                 df_sade_adam_Na3["Mean (1-F)"] - sem_sade,
#                 df_sade_adam_Na3["Mean (1-F)"] + sem_sade,
#                 color="red", alpha=0.3)

# PPO-RL curve for Na=3
# sem_ppoRL = compute_sem(df_ppoRL_Na3)
ax.plot(df_ppoRL_Na3["Episodes"], df_ppoRL_Na3["Mean (1-F)"],
        label="PPO-RL", color="green")
# ax.fill_between(df_ppoRL_Na3["Episodes"],
#                 df_ppoRL_Na3["Mean (1-F)"] - sem_ppoRL,
#                 df_ppoRL_Na3["Mean (1-F)"] + sem_ppoRL,
#                 color="green", alpha=0.3)

# End-to-End-RL curve for Na=3
# sem_end2end = compute_sem(df_end2end_Na3)
ax.plot(df_end2end_Na3["Episodes"], df_end2end_Na3["Mean (1-F)"],
        label="End-to-End-RL", color="blue")
# ax.fill_between(df_end2end_Na3["Episodes"],
#                 df_end2end_Na3["Mean (1-F)"] - sem_end2end,
#                 df_end2end_Na3["Mean (1-F)"] + sem_end2end,
#                 color="blue", alpha=0.3)

# Annotate final values for each curve in N_a = 3 subplot
# SADE-Adam final annotation
final_x = df_sade_adam_Na3["Episodes"].iloc[-1]
final_mean = df_sade_adam_Na3["Mean (1-F)"].iloc[-1]
# final_sem = sem_sade.iloc[-1]
ax.annotate(f"{final_mean:.3e}", xy=(final_x, final_mean), 
            xycoords='data', xytext=(15, 25), textcoords='offset points', 
            fontsize=23, color="red")
# PPO-RL final annotation
final_x = df_ppoRL_Na3["Episodes"].iloc[-1]
final_mean = df_ppoRL_Na3["Mean (1-F)"].iloc[-1]
# final_sem = sem_ppoRL.iloc[-1]
ax.annotate(f"{final_mean:.3e}", xy=(final_x, final_mean), 
            xycoords='data', xytext=(-240, -40), textcoords='offset points', 
            fontsize=23, color="green")
# End-to-End-RL final annotation
final_x = df_end2end_Na3["Episodes"].iloc[-1]
final_mean = df_end2end_Na3["Mean (1-F)"].iloc[-1]
# final_sem = sem_end2end.iloc[-1]
ax.annotate(f"{final_mean:.3e}", xy=(final_x, final_mean), 
            xycoords='data', xytext=(-30, -40), textcoords='offset points', 
            fontsize=23, color="blue")

ax.set_title("$N_g = 3$", fontsize=30)
ax.set_xlabel("Episodes (log scale)", fontsize=30)
ax.tick_params(axis="both", which="major", labelsize=30, width=2, length=6)
ax.tick_params(axis="both", which="minor", labelsize=24, width=1.5, length=4)
ax.legend(fontsize=26, loc='lower left')
ax.grid(False)
ax.set_xscale("log")  # X-axis log scale
ax.set_yscale("log")  # Y-axis log scale
ax.set_ylim(1e-4, 1)   # Set y-axis range from 10^-4 to 10^0

plt.tight_layout()
plt.savefig("learning_curves_Na2_Na3_1minusF_log.png", dpi=300, bbox_inches="tight")