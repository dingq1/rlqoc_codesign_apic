import os
import time
import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from jax import random

# Import functions from the other modules
from control_hardware_model import (
    calculate_Omega_rabi_prefactor, generate_grid, generate_atom_positions_equilateral,
    generate_dipoles, generate_slm_mod, generate_distances, generate_coupling_lengths, generate_n_eff_list,
    construct_V_smooth_with_carrier
)
from quantum_system_model import program_instruction
from sade_adam import optimize_multi_qubit_sade_adam

# --- Results Directory Setup ---
RESULTS_DIR = "results_sade_adam"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Logging Configuration ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Remove any existing handlers to avoid duplicate logs.
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
# Log file is now created inside the results folder.
log_file_path = os.path.join(RESULTS_DIR, 'simulation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def run_sadeadam():
    """
    Executes the multi-qubit gate optimization using the SADE+Adam algorithm.
    All results, log files, CSV outputs, and plots are saved in RESULTS_DIR.
    """
    logger.info(" --- Initial Setups --- ")

    # Laser intensity and detuning setup
    I_mW_per_cm2 = 20
    Detuning_MHz = 1000
    Omega_prefactor_MHz = calculate_Omega_rabi_prefactor(I_mW_per_cm2, Detuning_MHz)
    logger.info(f"Omega_prefactor_MHz: {Omega_prefactor_MHz} \n")

    # Control system parameters
    m, n0, lambda_0 = 600, 1.95, 780e-9
    L = m * lambda_0 / n0
    a0, t0, a1, t1 = 0.998, 0.998, 0.998, 0.998

    # Define frequencies in MHz
    omega_0, omega_r = 6.835e3, 6.835e3

    # Time parameters [us]
    tmin = 0
    tmax_fixed = 0.1
    t_steps = 100
    dt = (tmax_fixed - tmin) / t_steps
    logger.info(f"Chosen dt: {dt * 1e3:.2f} ns")

    # System and channel setup
    N_ch, N_scat_1, N_slm, N_ch_slm_in = 6, 4, 6, 4
    N_total = N_ch + N_scat_1
    N_scat_2, N_a = N_total - N_slm, 3

    selected_atoms = [0, 1, 2]

    # Crosstalk flag and related parameters
    enable_crosstalk = True

    if enable_crosstalk:
        base_distance = 1.0
        distance_variation = 0.1
        distances = generate_distances(N_ch, base_distance, random_variation=distance_variation, seed=42)
        base_coupling_length = 600.0
        length_variation = 60.0
        coupling_lengths = generate_coupling_lengths(N_ch, base_coupling_length, scaling_factor=1.1,
                                                      random_variation=length_variation, seed=42)
        base_n_eff = 1.75
        n_eff_variation = 0.05
        n_eff_list = generate_n_eff_list(N_ch, base_n_eff, random_variation=n_eff_variation, seed=42)
        kappa0 = 10.145
        alpha = 6.934
        logger.info("Crosstalk parameters initialized:")
        logger.info(f"Distances: {distances}")
        logger.info(f"Coupling lengths: {coupling_lengths}")
        logger.info(f"Effective refractive indices: {n_eff_list}")
        logger.info(f"kappa0: {kappa0}, alpha: {alpha}")
    else:
        logger.info("Crosstalk is disabled.")
        distances = None
        coupling_lengths = None
        n_eff_list = None
        kappa0 = None
        alpha = None

    # 2D atomic grid and atom positions
    atom_spacing = 3.0
    X, Y = generate_grid(grid_size=600, grid_range=(-6, 6))
    atom_positions = generate_atom_positions_equilateral(N_a, side_length=atom_spacing, center=(0, 0))
    logger.info(f"Atom positions: {atom_positions}, in [um]")

    # Beam and dipole parameters
    beam_centers, beam_waist = atom_positions, 2.0
    logger.info(f"Beam centers: {beam_centers}, Beam waist: {beam_waist} in [um]")
    dipoles = generate_dipoles(N_a)

    # SLM modulation parameters
    phase_mod, amp_mod = generate_slm_mod(N_slm)

    # Weak scattering parameter
    delta = 0.001

    # Control signal limits
    min_V_level, max_V_level = -15, 15

    # Initial fields for control signals
    a_pic = jnp.array([1.0] * N_ch)
    a_scat_1 = jnp.array([1.0] * N_scat_1)

    # Initialize control signals V0_t_list and V1_t_list
    num_pieces = t_steps
    V0_t_list, V1_t_list = [], []
    key = random.PRNGKey(256)

    for _ in range(N_ch):
        key, subkey = random.split(key)
        voltage_levels_V0 = random.uniform(subkey, shape=(num_pieces,), minval=-15.0, maxval=15.0)
        V0_t = construct_V_smooth_with_carrier(tmin, tmax_fixed, t_steps, voltage_levels_V0, omega_0)
        V0_t_list.append(V0_t)
        key, subkey = random.split(key)
        voltage_levels_V1 = random.uniform(subkey, shape=(num_pieces,), minval=-15.0, maxval=15.0)
        V1_t = construct_V_smooth_with_carrier(tmin, tmax_fixed, t_steps, voltage_levels_V1, omega_0)
        V1_t_list.append(V1_t)

    V0_t_list = jnp.array(V0_t_list)
    V1_t_list = jnp.array(V1_t_list)

    logger.info(f"V0_t_list shape: {V0_t_list.shape}")
    logger.info(f"V1_t_list shape: {V1_t_list.shape}")

    # Save initial control signals to CSV files in RESULTS_DIR.
    pd.DataFrame(np.array(V0_t_list)).to_csv(os.path.join(RESULTS_DIR, 'V0_initial_multi_qubit_withCT_3sQ_dwg_0p6um.csv'),
                                               index_label='Channel')
    pd.DataFrame(np.array(V1_t_list)).to_csv(os.path.join(RESULTS_DIR, 'V1_initial_multi_qubit_withCT_3sQ_dwg_0p6um.csv'),
                                               index_label='Channel')

    # Generate target gate using program_instruction
    gate_type = 'single'
    if gate_type == 'single':
        U_target = program_instruction(N_a, key_number=91, gate_type=gate_type, selected_atoms=selected_atoms)

    logger.info(f"U_target: {U_target}")

    # Define parameter dictionaries
    APIC_params = {
        'L': L, 'n0': n0, 'lambda_0': lambda_0, 'a0': a0, 't0': t0,
        'a1': a1, 't1': t1, 'phase_mod': phase_mod, 'amp_mod': amp_mod
    }

    atom_beam_params = {
        'atom_positions': atom_positions, 'dipoles': dipoles,
        'a_pic': a_pic, 'a_scat_1': a_scat_1, 'beam_centers': beam_centers,
        'beam_waist': beam_waist, 'X': X, 'Y': Y, 'Omega_prefactor_MHz': Omega_prefactor_MHz,
    }

    control_Vt_params = {
        'tmin': tmin,
        'tmax': tmax_fixed,
        't_steps': t_steps,
        'dt': dt,
        'min_V_level': min_V_level,
        'max_V_level': max_V_level
    }

    system_params = {
        'N_ch': N_ch, 'distances': distances,
        'coupling_lengths': coupling_lengths,
        'n_eff_list': n_eff_list,
        'enable_crosstalk': enable_crosstalk,
        'kappa0': kappa0, 'alpha': alpha,
        'N_slm': N_slm, 'N_ch_slm_in': N_ch_slm_in, 'N_scat_1': N_scat_1,
        'N_scat_2': N_scat_2, 'N_a': N_a, 'N_qubit_level': 2, 'omega_0': omega_0,
        'omega_r': omega_r, 'U_target': U_target, 'gate_type': gate_type, 'delta': delta
    }

    optimizer_params = {
        'num_generations': 500,
        'popsize': 10,
        'adam_steps': 500,
        'adam_lr': 0.0001,
        'fidelity_threshold': 0.95,
        'max_no_improvement': 5,
        'fidelity_decay_factors': [0.5, 0.2, 0.5, 0.2],
        'min_lr': 1e-6,
        'fidelity_decay_thresholds': [0.98, 0.99, 0.995, 0.997],
        'tol': 1e-3,
        'stability_steps': 10,
        'tmax_tolerance': 1e-6
    }

    start_time = time.time()
    logger.info("\n--- Starting multi-qubit gate optimization (SADE+Adam) with CrossTalk... ---\n")

    best_solution, fidelity_gens = optimize_multi_qubit_sade_adam(
        APIC_params, atom_beam_params, control_Vt_params, system_params,
        optimizer_params, V0_t_list, V1_t_list
    )

    half_len = best_solution.shape[0] // 2
    V0_opt = best_solution[:half_len].reshape(N_ch, t_steps)
    V1_opt = best_solution[half_len:].reshape(N_ch, t_steps)
    tmax_opt = control_Vt_params['tmax']

    execution_time = (time.time() - start_time) / 60.0
    logger.info(f"Optimization completed in {execution_time:.2f} minutes.")
    logger.info(f"Fixed tmax: {tmax_opt:.6f} us")
    logger.info(f"Final optimized fidelity: {fidelity_gens[-1]:.10f}")

    pd.DataFrame(np.array(V0_opt)).to_csv(os.path.join(RESULTS_DIR, 'V0_optimal_multi_qubit_withCT_3sQ_dwg_0p6um.csv'),
                                          index_label='Channel')
    pd.DataFrame(np.array(V1_opt)).to_csv(os.path.join(RESULTS_DIR, 'V1_optimal_multi_qubit_withCT_3sQ_dwg_0p6um.csv'),
                                          index_label='Channel')

    steps = list(range(1, len(fidelity_gens) + 1))
    df = pd.DataFrame({
        'Step': steps,
        'Fidelity': fidelity_gens
    })
    csv_filename = os.path.join(RESULTS_DIR, "fidelity_over_generations_multi_qubit_withCT_1sQ_XII.csv")
    df.to_csv(csv_filename, index=False)
    logger.info(f"Data successfully saved to {csv_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['Fidelity'], label="Multi-Qubit Fidelity")
    plt.title("Fidelity over Generations (SaDE + Adam)")
    plt.xlabel("Step")
    plt.ylabel("Fidelity")
    plt.legend()
    plot_filename = os.path.join(RESULTS_DIR, "fidelity_over_generations_multi_qubit_withCT_3sQ_dwg_0p6um.png")
    plt.savefig(plot_filename)
    logger.info(f"Plot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    run_sadeadam()
