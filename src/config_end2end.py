import os
import time
import logging
import jax
import jax.numpy as jnp
from control_hardware_model import (
    calculate_Omega_rabi_prefactor,
    generate_grid,
    generate_atom_positions_equilateral,
    generate_dipoles,
    generate_distances,
    generate_coupling_lengths,
    generate_n_eff_list
)
from quantum_system_model import program_instruction

# Define a results directory for end-to-end RL results.
RESULTS_DIR = "results_end2end_rl"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set up logging to write to a file inside RESULTS_DIR.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_file = os.path.join(RESULTS_DIR, "training_log.txt")
# Remove any pre-existing handlers.
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def get_config(simple=False):
    # APIC parameters
    APIC_params = {
        'L': 600 * 780e-9 / 1.95,
        'n0': 1.95,
        'lambda_0': 780e-9,
        'a0': 0.998,
        't0': 0.998,
        'a1': 0.998,
        't1': 0.998,
        'phase_mod': jnp.zeros(6),
        'amp_mod': jnp.ones(6)
    }
    # Generate grid and atom parameters
    X, Y = generate_grid(grid_size=600, grid_range=(-6, 6))
    atom_positions = generate_atom_positions_equilateral(3, side_length=3.0, center=(0, 0))
    dipoles = generate_dipoles(3)
    beam_centers = atom_positions
    beam_waist = 2.0
    Omega_prefactor_MHz = calculate_Omega_rabi_prefactor(20, 1000)
    atom_beam_params = {
        'atom_positions': atom_positions,
        'dipoles': dipoles,
        'beam_centers': beam_centers,
        'beam_waist': beam_waist,
        'X': X,
        'Y': Y,
        'Omega_prefactor_MHz': Omega_prefactor_MHz,
    }
    # Control voltage parameters
    t_steps = 10
    tmax_fixed = 0.1
    dt = tmax_fixed / t_steps
    control_Vt_params = {
        'tmin': 0.0,
        'tmax': tmax_fixed,
        't_steps': t_steps,
        'dt': dt,
        'min_V_level': -15.0,
        'max_V_level': 15.0
    }
    # System parameters
    a_pic = jnp.array([1.0] * 6)
    a_scat_1 = jnp.array([1.0] * 4)
    system_params = {
        'a_pic': a_pic,
        'a_scat_1': a_scat_1,
        'delta': 0.001,
        'N_ch': 6,
        'distances': generate_distances(6, 1.0, random_variation=0.1, seed=42),
        'coupling_lengths': generate_coupling_lengths(6, 600.0, scaling_factor=1.1, random_variation=60.0, seed=42),
        'n_eff_list': generate_n_eff_list(6, 1.75, random_variation=0.05, seed=42),
        'kappa0': 10.145,
        'alpha': 6.934,
        'enable_crosstalk': True,
        'N_slm': 6,
        'N_ch_slm_in': 4,
        'N_scat_1': 4,
        'N_scat_2': 4,
        'N_a': 3,
        'N_qubit_level': 2,
        'omega_0': 6.835e3,
        'omega_r': 6.835e3,
        'a_pic': a_pic,
        'a_scat_1': a_scat_1,
        'gate_type': 'single'
    }
    selected_atoms = [0, 1, 2]
    system_params['U_target'] = program_instruction(
        N_a=3,
        key_number=105,
        gate_type='single',
        selected_atoms=selected_atoms
    )
    if system_params['U_target'] is None:
        raise ValueError("U_target was not correctly assigned by program_instruction.")
    # Reward scaling parameters
    reward_scaling_params = {
        'a': 1.0,
        'b': 1.0,
        'scaling_methods': [
            {'method': 'log', 'min_ratio': 0.0, 'max_ratio': 0.5},
            {'method': 'linear', 'min_ratio': 0.5, 'max_ratio': 0.9},
            {'method': 'quadratic', 'min_ratio': 0.9, 'max_ratio': 1.0},
        ],
        'k': 5.0,
        'thresholds': [0.8, 0.9],
        'bonus': 5.0,
        'clip_min': -10.0,
        'clip_max': 10.0,
        'step_penalty': 0.01
    }
    # Final configuration dictionary
    config = {
        't_steps': t_steps,
        'N_ch': 6,
        'N_a': 3,
        'piecewise_segments': 10,
        'min_voltage': -15.0,
        'max_voltage': 15.0,
        'history_length': 5,
        'max_delta_voltage': 1.0,
        'APIC_params': APIC_params,
        'system_params': system_params,
        'atom_beam_params': atom_beam_params,
        'control_Vt_params': control_Vt_params,
        'reward_scaling_params': reward_scaling_params,
        'stagnant_threshold': 10,
        'stagnant_fidelity_min': 0.99,
        'target_fidelity': 0.999
    }
    return config

def run_end2end():
    start_time = time.perf_counter()
    
    # Use the dedicated results directory.
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("Starting end-to-end RL training...")
    
    # Configure JAX settings.
    jax.config.update("jax_platform_name", "gpu")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Get configuration.
    config = get_config()
    config["batch_size"] = 4
    config["results_dir"] = results_dir
    
    training_phases = [
        {"t_steps": 20, "piecewise_segments": 10, "num_iterations": 500},
        {"t_steps": 50, "piecewise_segments": 25, "num_iterations": 500},
        {"t_steps": 100, "piecewise_segments": 50, "num_iterations": 1000},
    ]
    
    # Import training routines from the core module.
    from end2end_rl import MLPPolicy, train_end_to_end_grad, load_trained_model
    
    policy = None
    rng_key = jax.random.PRNGKey(0)
    for phase_idx, phase in enumerate(training_phases, 1):
        t_steps = int(phase["t_steps"])
        piecewise_segments = int(phase["piecewise_segments"])
        num_iterations = phase["num_iterations"]
        config["control_Vt_params"]["t_steps"] = t_steps
        config["control_Vt_params"]["dt"] = config["control_Vt_params"]["tmax"] / t_steps
        logger.info(f"Starting Training Phase {phase_idx}: t_steps={t_steps}, piecewise_segments={piecewise_segments}, num_iterations={num_iterations}")
        if phase_idx == 1:
            policy = MLPPolicy(rng_key, piecewise_segments=piecewise_segments)
        else:
            rng_key, subkey = jax.random.split(rng_key)
            policy.update_piecewise_segments(piecewise_segments, subkey)
        trained_params = train_end_to_end_grad(
            policy=policy,
            config=config,
            num_iterations=num_iterations,
            initial_lr=1e-3,
            patience=200,
            min_lr=1e-6,
            lr_decay_factor=0.5,
            use_mixed_precision=True,
            use_checkpointing=True
        )
        policy.params = trained_params
    
    logger.info("Adaptive Training with Increasing t_steps and piecewise_segments Completed.")
    
    trained_model_path = os.path.join(config["results_dir"], "trained_model_params.pkl")
    if os.path.exists(trained_model_path):
        loaded_params = load_trained_model(trained_model_path)
        if loaded_params is not None:
            logger.info(f"Loaded trained model parameters from '{trained_model_path}'.")
            policy.params = loaded_params
            new_rng_key = jax.random.PRNGKey(123)
            optimal_Vt = policy.forward(loaded_params, new_rng_key)
            optimal_Vt_path = os.path.join(config["results_dir"], "optimal_Vt_new_seed.npy")
            import numpy as np
            np.save(optimal_Vt_path, optimal_Vt)
            logger.info(f"Optimal V(t) for new seed saved to '{optimal_Vt_path}'.")
        else:
            logger.error(f"Failed to load trained model parameters from '{trained_model_path}'.")
    else:
        logger.error(f"Trained model parameters file '{trained_model_path}' not found.")
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    logger.info(f"Adaptive training completed in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    run_end2end()
