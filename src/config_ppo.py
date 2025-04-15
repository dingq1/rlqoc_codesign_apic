import os
import time
import logging
import jax
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Define the logging configuration function first.
def configure_logging(log_file_path):
    # Ensure the directory exists.
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove any existing handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

# Create the logger.
logger = configure_logging(os.path.join("results", "training_log.txt"))

def create_config(simple=False):
    import jax.numpy as jnp
    from control_hardware_model import (
        generate_grid,
        generate_atom_positions_equilateral,
        generate_dipoles,
        calculate_Omega_rabi_prefactor,
        generate_distances,
        generate_coupling_lengths,
        generate_n_eff_list
    )
    from quantum_system_model import program_instruction
    X, Y = generate_grid(grid_size=600, grid_range=(-6, 6))
    atom_positions = generate_atom_positions_equilateral(3, side_length=3.0, center=(0, 0))
    dipoles = generate_dipoles(3)
    beam_centers = atom_positions
    beam_waist = 2.0
    Omega_prefactor_MHz = calculate_Omega_rabi_prefactor(20, 1000)
    a_pic = jnp.array([1.0] * 6)
    a_scat_1 = jnp.array([1.0] * 4)
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
    system_params['U_target'] = program_instruction(N_a=3, key_number=12, gate_type='single', selected_atoms=selected_atoms)
    atom_beam_params = {
        'atom_positions': atom_positions,
        'dipoles': dipoles,
        'beam_centers': beam_centers,
        'beam_waist': beam_waist,
        'X': X,
        'Y': Y,
        'Omega_prefactor_MHz': Omega_prefactor_MHz,
    }
    control_Vt_params = {
        'tmin': 0.0,
        'tmax': 0.1,
        't_steps': 100,
        'dt': 0.1 / 100,
        'min_V_level': -15.0,
        'max_V_level': 15.0
    }
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
    config = {
        't_steps': 100,
        'N_ch': 6,
        'N_a': 3,
        'piecewise_segments': 50,
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

def run_ppo():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    logger.info("Commencing training...")
    start_time = time.time()
    num_envs = 4
    jax.config.update("jax_platform_name", "cpu")
    os.environ["JAX_NUM_THREADS"] = str(max(1, os.cpu_count() // num_envs))
    jax.config.update("jax_enable_x64", True)

    from ppo_rl import make_env, QOCEnv, CustomActorCriticPolicy, RLLoggingCallback
    config = create_config()
    env_fns = [make_env(config, seed=i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO(
        CustomActorCriticPolicy,
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(results_dir, "ppo_qoc_tensorboard/"),
        device='cpu',
        ent_coef=0.1,
        learning_rate=1e-4,
        clip_range=0.2,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=64,
        max_grad_norm=0.5
    )

    rl_logging_callback = RLLoggingCallback(
        stagnant_threshold=config.get('stagnant_threshold', 100),
        stagnant_fidelity_min=config.get('stagnant_fidelity_min', 0.99),
        target_fidelity=config.get('target_fidelity', 0.999),
        results_dir=results_dir
    )

    total_timesteps = 10000
    model.learn(total_timesteps=total_timesteps, callback=rl_logging_callback)
    total_run_time = time.time() - start_time
    logger.info(f"Total run time: {total_run_time/60:.2f} minutes")

    model_save_path = os.path.join(results_dir, "ppo_qoc_model.zip")
    model.save(model_save_path)
    logger.info(f"Trained PPO model saved as '{model_save_path}'.")

    test_env = DummyVecEnv([make_env(config, seed=100)])
    test_model = PPO.load(model_save_path)
    test_results_dir = os.path.join(results_dir, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)
    # Adjust reset and step unpacking based on your environment's API.
    obs = test_env.reset()  # Assuming reset returns a single value.
    for episode in range(1, 6):
        done = False
        total_reward = 0.0
        while not done:
            action, _ = test_model.predict(obs, deterministic=True)
            # If step() returns 4 values: (obs, reward, done, info)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
        logger.info(f"Test Episode {episode}: Final Fidelity = {total_reward:.4f}")
        obs = test_env.reset()
    vec_env.close()
    test_env.close()

if __name__ == "__main__":
    run_ppo()

