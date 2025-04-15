# end2end_rl.py

import os
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import matplotlib.pyplot as plt
import logging
from stable_baselines3.common.callbacks import BaseCallback

from metrics import compute_multi_qubit_fidelity_closed_system

# --- RL Policy and Training Functions ---

class MLPPolicy:
    def __init__(self, rng_key, n_ch=6, piecewise_segments=10, hidden_dim=64, latent_dim=8):
        self.n_ch = n_ch
        self.piecewise_segments = piecewise_segments
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        rng_key, sub1, sub2 = jrandom.split(rng_key, 3)
        W1 = 0.01 * jrandom.normal(sub1, (latent_dim, hidden_dim))
        b1 = jnp.zeros((hidden_dim,), dtype=jnp.float32)
        W2 = 0.01 * jrandom.normal(sub2, (hidden_dim, n_ch * piecewise_segments))
        b2 = jnp.zeros((n_ch * piecewise_segments,), dtype=jnp.float32)
        self.params = dict(W1=W1, b1=b1, W2=W2, b2=b2)

    def forward(self, params, rng_key):
        z = jrandom.normal(rng_key, (self.latent_dim,)).astype(jnp.float32)
        h = jnp.tanh(z @ params["W1"] + params["b1"])
        out = h @ params["W2"] + params["b2"]
        return out.reshape((self.n_ch, self.piecewise_segments))

    def update_piecewise_segments(self, new_piecewise_segments, rng_key):
        if new_piecewise_segments <= self.piecewise_segments:
            logging.warning("New piecewise_segments must be greater than the current value.")
            return
        additional_segments = new_piecewise_segments - self.piecewise_segments
        rng_key, subkey1, subkey2 = jrandom.split(rng_key, 3)
        new_W2 = 0.01 * jrandom.normal(subkey1, (self.hidden_dim, self.n_ch * additional_segments))
        new_b2 = jnp.zeros((self.n_ch * additional_segments,), dtype=jnp.float32)
        updated_W2 = jnp.concatenate([self.params["W2"], new_W2], axis=1)
        updated_b2 = jnp.concatenate([self.params["b2"], new_b2], axis=0)
        self.params["W2"] = updated_W2
        self.params["b2"] = updated_b2
        self.piecewise_segments = new_piecewise_segments
        logging.info(f"Updated piecewise_segments to {new_piecewise_segments}.")

def single_seed_fidelity(params, rng_key, config, policy):
    voltages = policy.forward(params, rng_key)
    V0_t_list = voltages
    V1_t_list = voltages
    fidelity_all = compute_multi_qubit_fidelity_closed_system(
        V0_t_list, V1_t_list,
        config["APIC_params"]["L"],
        config["APIC_params"]["n0"],
        config["APIC_params"]["lambda_0"],
        config["APIC_params"]["a0"],
        config["APIC_params"]["t0"],
        config["APIC_params"]["a1"],
        config["APIC_params"]["t1"],
        config["APIC_params"]["phase_mod"],
        config["APIC_params"]["amp_mod"],
        config["system_params"]["delta"],
        config["atom_beam_params"]["atom_positions"],
        config["atom_beam_params"]["dipoles"],
        config["atom_beam_params"]["beam_centers"],
        config["atom_beam_params"]["beam_waist"],
        config["atom_beam_params"]["X"],
        config["atom_beam_params"]["Y"],
        config["atom_beam_params"]["Omega_prefactor_MHz"],
        config["control_Vt_params"]["t_steps"],
        config["control_Vt_params"]["dt"],
        config["system_params"]["N_ch"],
        config["system_params"]["distances"],
        config["system_params"]["coupling_lengths"],
        config["system_params"]["n_eff_list"],
        config["system_params"]["kappa0"],
        config["system_params"]["alpha"],
        config["system_params"]["enable_crosstalk"],
        config["system_params"]["N_slm"],
        config["system_params"]["N_ch_slm_in"],
        config["system_params"]["N_scat_1"],
        config["system_params"]["N_scat_2"],
        config["system_params"]["N_a"],
        config["system_params"]["N_qubit_level"],
        config["system_params"]["omega_0"],
        config["system_params"]["omega_r"],
        config["system_params"]["a_pic"],
        config["system_params"]["a_scat_1"],
        config["system_params"]["U_target"],
        config["system_params"].get("gate_type", "single"),
    )
    return fidelity_all[-1]

def build_loss_fn(policy, config):
    batch_size = config.get("batch_size", 2)
    def single_seed_loss(param, subkey):
        fid = single_seed_fidelity(param, subkey, config, policy)
        loss = jnp.where(fid <= 1.0, 1.0 - fid, (fid - 1.0) ** 2)
        return loss
    batched_seed_loss = jax.vmap(single_seed_loss, in_axes=(None, 0))
    def loss_fn(params, step_rng_key):
        rng_keys = jrandom.split(step_rng_key, batch_size)
        losses = batched_seed_loss(params, rng_keys)
        return jnp.mean(losses)
    return loss_fn

def train_end_to_end_grad(
    policy, config, num_iterations=500, initial_lr=1e-3, patience=50, min_lr=1e-6, lr_decay_factor=0.5,
    use_mixed_precision=True, use_checkpointing=True
):
    key = jrandom.PRNGKey(42)
    loss_fn = build_loss_fn(policy, config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(initial_lr)
    )
    opt_state = optimizer.init(policy.params)
    best_fidelity = 0.0
    best_voltage = None
    fidelity_history = []
    stagnant_counter = 0
    current_lr = initial_lr
    logging.info(f"[End-to-End Grad] Starting training for {num_iterations} iterations, initial_lr={initial_lr}, batch_size={config.get('batch_size', 4)}")
    if use_checkpointing:
        def loss_with_checkpoint(params, step_rng_key):
            return jax.checkpoint(loss_fn, prevent_cse=True)(params, step_rng_key)
        loss_fn_jit = jax.jit(loss_with_checkpoint)
    else:
        loss_fn_jit = jax.jit(loss_fn)
    for i in range(1, num_iterations + 1):
        key, subkey = jrandom.split(key)
        grads = jax.grad(loss_fn_jit)(policy.params, subkey)
        updates, opt_state = optimizer.update(grads, opt_state, policy.params)
        policy.params = optax.apply_updates(policy.params, updates)
        current_loss = loss_fn_jit(policy.params, subkey)
        fidelity = 1.0 - float(current_loss)
        if fidelity < 0.0 or fidelity > 1.0:
            continue
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_voltage = policy.forward(policy.params, subkey)
            stagnant_counter = 0
        else:
            stagnant_counter += 1
            logging.info(f"[End-to-End Grad] Iter={i}, no improvement in fidelity. Stagnant Counter: {stagnant_counter}/{patience}")
        fidelity_history.append(best_fidelity)
        if stagnant_counter >= patience:
            if current_lr > min_lr:
                current_lr = max(current_lr * lr_decay_factor, min_lr)
                optimizer = optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adam(current_lr)
                )
                opt_state = optimizer.init(policy.params)
                logging.info(f"[End-to-End Grad] Iter={i}, learning rate decayed to {current_lr}. Resetting stagnant counter.")
                stagnant_counter = 0
            else:
                logging.warning(f"[End-to-End Grad] Iter={i}, minimum learning rate reached ({min_lr}).")
        if (i % 10) == 0:
            logging.info(f"[End-to-End Grad] Iter={i}, loss={current_loss:.6f}, best_fidelity={best_fidelity:.6f}, current_lr={current_lr:.6f}")
            if best_fidelity > config.get("target_fidelity", 0.999):
                logging.info(f"[End-to-End Grad] Best fidelity > {config.get('target_fidelity', 0.999)} reached at iter={i}. Stopping early!")
                break
    fidelity_history_path_npy = os.path.join(config["results_dir"], "fidelity_history.npy")
    fidelity_history_path_csv = os.path.join(config["results_dir"], "fidelity_history.csv")
    np.save(fidelity_history_path_npy, np.array(fidelity_history))
    np.savetxt(fidelity_history_path_csv, np.array(fidelity_history), delimiter=",", header="Episode,Best_Fidelity")
    logging.info(f"Fidelity history saved to '{fidelity_history_path_npy}' and '{fidelity_history_path_csv}'.")
    _plot_fidelity_progress(fidelity_history, config["results_dir"])
    model_save_path = os.path.join(config.get('results_dir', 'results'), "trained_model_params.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(policy.params, f)
    logging.info(f"Trained model parameters saved to '{model_save_path}'.")
    return policy.params

def _plot_fidelity_progress(fidelity_history, results_dir):
    try:
        episodes = np.arange(1, len(fidelity_history) + 1)
        best_fidelity = np.array(fidelity_history)
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, best_fidelity, label='Best Fidelity')
        plt.title('Global Best Fidelity Progress Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Best Fidelity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fidelity_plot_path = os.path.join(results_dir, "fidelity_progress.png")
        plt.savefig(fidelity_plot_path)
        plt.close()
        logging.info(f"Fidelity progress plot saved as '{fidelity_plot_path}'.")
    except Exception as e:
        logging.error(f"Error while plotting or saving fidelity progress: {e}")

def load_trained_model(model_path):
    try:
        with open(model_path, "rb") as f:
            params = pickle.load(f)
        return params
    except Exception as e:
        logging.error(f"Failed to load trained model parameters from '{model_path}': {e}")
        return None

# --- Custom RL Logging Callback (for use with Stable-Baselines3) ---

class RLLoggingCallback(BaseCallback):
    def __init__(self, stagnant_threshold=100, stagnant_fidelity_min=0.99, target_fidelity=0.999, results_dir='results', verbose=0):
        super(RLLoggingCallback, self).__init__(verbose)
        self.best_fidelity = 0.0
        self.best_voltage = None
        self.best_fidelity_history = []
        self.episode_count = 0
        self.stagnant_counter = 0
        self.stagnant_threshold = stagnant_threshold
        self.stagnant_fidelity_min = stagnant_fidelity_min
        self.target_fidelity = target_fidelity
        self.start_time = time.perf_counter()
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        improved = False
        logging.info(f"[RLLoggingCallback] _on_step called. Infos count={len(infos)}. Current best fidelity={self.best_fidelity:.4f}")
        for env_id, info in enumerate(infos):
            if 'final_fidelity' in info:
                fidelity = info['final_fidelity']
                self.episode_count += 1
                if fidelity < 0.0 or fidelity > 1.0:
                    continue
                logging.info(f"[RLLoggingCallback] Env={env_id}, Episode={self.episode_count}, final_fidelity={fidelity:.6f}")
                if fidelity > self.best_fidelity:
                    self.best_fidelity = fidelity
                    self.stagnant_counter = 0
                    improved = True
                    new_best_voltage = info.get('best_voltage', None)
                    if new_best_voltage is not None:
                        self.best_voltage = new_best_voltage
                    logging.info(f"New best fidelity: {self.best_fidelity:.6f} achieved in Env {env_id} | Episode: {self.episode_count}")
        self.best_fidelity_history.append(self.best_fidelity)
        if not improved:
            if self.stagnant_fidelity_min <= self.best_fidelity < self.target_fidelity:
                self.stagnant_counter += 1
                logging.info(f"Stagnant Counter Incremented: {self.stagnant_counter}/{self.stagnant_threshold} | Best Fidelity: {self.best_fidelity:.6f}")
            else:
                self.stagnant_counter = 0
        if self.best_fidelity >= self.target_fidelity:
            elapsed_time = time.perf_counter() - self.start_time
            logging.info(f"Target fidelity of {self.best_fidelity:.6f} achieved in {self.episode_count} episodes. Training terminated after {elapsed_time:.2f} seconds.")
            if self.best_voltage is not None:
                best_voltage_final_path = os.path.join(self.results_dir, "best_voltage_final.npy")
                np.save(best_voltage_final_path, self.best_voltage)
                logging.info(f"Final best voltage saved to '{best_voltage_final_path}'.")
            best_fidelity_path = os.path.join(self.results_dir, "best_fidelity.txt")
            with open(best_fidelity_path, "w") as f:
                f.write(f"Best Fidelity: {self.best_fidelity:.6f}\n")
            logging.info(f"Best fidelity saved to '{best_fidelity_path}'.")
            best_fidelity_history_path_npy = os.path.join(self.results_dir, "best_fidelity_history.npy")
            best_fidelity_history_path_csv = os.path.join(self.results_dir, "best_fidelity_history.csv")
            np.save(best_fidelity_history_path_npy, np.array(self.best_fidelity_history))
            np.savetxt(best_fidelity_history_path_csv, np.array(self.best_fidelity_history), delimiter=",", header="Episode,Best_Fidelity")
            logging.info(f"Best fidelity history saved to '{best_fidelity_history_path_npy}' and '{best_fidelity_history_path_csv}'.")
            self._plot_fidelity_progress(self.best_fidelity_history, self.results_dir)
            return False
        if self.stagnant_counter >= self.stagnant_threshold:
            elapsed_time = time.perf_counter() - self.start_time
            logging.warning(f"Training is stagnating after {self.episode_count} episodes. Best Fidelity: {self.best_fidelity:.6f}. Terminating training after {elapsed_time:.2f} seconds.")
            if self.best_voltage is not None:
                best_voltage_final_path = os.path.join(self.results_dir, "best_voltage_final.npy")
                np.save(best_voltage_final_path, self.best_voltage)
                logging.info(f"Final best voltage saved to '{best_voltage_final_path}'.")
            best_fidelity_path = os.path.join(self.results_dir, "best_fidelity.txt")
            with open(best_fidelity_path, "w") as f:
                f.write(f"Best Fidelity: {self.best_fidelity:.6f}\n")
            logging.info(f"Best fidelity saved to '{best_fidelity_path}'.")
            best_fidelity_history_path_npy = os.path.join(self.results_dir, "best_fidelity_history.npy")
            best_fidelity_history_path_csv = os.path.join(self.results_dir, "best_fidelity_history.csv")
            np.save(best_fidelity_history_path_npy, np.array(self.best_fidelity_history))
            np.savetxt(best_fidelity_history_path_csv, np.array(self.best_fidelity_history), delimiter=",", header="Episode,Best_Fidelity")
            logging.info(f"Best fidelity history saved to '{best_fidelity_history_path_npy}' and '{best_fidelity_history_path_csv}'.")
            self._plot_fidelity_progress(self.best_fidelity_history, self.results_dir)
            return False
        elapsed_time = time.perf_counter() - self.start_time
        logging.info(f"Training Status | Episodes: {self.episode_count} | Best Fidelity: {self.best_fidelity:.6f} | Stagnant Counter: {self.stagnant_counter}/{self.stagnant_threshold} | Elapsed Time: {elapsed_time:.2f} seconds")
        return True

    def _plot_fidelity_progress(self, fidelity_history, results_dir):
        try:
            episodes = np.arange(1, len(fidelity_history) + 1)
            best_fidelity = np.array(fidelity_history)
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, best_fidelity, label='Best Fidelity')
            plt.title('Global Best Fidelity Progress Over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Best Fidelity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fidelity_plot_path = os.path.join(results_dir, "fidelity_progress.png")
            plt.savefig(fidelity_plot_path)
            plt.close()
            logging.info(f"Fidelity progress plot saved as '{fidelity_plot_path}'.")
            fidelity_progress_csv = os.path.join(results_dir, "fidelity_progress.csv")
            import pandas as pd
            df = pd.DataFrame({'Episode': episodes, 'Best_Fidelity': best_fidelity})
            df.to_csv(fidelity_progress_csv, index=False)
            logging.info(f"Fidelity progress data saved as '{fidelity_progress_csv}'.")
        except Exception as e:
            logging.error(f"Error while plotting or saving fidelity progress: {e}")