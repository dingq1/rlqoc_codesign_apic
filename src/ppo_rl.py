import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.linalg as jla
import jax
import numpy as np
import math
import time
from collections import deque
from typing import Optional
import os
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

# ========================================
# Import functions from other modules
# ========================================
from control_hardware_model import (
    generate_grid,
    generate_atom_positions_equilateral,
    generate_dipoles,
    calculate_Omega_rabi_prefactor,
    generate_distances,
    generate_coupling_lengths,
    generate_n_eff_list,
    generate_slm_mod,
    construct_V_smooth_with_carrier,
    U_drmzm_multi_channel,
    construct_U_multi_channel_slm,
    construct_I_prime,
    lg00_mode_profile,
    compute_E_field_for_channel,
    compute_total_E_field_profile,
    extract_E_field_at_atoms,
    compute_alpha_t
)
from quantum_system_model import program_instruction
from metrics import compute_multi_qubit_fidelity_closed_system

# ========================================
# Custom Gymnasium Environment for Quantum Optimal Control (QOCEnv)
# ========================================
class QOCEnv(gym.Env):
    """
    Custom Environment for Quantum Optimal Control using Gymnasium interface.
    Enhanced with warm-up initialization.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(QOCEnv, self).__init__()
        self.config = config

        # Configuration parameters
        self.t_steps = self.config.get('t_steps', 500)
        self.N_ch = self.config.get('N_ch', 6)
        self.N_a = self.config.get('N_a', 3)
        self.U_target = self.config.get('system_params', {}).get('U_target')
        if self.U_target is None:
            raise ValueError("U_target is not defined in the configuration.")

        self.piecewise_segments = self.config.get('piecewise_segments', 50)
        self.min_voltage = self.config.get('min_voltage', -15.0)
        self.max_voltage = self.config.get('max_voltage', 15.0)
        self.max_delta_voltage = self.config.get('max_delta_voltage', 1.0)
        self.history_length = self.config.get('history_length', 5)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N_ch,), dtype=np.float32)
        obs_shape = (self.history_length, self.N_ch, self.piecewise_segments + 2)
        self.observation_space = spaces.Box(
            low=np.full(obs_shape, self.min_voltage, dtype=np.float32),
            high=np.full(obs_shape, self.max_voltage, dtype=np.float32),
            shape=obs_shape, dtype=np.float32
        )

        self.current_step = 0
        self.done = False
        self.fidelity = 0.0
        self.max_steps = self.config.get('max_steps', 100)
        self.best_fidelity = 0.0
        self.best_voltage = np.zeros((self.N_ch, self.piecewise_segments), dtype=np.float32)

        self.system_params = self.config.get('system_params', {})
        self.APIC_params = self.config.get('APIC_params', {})
        self.atom_beam_params = self.config.get('atom_beam_params', {})
        self.control_Vt_params = self.config.get('control_Vt_params', {})

        self.current_voltage = self.np_random.uniform(
            low=self.min_voltage,
            high=self.max_voltage,
            size=(self.N_ch, self.piecewise_segments)
        ).astype(np.float32)

        self.voltage_history = deque(maxlen=self.history_length)
        fidelity_array = np.full((self.N_ch, 1), self.fidelity, dtype=np.float32)
        self.current_segment = 0
        segment_array = np.full((self.N_ch, 1), self.current_segment, dtype=np.float32)
        initial_observation = np.hstack((self.current_voltage, fidelity_array, segment_array))
        for _ in range(self.history_length):
            self.voltage_history.append(initial_observation.copy())

        self.scaling_methods = self.config.get('reward_scaling_params', {}).get('scaling_methods', [
            {'method': 'linear', 'min_ratio': 0.0, 'max_ratio': 0.6},
            {'method': 'quadratic', 'min_ratio': 0.6, 'max_ratio': 0.9},
            {'method': 'exponential', 'min_ratio': 0.9, 'max_ratio': 1.0},
        ])
        self.target_fidelity = self.config.get('target_fidelity', 0.999)
        self.a = self.config.get('reward_scaling_params', {}).get('a', 1.0)
        self.b = self.config.get('reward_scaling_params', {}).get('b', 1.0)
        self.method = 'linear'
        self.k = self.config.get('reward_scaling_params', {}).get('k', 5.0)
        self.thresholds = self.config.get('reward_scaling_params', {}).get('thresholds', [0.8, 0.9])
        self.bonus = self.config.get('reward_scaling_params', {}).get('bonus', 5.0)
        self.clip_min = self.config.get('reward_scaling_params', {}).get('clip_min', -10.0)
        self.clip_max = self.config.get('reward_scaling_params', {}).get('clip_max', 10.0)
        self.step_penalty = self.config.get('reward_scaling_params', {}).get('step_penalty', 0.01)

        self.logger = None  # To be optionally set externally
        self.seed()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def select_scaling_method(self, best_fidelity):
        ratio = best_fidelity / self.target_fidelity
        new_method = None
        for scaling in self.scaling_methods:
            if scaling['min_ratio'] <= ratio < scaling['max_ratio']:
                new_method = scaling['method']
                break
        if new_method is None and ratio >= self.scaling_methods[-1]['max_ratio']:
            new_method = self.scaling_methods[-1]['method']
        if new_method and new_method != self.method:
            if self.logger:
                self.logger.info(f"Scaling method changed from {self.method} to {new_method} (ratio: {ratio:.2f}).")
            self.method = new_method

    def compute_scaled_reward(self, final_fidelity, improvement):
        self.select_scaling_method(self.best_fidelity)
        if self.method == 'linear':
            scaled_fidelity = self.a * final_fidelity
        elif self.method == 'quadratic':
            scaled_fidelity = self.a * (final_fidelity ** 2)
        elif self.method == 'exponential':
            scaled_fidelity = self.a * (math.exp(self.k * final_fidelity) - 1)
        elif self.method == 'log':
            scaled_fidelity = self.a * (math.log(self.k * final_fidelity) - 1)
        elif self.method == 'sigmoid':
            c = 0.95
            scaled_fidelity = self.a * (1 / (1 + math.exp(-self.k * (final_fidelity - c))))
        else:
            raise ValueError("Unsupported scaling method")
        if improvement > 0:
            scaled_fidelity += self.b * improvement
        threshold_bonus = 0.0
        for threshold in self.thresholds:
            relative_threshold = threshold * self.target_fidelity
            if self.best_fidelity < relative_threshold <= final_fidelity:
                threshold_bonus += self.bonus
        scaled_fidelity += threshold_bonus
        return scaled_fidelity

    def step(self, action):
        if self.done:
            raise ValueError("Episode ended. Call reset().")
        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * self.max_delta_voltage
        self.current_voltage[:, self.current_segment] += scaled_action
        self.current_voltage[:, self.current_segment] = np.clip(
            self.current_voltage[:, self.current_segment],
            self.min_voltage, self.max_voltage
        )
        V0_t_list = jnp.array(self.current_voltage[:, :self.current_segment + 1])
        V1_t_list = jnp.array(self.current_voltage[:, :self.current_segment + 1])
        fidelity_all = compute_multi_qubit_fidelity_closed_system(
            V0_t_list, V1_t_list,
            self.APIC_params.get('L'), self.APIC_params.get('n0'),
            self.APIC_params.get('lambda_0'), self.APIC_params.get('a0'), self.APIC_params.get('t0'),
            self.APIC_params.get('a1'), self.APIC_params.get('t1'),
            self.APIC_params.get('phase_mod'), self.APIC_params.get('amp_mod'),
            self.system_params.get('delta'),
            self.atom_beam_params.get('atom_positions'),
            self.atom_beam_params.get('dipoles'),
            self.atom_beam_params.get('beam_centers'),
            self.atom_beam_params.get('beam_waist'),
            self.atom_beam_params.get('X'), self.atom_beam_params.get('Y'),
            self.atom_beam_params.get('Omega_prefactor_MHz'),
            self.control_Vt_params.get('t_steps'), self.control_Vt_params.get('dt'),
            self.system_params.get('N_ch'),
            self.system_params.get('distances'),
            self.system_params.get('coupling_lengths'),
            self.system_params.get('n_eff_list'),
            self.system_params.get('kappa0'),
            self.system_params.get('alpha'),
            self.system_params.get('enable_crosstalk'),
            self.system_params.get('N_slm'),
            self.system_params.get('N_ch_slm_in'),
            self.system_params.get('N_scat_1'),
            self.system_params.get('N_scat_2'),
            self.system_params.get('N_a'),
            self.system_params.get('N_qubit_level'),
            self.system_params.get('omega_0'),
            self.system_params.get('omega_r'),
            self.system_params.get('a_pic'),
            self.system_params.get('a_scat_1'),
            self.U_target,
            self.system_params.get('gate_type')
        )
        try:
            unclipped_fidelity = float(jnp.real(fidelity_all[-1]))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error computing fidelity: {e}")
            unclipped_fidelity = 0.0
        final_fidelity = np.clip(unclipped_fidelity, 0.0, 1.0)
        improvement = final_fidelity - self.best_fidelity
        scaled_reward = self.compute_scaled_reward(final_fidelity, improvement)
        scaled_reward = np.clip(scaled_reward, self.clip_min, self.clip_max)
        reward = scaled_reward
        if unclipped_fidelity > 1.0:
            self.current_voltage = self.best_voltage.copy()
            penalty = -1000.0
            reward = penalty
            self.done = True
            final_fidelity = self.best_fidelity
            self.fidelity = final_fidelity
        else:
            if improvement > 0:
                self.best_fidelity = final_fidelity
                self.best_voltage = self.current_voltage.copy()
            self.fidelity = final_fidelity
        self.current_segment += 1
        if self.current_segment >= self.piecewise_segments:
            self.done = True
        info = {
            'final_fidelity': final_fidelity,
            'current_step': self.current_step,
            'current_segment': self.current_segment
        }
        if improvement > 0:
            info['best_voltage'] = self.best_voltage.copy()
        fidelity_array = np.full((self.N_ch, 1), self.fidelity, dtype=np.float32)
        segment_array = np.full((self.N_ch, 1), self.current_segment, dtype=np.float32)
        observation_component = np.hstack((self.current_voltage, fidelity_array, segment_array))
        self.voltage_history.append(observation_component.copy())
        observation = np.stack(self.voltage_history, axis=0)
        terminated = False
        truncated = self.done
        if self.logger:
            self.logger.debug(f"Segment: {self.current_segment}, Reward: {reward:.4f}, Fidelity: {self.fidelity:.4f}, Done: {self.done}")
        return observation.astype(np.float32), reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Segment: {self.current_segment} | Fidelity: {self.fidelity:.4f}")

    def close(self):
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            super().reset(seed=seed)
            self.current_step = 0
            self.done = False
            self.fidelity = 0.0
            self.best_fidelity = 0.0
            self.best_voltage = np.zeros((self.N_ch, self.piecewise_segments), dtype=np.float32)
            self.current_voltage = self.np_random.normal(
                loc=0.0, scale=1.0, size=(self.N_ch, self.piecewise_segments)
            ).astype(np.float32)
            self.current_voltage = np.clip(self.current_voltage, self.min_voltage, self.max_voltage)
            fixed_segments = 5
            self.current_voltage[:, :fixed_segments] = 0.0
            self.voltage_history = deque(maxlen=self.history_length)
            fidelity_array = np.full((self.N_ch, 1), self.fidelity, dtype=np.float32)
            self.current_segment = 0
            segment_array = np.full((self.N_ch, 1), self.current_segment, dtype=np.float32)
            initial_observation = np.hstack((self.current_voltage, fidelity_array, segment_array))
            for _ in range(self.history_length):
                self.voltage_history.append(initial_observation.copy())
            observation = np.stack(self.voltage_history, axis=0)
            return observation.astype(np.float32), {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during reset: {e}")
            raise e

# ========================================
# Custom Policy and Callback Definitions
# ========================================

class CorrelationFeatureExtractor(nn.Module):
    """
    Custom feature extractor using convolutional layers.
    """
    def __init__(self, observation_space: gym.Space, *, features_dim: int = 256):
        super(CorrelationFeatureExtractor, self).__init__()
        assert isinstance(observation_space, spaces.Box), "Observation space must be Box"
        obs_shape = observation_space.shape  # (history_length, N_ch, piecewise_segments+2)
        history_length, N_ch, piecewise_segments_plus_2 = obs_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=history_length, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, history_length, N_ch, piecewise_segments_plus_2)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, features_dim),
            nn.ReLU()
        )
        self.features_dim = features_dim

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy with the CorrelationFeatureExtractor.
    """
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CorrelationFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 256, 256],
            activation_fn=nn.ReLU
        )

    def forward(self, obs, deterministic=False):
        actions, values, log_probs = super().forward(obs, deterministic=deterministic)
        return actions, values, log_probs

class RLLoggingCallback(BaseCallback):
    """
    Custom callback for logging training metrics and handling early termination.
    """
    def __init__(self, stagnant_threshold=100, stagnant_fidelity_min=0.99,
                 target_fidelity=0.999, results_dir='results', verbose=0):
        super(RLLoggingCallback, self).__init__(verbose)
        self.best_fidelity = 0.0
        self.best_voltage = None
        self.best_fidelity_history = []
        self.episode_count = 0
        self.stagnant_counter = 0
        self.stagnant_threshold = stagnant_threshold
        self.stagnant_fidelity_min = stagnant_fidelity_min
        self.target_fidelity = target_fidelity
        self.start_time = time.time()
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        import logging
        self._local_logger = logging.getLogger(__name__)
        self._local_logger.setLevel(logging.INFO)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        improved = False
        for env_id, info in enumerate(infos):
            if 'final_fidelity' in info:
                fidelity = info['final_fidelity']
                self.episode_count += 1
                if fidelity > self.best_fidelity:
                    self.best_fidelity = fidelity
                    self.stagnant_counter = 0
                    improved = True
                    self.best_voltage = info.get('best_voltage', None)
                    self._local_logger.info(f"New best fidelity: {self.best_fidelity:.4f} in Env {env_id} | Episode: {self.episode_count}")
                else:
                    if self.stagnant_fidelity_min <= self.best_fidelity < self.target_fidelity:
                        self.stagnant_counter += 1
                    else:
                        self.stagnant_counter = 0
        self.best_fidelity_history.append(self.best_fidelity)
        if self.best_fidelity >= self.target_fidelity:
            self._local_logger.info(f"Target fidelity {self.best_fidelity:.4f} reached in {self.episode_count} episodes. Training terminated.")
            if self.best_voltage is not None:
                np.save(os.path.join(self.results_dir, "best_voltage_final.npy"), self.best_voltage)
                self._local_logger.info("Final best voltage saved.")
            with open(os.path.join(self.results_dir, "best_fidelity.txt"), "w") as f:
                f.write(f"Best Fidelity: {self.best_fidelity:.6f}\n")
            np.save(os.path.join(self.results_dir, "best_fidelity_history.npy"), np.array(self.best_fidelity_history))
            np.savetxt(os.path.join(self.results_dir, "best_fidelity_history.csv"),
                       np.array(self.best_fidelity_history), delimiter=",", header="Episode,Best_Fidelity")
            self._plot_fidelity_progress()
            return False
        if self.stagnant_counter >= self.stagnant_threshold:
            self._local_logger.warning(f"Training stagnating after {self.episode_count} episodes. Terminating training.")
            if self.best_voltage is not None:
                np.save(os.path.join(self.results_dir, "best_voltage_final.npy"), self.best_voltage)
            with open(os.path.join(self.results_dir, "best_fidelity.txt"), "w") as f:
                f.write(f"Best Fidelity: {self.best_fidelity:.6f}\n")
            np.save(os.path.join(self.results_dir, "best_fidelity_history.npy"), np.array(self.best_fidelity_history))
            np.savetxt(os.path.join(self.results_dir, "best_fidelity_history.csv"),
                       np.array(self.best_fidelity_history), delimiter=",", header="Episode,Best_Fidelity")
            self._plot_fidelity_progress()
            return False
        self._local_logger.info(f"Status | Episodes: {self.episode_count} | Best Fidelity: {self.best_fidelity:.4f} | Stagnant: {self.stagnant_counter}/{self.stagnant_threshold}")
        return True

    def _plot_fidelity_progress(self):
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            episodes = np.arange(1, len(self.best_fidelity_history) + 1)
            best_fidelity = np.array(self.best_fidelity_history)
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, best_fidelity, label='Best Fidelity')
            plt.title('Global Best Fidelity Progress Over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Best Fidelity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(self.results_dir, "fidelity_progress.png")
            plt.savefig(plot_path)
            plt.close()
            self._local_logger.info(f"Fidelity progress plot saved as '{plot_path}'.")
            df = pd.DataFrame({'Episode': episodes, 'Best_Fidelity': best_fidelity})
            csv_path = os.path.join(self.results_dir, "fidelity_progress.csv")
            df.to_csv(csv_path, index=False)
            self._local_logger.info(f"Fidelity progress data saved as '{csv_path}'.")
        except Exception as e:
            self._local_logger.error(f"Error in plotting fidelity progress: {e}")

# ========================================
# Environment Creation Function
# ========================================
def make_env(config, seed):
    def _init():
        env = QOCEnv(config)
        env.seed(seed)
        return env
    return _init
