# control_hardware_model.py

import math
import os
import jax.numpy as jnp
import jax.random as jrandom

# --- Control and Hardware Functions ---

def calculate_Omega_rabi_prefactor(I_mW_per_cm2, Detuning_MHz):
    """
    Calculate the effective Rabi frequency prefactor (MHz) based on laser intensity and detuning.
    """
    hbar = 1.0545718e-34
    c = 3e8
    epsilon_0 = 8.854187817e-12
    mu0e = 3.336e-29
    mu1e = 3.400e-29

    I = I_mW_per_cm2 * 10
    E_0 = jnp.sqrt(2 * I / (c * epsilon_0))
    Omega_0e_prefactor = (mu0e * E_0) / hbar
    Omega_1e_prefactor = (mu1e * E_0) / hbar
    Omega_effRabi_prefactor = (Omega_0e_prefactor * Omega_1e_prefactor) / (2 * Detuning_MHz * 1e6)
    return Omega_effRabi_prefactor / 1e6  # in MHz

def generate_grid(grid_size=600, grid_range=(-6, 6)):
    x = jnp.linspace(grid_range[0], grid_range[1], grid_size - 1)
    y = jnp.linspace(grid_range[0], grid_range[1], grid_size - 1)
    X, Y = jnp.meshgrid(x, y)
    return X, Y

def generate_atom_positions_equilateral(N_a, side_length=3.0, center=(0, 0), triangle_spacing_factor=2.0):
    positions = []
    x_center, y_center = center
    height = (math.sqrt(3) / 2) * side_length
    num_full_triangles = N_a // 3
    for triangle_idx in range(num_full_triangles):
        current_x_center = x_center + triangle_idx * triangle_spacing_factor * side_length
        current_y_center = y_center
        positions.append((current_x_center - side_length / 2, current_y_center - height / 2))
        positions.append((current_x_center + side_length / 2, current_y_center - height / 2))
        positions.append((current_x_center, current_y_center + height / 2))
    remaining_atoms = N_a % 3
    if remaining_atoms > 0:
        last_x_center = x_center if num_full_triangles == 0 else x_center + (num_full_triangles - 1) * triangle_spacing_factor * side_length
        last_y_center = y_center
        if remaining_atoms >= 1:
            positions.append((last_x_center - side_length / 2, last_y_center - height / 2))
        if remaining_atoms == 2:
            positions.append((last_x_center + side_length / 2, last_y_center - height / 2))
    return positions

def generate_dipoles(N_a):
    return [jnp.array([1.0, 0]) for _ in range(N_a)]

def generate_slm_mod(N_slm):
    phase_mod = jnp.zeros(N_slm)
    amp_mod = jnp.ones(N_slm)
    return phase_mod, amp_mod

def generate_distances(num_channels, pitch, random_variation=0.0, seed=None):
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    distances = np.zeros((num_channels, num_channels), dtype=np.float32)
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                base_distance = abs(i - j) * pitch
                variation = np.random.uniform(-random_variation, random_variation)
                distances[i, j] = base_distance + variation
    return jnp.array(distances)

def generate_coupling_lengths(num_channels, base_length, scaling_factor=1.0, random_variation=0.0, seed=None):
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    coupling_lengths = np.zeros((num_channels, num_channels), dtype=np.float32)
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                base_length_ij = base_length * (scaling_factor**abs(i - j))
                variation = np.random.uniform(-random_variation, random_variation)
                coupling_lengths[i, j] = base_length_ij + variation
    return jnp.array(coupling_lengths)

def generate_n_eff_list(num_channels, base_n_eff, random_variation=0.0, seed=None):
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    variations = np.random.uniform(-random_variation, random_variation, num_channels)
    n_eff_list = base_n_eff + variations
    return jnp.array(n_eff_list)

# --- Control Signal and Unitary Matrix Construction Functions ---

def construct_V_smooth_with_carrier(tmin, tmax, t_steps, voltage_levels, omega_0, phi=0, max_step=30):
    time_points = jnp.linspace(tmin, tmax, t_steps)
    num_pieces = len(voltage_levels)
    piece_duration = t_steps // num_pieces
    voltage_levels = jnp.array(voltage_levels, dtype=jnp.float32)
    V_piecewise = jnp.zeros_like(time_points)
    for i in range(num_pieces):
        if i > 0:
            delta_V = voltage_levels[i] - voltage_levels[i - 1]
            if jnp.abs(delta_V) > max_step:
                voltage_levels = voltage_levels.at[i].set(voltage_levels[i - 1] + jnp.sign(delta_V) * max_step)
        start_idx = i * piece_duration
        end_idx = (i + 1) * piece_duration if i < num_pieces - 1 else t_steps
        V_piecewise = V_piecewise.at[start_idx:end_idx].set(jnp.full(end_idx - start_idx, voltage_levels[i]))
    carrier = jnp.cos(omega_0 * time_points + phi)
    return V_piecewise * carrier

def dn(V):
    return 4e-5 * V

def phi_func(L, n, dn_val, lambda_):
    return 2 * jnp.pi * L * (n + dn_val) / lambda_

def D_func(a, t_val, phi_val):
    return jnp.exp(1j * jnp.pi) * (a - t_val * jnp.exp(-1j * phi_val)) / (1 - t_val * a * jnp.exp(-1j * phi_val))

def U_drmzm_single_channel(V0, V1, L, n0, lambda_0, a0, t0, a1, t1, psi_0=0):
    dn_0 = dn(V0)
    dn_1 = dn(V1)
    phi0 = phi_func(L, n0, dn_0, lambda_0)
    phi1 = phi_func(L, n0, dn_1, lambda_0)
    D_00 = D_func(a0, t0, phi0) * jnp.exp(psi_0)
    D_11 = D_func(a1, t1, phi1)
    U_drmzm_matrix = jnp.array([[D_00, jnp.zeros_like(D_00)], [jnp.zeros_like(D_11), D_11]])
    U_bs = (1 / jnp.sqrt(2)) * jnp.array([[1, 1], [1, -1]])
    return jnp.dot(U_bs, jnp.dot(U_drmzm_matrix, U_bs))

def U_drmzm_multi_channel(
    V0_t_list, V1_t_list, L, n0, lambda_0, a0, t0, a1, t1,
    N_ch, distances, coupling_lengths, n_eff_list, kappa0, alpha, enable_crosstalk=True
):
    U_multi_channel_no_ct = jnp.zeros((N_ch, N_ch), dtype=jnp.complex64)
    for i in range(N_ch):
        V0_t = V0_t_list[i]
        V1_t = V1_t_list[i]
        U_single_channel = U_drmzm_single_channel(V0_t, V1_t, L, n0, lambda_0, a0, t0, a1, t1)
        U_multi_channel_no_ct = U_multi_channel_no_ct.at[i, i].set(U_single_channel[0, 0])
    if not enable_crosstalk:
        return U_multi_channel_no_ct

    U_wg_coupling_ct = jnp.eye(N_ch, dtype=jnp.complex64)
    k = (2.0 * jnp.pi) / lambda_0

    def compute_transfer_matrix(L_val, beta_1, beta_2, kappa):
        delta_beta = (beta_1 - beta_2) / 2.0
        kappa_eff = jnp.sqrt(kappa**2 + delta_beta**2)
        cos_term = jnp.cos(kappa_eff * L_val)
        sin_term = jnp.sin(kappa_eff * L_val)
        safe_kappa_eff = jnp.where(kappa_eff != 0, kappa_eff, 1e-12)
        delta_term = delta_beta / safe_kappa_eff
        return jnp.array([
            [cos_term - 1j * delta_term * sin_term, -1j * sin_term],
            [-1j * sin_term, cos_term + 1j * delta_term * sin_term]
        ], dtype=jnp.complex64)

    for i in range(N_ch):
        beta_i = k * n_eff_list[i]
        for j in range(i + 1, N_ch):
            beta_j = k * n_eff_list[j]
            dist_ij = distances[i, j]
            L_ij_full = coupling_lengths[i, j]
            L_ij = jnp.where(dist_ij > 0.0, L_ij_full, 0.0)
            kappa_ij = jnp.where(dist_ij > 0.0, kappa0 * jnp.exp(-alpha * dist_ij), 0.0)
            M = compute_transfer_matrix(L_ij, beta_i, beta_j, kappa_ij)
            U_wg_coupling_ct = U_wg_coupling_ct.at[i, j].set(M[0, 1])
            U_wg_coupling_ct = U_wg_coupling_ct.at[j, i].set(jnp.conj(M[0, 1]))
    return jnp.matmul(U_multi_channel_no_ct, U_wg_coupling_ct)

def construct_U_multi_channel_slm(N_slm, phase_mod, amp_mod, t_steps):
    U_slm = jnp.zeros((t_steps, N_slm, N_slm), dtype=jnp.complex64)
    for t in range(t_steps):
        for i in range(N_slm):
            phase = jnp.exp(1j * phase_mod[i])
            amplitude = amp_mod[i]
            U_slm = U_slm.at[t, i, i].set(amplitude * phase)
    return U_slm

def construct_I_prime(N_scat, delta, t_steps):
    I_prime = jnp.zeros((t_steps, N_scat, N_scat), dtype=jnp.complex64)
    for t in range(t_steps):
        I_prime = I_prime.at[t].set(jnp.eye(N_scat) + jnp.diag(jnp.full(N_scat, delta)))
    return I_prime

# --- E-Field Calculations ---

def lg00_mode_profile(X, Y, beam_center, beam_waist):
    cx, cy = beam_center
    r_squared = (X - cx)**2 + (Y - cy)**2
    return jnp.exp(-r_squared / (2 * beam_waist**2))

def compute_E_field_for_channel(X, Y, E_t, beam_center, beam_waist, t_idx):
    E_profile = lg00_mode_profile(X, Y, beam_center, beam_waist)
    return E_profile * (jnp.real(E_t) + 1j * jnp.imag(E_t))

def compute_total_E_field_profile(X, Y, b_slm_out, beam_centers, beam_waist):
    E_field_profiles = []
    for t_idx in range(b_slm_out.shape[0]):
        E_field_total = jnp.zeros_like(X, dtype=jnp.complex64)
        for atom_index, beam_center in enumerate(beam_centers):
            E_field_total += compute_E_field_for_channel(X, Y, b_slm_out[t_idx, atom_index], beam_center, beam_waist, t_idx)
        E_field_profiles.append(E_field_total)
    return jnp.array(E_field_profiles)

def extract_E_field_at_atoms(E_field_profiles, atom_positions, X, Y):
    E_field_at_atoms = []
    for t_idx in range(E_field_profiles.shape[0]):
        E_field_at_timestep = []
        for x0, y0 in atom_positions:
            x_idx = jnp.argmin(jnp.abs(X[0, :] - x0))
            y_idx = jnp.argmin(jnp.abs(Y[:, 0] - y0))
            E_field_at_timestep.append(E_field_profiles[t_idx][y_idx, x_idx])
        E_field_at_atoms.append(jnp.array(E_field_at_timestep))
    return jnp.array(E_field_at_atoms)

def compute_alpha_t(E_fields_at_atoms, dipoles, Omega_prefactor_MHz):
    alpha_t = []
    for t_idx in range(E_fields_at_atoms.shape[0]):
        alpha_t_timestep = []
        for atom_idx in range(len(dipoles)):
            E_field_atom = E_fields_at_atoms[t_idx, atom_idx]
            dipole = dipoles[atom_idx]
            alpha_t_timestep.append(Omega_prefactor_MHz * dipole[0] * E_field_atom)
        alpha_t.append(jnp.array(alpha_t_timestep))
    return jnp.array(alpha_t)