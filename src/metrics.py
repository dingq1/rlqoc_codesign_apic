# metrics.py

import jax.numpy as jnp
import jax
import numpy as np
import pickle
import os
from control_hardware_model import (
    U_drmzm_multi_channel,
    construct_I_prime,
    construct_U_multi_channel_slm,
    compute_total_E_field_profile,
    extract_E_field_at_atoms,
    compute_alpha_t
)
from quantum_system_model import (
    construct_H_time,
    compute_accumulated_propagator,
    trace_out_field_from_unitary,
    trace_out_field_from_unitary_t_all,
    compute_fidelity_unitary,
    compute_fidelity_unitary_t_all,
)

def compute_multi_qubit_fidelity_closed_system(
    V0_t_list, V1_t_list, L, n0, lambda_0, a0, t0, a1, t1, phase_mod, amp_mod, delta,
    atom_positions, dipoles, beam_centers, beam_waist, X, Y, Omega_prefactor_MHz,
    t_steps, dt,
    N_ch, distances, coupling_lengths, n_eff_list, kappa0, alpha, enable_crosstalk,
    N_slm, N_ch_slm_in, N_scat_1, N_scat_2, N_a, N_qubit_level, omega_0, omega_r, a_pic, a_scat_1,
    U_target, gate_type='single'
):
    U_system_multi_channel = jnp.array([
        U_drmzm_multi_channel(
            [V0_t[i] for V0_t in V0_t_list],
            [V1_t[i] for V1_t in V1_t_list],
            L, n0, lambda_0, a0, t0, a1, t1,
            N_ch, distances, coupling_lengths, n_eff_list,
            kappa0, alpha, enable_crosstalk
        )
        for i in range(t_steps)
    ])
    I1_prime = construct_I_prime(N_scat_1, delta, t_steps)
    U_tensor_product = jnp.array([jnp.kron(U_system_multi_channel[i], I1_prime[i]) for i in range(t_steps)])
    a_total = jnp.kron(a_pic, a_scat_1)
    output_modes = jnp.array([jnp.dot(U_tensor_product[i], a_total) for i in range(t_steps)])
    output_modes_reshaped = output_modes.reshape(t_steps, N_ch, N_scat_1)
    a_pic_out = jnp.sum(output_modes_reshaped, axis=-1)
    a_scat_1_out = jnp.sum(output_modes_reshaped, axis=-2)
    b_slm_in = jnp.zeros((t_steps, N_slm), dtype=jnp.complex64)
    b_scat2_in = jnp.zeros((t_steps, N_scat_2), dtype=jnp.complex64)
    for t in range(t_steps):
        b_slm_in = b_slm_in.at[t, :N_ch_slm_in].set(a_pic_out[t, :N_ch_slm_in])
        b_slm_in = b_slm_in.at[t, N_ch_slm_in:].set(a_scat_1_out[t, :(N_slm - N_ch_slm_in)])
        b_scat2_in = b_scat2_in.at[t, :(N_ch - N_ch_slm_in)].set(a_pic_out[t, N_ch_slm_in:])
        b_scat2_in = b_scat2_in.at[t, (N_ch - N_ch_slm_in):].set(a_scat_1_out[t, (N_scat_2 - (N_ch - N_ch_slm_in)):])
    b_total_in = jnp.array([jnp.kron(b_slm_in[t], b_scat2_in[t]) for t in range(t_steps)])
    U_multi_channel_slm = construct_U_multi_channel_slm(N_slm, phase_mod, amp_mod, t_steps)
    I2_prime = construct_I_prime(N_scat_2, delta, t_steps)
    U_tensor_product_stage_2 = jnp.array([jnp.kron(U_multi_channel_slm[t], I2_prime[t]) for t in range(t_steps)])
    b_total_out = jnp.array([jnp.dot(U_tensor_product_stage_2[t], b_total_in[t]) for t in range(t_steps)])
    b_total_out_reshaped = b_total_out.reshape(t_steps, N_slm, N_scat_2)
    b_slm_out = jnp.sum(b_total_out_reshaped, axis=-1)
    E_field_profiles = compute_total_E_field_profile(X, Y, b_slm_out, beam_centers, beam_waist)
    E_fields_at_atoms = extract_E_field_at_atoms(E_field_profiles, atom_positions, X, Y)
    alpha_t = compute_alpha_t(E_fields_at_atoms, dipoles, Omega_prefactor_MHz)
    g_real_t = alpha_t.real / 2
    g_imag_t = alpha_t.imag / 2
    H_t = construct_H_time(N_a, N_qubit_level, omega_0, omega_r, g_real_t, g_imag_t, atom_positions, gate_type)
    U_t_all = compute_accumulated_propagator(H_t, dt, N_a, N_qubit_level)
    U_t_traced = trace_out_field_from_unitary(U_t_all[-1], N_a, N_qubit_level)
    U_t_traced_all = trace_out_field_from_unitary_t_all(U_t_all, N_a, N_qubit_level)
    fidelity = compute_fidelity_unitary(U_t_traced, U_target)
    fidelity_all = compute_fidelity_unitary_t_all(U_t_traced_all, U_target)
    return fidelity_all