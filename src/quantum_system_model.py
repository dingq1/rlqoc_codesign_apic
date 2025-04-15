# quantum_system_model.py

import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax
from jax import random
import math

# --- Multi-Qubit Operators ---
s_plus = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64)
s_minus = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64)
s_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

def construct_multi_qubit_operator(single_qubit_op, N_a, N_qubit_level, qubit_idx):
    I = jnp.eye(N_qubit_level, dtype=jnp.complex64)
    op_list = [I] * N_a
    op_list[qubit_idx] = single_qubit_op
    multi_qubit_op = op_list[0]
    for op in op_list[1:]:
        multi_qubit_op = jnp.kron(multi_qubit_op, op)
    return multi_qubit_op

def construct_annihilation_operator(fock_dim):
    a = jnp.zeros((fock_dim, fock_dim), dtype=jnp.complex64)
    for n in range(1, fock_dim):
        a = a.at[n - 1, n].set(jnp.sqrt(n))
    a_dag = a.T.conj()
    return a, a_dag

# Setup Fock space
fock_dim = 5
a, a_dag = construct_annihilation_operator(fock_dim)
I_fock = jnp.eye(fock_dim, dtype=jnp.complex64)

# --- Hamiltonians and Time Evolution ---

def construct_H_0(N_a, omega_0, omega_r):
    H_0_qubits = sum(0.5 * omega_0 * construct_multi_qubit_operator(s_z, N_a, 2, i) for i in range(N_a))
    H_0_field = omega_r * jnp.kron(jnp.eye(2 ** N_a), a_dag @ a)
    return jnp.kron(H_0_qubits, I_fock) + H_0_field

def construct_H_control(N_a, N_qubit_level, g_real_t, g_imag_t):
    dim = N_qubit_level ** N_a * fock_dim
    H_control = jnp.zeros((dim, dim), dtype=jnp.complex64)
    for i in range(N_a):
        H_control += g_real_t[i] * (jnp.kron(construct_multi_qubit_operator(s_plus, N_a, N_qubit_level, i), a) +
                                    jnp.kron(construct_multi_qubit_operator(s_minus, N_a, N_qubit_level, i), a_dag))
        H_control += g_imag_t[i] * (1j * jnp.kron(construct_multi_qubit_operator(s_plus, N_a, N_qubit_level, i), a) -
                                    1j * jnp.kron(construct_multi_qubit_operator(s_minus, N_a, N_qubit_level, i), a_dag))
    return H_control

def construct_H_time(N_a, N_qubit_level, omega_0, omega_r, g_real_t, g_imag_t, atom_positions, gate_type='single'):
    H_t_list = []
    H_0 = construct_H_0(N_a, omega_0, omega_r)
    for t in range(len(g_real_t)):
        H_control = construct_H_control(N_a, N_qubit_level, g_real_t[t], g_imag_t[t])
        H_t_list.append(H_0 + H_control)
    return jnp.array(H_t_list)

def compute_accumulated_propagator(H_t, dt, N_a, N_qubit_level):
    dim = N_qubit_level ** N_a * fock_dim
    U_accumulated = jnp.eye(dim, dtype=jnp.complex64)
    U_t_all = []
    for H in H_t:
        U_t = jla.expm(-1j * H * dt)
        U_accumulated = U_t @ U_accumulated
        U_t_all.append(U_accumulated)
    return jnp.array(U_t_all)

def trace_out_field_from_unitary(U_t, N_a, N_qubit_level, field_dim=5):
    dim_qubits = N_qubit_level ** N_a
    U_t_reshaped = U_t.reshape(dim_qubits, field_dim, dim_qubits, field_dim)
    return jnp.sum(U_t_reshaped, axis=(1, 3))

def trace_out_field_from_unitary_t_all(U_t_all, N_a, N_qubit_level, field_dim=5):
    time_steps = U_t_all.shape[0]
    dim_qubits = N_qubit_level**N_a
    total_dim = U_t_all.shape[-1]
    assert dim_qubits * field_dim == total_dim, "Mismatch in total dimensions!"
    U_t_reshaped = U_t_all.reshape(time_steps, dim_qubits, field_dim, dim_qubits, field_dim)
    return jnp.sum(U_t_reshaped, axis=(2, 4))

# --- Fidelity and Gate Target Functions ---

def compute_fidelity_unitary(U_t_traced, U_target):
    d = U_target.shape[0]
    trace_overlap = jnp.abs(jnp.trace(U_target.conj().T @ U_t_traced))**2
    return jnp.real(trace_overlap / (d**2))

def compute_fidelity_unitary_t_all(U_t_traced_all, U_target):
    d = U_target.shape[0]
    def fidelity_at_timestep(U_t_traced):
        return jnp.abs(jnp.trace(U_target.conj().T @ U_t_traced))**2 / (d**2)
    return jax.vmap(fidelity_at_timestep)(U_t_traced_all)

def clifford_group_and_t_gate():
    I = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)
    H = (1 / jnp.sqrt(2)) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64)
    S = jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex64)
    X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    T = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=jnp.complex64)
    clifford_group = [
        ("X", X), ("I", I), ("H", H), ("S", S), ("Y", Y), ("Z", Z),
        ("HS", H @ S), ("SH", S @ H), ("HX", H @ X), ("SX", S @ X),
        ("SY", S @ Y), ("SZ", S @ Z), ("XH", X @ H), ("YH", Y @ H),
        ("ZH", Z @ H), ("XS", X @ S), ("YS", Y @ S), ("ZS", Z @ S),
        ("HSX", H @ S @ X), ("HSY", H @ S @ Y), ("HSZ", H @ S @ Z),
        ("XHS", X @ H @ S), ("YHS", Y @ H @ S), ("ZHS", Z @ H @ S),
    ]
    return clifford_group + [("T", T)]

def program_instruction(N_a, key_number, gate_type='single', selected_atoms=None, control_atom=None, target_atom=None):
    from jax import random
    import jax.random as jrandom
    import logging
    logger = logging.getLogger()
    if gate_type == 'single':
        clifford_t_gates = clifford_group_and_t_gate()
        U_target_multi_qubit = []
        key = jrandom.PRNGKey(key_number)
        for atom_idx in range(N_a):
            if selected_atoms is None or atom_idx in selected_atoms:
                key, subkey = jrandom.split(key)
                gate_idx = jrandom.randint(subkey, (), minval=0, maxval=len(clifford_t_gates))
                gate_name, gate = clifford_t_gates[gate_idx]
                logger.info(f"Applying {gate_name} gate to Atom {atom_idx}")
                U_target_multi_qubit.append(gate)
            else:
                logger.info(f"Applying Identity gate to Atom {atom_idx}")
                U_target_multi_qubit.append(jnp.eye(2))
        U_target = U_target_multi_qubit[0]
        for gate in U_target_multi_qubit[1:]:
            U_target = jnp.kron(U_target, gate)
        return U_target