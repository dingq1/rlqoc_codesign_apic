import numpy as np
import jax
from jax import grad
import optax
import logging

# Import the fidelity function from your metrics module.
# (It returns a time series of fidelity values.)
from metrics import compute_multi_qubit_fidelity_closed_system

logger = logging.getLogger(__name__)

# ========================================
# Part 5: Optimization : SADE-ADAM
# ========================================

def multi_qubit_objective(V_combined, APIC_params, atom_beam_params, control_Vt_params, system_params):
    """
    Computes the gate error objective for a given control signal.
    It calls the fidelity function (which returns a time series) and then takes
    the final time step fidelity as the final fidelity.
    Returns a scalar value: 1 - final_fidelity.
    """
    half_len = V_combined.shape[0] // 2
    V0_t_list = V_combined[:half_len].reshape(system_params['N_ch'], control_Vt_params['t_steps'])
    V1_t_list = V_combined[half_len:].reshape(system_params['N_ch'], control_Vt_params['t_steps'])
    
    # fidelity_series is assumed to be an array (or list) of fidelity values over time.
    fidelity_series = compute_multi_qubit_fidelity_closed_system(
        V0_t_list, V1_t_list,
        APIC_params['L'], APIC_params['n0'], APIC_params['lambda_0'],
        APIC_params['a0'], APIC_params['t0'], APIC_params['a1'], APIC_params['t1'],
        APIC_params['phase_mod'], APIC_params['amp_mod'],
        system_params['delta'],
        atom_beam_params['atom_positions'], atom_beam_params['dipoles'],
        atom_beam_params['beam_centers'], atom_beam_params['beam_waist'],
        atom_beam_params['X'], atom_beam_params['Y'], atom_beam_params['Omega_prefactor_MHz'],
        control_Vt_params['t_steps'], control_Vt_params['dt'],
        system_params['N_ch'], system_params['distances'], system_params['coupling_lengths'],
        system_params['n_eff_list'], system_params['kappa0'], system_params['alpha'],
        system_params['enable_crosstalk'],
        system_params['N_slm'], system_params['N_ch_slm_in'], system_params['N_scat_1'],
        system_params['N_scat_2'], system_params['N_a'], system_params['N_qubit_level'],
        system_params['omega_0'], system_params['omega_r'],
        atom_beam_params['a_pic'], atom_beam_params['a_scat_1'],
        system_params['U_target'], system_params['gate_type']
    )
    # Take the final time-step fidelity as the final fidelity.
    final_fidelity = fidelity_series[-1]
    return 1 - final_fidelity

def adam_optimization(V_combined_init, APIC_params, atom_beam_params, control_Vt_params,
                      system_params, optimizer_params, fidelity_gens=None):
    grad_obj = grad(lambda V: multi_qubit_objective(V, APIC_params, atom_beam_params, control_Vt_params, system_params))
    params = V_combined_init
    current_lr = optimizer_params['adam_lr']
    optimizer = optax.adam(current_lr)
    opt_state = optimizer.init(params)

    starting_fitness = multi_qubit_objective(params, APIC_params, atom_beam_params, control_Vt_params, system_params)
    starting_fidelity = float(1 - starting_fitness)
    logger.info(f"Initial Fidelity: {starting_fidelity:.10f}, Fixed tmax: {control_Vt_params['tmax']:.6f} us")

    best_fidelity = starting_fidelity
    fidelity_gens = fidelity_gens or [starting_fidelity]
    decay_idx = 0
    thresholds = optimizer_params.get('fidelity_decay_thresholds', [0.95, 0.99, 0.995])
    decay_factors = optimizer_params.get('fidelity_decay_factors', [0.10, 0.50, 0.20])
    min_lr = optimizer_params.get('min_lr', 1e-6)

    for step in range(optimizer_params['adam_steps']):
        grads = grad_obj(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        current_fitness = multi_qubit_objective(params, APIC_params, atom_beam_params, control_Vt_params, system_params)
        current_fidelity = float(1 - current_fitness)
        fidelity_gens.append(current_fidelity)

        if current_fidelity > 1.0:
            current_fidelity = -np.inf
            opt_state = optimizer.init(params)
            continue

        if step % 10 == 0:
            logger.info(f"Adam Step {step}: Fidelity = {current_fidelity:.10f}, LR = {current_lr:.6f}")

        if current_fidelity > best_fidelity:
            best_fidelity = current_fidelity

        if decay_idx < len(thresholds) and current_fidelity >= thresholds[decay_idx]:
            new_lr = max(current_lr * decay_factors[decay_idx], min_lr)
            logger.info(f"Fidelity {current_fidelity:.4f} exceeded threshold {thresholds[decay_idx]:.2f}. "
                        f"Reducing LR from {current_lr:.6f} to {new_lr:.6f}.")
            current_lr = new_lr
            optimizer = optax.adam(current_lr)
            opt_state = optimizer.init(params)
            decay_idx += 1

        if current_fidelity >= 1 - optimizer_params['tol']:
            logger.info(f"Early stopping at Adam Step {step}: Best Fidelity = {best_fidelity:.10f}")
            break

    return params, fidelity_gens

def evaluate_population_sade(population, APIC_params, atom_beam_params, control_Vt_params, system_params):
    scores = []
    for control_signals in population:
        half_len = control_signals.shape[0] // 2
        V0_t_list = control_signals[:half_len].reshape(system_params['N_ch'], control_Vt_params['t_steps'])
        V1_t_list = control_signals[half_len:].reshape(system_params['N_ch'], control_Vt_params['t_steps'])
        fidelity_series = compute_multi_qubit_fidelity_closed_system(
            V0_t_list, V1_t_list,
            APIC_params['L'], APIC_params['n0'], APIC_params['lambda_0'],
            APIC_params['a0'], APIC_params['t0'], APIC_params['a1'], APIC_params['t1'],
            APIC_params['phase_mod'], APIC_params['amp_mod'],
            system_params['delta'],
            atom_beam_params['atom_positions'], atom_beam_params['dipoles'],
            atom_beam_params['beam_centers'], atom_beam_params['beam_waist'],
            atom_beam_params['X'], atom_beam_params['Y'], atom_beam_params['Omega_prefactor_MHz'],
            control_Vt_params['t_steps'], control_Vt_params['dt'],
            system_params['N_ch'], system_params['distances'], system_params['coupling_lengths'],
            system_params['n_eff_list'], system_params['kappa0'], system_params['alpha'],
            system_params['enable_crosstalk'],
            system_params['N_slm'], system_params['N_ch_slm_in'], system_params['N_scat_1'],
            system_params['N_scat_2'], system_params['N_a'], system_params['N_qubit_level'],
            system_params['omega_0'], system_params['omega_r'],
            atom_beam_params['a_pic'], atom_beam_params['a_scat_1'],
            system_params['U_target'], system_params['gate_type']
        )
        # Use final time-step fidelity as final fidelity.
        final_fidelity = np.array(fidelity_series)[-1]
        if final_fidelity > 1.0:
            scores.append(np.inf)
        else:
            scores.append(1.0 - float(final_fidelity))
    return np.array(scores)

def optimize_multi_qubit_sade_adam(APIC_params, atom_beam_params, control_Vt_params,
                                   system_params, optimizer_params, V0_init, V1_init):
    def run_sade(V_combined_init):
        fixed_tmax = control_Vt_params['tmax']
        logger.info(f"Fixed tmax for SaDE: {fixed_tmax:.6f} us")
        population = [
            np.clip(
                V_combined_init + 0.2 * np.random.randn(*V_combined_init.shape),
                control_Vt_params['min_V_level'], control_Vt_params['max_V_level']
            )
            for _ in range(optimizer_params['popsize'])
        ]
        fitness = evaluate_population_sade(population, APIC_params, atom_beam_params, control_Vt_params, system_params)
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        fidelity_gens = [1.0 - best_fitness]
        logger.info(f"Initial Fidelity: {fidelity_gens[-1]:.10f}, Fixed tmax: {fixed_tmax:.6f} us")
        for generation in range(optimizer_params['num_generations']):
            new_population = []
            for i in range(optimizer_params['popsize']):
                idxs = np.random.choice([j for j in range(optimizer_params['popsize']) if j != i], 3, replace=False)
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                F = np.random.uniform(0.1, 0.9)
                mutant = np.clip(a + F * (b - c),
                                   control_Vt_params['min_V_level'], control_Vt_params['max_V_level'])
                CR = np.random.uniform(0.1, 0.9)
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(*mutant.shape) < CR
                trial[crossover_mask] = mutant[crossover_mask]
                trial_fitness = evaluate_population_sade([trial], APIC_params, atom_beam_params, control_Vt_params, system_params)[0]
                if trial_fitness == np.inf:
                    new_population.append(population[i])
                    continue
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
            population = new_population
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]
            current_fidelity = float(1.0 - best_fitness)
            fidelity_gens.append(current_fidelity)
            logger.info(f"Generation {generation}, Fidelity: {current_fidelity:.10f}, Fixed tmax: {fixed_tmax:.6f} us")
            if current_fidelity >= optimizer_params['fidelity_threshold']:
                logger.info(f"Switching to Adam after reaching fidelity threshold: {current_fidelity:.10f}")
                break
        return best_solution, fidelity_gens

    V_combined_init = np.concatenate([V0_init.flatten(), V1_init.flatten()])
    best_solution, fidelity_gens = run_sade(V_combined_init)
    if fidelity_gens[-1] >= optimizer_params['fidelity_threshold']:
        logger.info("\n--- Switching to Adam Optimizer ---\n")
        best_solution, fidelity_gens = adam_optimization(
            best_solution, APIC_params, atom_beam_params, control_Vt_params,
            system_params, optimizer_params, fidelity_gens
        )
    else:
        logger.info("\n--- Fidelity Threshold Not Reached. Skipping Adam Optimizer ---\n")
    return best_solution, fidelity_gens
