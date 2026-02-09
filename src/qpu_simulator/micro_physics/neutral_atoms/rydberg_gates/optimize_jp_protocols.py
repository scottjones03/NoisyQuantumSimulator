#!/usr/bin/env python3
"""
JP Protocol Parameter Optimization  [DEPRECATED]
==================================================

.. deprecated::
    This module is DEPRECATED. Use ``optimize_cz_gate.py`` instead, which:
    - Works for all 3 protocols (LP, JP bang-bang, smooth JP)
    - Passes parameters directly through simulation_inputs (no monkey-patching)
    - Includes memoization, apparatus constraints, and discrete variant search
    
    Migration example::
    
        from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.optimize_cz_gate import (
            optimize_cz_gate, ApparatusConstraints
        )
        result = optimize_cz_gate("smooth_jp", ApparatusConstraints())

WARNING: This module monkey-patches module-level globals in protocols.py,
which is fragile and was the root cause of optimization failures. The
simulation reads parameters from simulation_inputs dataclass fields, not
from module globals — so patching globals has NO EFFECT on the actual
simulation.

Author: Quantum Simulation Team
"""

import warnings
warnings.warn(
    "optimize_jp_protocols is deprecated. Use optimize_cz_gate instead. "
    "This module monkey-patches module globals which never reach the simulator. "
    "See optimize_cz_gate.py for the correct approach.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time

# Import simulation components
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    LPSimulationInputs,
    JPSimulationInputs,
    SmoothJPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default simulation parameters for reasonable V/Ω
DEFAULT_SPACING_FACTOR = 2.8   # Gives V/Ω ~ 50-100 for typical Rb87
DEFAULT_N_RYDBERG = 70
DEFAULT_TEMPERATURE = 2e-6     # 2 µK
DEFAULT_TWEEZER_POWER = 0.020  # 20 mW
DEFAULT_TWEEZER_WAIST = 0.8e-6 # 0.8 µm


def create_noiseless_config() -> Tuple[TwoPhotonExcitationConfig, NoiseSourceConfig]:
    """Create configuration with all noise disabled for coherent optimization."""
    # LaserParameters: power, waist, polarization, polarization_purity, linewidth_hz
    laser_1 = LaserParameters(
        power=50e-6,        # 50 µW (first leg, 780 nm)
        waist=50e-6,        # 50 µm
        polarization="pi",
        polarization_purity=1.0,  # Perfect polarization
        linewidth_hz=100,   # Very narrow
    )
    laser_2 = LaserParameters(
        power=0.3,          # 300 mW (second leg, 480 nm)
        waist=50e-6,        # 50 µm
        polarization="sigma+",
        polarization_purity=1.0,
        linewidth_hz=100,
    )
    # TwoPhotonExcitationConfig: laser_1, laser_2, Delta_e, counter_propagating
    excitation = TwoPhotonExcitationConfig(
        laser_1=laser_1,
        laser_2=laser_2,
        Delta_e=5e9,  # 5 GHz intermediate detuning (in Hz, NOT rad/s!)
        counter_propagating=True,
    )
    # NoiseSourceConfig: individual noise toggles
    noise = NoiseSourceConfig(
        include_spontaneous_emission=False,
        include_intermediate_scattering=False,
        include_motional_dephasing=False,
        include_doppler_dephasing=False,
        include_intensity_noise=False,
        intensity_noise_frac=0.0,
        include_laser_dephasing=False,
        include_magnetic_dephasing=False,
    )
    return excitation, noise


# =============================================================================
# PHASE EXTRACTION UTILITIES
# =============================================================================

def extract_phase_metrics(result) -> Dict[str, float]:
    """
    Extract phase metrics from SimulationResult.
    
    The simulation already computes controlled phase in result.phase_info.
    We just need to extract the relevant metrics.
    
    Returns dict with phase error and population info.
    """
    phase_info = result.phase_info
    fidelities = result.fidelities
    
    metrics = {
        'controlled_phase_deg': phase_info.get('controlled_phase_deg', np.nan),
        'controlled_phase_rad': phase_info.get('controlled_phase_rad', np.nan),
        'phase_error_deg': phase_info.get('phase_error_from_pi_deg', np.nan),
        'phase_error_rad': phase_info.get('phase_error_from_pi', np.nan),
        'cz_phase_fidelity': phase_info.get('cz_phase_fidelity', np.nan),
        'f11': fidelities.get('11', np.nan),
        'f00': fidelities.get('00', np.nan),
        'f01': fidelities.get('01', np.nan),
        'f10': fidelities.get('10', np.nan),
        'avg_fidelity': result.avg_fidelity,
    }
    
    return metrics


# =============================================================================
# JP BANG-BANG OPTIMIZATION
# =============================================================================

@dataclass
class JPBangBangOptResult:
    """Result of JP bang-bang optimization."""
    success: bool
    switching_times: List[float]
    phases: List[float]
    omega_tau: float
    controlled_phase_deg: float
    phase_error_deg: float
    f11_fidelity: float
    fidelity: float
    n_evaluations: int
    runtime_seconds: float
    message: str


def jp_bangbang_cost(
    params: np.ndarray,
    n_segments: int,
    excitation: TwoPhotonExcitationConfig,
    noise: NoiseSourceConfig,
    verbose: bool = False,
) -> float:
    """
    Cost function for JP bang-bang optimization.
    
    Parameters are:
    - switching_times: (n_segments - 1) values (dimensionless Ωt)
    - omega_tau: total pulse area
    
    Phases are fixed to symmetric pattern: [π/2, 0, -π/2, 0, π/2] for 5 segments
    """
    # Unpack parameters
    omega_tau = params[0]
    switching_times = list(params[1:])
    
    # Ensure switching times are sorted
    switching_times = sorted(switching_times)
    
    # Ensure last switching time < omega_tau
    if switching_times[-1] >= omega_tau:
        return 1e6  # Invalid configuration
    
    # Define symmetric phase pattern
    if n_segments == 5:
        phases = [np.pi/2, 0, -np.pi/2, 0, np.pi/2]
    elif n_segments == 7:
        phases = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]
    else:
        # General symmetric pattern
        phases = []
        for i in range(n_segments):
            if i % 2 == 0:
                phases.append(np.pi/2 * (1 - 2 * ((i // 2) % 2)))
            else:
                phases.append(0)
    
    # Create JPSimulationInputs with custom parameters
    # We need to modify the protocol parameters before simulation
    # For now, use the existing infrastructure
    from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.protocols import (
        JP_SWITCHING_TIMES_VALIDATED, JP_PHASES_VALIDATED, JP_OMEGA_TAU_VALIDATED
    )
    
    # Temporarily patch the constants (ugly but works for optimization)
    import src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.protocols as protocols
    old_switching = protocols.JP_SWITCHING_TIMES_VALIDATED
    old_phases = protocols.JP_PHASES_VALIDATED  
    old_omega_tau = protocols.JP_OMEGA_TAU_VALIDATED
    
    try:
        protocols.JP_SWITCHING_TIMES_VALIDATED = switching_times
        protocols.JP_PHASES_VALIDATED = phases
        protocols.JP_OMEGA_TAU_VALIDATED = omega_tau
        
        # Also patch JP_DEFAULT
        protocols.JP_DEFAULT = protocols.JPProtocolParameters(
            name="jandura_pupillo",
            omega_tau=omega_tau,
            switching_times=switching_times.copy(),
            phases=phases.copy(),
            n_pulses=1,
            reference="Optimization",
            adapted_for_V_over_Omega=200.0,
        )
        
        jp_inputs = JPSimulationInputs(excitation=excitation, noise=noise, omega_tau=omega_tau)
        result = simulate_CZ_gate(
            simulation_inputs=jp_inputs,
            spacing_factor=DEFAULT_SPACING_FACTOR,
            n_rydberg=DEFAULT_N_RYDBERG,
            temperature=DEFAULT_TEMPERATURE,
            tweezer_power=DEFAULT_TWEEZER_POWER,
            tweezer_waist=DEFAULT_TWEEZER_WAIST,
            include_noise=False,
            verbose=False,
        )
        
    finally:
        # Restore original constants
        protocols.JP_SWITCHING_TIMES_VALIDATED = old_switching
        protocols.JP_PHASES_VALIDATED = old_phases
        protocols.JP_OMEGA_TAU_VALIDATED = old_omega_tau
        protocols.JP_DEFAULT = protocols.JPProtocolParameters(
            name="jandura_pupillo",
            omega_tau=old_omega_tau,
            switching_times=old_switching.copy(),
            phases=old_phases.copy(),
            n_pulses=1,
            reference="Jandura & Pupillo, PRX Quantum 3, 010353 (2022)",
            adapted_for_V_over_Omega=200.0,
        )
    
    # Extract phases and compute cost
    metrics = extract_phase_metrics(result)
    
    if np.isnan(metrics['phase_error_deg']):
        return 1e6  # Failed to extract phases
    
    # Primary cost: phase error from ±π
    phase_error = metrics['phase_error_deg']
    
    # Secondary cost: |11⟩ fidelity (should be close to 1)
    f11 = metrics.get('f11', 0.0)
    if np.isnan(f11):
        f11 = 0.0
    population_penalty = 100 * (1 - f11)  # Penalize low fidelity
    
    # Total cost
    cost = phase_error + 0.1 * population_penalty
    
    if verbose:
        print(f"  Ωτ={omega_tau:.2f}, switch={switching_times}, "
              f"phase_err={phase_error:.1f}°, F11={f11:.3f}, cost={cost:.2f}")
    
    return cost


def optimize_jp_bangbang(
    n_segments: int = 5,
    excitation: TwoPhotonExcitationConfig = None,
    noise: NoiseSourceConfig = None,
    method: str = "differential_evolution",
    maxiter: int = 100,
    verbose: bool = True,
) -> JPBangBangOptResult:
    """
    Optimize JP bang-bang protocol parameters.
    
    Parameters
    ----------
    n_segments : int
        Number of phase segments (5 or 7 typically)
    excitation : TwoPhotonExcitationConfig
        Excitation configuration. If None, uses default noiseless.
    noise : NoiseSourceConfig
        Noise configuration. If None, uses noiseless.
    method : str
        Optimization method: "differential_evolution", "dual_annealing", "nelder_mead"
    maxiter : int
        Maximum iterations
    verbose : bool
        Print progress
        
    Returns
    -------
    JPBangBangOptResult
        Optimization result with optimal parameters
    """
    if excitation is None or noise is None:
        excitation, noise = create_noiseless_config()
    
    start_time = time.time()
    n_evals = [0]
    
    def wrapped_cost(params):
        n_evals[0] += 1
        return jp_bangbang_cost(params, n_segments, excitation, noise, verbose=False)
    
    # Parameter bounds
    # omega_tau typically 6-25, switching times within [0, omega_tau]
    n_switches = n_segments - 1
    
    # Initial guess based on equal spacing
    omega_tau_init = 15.0
    switch_init = np.linspace(1.0, omega_tau_init - 1.0, n_switches)
    x0 = np.concatenate([[omega_tau_init], switch_init])
    
    # Bounds
    bounds = [(6.0, 30.0)]  # omega_tau
    for i in range(n_switches):
        bounds.append((0.5, 29.0))  # switching times
    
    if verbose:
        print(f"Optimizing JP bang-bang with {n_segments} segments...")
        print(f"  Parameters: omega_tau + {n_switches} switching times")
        print(f"  Bounds: Ωτ ∈ [6, 30], switches ∈ [0.5, 29]")
        print(f"  Method: {method}")
    
    # Run optimization
    if method == "differential_evolution":
        result = differential_evolution(
            wrapped_cost,
            bounds=bounds,
            maxiter=maxiter,
            seed=42,
            polish=True,
            disp=verbose,
        )
        opt_params = result.x
        success = result.success
        message = result.message
        
    elif method == "dual_annealing":
        result = dual_annealing(
            wrapped_cost,
            bounds=bounds,
            maxiter=maxiter,
            seed=42,
        )
        opt_params = result.x
        success = result.success
        message = str(result.message)
        
    else:  # nelder_mead or other
        result = minimize(
            wrapped_cost,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'disp': verbose}
        )
        opt_params = result.x
        success = result.success
        message = result.message
    
    # Extract optimal parameters
    omega_tau_opt = opt_params[0]
    switching_times_opt = sorted(opt_params[1:])
    
    # Final evaluation for detailed metrics
    if n_segments == 5:
        phases_opt = [np.pi/2, 0, -np.pi/2, 0, np.pi/2]
    elif n_segments == 7:
        phases_opt = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]
    else:
        phases_opt = [np.pi/2 * (1 - 2 * ((i // 2) % 2)) if i % 2 == 0 else 0 
                      for i in range(n_segments)]
    
    # Run final simulation with optimal parameters
    import src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.protocols as protocols
    old_switching = protocols.JP_SWITCHING_TIMES_VALIDATED
    old_phases = protocols.JP_PHASES_VALIDATED
    old_omega_tau = protocols.JP_OMEGA_TAU_VALIDATED
    
    try:
        protocols.JP_SWITCHING_TIMES_VALIDATED = list(switching_times_opt)
        protocols.JP_PHASES_VALIDATED = phases_opt
        protocols.JP_OMEGA_TAU_VALIDATED = omega_tau_opt
        protocols.JP_DEFAULT = protocols.JPProtocolParameters(
            name="jandura_pupillo",
            omega_tau=omega_tau_opt,
            switching_times=list(switching_times_opt),
            phases=phases_opt.copy(),
            n_pulses=1,
            reference="Optimization",
            adapted_for_V_over_Omega=200.0,
        )
        
        jp_inputs = JPSimulationInputs(excitation=excitation, noise=noise, omega_tau=omega_tau_opt)
        final_result = simulate_CZ_gate(
            simulation_inputs=jp_inputs,
            spacing_factor=DEFAULT_SPACING_FACTOR,
            n_rydberg=DEFAULT_N_RYDBERG,
            temperature=DEFAULT_TEMPERATURE,
            tweezer_power=DEFAULT_TWEEZER_POWER,
            tweezer_waist=DEFAULT_TWEEZER_WAIST,
            include_noise=False,
            verbose=False,
        )
        
    finally:
        protocols.JP_SWITCHING_TIMES_VALIDATED = old_switching
        protocols.JP_PHASES_VALIDATED = old_phases
        protocols.JP_OMEGA_TAU_VALIDATED = old_omega_tau
        protocols.JP_DEFAULT = protocols.JPProtocolParameters(
            name="jandura_pupillo",
            omega_tau=old_omega_tau,
            switching_times=old_switching.copy(),
            phases=old_phases.copy(),
            n_pulses=1,
            reference="Jandura & Pupillo, PRX Quantum 3, 010353 (2022)",
            adapted_for_V_over_Omega=200.0,
        )
    
    metrics = extract_phase_metrics(final_result)
    
    runtime = time.time() - start_time
    
    opt_result = JPBangBangOptResult(
        success=success,
        switching_times=list(switching_times_opt),
        phases=phases_opt,
        omega_tau=omega_tau_opt,
        controlled_phase_deg=metrics.get('controlled_phase_deg', np.nan),
        phase_error_deg=metrics.get('phase_error_deg', np.nan),
        f11_fidelity=metrics.get('f11', np.nan),
        fidelity=metrics.get('avg_fidelity', np.nan),
        n_evaluations=n_evals[0],
        runtime_seconds=runtime,
        message=str(message),
    )
    
    if verbose:
        print(f"\n=== JP Bang-Bang Optimization Complete ===")
        print(f"Success: {success}")
        print(f"Optimal Ωτ: {omega_tau_opt:.4f}")
        print(f"Switching times: {[f'{t:.4f}' for t in switching_times_opt]}")
        print(f"Phases: {phases_opt}")
        print(f"Controlled phase: {opt_result.controlled_phase_deg:.1f}° (target: ±180°)")
        print(f"Phase error: {opt_result.phase_error_deg:.1f}°")
        print(f"|11⟩ fidelity: {opt_result.f11_fidelity:.4f}")
        print(f"Avg fidelity: {opt_result.fidelity:.4f}")
        print(f"Evaluations: {n_evals[0]}, Runtime: {runtime:.1f}s")
    
    return opt_result


# =============================================================================
# SMOOTH JP OPTIMIZATION
# =============================================================================

@dataclass
class SmoothJPOptResult:
    """Result of smooth JP optimization."""
    success: bool
    A: float
    omega_mod_ratio: float
    phi_offset: float
    delta_over_omega: float
    omega_tau: float
    controlled_phase_deg: float
    phase_error_deg: float
    f11_fidelity: float
    fidelity: float
    n_evaluations: int
    runtime_seconds: float
    message: str


def smooth_jp_cost(
    params: np.ndarray,
    excitation: TwoPhotonExcitationConfig,
    noise: NoiseSourceConfig,
    verbose: bool = False,
) -> float:
    """
    Cost function for smooth JP optimization.
    
    Parameters:
    - A: phase amplitude (radians)
    - omega_mod_ratio: ω_mod/Ω ratio
    - phi_offset: phase offset (radians)
    - delta_over_omega: δ₀/Ω ratio
    - omega_tau: pulse area
    """
    A, omega_mod_ratio, phi_offset, delta_over_omega, omega_tau = params
    
    # Create SmoothJPSimulationInputs with these parameters
    smooth_inputs = SmoothJPSimulationInputs(
        excitation=excitation,
        noise=noise,
        omega_tau=omega_tau,
        A=A,
        omega_mod_ratio=omega_mod_ratio,
        phi_offset=phi_offset,
        delta_over_omega=delta_over_omega,
    )
    
    try:
        result = simulate_CZ_gate(
            simulation_inputs=smooth_inputs,
            spacing_factor=DEFAULT_SPACING_FACTOR,
            n_rydberg=DEFAULT_N_RYDBERG,
            temperature=DEFAULT_TEMPERATURE,
            tweezer_power=DEFAULT_TWEEZER_POWER,
            tweezer_waist=DEFAULT_TWEEZER_WAIST,
            include_noise=False,
            verbose=False,
        )
    except Exception as e:
        if verbose:
            print(f"  Simulation failed: {e}")
        return 1e6
    
    # Extract metrics
    metrics = extract_phase_metrics(result)
    
    if np.isnan(metrics['phase_error_deg']):
        return 1e6
    
    # Primary cost: phase error from ±π
    phase_error = metrics['phase_error_deg']
    
    # Secondary cost: |11⟩ fidelity
    f11 = metrics.get('f11', 0.0)
    if np.isnan(f11):
        f11 = 0.0
    population_penalty = 100 * (1 - f11)
    
    # Total cost
    cost = phase_error + 0.1 * population_penalty
    
    if verbose:
        print(f"  A={A:.3f}, ω_mod={omega_mod_ratio:.3f}, ϕ={phi_offset:.3f}, "
              f"δ/Ω={delta_over_omega:.4f}, Ωτ={omega_tau:.2f} -> "
              f"phase_err={phase_error:.1f}°, F11={f11:.3f}, cost={cost:.2f}")
    
    return cost


def optimize_smooth_jp(
    excitation: TwoPhotonExcitationConfig = None,
    noise: NoiseSourceConfig = None,
    method: str = "differential_evolution",
    maxiter: int = 100,
    verbose: bool = True,
) -> SmoothJPOptResult:
    """
    Optimize smooth JP protocol parameters.
    
    Parameters
    ----------
    excitation : TwoPhotonExcitationConfig
        Excitation configuration. If None, uses default noiseless.
    noise : NoiseSourceConfig
        Noise configuration. If None, uses noiseless.
    method : str
        Optimization method
    maxiter : int
        Maximum iterations
    verbose : bool
        Print progress
        
    Returns
    -------
    SmoothJPOptResult
        Optimization result with optimal parameters
    """
    if excitation is None or noise is None:
        excitation, noise = create_noiseless_config()
    
    start_time = time.time()
    n_evals = [0]
    
    def wrapped_cost(params):
        n_evals[0] += 1
        return smooth_jp_cost(params, excitation, noise, verbose=False)
    
    # Parameter bounds based on validated ranges and physics
    # A: amplitude, typically 0 to π (0 to ~3.14)
    # omega_mod_ratio: typically 0.5 to 2 (modulation freq ~ Rabi freq)
    # phi_offset: 0 to 2π
    # delta_over_omega: -0.1 to 0.1 (small detuning)
    # omega_tau: 5 to 20
    bounds = [
        (0.1, np.pi),      # A
        (0.5, 2.5),        # omega_mod_ratio
        (0.0, 2*np.pi),    # phi_offset
        (-0.1, 0.1),       # delta_over_omega
        (5.0, 20.0),       # omega_tau
    ]
    
    # Initial guess from validated Bluvstein parameters
    x0 = [
        0.311 * np.pi,  # A
        1.242,          # omega_mod_ratio
        4.696,          # phi_offset
        0.0205,         # delta_over_omega
        10.09,          # omega_tau
    ]
    
    if verbose:
        print(f"Optimizing smooth JP protocol...")
        print(f"  Parameters: A, ω_mod/Ω, ϕ, δ/Ω, Ωτ")
        print(f"  Initial (Bluvstein): A={x0[0]/np.pi:.3f}π, ω_mod/Ω={x0[1]:.3f}, "
              f"ϕ={x0[2]:.3f}, δ/Ω={x0[3]:.4f}, Ωτ={x0[4]:.2f}")
        print(f"  Method: {method}")
    
    # Run optimization
    if method == "differential_evolution":
        result = differential_evolution(
            wrapped_cost,
            bounds=bounds,
            maxiter=maxiter,
            seed=42,
            polish=True,
            disp=verbose,
        )
        opt_params = result.x
        success = result.success
        message = result.message
        
    elif method == "dual_annealing":
        result = dual_annealing(
            wrapped_cost,
            bounds=bounds,
            maxiter=maxiter,
            seed=42,
        )
        opt_params = result.x
        success = result.success
        message = str(result.message)
        
    else:
        result = minimize(
            wrapped_cost,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'disp': verbose}
        )
        opt_params = result.x
        success = result.success
        message = result.message
    
    # Extract optimal parameters
    A_opt, omega_mod_opt, phi_opt, delta_opt, omega_tau_opt = opt_params
    
    # Final evaluation
    smooth_inputs = SmoothJPSimulationInputs(
        excitation=excitation,
        noise=noise,
        omega_tau=omega_tau_opt,
        A=A_opt,
        omega_mod_ratio=omega_mod_opt,
        phi_offset=phi_opt,
        delta_over_omega=delta_opt,
    )
    final_result = simulate_CZ_gate(
        simulation_inputs=smooth_inputs,
        spacing_factor=DEFAULT_SPACING_FACTOR,
        n_rydberg=DEFAULT_N_RYDBERG,
        temperature=DEFAULT_TEMPERATURE,
        tweezer_power=DEFAULT_TWEEZER_POWER,
        tweezer_waist=DEFAULT_TWEEZER_WAIST,
        include_noise=False,
        verbose=False,
    )
    metrics = extract_phase_metrics(final_result)
    
    runtime = time.time() - start_time
    
    opt_result = SmoothJPOptResult(
        success=success,
        A=A_opt,
        omega_mod_ratio=omega_mod_opt,
        phi_offset=phi_opt,
        delta_over_omega=delta_opt,
        omega_tau=omega_tau_opt,
        controlled_phase_deg=metrics.get('controlled_phase_deg', np.nan),
        phase_error_deg=metrics.get('phase_error_deg', np.nan),
        f11_fidelity=metrics.get('f11', np.nan),
        fidelity=metrics.get('avg_fidelity', np.nan),
        n_evaluations=n_evals[0],
        runtime_seconds=runtime,
        message=str(message),
    )
    
    if verbose:
        print(f"\n=== Smooth JP Optimization Complete ===")
        print(f"Success: {success}")
        print(f"Optimal parameters:")
        print(f"  A = {A_opt:.4f} rad ({A_opt/np.pi:.4f}π)")
        print(f"  ω_mod/Ω = {omega_mod_opt:.4f}")
        print(f"  ϕ = {phi_opt:.4f} rad")
        print(f"  δ/Ω = {delta_opt:.5f}")
        print(f"  Ωτ = {omega_tau_opt:.4f}")
        print(f"Controlled phase: {opt_result.controlled_phase_deg:.1f}° (target: ±180°)")
        print(f"Phase error: {opt_result.phase_error_deg:.1f}°")
        print(f"|11⟩ fidelity: {opt_result.f11_fidelity:.4f}")
        print(f"Avg fidelity: {opt_result.fidelity:.4f}")
        print(f"Evaluations: {n_evals[0]}, Runtime: {runtime:.1f}s")
    
    return opt_result


# =============================================================================
# LP BASELINE
# =============================================================================

def evaluate_lp_baseline(verbose: bool = True) -> dict:
    """
    Evaluate LP protocol as baseline for comparison.
    """
    excitation, noise = create_noiseless_config()
    lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise)
    result = simulate_CZ_gate(
        simulation_inputs=lp_inputs,
        spacing_factor=DEFAULT_SPACING_FACTOR,
        n_rydberg=DEFAULT_N_RYDBERG,
        temperature=DEFAULT_TEMPERATURE,
        tweezer_power=DEFAULT_TWEEZER_POWER,
        tweezer_waist=DEFAULT_TWEEZER_WAIST,
        include_noise=False,
        verbose=False,
    )
    metrics = extract_phase_metrics(result)
    
    if verbose:
        print(f"\n=== LP Protocol Baseline (Noise-Free) ===")
        print(f"V/Ω = {result.V_over_Omega:.1f}")
        print(f"Fidelity: {metrics.get('avg_fidelity', np.nan):.4f}")
        print(f"Controlled phase: {metrics.get('controlled_phase_deg', np.nan):.1f}°")
        print(f"Phase error: {metrics.get('phase_error_deg', np.nan):.1f}°")
        print(f"|11⟩ fidelity: {metrics.get('f11', np.nan):.4f}")
    
    return {
        'fidelity': metrics.get('avg_fidelity', np.nan),
        'controlled_phase_deg': metrics.get('controlled_phase_deg', np.nan),
        'phase_error_deg': metrics.get('phase_error_deg', np.nan),
        'f11_fidelity': metrics.get('f11', np.nan),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("JP Protocol Parameter Optimization")
    print("=" * 70)
    
    # First, evaluate LP baseline
    lp_baseline = evaluate_lp_baseline()
    
    # Optimize smooth JP (usually faster and already close to optimal)
    print("\n" + "=" * 70)
    smooth_result = optimize_smooth_jp(
        method="differential_evolution",
        maxiter=50,
        verbose=True,
    )
    
    # Optimize JP bang-bang (this is the problematic one)
    print("\n" + "=" * 70)
    bangbang_result = optimize_jp_bangbang(
        n_segments=5,
        method="differential_evolution",
        maxiter=50,
        verbose=True,
    )
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON (Noise-Free)")
    print("=" * 70)
    print(f"{'Protocol':<20} {'Fidelity':>12} {'Phase Err':>12} {'|11⟩ F':>12}")
    print("-" * 58)
    print(f"{'LP (baseline)':<20} {lp_baseline['fidelity']:>12.4f} "
          f"{lp_baseline['phase_error_deg']:>11.1f}° {lp_baseline['f11_fidelity']:>12.4f}")
    print(f"{'Smooth JP (opt)':<20} {smooth_result.fidelity:>12.4f} "
          f"{smooth_result.phase_error_deg:>11.1f}° {smooth_result.f11_fidelity:>12.4f}")
    print(f"{'JP Bang-Bang (opt)':<20} {bangbang_result.fidelity:>12.4f} "
          f"{bangbang_result.phase_error_deg:>11.1f}° {bangbang_result.f11_fidelity:>12.4f}")
    
    # Generate code for updating protocols.py
    print("\n" + "=" * 70)
    print("SUGGESTED UPDATES FOR protocols.py")
    print("=" * 70)
    
    if smooth_result.phase_error_deg < 5.0:  # Within 5 degrees of correct
        print("\n# Optimized Smooth JP Parameters:")
        print(f"SMOOTH_JP_PARAMS_OPTIMIZED = {{")
        print(f'    "A": {smooth_result.A:.6f},  # {smooth_result.A/np.pi:.4f}π rad')
        print(f'    "omega_mod_ratio": {smooth_result.omega_mod_ratio:.6f},')
        print(f'    "phi_offset": {smooth_result.phi_offset:.6f},')
        print(f'    "delta_over_omega": {smooth_result.delta_over_omega:.6f},')
        print(f'    "omega_tau": {smooth_result.omega_tau:.6f},')
        print(f"    # Phase error: {smooth_result.phase_error_deg:.1f}°, Fidelity: {smooth_result.fidelity:.4f}")
        print(f"}}")
    
    if bangbang_result.phase_error_deg < 30.0:  # Reasonable improvement
        print("\n# Optimized JP Bang-Bang Parameters:")
        print(f"JP_SWITCHING_TIMES_OPTIMIZED = {bangbang_result.switching_times}")
        print(f"JP_PHASES_OPTIMIZED = {bangbang_result.phases}")
        print(f"JP_OMEGA_TAU_OPTIMIZED = {bangbang_result.omega_tau:.6f}")
        print(f"# Phase error: {bangbang_result.phase_error_deg:.1f}°, Fidelity: {bangbang_result.fidelity:.4f}")
