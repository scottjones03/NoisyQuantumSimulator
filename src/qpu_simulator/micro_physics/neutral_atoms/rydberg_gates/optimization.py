"""
CZ Gate Hardware Parameter Optimization (Inverse Problem)
=========================================================

This module solves the **inverse problem**: given a *target* gate fidelity
and gate time, find the hardware configuration (laser powers, temperature,
atom spacing, etc.) that achieves them.

Contrast with :mod:`optimize_cz_gate` which solves the **forward problem**:
given fixed apparatus constraints, find the best *protocol* parameters.

Supported Protocols
-------------------
- **Levine-Pichler (LP)** — aliases: ``"levine_pichler"``, ``"lp"``,
  ``"two_pulse"``.  Two-pulse Rydberg excitation with static detuning.
  Protocol params (δ/Ω, Ωτ) can optionally be co-optimised.

- **Jandura-Pupillo / Smooth JP** — aliases: ``"jandura_pupillo"``,
  ``"jp"``, ``"smooth_jp"``, ``"single_pulse"``, ``"time_optimal"``.
  Internally uses the smooth JP solver (continuous sinusoidal modulation).
  Protocol param (Ωτ) can optionally be co-optimised.

.. note::

   For direct control over smooth-JP-specific parameters (``A``,
   ``omega_mod_ratio``, ``phi_offset``), use :func:`optimize_cz_gate.optimize_cz_gate`
   instead — it accepts ``SmoothJPSimulationInputs`` and can tune all
   five JP shape parameters.

Key Components
--------------
1. **optimize_CZ_parameters()**: Inverse optimisation via differential evolution
   - Finds hardware params for target fidelity/time
   - Optimises laser powers, linewidth, temperature, spacing

2. **explore_parameter_space()**: Efficient Pareto exploration
   - Caches ALL evaluations during optimisation
   - Extracts Pareto front post-hoc
   - Much faster than grid search

3. **ExplorationResult**: Container for exploration data
   - Stores all evaluated points
   - Computes Pareto frontier
   - Supports filtering and analysis

Usage
-----
>>> # Single optimisation — LP protocol
>>> result = optimize_CZ_parameters(target_fidelity=0.99, target_gate_time_ns=300)
>>> print(result.optimal_parameters)

>>> # Single optimisation — JP protocol (routes through smooth JP internally)
>>> result = optimize_CZ_parameters(protocol="smooth_jp", target_fidelity=0.995)

>>> # Pareto exploration
>>> exploration = explore_parameter_space(protocol="levine_pichler", maxiter=50)
>>> print(exploration.summary())
>>> best = exploration.get_best_for_target(target_fidelity=0.995)

References
----------
- Uses scipy.optimize.differential_evolution for global optimisation
- Handles non-convex optimisation landscape from quantum physics
"""

from __future__ import annotations

import time
import pickle
import warnings
import numpy as np
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple, Optional, Any, Union

# Local imports
from .simulation import simulate_CZ_gate
from .configurations import (
    LPSimulationInputs,
    JPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)


# =============================================================================
# OPTIMIZATION RESULT DATACLASS
# =============================================================================

@dataclass
class HardwareOptimizationResult:
    """
    Container for hardware parameter optimization results.
    
    This is the result of the *inverse* problem: given a target fidelity and
    gate time, find the hardware parameters (laser power, linewidth, etc.)
    that achieve them. See also ``optimize_cz_gate.OptimizationResult`` for
    the *forward* problem result (given apparatus constraints, find the best
    protocol parameters).
    """
    success: bool
    target_fidelity: float
    target_gate_time_ns: float
    achieved_fidelity: float
    achieved_gate_time_ns: float
    fidelity_error_pct: float
    gate_time_error_pct: float
    optimal_parameters: Dict[str, float]
    V_over_Omega: float
    noise_breakdown: Dict[str, float]
    n_evaluations: int
    final_cost: float
    message: str
    
    def __repr__(self):
        return f"""HardwareOptimizationResult(
  Target:   F={self.target_fidelity:.4f}, t={self.target_gate_time_ns:.1f} ns
  Achieved: F={self.achieved_fidelity:.4f}, t={self.achieved_gate_time_ns:.1f} ns
  Errors:   ΔF={self.fidelity_error_pct:+.2f}%, Δt={self.gate_time_error_pct:+.2f}%
  V/Ω={self.V_over_Omega:.1f}, Evals={self.n_evaluations}, Cost={self.final_cost:.2e}
  Success: {self.success}
)"""


# =============================================================================
# EXPLORATION DATACLASSES
# =============================================================================

@dataclass
class EvaluatedPoint:
    """A single evaluated parameter configuration."""
    # Physical parameters
    Omega_MHz: float
    laser_linewidth_kHz: float
    V_over_Omega: float
    
    # Achieved metrics
    fidelity: float
    gate_time_ns: float
    infidelity: float  # 1 - fidelity
    
    # Noise breakdown
    noise_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Protocol info
    protocol: str = ""
    species: str = "Rb87"


@dataclass
class ExplorationResult:
    """Container for all evaluated points from parameter space exploration."""
    
    protocol: str
    species: str
    
    # All evaluated points
    points: List[EvaluatedPoint] = field(default_factory=list)
    
    # Pareto frontier (computed after exploration)
    pareto_front: List[EvaluatedPoint] = field(default_factory=list)
    
    # Metadata
    n_evaluations: int = 0
    runtime_seconds: float = 0.0
    optimizer_settings: Dict[str, Any] = field(default_factory=dict)
    
    def add_point(self, point: EvaluatedPoint):
        """Add an evaluated point."""
        self.points.append(point)
        self.n_evaluations += 1
    
    def compute_pareto_front(self):
        """
        Compute Pareto-optimal points for the fidelity vs speed trade-off.
        A point is Pareto-optimal if no other point is better in BOTH fidelity AND speed.
        
        Note: No V/Ω filtering is applied. With correct physics (V in rad/s),
        weak blockade configurations naturally have low fidelity and won't 
        appear on the Pareto front unless they're actually optimal.
        """
        if not self.points:
            return
        
        # Sort by gate time (ascending)
        sorted_points = sorted(self.points, key=lambda p: p.gate_time_ns)
        
        pareto = []
        best_fidelity_so_far = -1
        
        for point in sorted_points:
            # A point is Pareto-optimal if it has better fidelity than all faster points
            if point.fidelity > best_fidelity_so_far:
                pareto.append(point)
                best_fidelity_so_far = point.fidelity
        
        self.pareto_front = pareto
    
    def get_points_above_fidelity(self, min_fidelity: float) -> List[EvaluatedPoint]:
        """Filter points above a minimum fidelity threshold."""
        return [p for p in self.points if p.fidelity >= min_fidelity]
    
    def get_points_below_time(self, max_time_ns: float) -> List[EvaluatedPoint]:
        """Filter points below a maximum gate time."""
        return [p for p in self.points if p.gate_time_ns <= max_time_ns]
    
    def get_best_for_target(
        self, 
        target_fidelity: float = None, 
        target_time_ns: float = None
    ) -> Optional[EvaluatedPoint]:
        """
        Find the best point for a given target.
        If target_fidelity is specified: find fastest point meeting that fidelity.
        If target_time_ns is specified: find highest fidelity point within that time.
        """
        if target_fidelity is not None:
            candidates = [p for p in self.points if p.fidelity >= target_fidelity]
            if not candidates:
                return None
            return min(candidates, key=lambda p: p.gate_time_ns)
        
        if target_time_ns is not None:
            candidates = [p for p in self.points if p.gate_time_ns <= target_time_ns]
            if not candidates:
                return None
            return max(candidates, key=lambda p: p.fidelity)
        
        return None
    
    def summary(self) -> str:
        """Generate summary string."""
        if not self.points:
            return "No points evaluated."
        
        fidelities = [p.fidelity for p in self.points]
        times = [p.gate_time_ns for p in self.points]
        
        lines = [
            f"{'='*60}",
            f"EXPLORATION RESULTS: {self.protocol.upper()}",
            f"{'='*60}",
            f"Total evaluations: {self.n_evaluations}",
            f"Runtime: {self.runtime_seconds:.1f}s ({self.runtime_seconds/60:.1f} min)",
            f"",
            f"Fidelity range: {min(fidelities)*100:.2f}% - {max(fidelities)*100:.2f}%",
            f"Gate time range: {min(times):.1f} - {max(times):.1f} ns",
            f"",
            f"Pareto front: {len(self.pareto_front)} points",
        ]
        
        if self.pareto_front:
            lines.append("")
            lines.append("Key Pareto points:")
            for p in self.pareto_front[:5]:  # Show first 5
                lines.append(f"  F={p.fidelity*100:.2f}% @ {p.gate_time_ns:.1f}ns (V/Ω={p.V_over_Omega:.1f})")
            if len(self.pareto_front) > 5:
                lines.append(f"  ... and {len(self.pareto_front)-5} more")
        
        return "\n".join(lines)
    
    def save(self, filepath: str):
        """Save results to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Saved {self.n_evaluations} points to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ExplorationResult':
        """Load results from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

def optimize_CZ_parameters(
    target_fidelity: float = 0.99,
    target_gate_time_ns: float = 300.0,
    
    # Protocol selection
    protocol: str = "levine_pichler",
    
    # Weight parameters for objective function
    weight_fidelity: float = 1.0,
    weight_time: float = 0.5,
    constraint_penalty: float = 100.0,
    
    # Fixed parameters (passed through to simulate_CZ_gate)
    species: str = "Rb87",
    background_loss_rate_hz: float = 10.0,
    include_noise: bool = True,
    include_motional_dephasing: bool = True,
    
    # === HARDWARE PARAMETER BOUNDS ===
    bounds_rydberg_power_2: Tuple[float, float] = (0.5, 50.0),
    bounds_rydberg_power_1: Tuple[float, float] = (0.1e-3, 20e-3),
    bounds_temperature: Tuple[float, float] = (0.1e-6, 20e-6),
    bounds_spacing_factor: Tuple[float, float] = (1.8, 6.0),
    bounds_n_rydberg: Tuple[int, int] = (40, 100),
    bounds_tweezer_power: Tuple[float, float] = (5e-3, 200e-3),
    bounds_tweezer_waist: Tuple[float, float] = (0.4e-6, 3.0e-6),
    bounds_Delta_e: Tuple[float, float] = (0.5e9, 15e9),
    bounds_laser_linewidth: Tuple[float, float] = (100.0, 50e3),
    
    # === PROTOCOL-SPECIFIC BOUNDS ===
    bounds_delta_over_omega: Tuple[float, float] = (0.30, 0.45),
    bounds_omega_tau_lp: Tuple[float, float] = (3.8, 5.0),
    bounds_omega_tau_jp: Tuple[float, float] = (5.5, 8.5),
    
    # Control which protocol params to optimize
    optimize_protocol_params: bool = True,
    
    # Power coupling mode
    couple_powers: bool = False,
    power_ratio_780_480: float = 0.001,
    
    # Optimizer settings
    maxiter: int = 100,
    tol: float = 1e-5,
    seed: Optional[int] = 42,
    polish: bool = True,
    workers: int = 1,
    popsize: int = 15,
    
    # Optional: fix certain parameters
    fixed_params: Optional[Dict[str, float]] = None,
    
    # Callback for progress reporting
    callback: Optional[Callable[[int, float, Dict], None]] = None,
    verbose: bool = True
    
) -> HardwareOptimizationResult:
    """
    Find optimal hardware parameters to achieve target gate fidelity and time.
    
    This is the **inverse problem**: given desired performance metrics, find the
    hardware configuration that achieves them (or gets as close as possible).
    The optimizer tunes laser powers, temperature, atom spacing, linewidth,
    and (optionally) protocol timing parameters.
    
    Parameters
    ----------
    target_fidelity : float
        Target average gate fidelity (0-1). Default 0.99 (99%).
    target_gate_time_ns : float
        Target gate time in nanoseconds. Default 300 ns.
    protocol : str
        Which CZ protocol to use.  Accepted values:
        
        - Levine-Pichler: ``"levine_pichler"``, ``"lp"``, ``"two_pulse"``
        - Jandura-Pupillo / smooth JP: ``"jandura_pupillo"``, ``"jp"``,
          ``"smooth_jp"``, ``"single_pulse"``, ``"time_optimal"``
        
        Both JP aliases route through the smooth JP solver internally.
        For full control over smooth-JP shape parameters (A,
        omega_mod_ratio, phi_offset), use :func:`optimize_cz_gate.optimize_cz_gate`
        instead.
    weight_fidelity : float
        Weight for fidelity error in objective. Default 1.0.
    weight_time : float
        Weight for timing error in objective. Default 0.5.
    species : str
        Atomic species ("Rb87", "Cs133", etc.).
    bounds_* : tuple
        (min, max) bounds for each optimisable parameter.
    couple_powers : bool
        If True, optimise a single "total power" and derive both powers
        from a fixed ratio.
    maxiter : int
        Maximum optimiser iterations.
    fixed_params : dict, optional
        Parameters to hold fixed (not optimise).
        
    Returns
    -------
    HardwareOptimizationResult
        Dataclass containing optimal parameters and achieved performance.
        
    Examples
    --------
    >>> # LP protocol
    >>> result = optimize_CZ_parameters(target_fidelity=0.995, target_gate_time_ns=200)
    >>> print(result.optimal_parameters)
    
    >>> # JP / smooth JP protocol
    >>> result = optimize_CZ_parameters(protocol="smooth_jp", target_fidelity=0.99)
    """
    
    if fixed_params is None:
        fixed_params = {}
    
    # Determine protocol type
    is_lp = protocol.lower() in ["levine_pichler", "lp", "two_pulse"]
    is_jp = protocol.lower() in ["jandura_pupillo", "jp", "smooth_jp", "single_pulse", "time_optimal"]
    
    if not is_lp and not is_jp:
        raise ValueError(
            f"Unknown protocol '{protocol}'. Use 'levine_pichler' (or 'lp') "
            f"or 'jandura_pupillo' / 'smooth_jp' (or 'jp')."
        )
    
    # Define parameter names and their bounds
    if couple_powers:
        param_config = {
            'total_power': (bounds_rydberg_power_2[0], bounds_rydberg_power_2[1]),
            'temperature': bounds_temperature,
            'spacing_factor': bounds_spacing_factor,
            'n_rydberg': bounds_n_rydberg,
            'tweezer_power': bounds_tweezer_power,
            'tweezer_waist': bounds_tweezer_waist,
            'laser_linewidth': bounds_laser_linewidth,
        }
    else:
        param_config = {
            'rydberg_power_2': bounds_rydberg_power_2,
            'rydberg_power_1': bounds_rydberg_power_1,
            'temperature': bounds_temperature,
            'spacing_factor': bounds_spacing_factor,
            'n_rydberg': bounds_n_rydberg,
            'tweezer_power': bounds_tweezer_power,
            'tweezer_waist': bounds_tweezer_waist,
            'laser_linewidth': bounds_laser_linewidth,
        }
    
    # Add protocol-specific parameters
    if is_lp:
        param_config['Delta_e'] = bounds_Delta_e
        if optimize_protocol_params:
            param_config['delta_over_omega'] = bounds_delta_over_omega
            param_config['omega_tau'] = bounds_omega_tau_lp
    elif is_jp:
        param_config['Delta_e'] = bounds_Delta_e
        if optimize_protocol_params:
            param_config['omega_tau'] = bounds_omega_tau_jp
    
    # Remove fixed parameters from optimization
    opt_params = {k: v for k, v in param_config.items() if k not in fixed_params}
    param_names = list(opt_params.keys())
    bounds = [opt_params[name] for name in param_names]
    
    # Normalize protocol name for display
    proto_display = "LP (Levine-Pichler)" if is_lp else "JP (Jandura-Pupillo)"
    
    if verbose:
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║     CZ Gate Multi-Parameter Optimization                     ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║ Protocol: {proto_display:40s}         ║")
        print(f"║ Target Fidelity: {target_fidelity*100:6.2f}%                                  ║")
        print(f"║ Target Gate Time: {target_gate_time_ns:6.1f} ns                               ║")
        print(f"║ Species: {species:6s}                                           ║")
        print(f"║ Optimizing {len(param_names)} parameters                                    ║")
        print(f"║ Power coupling: {'ON ' if couple_powers else 'OFF'}                                        ║")
        fixed_str = str(list(fixed_params.keys()))[:45] if fixed_params else 'None'
        print(f"║ Fixed: {fixed_str:50s} ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")
    
    # Evaluation counter
    eval_count = [0]
    best_cost = [float('inf')]
    best_params = [None]
    
    def objective(x: np.ndarray) -> float:
        """Objective function to minimize."""
        eval_count[0] += 1
        
        # Build parameter dictionary
        params = dict(zip(param_names, x))
        params.update(fixed_params)
        
        # Handle coupled power mode
        if couple_powers and 'total_power' in params:
            power_2 = params.pop('total_power')
            params['rydberg_power_2'] = power_2
            params['rydberg_power_1'] = power_2 * power_ratio_780_480
        
        # Handle integer parameters
        if 'n_rydberg' in params:
            params['n_rydberg'] = int(round(params['n_rydberg']))
        
        # Extract laser linewidth
        laser_lw = params.pop('laser_linewidth', 1000.0)
        
        # Run simulation
        try:
            Delta_e_rads = 2 * np.pi * params.get('Delta_e', 5e9)
            proto_delta_over_omega = params.get('delta_over_omega', None)
            proto_omega_tau = params.get('omega_tau', None)
            
            # Build laser configuration
            laser_1 = LaserParameters(
                power=params.get('rydberg_power_1', 50e-6),
                waist=50e-6,  # Fixed waist
                linewidth_hz=laser_lw,
            )
            laser_2 = LaserParameters(
                power=params.get('rydberg_power_2', 500e-3),
                waist=50e-6,
                linewidth_hz=laser_lw,
            )
            excitation = TwoPhotonExcitationConfig(
                laser_1=laser_1,
                laser_2=laser_2,
                Delta_e=Delta_e_rads,
            )
            
            # Build noise configuration
            noise = NoiseSourceConfig(
                include_motional_dephasing=include_motional_dephasing,
            )
            
            # Create protocol-specific inputs
            if protocol in ("levine_pichler", "lp"):
                simulation_inputs = LPSimulationInputs(
                    excitation=excitation,
                    noise=noise,
                    delta_over_omega=proto_delta_over_omega,
                    omega_tau=proto_omega_tau,
                )
            else:
                simulation_inputs = JPSimulationInputs(
                    excitation=excitation,
                    noise=noise,
                    omega_tau=proto_omega_tau,
                )
            
            result = simulate_CZ_gate(
                simulation_inputs=simulation_inputs,
                species=species,
                n_rydberg=params.get('n_rydberg', 70),
                temperature=params.get('temperature', 5e-6),
                spacing_factor=params.get('spacing_factor', 3.0),
                tweezer_power=params.get('tweezer_power', 50e-3),
                tweezer_waist=params.get('tweezer_waist', 1e-6),
                background_loss_rate_hz=background_loss_rate_hz,
                include_noise=include_noise,
                verbose=False,
                return_dataclass=False,
            )
        except Exception as e:
            if verbose and eval_count[0] % 50 == 0:
                print(f"  [Eval {eval_count[0]}] Simulation error: {str(e)[:50]}")
            return 1e6
        
        fidelity = result['avg_fidelity']
        gate_time_ns = result['tau_total'] * 1e9
        V_over_Omega = result['V_over_Omega']
        
        # Compute objective components
        fid_error = (1 - fidelity / target_fidelity) ** 2
        time_error = ((gate_time_ns - target_gate_time_ns) / target_gate_time_ns) ** 2
        
        # Constraint penalties
        penalty = 0.0
        if V_over_Omega < 10:
            penalty += (10 - V_over_Omega) ** 2
        
        spacing = params.get('spacing_factor', 3.0) * params.get('tweezer_waist', 1e-6)
        waist = params.get('tweezer_waist', 1e-6)
        if spacing < 2 * waist:
            penalty += ((2 * waist - spacing) / waist) ** 2
        
        temp = params.get('temperature', 5e-6)
        if temp < 0.05e-6:
            penalty += ((0.05e-6 - temp) / 1e-6) ** 2
        
        cost = weight_fidelity * fid_error + weight_time * time_error + constraint_penalty * penalty
        
        # Track best
        if cost < best_cost[0]:
            best_cost[0] = cost
            best_params[0] = params.copy()
            best_params[0]['laser_linewidth'] = laser_lw
            best_params[0]['_fidelity'] = fidelity
            best_params[0]['_gate_time_ns'] = gate_time_ns
            best_params[0]['_V_over_Omega'] = V_over_Omega
            best_params[0]['_noise'] = result.get('noise_breakdown', {})
        
        if callback is not None:
            callback(eval_count[0], cost, params)
        
        if verbose and eval_count[0] % 20 == 0:
            print(f"  [Eval {eval_count[0]:4d}] F={fidelity:.4f}, t={gate_time_ns:.1f}ns, "
                  f"V/Ω={V_over_Omega:.1f}, cost={cost:.2e}")
        
        return cost
    
    # Run global optimization
    if verbose:
        print(f"\nStarting differential evolution (maxiter={maxiter}, popsize={popsize})...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        opt_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            polish=polish,
            workers=workers,
            popsize=popsize,
            disp=False,
            updating='deferred' if workers > 1 else 'immediate'
        )
    
    # Extract final results
    if best_params[0] is not None:
        final_params = {k: v for k, v in best_params[0].items() if not k.startswith('_')}
        final_fidelity = best_params[0]['_fidelity']
        final_gate_time = best_params[0]['_gate_time_ns']
        final_V_over_Omega = best_params[0]['_V_over_Omega']
        final_noise = best_params[0]['_noise']
    else:
        # Fallback
        final_params = dict(zip(param_names, opt_result.x))
        final_params.update(fixed_params)
        if 'n_rydberg' in final_params:
            final_params['n_rydberg'] = int(round(final_params['n_rydberg']))
        
        laser_lw = final_params.pop('laser_linewidth', 1000.0)
        
        if couple_powers and 'total_power' in final_params:
            power_2 = final_params.pop('total_power')
            final_params['rydberg_power_2'] = power_2
            final_params['rydberg_power_1'] = power_2 * power_ratio_780_480
        
        # Build laser configuration for final simulation
        laser_1 = LaserParameters(
            power=final_params.get('rydberg_power_1', 50e-6),
            waist=50e-6,
            linewidth_hz=laser_lw,
        )
        laser_2 = LaserParameters(
            power=final_params.get('rydberg_power_2', 500e-3),
            waist=50e-6,
            linewidth_hz=laser_lw,
        )
        excitation = TwoPhotonExcitationConfig(
            laser_1=laser_1,
            laser_2=laser_2,
            Delta_e=2*np.pi*final_params.get('Delta_e', 5e9),
        )
        noise = NoiseSourceConfig(include_motional_dephasing=include_motional_dephasing)
        
        if protocol in ("levine_pichler", "lp"):
            simulation_inputs = LPSimulationInputs(
                excitation=excitation,
                noise=noise,
                delta_over_omega=final_params.get('delta_over_omega', None),
                omega_tau=final_params.get('omega_tau', None),
            )
        else:
            simulation_inputs = JPSimulationInputs(
                excitation=excitation,
                noise=noise,
                omega_tau=final_params.get('omega_tau', None),
            )
        
        final_result = simulate_CZ_gate(
            simulation_inputs=simulation_inputs,
            species=species,
            n_rydberg=final_params.get('n_rydberg', 70),
            temperature=final_params.get('temperature', 5e-6),
            spacing_factor=final_params.get('spacing_factor', 3.0),
            tweezer_power=final_params.get('tweezer_power', 50e-3),
            tweezer_waist=final_params.get('tweezer_waist', 1e-6),
            background_loss_rate_hz=background_loss_rate_hz,
            include_noise=include_noise,
            verbose=False,
            return_dataclass=False,
        )
        final_params['laser_linewidth'] = laser_lw
        final_fidelity = final_result['avg_fidelity']
        final_gate_time = final_result['tau_total'] * 1e9
        final_V_over_Omega = final_result['V_over_Omega']
        final_noise = final_result.get('noise_breakdown', {})
    
    # Compute errors
    fid_error_pct = (final_fidelity / target_fidelity - 1) * 100
    time_error_pct = (final_gate_time / target_gate_time_ns - 1) * 100
    
    # Determine success
    fid_tolerance = 2.0 if target_fidelity > 0.995 else 5.0
    success = (
        abs(fid_error_pct) < fid_tolerance and
        abs(time_error_pct) < 30.0 and
        final_V_over_Omega >= 8
    )
    
    # Build result message
    if success:
        message = "Optimization converged successfully"
    else:
        issues = []
        if abs(fid_error_pct) >= fid_tolerance:
            issues.append(f"fidelity error {fid_error_pct:+.1f}%")
        if abs(time_error_pct) >= 30.0:
            issues.append(f"time error {time_error_pct:+.1f}%")
        if final_V_over_Omega < 8:
            issues.append(f"V/Ω={final_V_over_Omega:.1f} < 8")
        message = f"Optimization incomplete: {', '.join(issues)}"
    
    if verbose:
        print(f"\n{'═'*64}")
        print(f"Optimization Complete!")
        print(f"  Evaluations: {eval_count[0]}")
        print(f"  Target:   F={target_fidelity:.4f}, t={target_gate_time_ns:.1f} ns")
        print(f"  Achieved: F={final_fidelity:.4f}, t={final_gate_time:.1f} ns")
        print(f"  Errors:   ΔF={fid_error_pct:+.2f}%, Δt={time_error_pct:+.2f}%")
        print(f"  V/Ω = {final_V_over_Omega:.1f}")
        print(f"  Laser linewidth: {final_params.get('laser_linewidth', 'N/A'):.0f} Hz")
        if 'delta_over_omega' in final_params:
            print(f"  Protocol Δ/Ω = {final_params['delta_over_omega']:.4f}")
        if 'omega_tau' in final_params:
            print(f"  Protocol Ωτ = {final_params['omega_tau']:.3f}")
        print(f"  Status: {message}")
        print(f"{'═'*64}")
    
    return HardwareOptimizationResult(
        success=success,
        target_fidelity=target_fidelity,
        target_gate_time_ns=target_gate_time_ns,
        achieved_fidelity=final_fidelity,
        achieved_gate_time_ns=final_gate_time,
        fidelity_error_pct=fid_error_pct,
        gate_time_error_pct=time_error_pct,
        optimal_parameters=final_params,
        V_over_Omega=final_V_over_Omega,
        noise_breakdown=final_noise,
        n_evaluations=eval_count[0],
        final_cost=best_cost[0],
        message=message
    )


# =============================================================================
# PARAMETER SPACE EXPLORATION (EFFICIENT PARETO)
# =============================================================================

def explore_parameter_space(
    protocol: str = "levine_pichler",
    species: str = "Rb87",
    n_runs: int = 1,
    maxiter: int = 30,
    popsize: int = 10,
    seeds: List[int] = None,
    verbose: bool = True,
    # === PARAMETER BOUNDS (all tunable knobs) ===
    bounds_rydberg_power_2: Tuple[float, float] = (0.5, 30.0),
    bounds_rydberg_power_1: Tuple[float, float] = (0.5e-3, 10e-3),
    bounds_temperature: Tuple[float, float] = (1e-6, 15e-6),
    bounds_spacing_factor: Tuple[float, float] = (2.0, 5.0),
    bounds_n_rydberg: Tuple[int, int] = (50, 90),
    bounds_tweezer_power: Tuple[float, float] = (10e-3, 100e-3),
    bounds_tweezer_waist: Tuple[float, float] = (0.5e-6, 2.0e-6),
    bounds_laser_linewidth: Tuple[float, float] = (100.0, 10e3),
    # Protocol-specific
    bounds_delta_over_omega: Tuple[float, float] = (0.32, 0.42),
    bounds_omega_tau: Tuple[float, float] = (3.9, 4.8),
) -> ExplorationResult:
    """
    Explore the FULL parameter space by optimising ALL tunable hardware knobs.
    
    This explores the TRUE Pareto front by varying all 8+ parameters
    simultaneously, not just 1-2 parameters. Each evaluation is cached for
    post-hoc analysis.
    
    Parameters
    ----------
    protocol : str
        Which CZ protocol to use.  Accepted values:
        
        - Levine-Pichler: ``"levine_pichler"``, ``"lp"``, ``"two_pulse"``
        - Jandura-Pupillo / smooth JP: ``"jandura_pupillo"``, ``"jp"``,
          ``"smooth_jp"``, ``"single_pulse"``, ``"time_optimal"``
    species : str
        "Rb87" or "Cs133"
    n_runs : int
        Number of optimisation runs with different seeds
    maxiter : int
        Max iterations per run (default: 30 for efficiency)
    popsize : int
        Population size (default: 10 for efficiency)
    bounds_* : tuple
        (min, max) bounds for each optimisable parameter
    verbose : bool
        Print progress
        
    Returns
    -------
    ExplorationResult
        Container with all evaluated points and TRUE Pareto front
    """
    
    if seeds is None:
        seeds = [42 + i*111 for i in range(n_runs)]
    
    result = ExplorationResult(
        protocol=protocol,
        species=species,
        optimizer_settings={
            'n_runs': n_runs,
            'maxiter': maxiter,
            'popsize': popsize,
            'seeds': seeds,
        }
    )
    
    start_time = time.time()
    
    is_lp = protocol.lower() in ["levine_pichler", "lp", "two_pulse"]
    # Note: anything that isn't LP is treated as JP / smooth JP.
    # Smooth JP aliases ("smooth_jp", "jp", "jandura_pupillo", etc.) all
    # route through the smooth JP solver via JPSimulationInputs.
    
    if verbose:
        print(f"{'='*60}")
        print(f"FULL PARAMETER SPACE EXPLORATION: {protocol.upper()}")
        print(f"{'='*60}")
        print(f"Species: {species}")
        print(f"Runs: {n_runs} × (maxiter={maxiter}, popsize={popsize})")
        print(f"Optimizing ALL parameters: power, temp, spacing, n, linewidth, protocol")
        print()
    
    # Full parameter bounds - 8 dimensions for LP protocol
    param_names = [
        'rydberg_power_2', 'rydberg_power_1', 'temperature', 'spacing_factor',
        'n_rydberg', 'tweezer_power', 'tweezer_waist', 'laser_linewidth'
    ]
    bounds = [
        bounds_rydberg_power_2,
        bounds_rydberg_power_1,
        bounds_temperature,
        bounds_spacing_factor,
        bounds_n_rydberg,
        bounds_tweezer_power,
        bounds_tweezer_waist,
        bounds_laser_linewidth,
    ]
    
    # Add protocol-specific params
    if is_lp:
        param_names.extend(['delta_over_omega', 'omega_tau'])
        bounds.extend([bounds_delta_over_omega, bounds_omega_tau])
    
    eval_count = [0]
    best_fid = [0.0]
    fastest_time = [float('inf')]
    
    def objective_and_cache(x):
        """Evaluate full parameter set and cache result."""
        params = dict(zip(param_names, x))
        
        # Convert integer params
        n_ryd = int(round(params['n_rydberg']))
        
        try:
            # Build laser configuration
            laser_1 = LaserParameters(
                power=params['rydberg_power_1'],
                waist=1.0e-6,
                linewidth_hz=params['laser_linewidth'],
            )
            laser_2 = LaserParameters(
                power=params['rydberg_power_2'],
                waist=10e-6,
                linewidth_hz=params['laser_linewidth'],
            )
            excitation = TwoPhotonExcitationConfig(
                laser_1=laser_1,
                laser_2=laser_2,
            )
            
            # Noise configuration
            noise = NoiseSourceConfig(
                include_motional_dephasing=True,
                include_doppler_dephasing=True,
                include_intensity_noise=True,
                intensity_noise_frac=0.01,
            )
            
            # Create protocol-specific inputs
            if is_lp:
                simulation_inputs = LPSimulationInputs(
                    excitation=excitation,
                    noise=noise,
                    delta_over_omega=params.get('delta_over_omega', 0.377371),
                    omega_tau=params.get('omega_tau', 4.29268),
                )
            else:
                simulation_inputs = JPSimulationInputs(
                    excitation=excitation,
                    noise=noise,
                )
            
            sim_result = simulate_CZ_gate(
                simulation_inputs=simulation_inputs,
                species=species,
                n_rydberg=n_ryd,
                temperature=params['temperature'],
                spacing_factor=params['spacing_factor'],
                tweezer_power=params['tweezer_power'],
                tweezer_waist=params['tweezer_waist'],
                include_noise=True,
                verbose=False,
                return_dataclass=False,
            )
            
            fidelity = sim_result['avg_fidelity']
            gate_time_ns = sim_result['tau_total'] * 1e9
            V_over_Omega = sim_result['V_over_Omega']
            Omega_MHz = sim_result['Omega'] / (2 * np.pi * 1e6)
            noise_breakdown = sim_result.get('noise_breakdown', {})
            
            # Cache this evaluation
            point = EvaluatedPoint(
                Omega_MHz=Omega_MHz,
                laser_linewidth_kHz=params['laser_linewidth'] / 1e3,
                V_over_Omega=V_over_Omega,
                fidelity=fidelity,
                gate_time_ns=gate_time_ns,
                infidelity=1 - fidelity,
                noise_breakdown=noise_breakdown,
                protocol=protocol,
                species=species,
            )
            result.add_point(point)
            
            eval_count[0] += 1
            if fidelity > best_fid[0]:
                best_fid[0] = fidelity
            if gate_time_ns < fastest_time[0] and fidelity > 0.95:
                fastest_time[0] = gate_time_ns
            
            if verbose and eval_count[0] % 25 == 0:
                print(f"  [{eval_count[0]:4d}] best F={best_fid[0]*100:.2f}%, "
                      f"fastest (F>95%)={fastest_time[0]:.0f}ns")
            
            # Multi-objective: minimize infidelity (primary) with gate time as tiebreaker
            return (1 - fidelity) + 0.001 * (gate_time_ns / 1000)
            
        except Exception as e:
            eval_count[0] += 1
            return 1.0
    
    # Run optimization(s) with different seeds to explore broadly
    for run_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nRun {run_idx + 1}/{n_runs} (seed={seed})...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            differential_evolution(
                objective_and_cache,
                bounds=bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
                disp=False,
                workers=1,
                updating='deferred',
                mutation=(0.5, 1.0),  # Broader exploration
                recombination=0.7,
            )
    
    # Compute TRUE Pareto front from all cached evaluations
    result.compute_pareto_front()
    result.runtime_seconds = time.time() - start_time
    
    if verbose:
        print(f"\n{result.summary()}")
    
    return result


def combine_explorations(*results: ExplorationResult) -> ExplorationResult:
    """Combine multiple exploration results into one."""
    if not results:
        raise ValueError("No results to combine")
    
    combined = ExplorationResult(
        protocol=results[0].protocol,
        species=results[0].species,
    )
    
    for r in results:
        combined.points.extend(r.points)
        combined.runtime_seconds += r.runtime_seconds
    
    combined.n_evaluations = len(combined.points)
    combined.compute_pareto_front()
    
    return combined


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Optimization
    "HardwareOptimizationResult",
    "optimize_CZ_parameters",
    
    # Exploration
    "EvaluatedPoint",
    "ExplorationResult",
    "explore_parameter_space",
    "combine_explorations",
]
