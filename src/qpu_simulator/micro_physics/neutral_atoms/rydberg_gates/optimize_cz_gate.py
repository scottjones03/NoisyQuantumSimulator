#!/usr/bin/env python3
"""
Protocol-Agnostic CZ Gate Optimizer
====================================

Co-optimizes gate fidelity (primary) and gate time (secondary) for any
of the three CZ gate protocols (LP, JP bang-bang, and smooth JP) under
user-specified apparatus constraints.

Design Principles
-----------------
1. **Protocol-agnostic**: Same optimizer works for LP, JP bang-bang, and smooth JP
2. **Apparatus constraints**: Laser power, temperature, spacing are fixed inputs
3. **Direct parameter injection**: Parameters flow through simulation_inputs
   dataclass fields — no monkey-patching of module globals
4. **Discrete search**: For bang-bang JP, optimises both 5-segment and
   7-segment variants and picks best
5. **Memoization**: Cache results keyed by (protocol, rounded params) for
   reuse across runs

   
                       ╔═══════════════════════════════════════════════════╗
                    ║         END-TO-END CZ GATE OPTIMIZATION          ║
                    ╚═══════════════════════════════════════════════════╝

    USER INPUTS                          ENTRY POINT                         OUTPUT
   ┌──────────────┐                ┌─────────────────────┐           ┌──────────────────┐
   │ Protocol:    │                │                     │           │ OptimizationResult│
   │  "lp"        │──┐             │  optimize_cz_gate() │──────────▶│  .best_fidelity   │
   │  "smooth_jp" │  │             │  [optimize_cz_gate  │           │  .best_params     │
   │  "jp_bangbang"│ │             │        .py]         │           │  .gate_time_us    │
   │              │  │             │                     │           │  .cost            │
   └──────────────┘  │             └─────────┬───────────┘           │  .phase_error_deg │
   ┌──────────────┐  │                       │                      └──────────────────┘
   │ Apparatus:   │──┤   ┌───────────────────┼───────────────────┐
   │  laser_power │  │   │   FOR EACH CANDIDATE (DE optimizer)   │
   │  temperature │  │   │                                       │
   │  n_rydberg   │──┘   │  ┌─────────────┐   ┌──────────────┐  │
   │  tweezer_*   │      │  │ Build Sim    │   │ Run          │  │
   │  B_field     │      │  │ Inputs from  │──▶│ simulate_CZ  │  │
   └──────────────┘      │  │ apparatus +  │   │ _gate()      │  │
   ┌──────────────┐      │  │ trial params │   │ [simulation  │  │
   │ NoiseConfig: │──────┤  │ [configs.py] │   │       .py]   │  │
   │  motional    │      │  └─────────────┘   └──────┬───────┘  │
   │  doppler     │      │                           │          │
   │  intensity   │      │  ┌─────────────┐   ┌──────▼───────┐  │
   │  scattering  │      │  │SimulationCa-│   │ extract_     │  │
   └──────────────┘      │  │che (disk JSON│◀──│ metrics() +  │  │
                         │  │ memoisation) │   │ compute_     │  │
                         │  └─────────────┘   │ cost()       │  │
                         │                     └──────────────┘  │
                         └───────────────────────────────────────┘

                    ┌───────────────────────────────────────────────────┐
                    │          PHYSICS PIPELINE (simulate_CZ_gate)      │
                    │                  [simulation.py]                  │
                    │                                                   │
                    │  AtomicConfig ──▶ Laser/Trap Physics ──▶ Ω, V    │
                    │  [configs.py]    [laser_physics.py]               │
                    │                  [trap_physics.py]                │
                    │                                                   │
                    │  Protocol Params ──▶ Δ/Ω, Ωτ, switching_times   │
                    │  [protocols.py]      (from lookup or optimizer)   │
                    │                                                   │
                    │  Noise Rates ──▶ Collapse operators               │
                    │  [noise_models.py]                                │
                    │                                                   │
                    │  Hamiltonians ──▶ H(t) with pulse shaping        │
                    │  [hamiltonians.py]  [pulse_shaping.py]           │
                    │                                                   │
                    │  Time Evolution ──▶ Fidelity + Phase extraction  │
                    │  (QuTiP mesolve)                                 │
                    │                                                   │
                    │  Constants: [constants.py] [atom_database.py]    │
                    └───────────────────────────────────────────────────┘

                    ┌───────────────────────────────────────────────────┐
                    │     COMPLEMENTARY TOOLS (keep but update)         │
                    │                                                   │
                    │  optimization.py ── Pareto exploration of         │
                    │                     HARDWARE params (inverse)     │
                    │  visualization.py ── Plot Pareto fronts,         │
                    │                      heatmaps, noise breakdowns  │
                    └───────────────────────────────────────────────────┘
Usage
-----
    from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.optimize_cz_gate import (
        optimize_cz_gate, ApparatusConstraints
    )
    
    apparatus = ApparatusConstraints(
        laser_1_power=50e-6, laser_1_waist=50e-6,
        laser_2_power=0.3, laser_2_waist=50e-6,
        Delta_e=2 * np.pi * 5e9,
        temperature=2e-6, spacing_factor=2.8,
        n_rydberg=70,
    )
    
    result = optimize_cz_gate(
        protocol="smooth_jp",
        apparatus=apparatus,
        include_noise=False,
        maxiter=80,
        verbose=True,
    )
    print(result)

References
----------
- Levine et al., PRL 123, 170503 (2019) — LP protocol
- Jandura & Pupillo, PRX Quantum 3, 010353 (2022) — JP bang-bang
- Evered et al., Nature 622, 268 (2023) — Smooth JP (Bluvstein-form)

Author: Quantum Simulation Team
"""

import numpy as np
import json
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from scipy.optimize import differential_evolution

# Simulation imports
from .simulation import simulate_CZ_gate, SimulationResult
from .configurations import (
    LPSimulationInputs,
    JPSimulationInputs,
    SmoothJPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)
from .protocols import (
    SMOOTH_JP_PARAMS,
    LEVINE_PICHLER_PARAMS,
    JP_SWITCHING_TIMES_VALIDATED,
    JP_PHASES_VALIDATED,
    JP_OMEGA_TAU_VALIDATED,
    JP_SWITCHING_TIMES_DEFAULT,
    JP_PHASES_DEFAULT,
)


# =============================================================================
# APPARATUS CONSTRAINTS
# =============================================================================

@dataclass
class ApparatusConstraints:
    """
    Fixed experimental apparatus parameters that the optimizer cannot change.
    
    These represent hardware limitations: how much laser power is available,
    what temperature we can cool to, etc.
    
    Attributes
    ----------
    laser_1_power : float
        First-leg laser power (W). Typically 10-500 µW.
    laser_1_waist : float
        First-leg beam waist (m). Typically 20-100 µm.
    laser_2_power : float
        Second-leg laser power (W). Typically 100 mW - 5 W.
    laser_2_waist : float
        Second-leg beam waist (m). Typically 20-100 µm.
    Delta_e : float
        Intermediate state detuning (rad/s). Typically 2π × 1-10 GHz.
    laser_1_linewidth_hz : float
        First-leg laser linewidth (Hz).
    laser_2_linewidth_hz : float
        Second-leg laser linewidth (Hz).
    temperature : float
        Atom temperature (K). Typically 2-20 µK.
    spacing_factor : float
        Atom spacing in units of λ/(2·NA).
    n_rydberg : int
        Rydberg principal quantum number.
    species : str
        Atomic species ("Rb87" or "Cs133").
    tweezer_power : float
        Optical tweezer power (W).
    tweezer_waist : float
        Tweezer beam waist (m).
    B_field : float
        Magnetic field (T).
    NA : float
        Numerical aperture.
    counter_propagating : bool
        Whether Rydberg lasers are counter-propagating.
    """
    # Laser parameters
    laser_1_power: float = 50e-6       # 50 µW
    laser_1_waist: float = 50e-6       # 50 µm
    laser_2_power: float = 0.3         # 300 mW
    laser_2_waist: float = 50e-6       # 50 µm
    Delta_e: float = 2 * np.pi * 1e9   # 1 GHz in rad/s (matches TwoPhotonExcitationConfig default)
    laser_1_linewidth_hz: float = 100.0
    laser_2_linewidth_hz: float = 100.0
    
    # Atom / trap
    temperature: float = 2e-6          # 2 µK
    spacing_factor: float = 2.8
    n_rydberg: int = 70
    species: str = "Rb87"
    tweezer_power: float = 0.020       # 20 mW
    tweezer_waist: float = 0.8e-6      # 0.8 µm
    B_field: float = 1e-4              # 1 Gauss
    NA: float = 0.5
    counter_propagating: bool = True
    
    def fingerprint(self) -> str:
        """Short hash of apparatus parameters for cache key uniqueness."""
        key_vals = (
            round(self.laser_1_power, 8), round(self.laser_1_waist, 8),
            round(self.laser_2_power, 8), round(self.laser_2_waist, 8),
            round(self.Delta_e, 2), self.n_rydberg,
            round(self.spacing_factor, 4), round(self.temperature, 10),
            self.species, round(self.tweezer_power, 6),
            round(self.tweezer_waist, 8), round(self.NA, 3),
        )
        return hashlib.md5(str(key_vals).encode()).hexdigest()[:12]

    def make_excitation_config(
        self,
        pol_purity: float = 1.0,
    ) -> TwoPhotonExcitationConfig:
        """Build TwoPhotonExcitationConfig from apparatus constraints."""
        return TwoPhotonExcitationConfig(
            laser_1=LaserParameters(
                power=self.laser_1_power,
                waist=self.laser_1_waist,
                polarization="pi",
                polarization_purity=pol_purity,
                linewidth_hz=self.laser_1_linewidth_hz,
            ),
            laser_2=LaserParameters(
                power=self.laser_2_power,
                waist=self.laser_2_waist,
                polarization="sigma+",
                polarization_purity=pol_purity,
                linewidth_hz=self.laser_2_linewidth_hz,
            ),
            Delta_e=self.Delta_e,
            counter_propagating=self.counter_propagating,
        )
    
    @staticmethod
    def make_noiseless() -> NoiseSourceConfig:
        """Noise config with everything disabled."""
        return NoiseSourceConfig(
            include_spontaneous_emission=False,
            include_intermediate_scattering=False,
            include_motional_dephasing=False,
            include_doppler_dephasing=False,
            include_intensity_noise=False,
            intensity_noise_frac=0.0,
            include_laser_dephasing=False,
            include_magnetic_dephasing=False,
        )
    
    @staticmethod
    def make_full_noise() -> NoiseSourceConfig:
        """Noise config with everything enabled (realistic)."""
        return NoiseSourceConfig(
            include_spontaneous_emission=True,
            include_intermediate_scattering=True,
            include_motional_dephasing=True,
            include_doppler_dephasing=True,
            include_intensity_noise=True,
            intensity_noise_frac=0.01,
            include_laser_dephasing=True,
            include_magnetic_dephasing=True,
        )


# =============================================================================
# MEMOIZATION CACHE
# =============================================================================

class SimulationCache:
    """
    Memoization cache for simulation results.
    
    Keys are rounded parameter tuples. Values are (cost, metrics) pairs.
    Can be saved/loaded from disk for cross-run memoization.
    
    Usage
    -----
        cache = SimulationCache(precision=4)
        key = cache.make_key("smooth_jp", [10.09, 0.311, 1.242, 4.696, 0.0205])
        
        if key in cache:
            cost, metrics = cache[key]
        else:
            # ... run simulation ...
            cache[key] = (cost, metrics)
        
        cache.save("cache_smooth_jp.json")
    """
    
    def __init__(self, precision: int = 4):
        self._store: Dict[str, Tuple[float, Dict]] = {}
        self.precision = precision
        self.hits = 0
        self.misses = 0
    
    def make_key(self, protocol: str, params: list, apparatus_hash: str = "") -> str:
        """Create a hashable key from protocol name, rounded params, and apparatus."""
        rounded = tuple(round(float(p), self.precision) for p in params)
        return f"{apparatus_hash}|{protocol}|{rounded}"
    
    def __contains__(self, key: str) -> bool:
        return key in self._store
    
    def __getitem__(self, key: str) -> Tuple[float, Dict]:
        self.hits += 1
        return self._store[key]
    
    def __setitem__(self, key: str, value: Tuple[float, Dict]):
        self._store[key] = value
    
    def __len__(self) -> int:
        return len(self._store)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def save(self, path: str):
        """Save cache to JSON file."""
        data = {
            "precision": self.precision,
            "entries": {k: {"cost": v[0], "metrics": v[1]} for k, v in self._store.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, path: str):
        """Load cache from JSON file."""
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.precision = data.get("precision", self.precision)
        for k, v in data.get("entries", {}).items():
            self._store[k] = (v["cost"], v["metrics"])


# Global cache instance
_global_cache = SimulationCache(precision=4)


# =============================================================================
# COST FUNCTION
# =============================================================================

def compute_cost(
    metrics: Dict[str, float],
    gate_time_us: float = 0.0,
    time_weight: float = 0.01,
) -> float:
    """
    Unified cost function for CZ gate optimization.
    
    All penalty terms are expressed in **percentage infidelity** units
    (0–100 scale) so they are naturally comparable.  The previous version
    used ``phase_error_deg²`` (up to 32 400) alongside ``infidelity_pct²``
    (typically < 1), causing the optimizer to sacrifice fidelity to reduce
    phase error — that bug is fixed here.
    
    Penalty hierarchy (by weight):
    
    1. ``avg_fidelity``        — overall gate quality              (×10)
    2. ``F(|11⟩)``            — hardest computational-basis state  (×5)
    3. ``cz_phase_fidelity``  — phase-accuracy signal booster      (×2)
    4. gate time              — very secondary                     (×time_weight)
    
    Notes
    -----
    * ``avg_fidelity`` already includes the phase penalty through
      ``F11_with_phase = F11_population × cz_phase_fidelity``, so the
      explicit ``cz_phase_fidelity`` term only strengthens the gradient
      signal on the controlled phase.
    * ``cz_phase_fidelity = cos²(phase_error_from_π / 2)`` ∈ [0, 1],
      which maps naturally to percentage-infidelity units.
    
    Parameters
    ----------
    metrics : dict
        Must contain: avg_fidelity, f11, cz_phase_fidelity.
    gate_time_us : float
        Total gate time in µs (secondary objective).
    time_weight : float
        Weight on gate time (default 0.01).
    
    Returns
    -------
    float
        Total cost (lower is better).  A perfect CZ gate has cost ≈ 0.
    """
    avg_fid = metrics.get("avg_fidelity", 0.0)
    f11 = metrics.get("f11", 0.0)
    cz_phase_fid = metrics.get("cz_phase_fidelity", 0.0)
    
    # NaN guard
    if any(np.isnan(x) for x in [avg_fid, f11, cz_phase_fid]):
        return 1e6
    
    # Hard floor — truly hopeless region, return large linear penalty
    # so the optimizer gets a consistent gradient *away* from here.
    if avg_fid < 0.50:
        return 1e6
    
    # ── All terms in *percentage infidelity* [0, 100] ──────────────
    infidelity_pct = (1.0 - avg_fid) * 100.0        # overall gate
    f11_infidelity_pct = (1.0 - f11) * 100.0         # |11⟩ state
    phase_infidelity_pct = (1.0 - cz_phase_fid) * 100.0  # phase accuracy
    
    # Quadratic penalties — smooth landscape with correct hierarchy
    cost = (
        10.0 * infidelity_pct ** 2           # dominant: overall fidelity
        + 5.0 * f11_infidelity_pct ** 2      # strong: CZ-critical state
        + 2.0 * phase_infidelity_pct ** 2    # moderate: phase signal
        + time_weight * gate_time_us         # mild: gate speed
    )
    return cost


def extract_metrics(result: SimulationResult) -> Dict[str, float]:
    """Extract optimisation-relevant metrics from SimulationResult."""
    phase_info = result.phase_info
    fidelities = result.fidelities
    
    return {
        "controlled_phase_deg": phase_info.get("controlled_phase_deg", np.nan),
        "phase_error_deg": phase_info.get("phase_error_from_pi_deg", np.nan),
        "cz_phase_fidelity": phase_info.get("cz_phase_fidelity", np.nan),
        "f00": fidelities.get("00", np.nan),
        "f01": fidelities.get("01", np.nan),
        "f10": fidelities.get("10", np.nan),
        "f11": fidelities.get("11", np.nan),
        "avg_fidelity": result.avg_fidelity,
        "gate_time_us": result.tau_total * 1e6,
        "V_over_Omega": result.V_over_Omega,
        "Omega_MHz": result.Omega / (2 * np.pi * 1e6),
    }


# =============================================================================
# PROTOCOL-SPECIFIC PARAMETER BUILDERS
# =============================================================================

def _build_lp_inputs(
    params: np.ndarray,
    excitation: TwoPhotonExcitationConfig,
    noise: NoiseSourceConfig,
) -> LPSimulationInputs:
    """
    Build LPSimulationInputs from optimiser parameter vector.
    
    params = [delta_over_omega, omega_tau]
    """
    return LPSimulationInputs(
        excitation=excitation,
        noise=noise,
        delta_over_omega=float(params[0]),
        omega_tau=float(params[1]),
        pulse_shape="square",
    )


def _build_jp_bangbang_inputs(
    params: np.ndarray,
    excitation: TwoPhotonExcitationConfig,
    noise: NoiseSourceConfig,
    n_segments: int = 5,
    spacing_factor_idx: Optional[int] = None,
) -> JPSimulationInputs:
    """
    Build JPSimulationInputs from optimiser parameter vector.
    
    **Fractional parameterisation** — switching times are encoded as gap
    fractions of omega_tau so that ordering is guaranteed by construction:
    
        params = [omega_tau, f1, f2, ..., f_{N-1}, phi0, ..., phi_{N-1}]
        
        where each f_i ∈ (0, 1) is the fractional position within [0, Ωτ].
        We sort the fractions and convert: t_i = sorted(f_i) * omega_tau.
    
    This avoids the hard constraint t₁ < t₂ < ... < t_{N-1} and makes the
    search space uniform regardless of omega_tau.
    
    If spacing_factor_idx is set, params[-1] is spacing_factor (handled
    externally in the objective function, not here).
    """
    omega_tau = float(params[0])
    n_switch = n_segments - 1
    # Extract raw fractions and sort to guarantee ordering
    raw_fracs = [float(params[1 + i]) for i in range(n_switch)]
    sorted_fracs = sorted(raw_fracs)
    # Convert fractions to absolute dimensionless times
    switching_times = [f * omega_tau for f in sorted_fracs]
    phases = [float(params[1 + n_switch + i]) for i in range(n_segments)]
    
    return JPSimulationInputs(
        excitation=excitation,
        noise=noise,
        omega_tau=omega_tau,
        switching_times=switching_times,
        phases=phases,
    )


def _build_smooth_jp_inputs(
    params: np.ndarray,
    excitation: TwoPhotonExcitationConfig,
    noise: NoiseSourceConfig,
) -> SmoothJPSimulationInputs:
    """
    Build SmoothJPSimulationInputs from optimiser parameter vector.
    
    params = [omega_tau, A, omega_mod_ratio, phi_offset, delta_over_omega]
    """
    return SmoothJPSimulationInputs(
        excitation=excitation,
        noise=noise,
        omega_tau=float(params[0]),
        A=float(params[1]),
        omega_mod_ratio=float(params[2]),
        phi_offset=float(params[3]),
        delta_over_omega=float(params[4]),
    )


# =============================================================================
# DEFAULT BOUNDS AND STARTING POINTS
# =============================================================================

def _get_lp_bounds_and_x0() -> Tuple[list, np.ndarray]:
    """
    Default bounds and starting point for LP optimisation.
    
    params = [delta_over_omega, omega_tau]
    """
    bounds = [
        (0.20, 0.50),   # delta_over_omega: LP needs non-zero detuning
        (3.5, 5.5),     # omega_tau: per-pulse area
    ]
    x0 = np.array([
        LEVINE_PICHLER_PARAMS["delta_over_omega"],  # 0.377
        LEVINE_PICHLER_PARAMS["omega_tau"],          # 4.293
    ])
    return bounds, x0


def _get_jp_bangbang_bounds_and_x0(
    n_segments: int = 5,
) -> Tuple[list, np.ndarray]:
    """
    Default bounds and starting point for JP bang-bang optimisation.
    
    Uses **fractional parameterisation**: switching times are encoded as
    fractions f_i ∈ (0.01, 0.99) of omega_tau. The builder sorts them
    and converts to absolute times: t_i = sorted(f_i) * omega_tau.
    
    This makes the search space independent of omega_tau and V/Ω.
    
    5-segment: params = [omega_tau, f1, f2, f3, f4, phi0..phi4]  (10D)
    7-segment: params = [omega_tau, f1..f6, phi0..phi6]           (14D)
    """
    frac_bounds = (0.01, 0.99)  # each switching fraction
    phase_bounds = (-np.pi, np.pi)  # bang-bang phases — full 2π range
    
    if n_segments == 5:
        # Convert validated absolute times to fractions of omega_tau
        ot0 = JP_OMEGA_TAU_VALIDATED  # 22.08
        f0 = [t / ot0 for t in JP_SWITCHING_TIMES_VALIDATED]
        # f0 ≈ [0.100, 0.400, 0.600, 0.900]
        
        bounds = [
            (5.0, 40.0),   # omega_tau — wide range for any V/Ω
        ] + [frac_bounds] * 4 + [phase_bounds] * 5
        
        x0 = np.array([
            ot0,           # 22.08
            f0[0], f0[1], f0[2], f0[3],
            JP_PHASES_VALIDATED[0],   # π/2
            JP_PHASES_VALIDATED[1],   # 0
            JP_PHASES_VALIDATED[2],   # -π/2
            JP_PHASES_VALIDATED[3],   # 0
            JP_PHASES_VALIDATED[4],   # π/2
        ])
    elif n_segments == 7:
        ot0 = 7.0
        f0 = [t / ot0 for t in JP_SWITCHING_TIMES_DEFAULT]
        
        bounds = [
            (3.0, 30.0),   # omega_tau — wide range
        ] + [frac_bounds] * 6 + [phase_bounds] * 7
        
        x0 = np.array([
            ot0,
            f0[0], f0[1], f0[2], f0[3], f0[4], f0[5],
            JP_PHASES_DEFAULT[0],
            JP_PHASES_DEFAULT[1],
            JP_PHASES_DEFAULT[2],
            JP_PHASES_DEFAULT[3],
            JP_PHASES_DEFAULT[4],
            JP_PHASES_DEFAULT[5],
            JP_PHASES_DEFAULT[6],
        ])
    else:
        raise ValueError(f"Unsupported n_segments: {n_segments}. Use 5 or 7.")
    
    return bounds, x0


def _get_smooth_jp_bounds_and_x0() -> Tuple[list, np.ndarray]:
    """
    Default bounds and starting point for smooth JP optimisation.
    
    params = [omega_tau, A, omega_mod_ratio, phi_offset, delta_over_omega]
    """
    bounds = [
        (5.0, 25.0),        # omega_tau
        (0.05 * np.pi, 1.0 * np.pi),  # A (phase modulation amplitude)
        (0.5, 3.0),          # omega_mod_ratio (ω_mod/Ω)
        (0.0, 2 * np.pi),   # phi_offset
        (0.001, 0.10),       # delta_over_omega (magnitude; sign set by dark state)
    ]
    x0 = np.array([
        SMOOTH_JP_PARAMS.get("omega_tau", 10.09),
        SMOOTH_JP_PARAMS.get("A", 0.311 * np.pi),
        SMOOTH_JP_PARAMS.get("omega_mod_ratio", 1.242),
        SMOOTH_JP_PARAMS.get("phi_offset", 4.696),
        abs(SMOOTH_JP_PARAMS.get("delta_over_omega", 0.0205)),
    ])
    return bounds, x0


# =============================================================================
# OPTIMISATION RESULT
# =============================================================================

@dataclass
class OptimizationResult:
    """
    Result of CZ gate optimisation.
    
    Attributes
    ----------
    success : bool
        Whether optimisation found a satisfactory solution.
    protocol : str
        Protocol name.
    best_params : np.ndarray
        Optimal parameter vector.
    param_names : List[str]
        Names corresponding to best_params entries.
    best_cost : float
        Final cost function value.
    best_metrics : Dict[str, float]
        Gate metrics at optimum (fidelity, phase error, etc.)
    n_evaluations : int
        Number of cost function evaluations.
    runtime_s : float
        Wall-clock time in seconds.
    discrete_variant : str
        Which discrete variant was chosen (e.g., "7-segment" vs "5-segment").
    all_variants : Dict[str, Any]
        Results for all discrete variants tried.
    cache_hits : int
        Number of cache hits during optimisation.
    """
    success: bool
    protocol: str
    best_params: np.ndarray
    param_names: List[str]
    best_cost: float
    best_metrics: Dict[str, float]
    n_evaluations: int
    runtime_s: float
    discrete_variant: str = ""
    all_variants: Dict[str, Any] = field(default_factory=dict)
    cache_hits: int = 0
    
    def __repr__(self) -> str:
        lines = [
            "=" * 70,
            f"  CZ Gate Optimisation Result — {self.protocol}",
            "=" * 70,
            f"  Success:          {self.success}",
            f"  Variant:          {self.discrete_variant}",
            f"  Cost:             {self.best_cost:.4f}",
            f"  Avg fidelity:     {self.best_metrics.get('avg_fidelity', 0):.6f}"
            f"  ({(1-self.best_metrics.get('avg_fidelity', 0))*100:.4f}% error)",
            f"  F(|11⟩):          {self.best_metrics.get('f11', 0):.6f}",
            f"  CZ phase fid:     {self.best_metrics.get('cz_phase_fidelity', 0):.6f}",
            f"  Phase error:      {self.best_metrics.get('phase_error_deg', 999):.2f}°",
            f"  Controlled phase: {self.best_metrics.get('controlled_phase_deg', 0):.2f}°",
            f"  Gate time:        {self.best_metrics.get('gate_time_us', 0):.3f} µs",
            f"  V/Ω:              {self.best_metrics.get('V_over_Omega', 0):.1f}",
            f"  Ω/2π:             {self.best_metrics.get('Omega_MHz', 0):.3f} MHz",
            f"  Evaluations:      {self.n_evaluations}",
            f"  Runtime:          {self.runtime_s:.1f} s",
            f"  Cache hits:       {self.cache_hits}",
            "-" * 70,
            "  Optimal parameters:",
        ]
        for name, val in zip(self.param_names, self.best_params):
            lines.append(f"    {name:25s} = {val:.6f}")
        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# MAIN OPTIMISATION FUNCTION
# =============================================================================

def optimize_cz_gate(
    protocol: str,
    apparatus: ApparatusConstraints,
    include_noise: bool = False,
    noise_config: Optional[NoiseSourceConfig] = None,
    time_weight: float = 0.01,
    optimize_spacing: bool = False,
    spacing_bounds: Optional[Tuple[float, float]] = None,
    maxiter: int = 80,
    popsize: int = 15,
    tol: float = 1e-6,
    seed: int = 42,
    bounds: Optional[list] = None,
    x0: Optional[np.ndarray] = None,
    cache: Optional[SimulationCache] = None,
    cache_path: Optional[str] = None,
    strategy: str = "standard",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Protocol-agnostic CZ gate parameter optimisation.
    
    Co-optimizes gate fidelity (primary) and gate time (secondary) using
    differential evolution.
    
    Parameters
    ----------
    protocol : str
        Protocol to optimise: "lp", "smooth_jp", or "jp_bangbang"
        (also accepts "jp"/"jandura_pupillo" as aliases for bang-bang).
    apparatus : ApparatusConstraints
        Fixed experimental parameters.
    include_noise : bool
        Whether to include decoherence in simulations.
    noise_config : NoiseSourceConfig, optional
        Custom noise config. If None, uses noiseless (include_noise=False)
        or full noise (include_noise=True).
    time_weight : float
        Weight on gate time in µs (default 0.01 — very secondary).
    optimize_spacing : bool
        If True, also optimise the atom spacing_factor to find the best
        V/Ω ratio. This adds one extra dimension to the search.
        Useful when the apparatus V/Ω isn't matched to the protocol's
        sweet spot. Default: False (uses apparatus.spacing_factor).
    spacing_bounds : tuple of (float, float), optional
        Bounds for spacing_factor when optimize_spacing=True.
        Default: (1.5, 5.0).  Typical experimental range: 2.0-4.0.
    maxiter : int
        Maximum differential_evolution iterations.
    popsize : int
        Population size for differential_evolution.
    tol : float
        Convergence tolerance.
    seed : int
        Random seed for reproducibility.
    bounds : list, optional
        Custom parameter bounds. If None, uses protocol defaults.
    x0 : np.ndarray, optional
        Custom starting point (used to seed initial population).
    cache : SimulationCache, optional
        Shared cache. If None, uses global cache.
    cache_path : str, optional
        Path to save/load cache. If None, no disk persistence.
    strategy : str
        Optimization strategy: "standard" (default) or "two_phase".
        Two-phase first optimizes omega_tau alone, then fine-tunes all
        parameters.  Most useful for smooth_jp in unfamiliar V/Ω regimes.
    verbose : bool
        Print progress.
    
    Returns
    -------
    OptimizationResult
        Optimal parameters and gate metrics.
    """
    protocol_norm = protocol.lower().replace("-", "_").replace(" ", "_")
    
    # Validate protocol
    valid_protocols = {
        "lp", "levine_pichler",
        "jp_bangbang", "jp", "jandura_pupillo",
        "smooth_jp", "dark_state",
    }
    if protocol_norm not in valid_protocols:
        raise ValueError(
            f"Unknown protocol: {protocol}. "
            f"Use 'lp', 'jp_bangbang', or 'smooth_jp'."
        )
    
    is_lp = protocol_norm in ("lp", "levine_pichler")
    is_jp_bangbang = protocol_norm in ("jp_bangbang", "jp", "jandura_pupillo")
    is_smooth_jp = protocol_norm in ("smooth_jp", "dark_state")
    
    # Build configs from apparatus constraints
    excitation = apparatus.make_excitation_config(
        pol_purity=1.0 if not include_noise else 0.99,
    )
    if noise_config is not None:
        noise = noise_config
    elif include_noise:
        noise = apparatus.make_full_noise()
    else:
        noise = apparatus.make_noiseless()
    
    # Setup cache
    if cache is None:
        cache = _global_cache
    if cache_path:
        cache.load(cache_path)
    
    # Determine discrete variants
    if is_jp_bangbang:
        # Try both 5-segment and 7-segment parameterisations
        variants = {"5-segment": 5, "7-segment": 7}
    else:
        # LP and smooth JP have a single "default" variant
        variants = {"default": None}
    
    if verbose:
        print("=" * 70)
        print(f"  CZ Gate Optimisation — {protocol}")
        print("=" * 70)
        print(f"  Species: {apparatus.species}, n_rydberg: {apparatus.n_rydberg}")
        print(f"  Spacing factor: {apparatus.spacing_factor}"
              f"{' (will be optimised)' if optimize_spacing else ''}")
        print(f"  Noise: {'ON' if include_noise else 'OFF'}")
        print(f"  Laser 1: {apparatus.laser_1_power*1e6:.0f} µW, "
              f"waist {apparatus.laser_1_waist*1e6:.0f} µm")
        print(f"  Laser 2: {apparatus.laser_2_power*1e3:.0f} mW, "
              f"waist {apparatus.laser_2_waist*1e6:.0f} µm")
        print(f"  Delta_e/2π: {apparatus.Delta_e/(2*np.pi)/1e9:.2f} GHz")
        print(f"  Temperature: {apparatus.temperature*1e6:.1f} µK")
        print("-" * 70)
    
    # Run optimization for each discrete variant
    variant_results: Dict[str, OptimizationResult] = {}
    
    for variant_name, n_seg in variants.items():
        if verbose and len(variants) > 1:
            print(f"\n  >>> Variant: {variant_name}")
        
        result = _optimize_single_variant(
            protocol_norm=protocol_norm,
            is_lp=is_lp,
            is_jp_bangbang=is_jp_bangbang,
            is_smooth_jp=is_smooth_jp,
            n_segments=n_seg,
            excitation=excitation,
            noise=noise,
            apparatus=apparatus,
            apparatus_hash=apparatus.fingerprint(),
            include_noise=include_noise,
            time_weight=time_weight,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            seed=seed,
            bounds=bounds,
            x0=x0,
            cache=cache,
            strategy=strategy,
            verbose=verbose,
            optimize_spacing=optimize_spacing,
            spacing_bounds=spacing_bounds,
        )
        result.discrete_variant = variant_name
        variant_results[variant_name] = result
        
        if verbose:
            print(f"    → {variant_name}: cost={result.best_cost:.4f}, "
                  f"F={result.best_metrics.get('avg_fidelity', 0):.6f}, "
                  f"phase_err={result.best_metrics.get('phase_error_deg', 999):.2f}°")
    
    # Pick best variant
    best_name = min(variant_results, key=lambda k: variant_results[k].best_cost)
    best = variant_results[best_name]
    best.all_variants = {k: {
        "cost": v.best_cost,
        "avg_fidelity": v.best_metrics.get("avg_fidelity", 0),
        "phase_error_deg": v.best_metrics.get("phase_error_deg", 999),
        "params": v.best_params.tolist(),
    } for k, v in variant_results.items()}
    
    # Save cache
    if cache_path:
        cache.save(cache_path)
    
    if verbose:
        print(f"\n  Best variant: {best_name}")
        print(best)
    
    return best


def _optimize_single_variant(
    protocol_norm: str,
    is_lp: bool,
    is_jp_bangbang: bool,
    is_smooth_jp: bool,
    n_segments: Optional[int],
    excitation: TwoPhotonExcitationConfig,
    noise: NoiseSourceConfig,
    apparatus: ApparatusConstraints,
    apparatus_hash: str,
    include_noise: bool,
    time_weight: float,
    maxiter: int,
    popsize: int,
    tol: float,
    seed: int,
    bounds: Optional[list],
    x0: Optional[np.ndarray],
    cache: SimulationCache,
    strategy: str,
    verbose: bool,
    optimize_spacing: bool = False,
    spacing_bounds: Optional[Tuple[float, float]] = None,
) -> OptimizationResult:
    """Run differential_evolution for a single discrete variant."""
    
    # Get default bounds and x0
    if is_lp:
        default_bounds, default_x0 = _get_lp_bounds_and_x0()
        param_names = ["delta_over_omega", "omega_tau"]
        cache_protocol = "lp"
    elif is_jp_bangbang:
        _n_seg = n_segments if n_segments is not None else 5
        default_bounds, default_x0 = _get_jp_bangbang_bounds_and_x0(n_segments=_n_seg)
        n_switch = _n_seg - 1
        param_names = (
            ["omega_tau"]
            + [f"frac{i+1}" for i in range(n_switch)]
            + [f"phi{i}" for i in range(_n_seg)]
        )
        cache_protocol = f"jp_bangbang_{_n_seg}seg"
    elif is_smooth_jp:
        default_bounds, default_x0 = _get_smooth_jp_bounds_and_x0()
        param_names = ["omega_tau", "A", "omega_mod_ratio", "phi_offset", "delta_over_omega"]
        cache_protocol = "smooth_jp"
    else:
        raise ValueError(f"Unknown protocol: {protocol_norm}")
    
    if bounds is None:
        bounds = default_bounds
    if x0 is None:
        x0 = default_x0
    
    # ── Append spacing_factor dimension when optimising V/Ω ────────────
    if optimize_spacing:
        _spacing_bounds = spacing_bounds or (1.5, 5.0)
        bounds = list(bounds) + [_spacing_bounds]
        x0 = np.append(x0, apparatus.spacing_factor)
        param_names.append("spacing_factor")
        if verbose:
            print(f"    Spacing optimisation ON: bounds={_spacing_bounds}, "
                  f"x0={apparatus.spacing_factor:.2f}")
    
    # Index of spacing_factor in params (or None if not optimising)
    _spacing_idx = len(bounds) - 1 if optimize_spacing else None
    
    # Evaluation counter
    eval_count = [0]
    best_so_far = [1e9]
    cache_hits = [0]
    t_start = time.time()
    
    v_omega_warned = [False]  # track whether V/Ω warning has been issued
    
    # Build a noise fingerprint so cache keys differ for different noise configs
    _noise_flags = (
        f"se{int(noise.include_spontaneous_emission)}"
        f"is{int(noise.include_intermediate_scattering)}"
        f"md{int(noise.include_motional_dephasing)}"
        f"dd{int(noise.include_doppler_dephasing)}"
        f"in{int(noise.include_intensity_noise)}"
        f"ld{int(noise.include_laser_dephasing)}"
        f"mg{int(noise.include_magnetic_dephasing)}"
    )
    if noise.include_intensity_noise and noise.intensity_noise_frac:
        _noise_flags += f"_inf{noise.intensity_noise_frac:.4f}"
    _noise_hash = hashlib.md5(_noise_flags.encode()).hexdigest()[:8]

    def objective(params: np.ndarray) -> float:
        """Objective function for differential_evolution."""
        eval_count[0] += 1
        
        # ── Extract spacing_factor if optimising it ────────────────────
        if optimize_spacing:
            _spacing = float(params[_spacing_idx])
            protocol_params = params[:_spacing_idx]
        else:
            _spacing = apparatus.spacing_factor
            protocol_params = params
        
        # Check cache (key includes spacing & noise so different configs don't collide)
        _cache_hash = (apparatus_hash if not optimize_spacing
                       else f"{apparatus_hash}_sf{_spacing:.6f}")
        _cache_hash = f"{_cache_hash}_n{_noise_hash}"
        cache_key = cache.make_key(
            cache_protocol, protocol_params.tolist(), _cache_hash)
        if cache_key in cache:
            cache_hits[0] += 1
            cost, _ = cache[cache_key]
            return cost
        else:
            cache.misses += 1
        
        try:
            # Build protocol-specific simulation inputs
            if is_lp:
                sim_inputs = _build_lp_inputs(protocol_params, excitation, noise)
            elif is_jp_bangbang:
                sim_inputs = _build_jp_bangbang_inputs(
                    protocol_params, excitation, noise, n_segments=_n_seg)
            elif is_smooth_jp:
                sim_inputs = _build_smooth_jp_inputs(protocol_params, excitation, noise)
            
            # Run simulation
            result = simulate_CZ_gate(
                simulation_inputs=sim_inputs,
                species=apparatus.species,
                n_rydberg=apparatus.n_rydberg,
                spacing_factor=_spacing,
                temperature=apparatus.temperature,
                tweezer_power=apparatus.tweezer_power,
                tweezer_waist=apparatus.tweezer_waist,
                B_field=apparatus.B_field,
                NA=apparatus.NA,
                include_noise=include_noise,
                verbose=False,
                return_dataclass=True,
            )
            
            metrics = extract_metrics(result)
            cost = compute_cost(
                metrics,
                gate_time_us=metrics.get("gate_time_us", 0),
                time_weight=time_weight,
            )
            
            # V/Ω regime warning (once per variant)
            if not v_omega_warned[0] and verbose:
                v_omega = metrics.get("V_over_Omega", 0)
                if v_omega > 200 and is_smooth_jp:
                    print(f"    ⚠ V/Ω = {v_omega:.0f} > 200: "
                          f"outside JP lookup table range [10-200]. "
                          f"Optimizer will search for adapted params.")
                elif v_omega < 10:
                    print(f"    ⚠ V/Ω = {v_omega:.1f} < 10: "
                          f"blockade too weak for reliable CZ gate!")
                v_omega_warned[0] = True
            
            # Cache result
            cache[cache_key] = (cost, metrics)
            
            # Progress logging
            if cost < best_so_far[0]:
                best_so_far[0] = cost
                if verbose:
                    _vomega_str = (
                        f"  V/Ω={metrics.get('V_over_Omega', 0):.1f}"
                        if optimize_spacing else ""
                    )
                    print(
                        f"    [{eval_count[0]:4d}] cost={cost:10.4f}  "
                        f"F={metrics.get('avg_fidelity', 0):.6f}  "
                        f"F11={metrics.get('f11', 0):.6f}  "
                        f"CZφ={metrics.get('cz_phase_fidelity', 0):.4f}  "
                        f"φ_err={metrics.get('phase_error_deg', 999):.2f}°  "
                        f"t={metrics.get('gate_time_us', 0):.3f}µs"
                        f"{_vomega_str}"
                    )
            
            return cost
            
        except Exception as e:
            if verbose:
                print(f"    [{eval_count[0]:4d}] ERROR: {e}")
            return 1e6
    
    # ── Two-phase strategy (optional) ──────────────────────────────────
    # Phase 1: optimize omega_tau alone (fast, 1-D) at literature defaults
    # Phase 2: fine-tune all params with tighter bounds around Phase 1 optimum
    # Most useful for smooth JP in unfamiliar V/Ω regimes.
    
    if strategy == "two_phase" and is_smooth_jp:
        if verbose:
            print("    ── Phase 1: coarse sweep (omega_tau only) ──")
        
        # Fix all params except omega_tau at literature defaults
        fixed_A = x0[1]
        fixed_omr = x0[2]
        fixed_phi = x0[3]
        fixed_doo = x0[4]
        # If optimising spacing, also sweep it alongside omega_tau in phase 1
        _fixed_spacing = x0[_spacing_idx] if optimize_spacing else None
        
        def phase1_obj(sweep_arr):
            """Sweep omega_tau (and optionally spacing_factor) in phase 1."""
            base = np.array([
                sweep_arr[0], fixed_A, fixed_omr, fixed_phi, fixed_doo
            ])
            if optimize_spacing:
                # sweep_arr = [omega_tau, spacing_factor]
                return objective(np.append(base, sweep_arr[1]))
            return objective(base)
        
        _phase1_bounds = [bounds[0]]  # omega_tau bounds
        _phase1_x0 = [x0[0]]
        if optimize_spacing:
            _phase1_bounds.append(bounds[_spacing_idx])
            _phase1_x0.append(x0[_spacing_idx])
        
        phase1_result = differential_evolution(
            phase1_obj,
            bounds=_phase1_bounds,
            x0=np.array(_phase1_x0),
            maxiter=max(20, maxiter // 4),
            popsize=10,
            tol=tol,
            seed=seed,
            polish=True,
            disp=False,
        )
        
        # Update x0 for phase 2 with best omega_tau (and spacing if applicable)
        x0_base = np.array([
            phase1_result.x[0], fixed_A, fixed_omr, fixed_phi, fixed_doo])
        if optimize_spacing:
            x0 = np.append(x0_base, phase1_result.x[1])
        else:
            x0 = x0_base
        
        # Tighten bounds around phase 1 optimum (±30% for non-angular params)
        ot_best = phase1_result.x[0]
        tighter_ot_bounds = (max(bounds[0][0], ot_best * 0.7),
                             min(bounds[0][1], ot_best * 1.3))
        bounds = [tighter_ot_bounds] + list(bounds[1:])
        
        if verbose:
            print(f"    Phase 1 best Ωτ = {ot_best:.3f}, "
                  f"cost = {phase1_result.fun:.4f}")
            print(f"    ── Phase 2: fine-tune all {len(bounds)}D ──")
    
    if verbose:
        print(f"    Starting differential_evolution: "
              f"maxiter={maxiter}, popsize={popsize}, "
              f"dim={len(bounds)}, strategy={strategy}")
        print(f"    x0 = {x0}")
    
    # Run optimisation
    de_result = differential_evolution(
        objective,
        bounds=bounds,
        x0=x0,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=seed,
        polish=True,
        disp=False,
    )
    
    runtime = time.time() - t_start
    
    # Get final metrics — extract spacing from best params if optimising
    if optimize_spacing:
        _final_spacing = float(de_result.x[_spacing_idx])
        _final_protocol_params = de_result.x[:_spacing_idx]
        _final_cache_hash = f"{apparatus_hash}_sf{_final_spacing:.6f}"
    else:
        _final_spacing = apparatus.spacing_factor
        _final_protocol_params = de_result.x
        _final_cache_hash = apparatus_hash
    
    final_key = cache.make_key(
        cache_protocol, _final_protocol_params.tolist(), _final_cache_hash)
    if final_key in cache:
        _, final_metrics = cache[final_key]
    else:
        # Re-evaluate to get metrics
        try:
            if is_lp:
                sim_inputs = _build_lp_inputs(_final_protocol_params, excitation, noise)
            elif is_jp_bangbang:
                sim_inputs = _build_jp_bangbang_inputs(
                    _final_protocol_params, excitation, noise, n_segments=_n_seg)
            elif is_smooth_jp:
                sim_inputs = _build_smooth_jp_inputs(
                    _final_protocol_params, excitation, noise)
            
            result = simulate_CZ_gate(
                simulation_inputs=sim_inputs,
                species=apparatus.species,
                n_rydberg=apparatus.n_rydberg,
                spacing_factor=_final_spacing,
                temperature=apparatus.temperature,
                tweezer_power=apparatus.tweezer_power,
                tweezer_waist=apparatus.tweezer_waist,
                B_field=apparatus.B_field,
                NA=apparatus.NA,
                include_noise=include_noise,
                verbose=False,
                return_dataclass=True,
            )
            final_metrics = extract_metrics(result)
        except Exception:
            final_metrics = {"avg_fidelity": 0, "phase_error_deg": 999, "f11": 0}
    
    # Determine success: F ≥ 99% and phase fidelity ≥ 99%
    avg_f = final_metrics.get("avg_fidelity", 0)
    cz_pf = final_metrics.get("cz_phase_fidelity", 0)
    phase_err = final_metrics.get("phase_error_deg", 999)
    success = avg_f >= 0.99 and cz_pf >= 0.99 and phase_err < 10.0
    
    return OptimizationResult(
        success=success,
        protocol=protocol_norm,
        best_params=de_result.x,
        param_names=param_names,
        best_cost=de_result.fun,
        best_metrics=final_metrics,
        n_evaluations=eval_count[0],
        runtime_s=runtime,
        cache_hits=cache_hits[0],
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_baseline(
    protocol: str,
    apparatus: ApparatusConstraints,
    include_noise: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run a single simulation with default protocol parameters (no optimisation).
    
    Useful for establishing a baseline before optimising.
    
    Parameters
    ----------
    protocol : str
        "lp", "jp_bangbang", or "smooth_jp"
    apparatus : ApparatusConstraints
        Fixed experimental parameters.
    include_noise : bool
        Whether to include decoherence.
    verbose : bool
        Print results.
    
    Returns
    -------
    dict
        Metrics: avg_fidelity, phase_error_deg, f11, gate_time_us, etc.
    """
    protocol_norm = protocol.lower().replace("-", "_").replace(" ", "_")
    
    excitation = apparatus.make_excitation_config(
        pol_purity=1.0 if not include_noise else 0.99,
    )
    noise = apparatus.make_noiseless() if not include_noise else apparatus.make_full_noise()
    
    is_lp = protocol_norm in ("lp", "levine_pichler")
    is_jp = protocol_norm in ("jp_bangbang", "jp", "jandura_pupillo")
    is_smooth = protocol_norm in ("smooth_jp", "dark_state")
    
    if is_lp:
        sim_inputs = LPSimulationInputs(excitation=excitation, noise=noise)
    elif is_jp:
        sim_inputs = JPSimulationInputs(excitation=excitation, noise=noise)
    elif is_smooth:
        sim_inputs = SmoothJPSimulationInputs(excitation=excitation, noise=noise)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    
    result = simulate_CZ_gate(
        simulation_inputs=sim_inputs,
        species=apparatus.species,
        n_rydberg=apparatus.n_rydberg,
        spacing_factor=apparatus.spacing_factor,
        temperature=apparatus.temperature,
        tweezer_power=apparatus.tweezer_power,
        tweezer_waist=apparatus.tweezer_waist,
        B_field=apparatus.B_field,
        NA=apparatus.NA,
        include_noise=include_noise,
        verbose=verbose,
        return_dataclass=True,
    )
    
    metrics = extract_metrics(result)
    
    if verbose:
        print(f"\n  Baseline — {protocol}")
        print(f"  Avg fidelity:     {metrics['avg_fidelity']:.6f}")
        print(f"  F(|11⟩):          {metrics['f11']:.6f}")
        print(f"  CZ phase fid:     {metrics['cz_phase_fidelity']:.6f}")
        print(f"  Phase error:      {metrics['phase_error_deg']:.2f}°")
        print(f"  Ctrl phase:       {metrics['controlled_phase_deg']:.2f}°")
        print(f"  Gate time:        {metrics['gate_time_us']:.3f} µs")
        print(f"  V/Ω:              {metrics['V_over_Omega']:.1f}")
        print(f"  Ω/2π:             {metrics['Omega_MHz']:.3f} MHz")
        print(f"  Cost (ref):       {compute_cost(metrics, metrics['gate_time_us']):.4f}")
    
    return metrics


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

def main():
    """Run full optimisation suite for all three protocols."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CZ Gate Optimizer")
    parser.add_argument("--protocol", default="all",
                       choices=["lp", "jp_bangbang", "smooth_jp", "all"],
                       help="Protocol to optimise (default: all)")
    parser.add_argument("--noise", action="store_true",
                       help="Include noise in simulations")
    parser.add_argument("--maxiter", type=int, default=60,
                       help="Max iterations (default: 60)")
    parser.add_argument("--popsize", type=int, default=15,
                       help="Population size (default: 15)")
    parser.add_argument("--cache-dir", default=None,
                       help="Directory for cache files")
    parser.add_argument("--spacing", type=float, default=2.8,
                       help="Spacing factor (default: 2.8)")
    parser.add_argument("--n-rydberg", type=int, default=70,
                       help="Rydberg n (default: 70)")
    args = parser.parse_args()
    
    # Build apparatus from args
    apparatus = ApparatusConstraints(
        spacing_factor=args.spacing,
        n_rydberg=args.n_rydberg,
    )
    
    protocols = ["lp", "smooth_jp", "jp_bangbang"] if args.protocol == "all" else [args.protocol]
    
    print("\n" + "=" * 70)
    print("  BASELINE PERFORMANCE (default parameters)")
    print("=" * 70)
    
    baselines = {}
    for p in protocols:
        print(f"\n  --- {p} ---")
        baselines[p] = run_baseline(
            p, apparatus, include_noise=args.noise, verbose=True
        )
    
    print("\n\n" + "=" * 70)
    print("  OPTIMISATION")
    print("=" * 70)
    
    results = {}
    for p in protocols:
        cache_path = None
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            cache_path = os.path.join(args.cache_dir, f"cache_{p}.json")
        
        results[p] = optimize_cz_gate(
            protocol=p,
            apparatus=apparatus,
            include_noise=args.noise,
            maxiter=args.maxiter,
            popsize=args.popsize,
            cache_path=cache_path,
            verbose=True,
        )
    
    # Final summary
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Protocol':<15} {'Base F':>10} {'Opt F':>10} "
          f"{'CZ φ-fid':>10} {'φ err':>8} {'t_gate':>8} {'Cost':>10} {'OK':>4}")
    print("-" * 75)
    for p in protocols:
        bl = baselines[p]
        opt = results[p].best_metrics
        status = "✓" if results[p].success else "✗"
        print(f"  {p:<15} {bl['avg_fidelity']:10.6f} {opt['avg_fidelity']:10.6f} "
              f"{opt.get('cz_phase_fidelity', 0):10.6f} "
              f"{opt['phase_error_deg']:7.2f}° "
              f"{opt['gate_time_us']:7.3f}µs "
              f"{results[p].best_cost:10.4f} {status:>4}")
    print("=" * 75)


if __name__ == "__main__":
    main()
