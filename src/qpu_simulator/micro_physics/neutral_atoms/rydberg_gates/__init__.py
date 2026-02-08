"""
Rydberg Gate Physics Module
===========================

This module implements the physics of Rydberg-blockade-based quantum gates for
neutral atom quantum computing. It provides a complete simulation framework
for two-qubit CZ gates using Rydberg atom arrays.

PHYSICS OVERVIEW FOR BEGINNERS
------------------------------

**What is a Rydberg atom?**
    A Rydberg atom is an atom where one electron has been excited to a very high
    energy level (high principal quantum number n, typically n=50-100). These atoms
    have exaggerated properties:
    
    - HUGE SIZE: The electron orbits ~1000× farther from nucleus (radius ~ n² a₀)
    - GIANT DIPOLE MOMENT: Strong interactions with light and other atoms
    - LONG LIFETIME: Can persist for ~100 μs (useful for gates)
    - STRONG INTERACTIONS: Two Rydberg atoms at μm separation interact strongly
    
**What is Rydberg blockade?**
    When two atoms are close together and both excited to Rydberg states, they
    experience a strong van der Waals interaction:
    
        V = C₆ / R⁶
    
    where C₆ scales as n¹¹ (!) and R is the separation. This interaction SHIFTS
    the energy of the doubly-excited state |rr⟩, making it off-resonant with the
    laser. Result: only ONE atom can be excited at a time = BLOCKADE.
    
    This blockade is the key to making a CZ gate! When both atoms are in state |1⟩,
    the blockade prevents certain transitions, accumulating a π phase shift.

**How does a CZ gate work with Rydberg atoms?**
    The CZ (controlled-Z) gate applies a π phase only when both qubits are |1⟩:
    
        |00⟩ → |00⟩
        |01⟩ → |01⟩
        |10⟩ → |10⟩
        |11⟩ → -|11⟩   ← The minus sign is the π phase!
    
    Rydberg implementation:
    1. Drive |1⟩ → |r⟩ (Rydberg) transition with laser
    2. For |01⟩ or |10⟩: One atom does a 2π Rabi cycle, returns to |1⟩
    3. For |11⟩: Blockade prevents double excitation, atoms evolve differently
    4. The different evolution paths give |11⟩ an extra π phase = CZ gate!

MODULE STRUCTURE
----------------

Core Physics:
    - constants: Physical constants and unit conversions
    - atom_database: Properties of Rb87 and Cs133 atoms
    - configurations: Dataclasses for simulation parameters

Laser Physics:
    - laser_physics: Rabi frequencies, two-photon excitation, Clebsch-Gordan coefficients

Trap Physics:
    - trap_physics: Optical tweezers, trap depth, position uncertainty

Noise Models:
    - noise_models: Decoherence, decay, dephasing, loss mechanisms

Quantum Evolution:
    - hamiltonians: Hamiltonian construction for 3-level and 4-level models
    - pulse_shaping: Pulse envelopes (square, Gaussian, DRAG)
    - protocols: Levine-Pichler and Jandura-Pupillo CZ protocols

Main Simulation:
    - simulation: The simulate_CZ_gate() function and fidelity computation

References
----------
[1] Saffman, Walker & Mølmer, "Quantum information with Rydberg atoms", 
    Rev. Mod. Phys. 82, 2313 (2010) - The "bible" of Rydberg physics
    
[2] Levine et al., "High-Fidelity Control and Entanglement of Rydberg Atom Qubits",
    Phys. Rev. Lett. 121, 123603 (2018) - First high-fidelity Rydberg gates
    
[3] Bluvstein, D., PhD Thesis, Harvard University (2024) - State-of-the-art
    error analysis and gate optimization

[4] Jandura & Pupillo, "Time-optimal two- and three-qubit gates for Rydberg atoms",
    PRX Quantum 3, 010353 (2022) - Time-optimal protocols

Authors: Based on notebook by [original author], modularized 2025
"""

from .constants import (
    # Fundamental constants
    HBAR, EPS0, C, E_CHARGE, A0, KB, MU_B,
    RY_JOULES, RY_EV,
    # Nuclear g-factors
    G_I_RB87, G_I_CS133, G_E,
)

from .atom_database import (
    ATOM_DB,
    get_atom_properties,
    get_quantum_defect,
    effective_n,
    get_rydberg_energy,
    get_C6,
    get_rydberg_lifetime,
    get_default_intermediate_state,
)

from .configurations import (
    LaserParameters,
    TweezerParameters,
    EnvironmentParameters,
    AtomicConfiguration,
    get_standard_rb87_config,
    get_standard_cs133_config,
    # Protocol-specific simulation inputs
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LPSimulationInputs,
    JPSimulationInputs,
    SmoothJPSimulationInputs,
)

from .laser_physics import (
    laser_E0,
    single_photon_rabi,
    two_photon_rabi,
    rydberg_blockade,
    blockade_radius,
    CLEBSCH_GORDAN_D2,
    get_clebsch_gordan,
)

from .trap_physics import (
    tweezer_spacing,
    trap_depth,
    trap_frequencies,
    position_uncertainty,
    anti_trap_potential,
    atom_loss_probability,
    compute_trap_dependent_noise,
    calculate_zeeman_shift,
    calculate_stark_shift,
)

from .noise_models import (
    build_all_noise_operators,
    build_decay_operators,
    build_dephasing_operators,
    build_loss_operators,
    build_scatter_operators,
    intermediate_state_scattering_rate,
    leakage_rate_to_adjacent_states,
    NoiseRates,
    compute_noise_rates,
)

from .hamiltonians import (
    # Hilbert space
    HilbertSpace,
    build_hilbert_space,
    HS3, HS4,
    # Hamiltonian builders
    build_laser_hamiltonian,
    build_detuning_hamiltonian,
    build_interaction_hamiltonian,
    build_full_hamiltonian,
)

from .pulse_shaping import (
    get_pulse_envelope,
    pulse_envelope_gaussian,
    pulse_envelope_cosine,
    pulse_envelope_blackman,
    pulse_envelope_drag,
    compute_leakage_detuning,
    spectral_leakage_factor,
    PULSE_SHAPES,
)

from .protocols import (
    # Protocol dataclasses
    ProtocolParameters,
    LPProtocolParameters,
    JPProtocolParameters,
    # Factory functions
    get_lp_protocol,
    get_jp_protocol,  # deprecated, returns smooth JP
    # Default instances
    LP_DEFAULT,
    JP_DEFAULT,
    # Protocol constants
    JP_SWITCHING_TIMES_DEFAULT,
    JP_PHASES_DEFAULT,
    LP_OMEGA_TAU_DEFAULT,
    LP_DELTA_OVER_OMEGA_DEFAULT,
    LP_XI_DEFAULT,
    # Smooth JP (recommended)
    SMOOTH_JP_PARAMS,
    # Parameter retrieval
    get_protocol_params,
    compute_phase_shift_xi,
    # Legacy dicts
    LEVINE_PICHLER_PARAMS,
    CZ_OPTIMAL_PARAMS,
)

from .simulation import (
    simulate_CZ_gate,
    compute_CZ_fidelity,
    compute_state_fidelity,
    SimulationResult,
)

from .optimization import (
    HardwareOptimizationResult,
    optimize_CZ_parameters,
    EvaluatedPoint,
    ExplorationResult,
    explore_parameter_space,
    combine_explorations,
)

from .optimize_cz_gate import (
    optimize_cz_gate,
    ApparatusConstraints,
    SimulationCache,
    run_baseline,
    compute_cost,
    extract_metrics,
    OptimizationResult,
)

from .visualization import (
    plot_exploration_results,
    plot_pareto_comparison,
    plot_parameter_heatmap,
    plot_noise_breakdown,
)

__version__ = "1.0.0"
__all__ = [
    # =========================================================================
    # END-TO-END CZ GATE OPTIMIZATION FLOW
    # =========================================================================
    # 1. Configure apparatus constraints
    "ApparatusConstraints",
    # 2. Run optimization (or baseline)
    "optimize_cz_gate", "run_baseline", "compute_cost", "extract_metrics",
    "SimulationCache", "OptimizationResult",
    # 3. Or run raw simulation
    "simulate_CZ_gate", "SimulationResult",
    "compute_CZ_fidelity", "compute_state_fidelity",
    
    # =========================================================================
    # PROTOCOL PARAMETERS
    # =========================================================================
    "ProtocolParameters", "LPProtocolParameters", "JPProtocolParameters",
    "get_lp_protocol", "get_jp_protocol", "LP_DEFAULT", "JP_DEFAULT",
    "SMOOTH_JP_PARAMS",
    "JP_SWITCHING_TIMES_DEFAULT", "JP_PHASES_DEFAULT",
    "LP_OMEGA_TAU_DEFAULT", "LP_DELTA_OVER_OMEGA_DEFAULT", "LP_XI_DEFAULT",
    "LEVINE_PICHLER_PARAMS",
    "get_protocol_params", "compute_phase_shift_xi",
    "CZ_OPTIMAL_PARAMS",
    
    # =========================================================================
    # SIMULATION CONFIGURATION
    # =========================================================================
    "TwoPhotonExcitationConfig", "NoiseSourceConfig",
    "LPSimulationInputs", "JPSimulationInputs", "SmoothJPSimulationInputs",
    "LaserParameters", "TweezerParameters", "EnvironmentParameters",
    "AtomicConfiguration",
    "get_standard_rb87_config", "get_standard_cs133_config",
    
    # =========================================================================
    # PHYSICS BUILDING BLOCKS
    # =========================================================================
    # Constants
    "HBAR", "EPS0", "C", "E_CHARGE", "A0", "KB", "MU_B",
    "RY_JOULES", "RY_EV", "G_I_RB87", "G_I_CS133", "G_E",
    # Atom database
    "ATOM_DB", "get_atom_properties", "get_quantum_defect", "effective_n",
    "get_rydberg_energy", "get_C6", "get_rydberg_lifetime",
    # Laser physics
    "laser_E0", "single_photon_rabi", "two_photon_rabi",
    "rydberg_blockade", "blockade_radius",
    "CLEBSCH_GORDAN_D2", "get_clebsch_gordan",
    # Trap physics
    "tweezer_spacing", "trap_depth", "trap_frequencies", "position_uncertainty",
    "anti_trap_potential", "atom_loss_probability", "compute_trap_dependent_noise",
    "calculate_zeeman_shift", "calculate_stark_shift",
    # Noise models
    "build_all_noise_operators",
    "build_decay_operators", "build_dephasing_operators", "build_loss_operators",
    "build_scatter_operators",
    "intermediate_state_scattering_rate", "leakage_rate_to_adjacent_states",
    "NoiseRates", "compute_noise_rates",
    # Hamiltonians
    "HilbertSpace", "build_hilbert_space", "HS3", "HS4",
    "build_laser_hamiltonian", "build_detuning_hamiltonian",
    "build_interaction_hamiltonian", "build_full_hamiltonian",
    # Pulse shaping
    "get_pulse_envelope", "pulse_envelope_gaussian", "pulse_envelope_cosine",
    "pulse_envelope_blackman", "pulse_envelope_drag",
    "compute_leakage_detuning", "spectral_leakage_factor", "PULSE_SHAPES",
    
    # =========================================================================
    # HARDWARE PARAMETER EXPLORATION (Inverse problem + Pareto analysis)
    # =========================================================================
    "HardwareOptimizationResult", "optimize_CZ_parameters",
    "EvaluatedPoint", "ExplorationResult", "explore_parameter_space",
    "combine_explorations",
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    "plot_exploration_results", "plot_pareto_comparison",
    "plot_parameter_heatmap", "plot_noise_breakdown",
]
