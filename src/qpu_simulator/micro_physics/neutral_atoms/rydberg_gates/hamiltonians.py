"""
Hamiltonians for Rydberg Gate Simulation
========================================

This module provides functions to build the Hamiltonians (energy operators)
for simulating two-qubit CZ gates using Rydberg interactions. The Hamiltonians
describe how atoms evolve under laser driving, Rydberg blockade, and various
detuning mechanisms.

What is a Hamiltonian?
----------------------
In quantum mechanics, the Hamiltonian H is the operator that determines how
quantum states evolve in time. For a state |ψ⟩, the Schrödinger equation gives:

    i ℏ d|ψ⟩/dt = H |ψ⟩

The Hamiltonian contains all the physics: laser couplings, detunings, and
atom-atom interactions. Different Hamiltonian terms correspond to different
physical effects.

How Hamiltonians Drive Evolution: The Key Insight
-------------------------------------------------
**This section explains WHY different Hamiltonian terms cause different effects.**

The Schrödinger equation i ℏ d|ψ⟩/dt = H |ψ⟩ has a formal solution:

    |ψ(t)⟩ = e^{-iHt/ℏ} |ψ(0)⟩

The exponential of an operator is defined by its Taylor series. To understand
what this means physically, we need to look at EIGENSTATES of H.

**Rule 1: DIAGONAL terms (energy shifts) cause PHASE accumulation**

If H|n⟩ = Eₙ|n⟩ (i.e., |n⟩ is an eigenstate with energy Eₙ), then:

    |n(t)⟩ = e^{-iEₙt/ℏ} |n(0)⟩

The state just picks up a phase! The phase grows linearly with time at rate Eₙ/ℏ.

Example: H_detuning = -Δ|r⟩⟨r| acting on |r⟩:
    - |r⟩ is an eigenstate with eigenvalue -Δ
    - |r(t)⟩ = e^{+iΔt} |r(0)⟩  (phase grows at rate Δ)

Example: H_zeeman = δ_B|1⟩⟨1| acting on |1⟩:
    - |1⟩ is an eigenstate with eigenvalue δ_B
    - |1(t)⟩ = e^{-iδ_B t} |1(0)⟩  (phase grows at rate -δ_B)

**Rule 2: OFF-DIAGONAL terms (couplings) cause STATE MIXING (oscillations)**

If H has off-diagonal elements connecting |a⟩ and |b⟩, the eigenstates are
SUPERPOSITIONS of |a⟩ and |b⟩, not the original states themselves.

Example: H_laser = (Ω/2)(|r⟩⟨1| + |1⟩⟨r|) for a two-level system:
    - Eigenstates are |+⟩ = (|1⟩+|r⟩)/√2 and |-⟩ = (|1⟩-|r⟩)/√2
    - Eigenvalues: E₊ = +Ω/2, E₋ = -Ω/2
    - Starting in |1⟩ = (|+⟩ + |-⟩)/√2:
      |ψ(t)⟩ = (e^{-iΩt/2}|+⟩ + e^{+iΩt/2}|-⟩)/√2
             = cos(Ωt/2)|1⟩ - i·sin(Ωt/2)|r⟩
    - Population in |r⟩: |⟨r|ψ⟩|² = sin²(Ωt/2)  → Rabi oscillations!

**Rule 3: COMBINED diagonal + off-diagonal gives the "generalized Rabi frequency"**

For a two-level system with coupling Ω and energy difference Δ:

    H = (Ω/2)(|r⟩⟨1| + |1⟩⟨r|) - Δ|r⟩⟨r|
    
      = [ 0    Ω/2 ]   in the {|1⟩, |r⟩} basis
        [ Ω/2  -Δ  ]

The eigenvalues are E± = -Δ/2 ± Ω_gen/2, where:

    Ω_gen = √(Ω² + Δ²)    ← "generalized Rabi frequency"

The oscillation frequency is set by the EIGENVALUE DIFFERENCE:

    ΔE = E₊ - E₋ = Ω_gen = √(Ω² + Δ²)

So with detuning Δ ≠ 0:
- Oscillations are FASTER: frequency Ω_gen > Ω
- But INCOMPLETE: max population in |r⟩ is only Ω²/Ω_gen² = Ω²/(Ω²+Δ²) < 1
- And the state picks up a PHASE that depends on both Ω and Δ

**⚠️ CRITICAL: Ω_gen only applies to TWO-LEVEL systems!**

The |01⟩ ↔ |0r⟩ subspace IS a two-level system (atom 2 is spectator).
The |11⟩ ↔ |1r⟩+|r1⟩ ↔ |rr⟩ subspace is a THREE-level system with blockade.
The dynamics are MORE COMPLEX and Ω_gen doesn't directly apply to |11⟩.

**Rule 4: Different TOTAL energies → Different PHASES → CZ gate!**

The key insight for understanding the CZ gate:

    If two states have different energies, they accumulate different phases.
    
    If we engineer the phase difference to be exactly π, we get a CZ gate!

Hilbert Space Structure
-----------------------
We work with either a 3-level or 4-level model for EACH atom:

**3-level model** (standard):
    |0⟩ = ground state "0" (e.g., F=1, mF=0 for Rb87)
    |1⟩ = ground state "1" (e.g., F=2, mF=0 for Rb87)
    |r⟩ = Rydberg state (e.g., 70S₁/₂)
    
    Laser coupling: |1⟩ ↔ |r⟩ (two-photon via intermediate P state)
    |0⟩ is spectator (dark to excitation laser)

**4-level model** (for polarization impurity):
    |0⟩ = ground state "0"
    |1⟩ = ground state "1"
    |r+⟩ = Rydberg state with mJ = +1/2
    |r-⟩ = Rydberg state with mJ = -1/2
    
    σ+ light couples |1⟩ → |r+⟩
    σ- light couples |1⟩ → |r-⟩
    Polarization impurity causes small |r-⟩ population

For TWO atoms, the Hilbert space is the tensor product:
    dim = 3×3 = 9 (3-level) or 4×4 = 16 (4-level)

The Rotating Wave Approximation (RWA)
-------------------------------------
Real laser fields oscillate at optical frequencies (~400 THz). We transform
to a "rotating frame" where these fast oscillations are removed, leaving only
the slow dynamics we care about. This is valid when:

    Ω << ω_optical    (Rabi frequency << optical frequency)

For typical Ω ~ 1-10 MHz and ω ~ 400 THz, the ratio is ~10⁻⁸, so RWA is
excellent. The counter-rotating terms we neglect would contribute corrections
of order (Ω/ω)² ~ 10⁻¹⁶, completely negligible.

Hamiltonian Terms
-----------------
The full two-atom Hamiltonian in the rotating frame is:

    H = H_laser + H_detuning + H_interaction + H_zeeman + H_stark

**Understanding the tensor product structure:**

Each single-atom operator acts on ONE atom. For two atoms, we write:

    O₁ ⊗ I₂ = "O acts on atom 1, identity on atom 2"
    I₁ ⊗ O₂ = "identity on atom 1, O acts on atom 2"

When we sum: O₁ ⊗ I₂ + I₁ ⊗ O₂, we're saying "O acts on atom 1 OR atom 2".

**Why |rr⟩ gets -2Δ from H_detuning (the factor of 2 explained):**

    H_detuning = -Δ (|r⟩⟨r|₁ ⊗ I₂ + I₁ ⊗ |r⟩⟨r|₂)

Acting on |rr⟩ = |r⟩₁|r⟩₂:
- First term: |r⟩⟨r|₁ ⊗ I₂ gives -Δ (atom 1 contributes)
- Second term: I₁ ⊗ |r⟩⟨r|₂ gives -Δ (atom 2 contributes)
- Total: -2Δ

Acting on |1r⟩ = |1⟩₁|r⟩₂:
- First term: |r⟩⟨r|₁|1⟩₁ = 0 (atom 1 is in |1⟩, not |r⟩)
- Second term: I₁ ⊗ |r⟩⟨r|₂|r⟩₂ = -Δ (atom 2 contributes)
- Total: -Δ

So each atom in |r⟩ contributes -Δ to the total energy. Two atoms → 2×(-Δ).

1. **H_laser** (laser coupling):
   Drives |1⟩ ↔ |r⟩ transitions. For two atoms:
   
       H_laser = (Ω/2)(|r⟩⟨1|₁ ⊗ I₂ + I₁ ⊗ |r⟩⟨1|₂) + h.c.
   
   where Ω is the (complex) Rabi frequency including laser phase.
   
   **How it drives evolution:** This off-diagonal coupling creates coherent
   oscillations (Rabi flopping) between |1⟩ and |r⟩. Starting in |1⟩, the
   population oscillates as sin²(Ωt/2), reaching |r⟩ at t = π/Ω (a "π-pulse").
   The laser effectively "rotates" the state on the Bloch sphere connecting
   |1⟩ and |r⟩. Larger Ω = faster rotation = faster gates.
   
   **Role in LP protocol:** Two pulses with different phases (ξ shift).
   **Role in JP protocol:** Constant Ω, but PHASE is modulated in time.

2. **H_detuning** (two-photon detuning):
   Energy shift of Rydberg state from laser frequency:
   
       H_detuning = -Δ (|r⟩⟨r|₁ ⊗ I₂ + I₁ ⊗ |r⟩⟨r|₂)
   
   Δ > 0 means laser is red-detuned (below resonance).
   
   **⚠️ IMPORTANT: Two different Δ symbols in this codebase!**
   
   - **Δₑ (intermediate detuning, ~GHz)**: Detuning from the P state in the
     two-photon ladder |1⟩ → |P⟩ → |r⟩. This is LARGE to avoid populating
     the short-lived P state. It determines Ω_eff = Ω₁Ω₂/(2Δₑ).
     See laser_physics.py for this.
   
   - **Δ (two-photon detuning, ~MHz)**: Detuning of the EFFECTIVE two-photon
     transition from the |r⟩ resonance. This is SMALL and used for gate control.
     This H_detuning term represents THIS Δ.
   
   **How Δ affects evolution (see "How Hamiltonians Drive Evolution" above):**
   
   The detuning Δ adds a diagonal energy -Δ to any state with an atom in |r⟩.
   This causes that state to accumulate phase at rate Δ relative to ground states.
   
   For a SINGLE atom (or the |01⟩↔|0r⟩ two-level subsystem):
   - Generalized Rabi frequency: Ω_gen = √(Ω² + Δ²)
   - Max Rydberg population: Ω²/(Ω² + Δ²) < 1
   - Accumulated phase depends on both Ω and Δ
   
   **Why |11⟩ evolves differently from |01⟩/|10⟩ (THE KEY INSIGHT):**
   
   The TOTAL energy of each two-atom state is different:
   
   | Two-atom state | Energy from H_detuning | Energy from H_interaction |
   |----------------|------------------------|---------------------------|
   | |00⟩           | 0                      | 0                         |
   | |01⟩, |10⟩     | 0                      | 0                         |
   | |0r⟩, |r0⟩     | -Δ                     | 0                         |
   | |11⟩           | 0                      | 0                         |
   | |1r⟩, |r1⟩     | -Δ                     | 0                         |
   | |rr⟩           | -2Δ                    | +V (BLOCKADE!)            |
   
   For |01⟩ or |10⟩ (one atom in |0⟩, one in |1⟩):
   
       |01⟩ ←─Ω─→ |0r⟩      (simple two-level system)
              ↓
         energy: -Δ
   
   This IS a two-level system. The |0⟩ atom just sits there. The |1⟩ atom
   oscillates with frequency Ω_gen = √(Ω² + Δ²) and picks up a phase φ₀₁.
   
   For |11⟩ (both atoms in |1⟩):
   
       |11⟩ ←─√2·Ω─→ |ψ_sym⟩ ←─√2·Ω─→ |rr⟩
                ↓                  ↓
           energy: -Δ        energy: -2Δ + V
   
   where |ψ_sym⟩ = (|1r⟩ + |r1⟩)/√2 is the symmetric single-excitation state.
   (The √2 enhancement comes from constructive interference of both atoms.)
   
   This is a THREE-level system! The dynamics are NOT simple Rabi oscillations.
   When V >> Ω (strong blockade):
   - |rr⟩ is so far off-resonance (energy shifted by V) it's inaccessible
   - System oscillates mostly in {|11⟩, |ψ_sym⟩} subspace
   - But the PRESENCE of |rr⟩ (even unpopulated) shifts the dynamics
   - This modified dynamics gives a DIFFERENT phase φ₁₁ ≠ 2×φ₀₁
   
   **The CZ gate condition:**
   
   By choosing Δ/Ω correctly:
   - LP protocol: Δ/Ω ≈ 0.377 → φ₁₁ - 2φ₀₁ = π
   - JP protocol: Uses Δ = 0 but modulates PHASE to achieve same condition
   
   **Role in LP vs JP protocols:**
   - LP (Levine-Pichler): Uses FIXED Δ/Ω ≈ 0.377 throughout, two pulses with phase shift
   - JP (Jandura-Pupillo): Uses Δ = 0, achieves phase control via laser PHASE modulation
     φ(t) ∈ {-π/2, 0, +π/2}. This is "bang-bang" optimal control.

3. **H_interaction** (Rydberg blockade):
   Van der Waals interaction when BOTH atoms are excited:
   
       H_interaction = V |rr⟩⟨rr|
   
   where V = C₆/R⁶. This is the key term that creates entanglement!
   
   **How it affects evolution:**
   
   This adds energy +V to the |rr⟩ state ONLY. From the energy table above,
   |rr⟩ has total energy (-2Δ + V). When V >> Ω, Δ:
   - |rr⟩ is far off-resonance from the laser
   - Population cannot reach |rr⟩ → "blockade"
   - |11⟩ dynamics are confined to {|11⟩, |1r⟩+|r1⟩} subspace
   
   **Blockade mechanism:** When V >> Ω, the state |rr⟩ is shifted so far
   off-resonance that the laser cannot excite both atoms simultaneously.
   This creates the controlled-phase gate.
   
   **Why this creates entanglement:** For |11⟩, the system TRIES to go
   |11⟩ → |1r⟩+|r1⟩ → |rr⟩, but blockade prevents reaching |rr⟩. The 
   population oscillates in a DIFFERENT pattern than |01⟩ or |10⟩ (which 
   freely excite to |0r⟩ or |r0⟩). Different dynamics = different phase.
   
   **Role in both LP and JP:** Blockade is essential for BOTH protocols.
   It's what makes the |11⟩ phase different from twice the |01⟩ phase.

4. **H_zeeman** (differential Zeeman shift):
   Energy shift of qubit states due to magnetic field:
   
       H_zeeman = δ_B (|1⟩⟨1|₁ ⊗ I₂ + I₁ ⊗ |1⟩⟨1|₂)
   
   For clock states (mF=0 → mF=0), this is suppressed.
   
   **How it affects evolution:**
   
   This is a DIAGONAL term, so it causes pure phase accumulation (Rule 1).
   Any state with atom(s) in |1⟩ accumulates phase at rate δ_B per |1⟩ atom.
   
   | State | Phase accumulation rate |
   |-------|-------------------------|
   | |00⟩  | 0                       |
   | |01⟩  | δ_B                     |
   | |10⟩  | δ_B                     |
   | |11⟩  | 2·δ_B                   |
   
   After time t: |01⟩ → e^{-iδ_B·t}|01⟩, |11⟩ → e^{-2iδ_B·t}|11⟩
   
   **Why it's a problem:** If δ_B fluctuates (due to magnetic field noise),
   the phase becomes random → decoherence. Even if stable, an UNKNOWN δ_B
   causes systematic phase errors. Clock states (mF=0) have δ_B ∝ B² instead
   of ∝ B, making them ~1000× less sensitive to B-field fluctuations.

5. **H_stark** (differential AC Stark shift):
   Light shift from trap laser on qubit states:
   
       H_stark = δ_AC (|1⟩⟨1|₁ ⊗ I₂ + I₁ ⊗ |1⟩⟨1|₂)
   
   Typically ~100 kHz for ~1 mK traps.
   
   **How it affects evolution:**
   
   Same structure as H_zeeman → same phase accumulation pattern.
   The trap laser's electric field shifts |0⟩ and |1⟩ by different amounts
   (different AC polarizabilities), creating a differential energy δ_AC.
   
   **Why it matters for gates:** During a ~200 ns gate with δ_AC = 2π×100 kHz,
   the accumulated phase error is δ_AC × t ≈ 0.13 rad ≈ 7°. This systematic
   shift must be calibrated out or the trap turned off during the gate.
   
   **LP vs JP consideration:** Both protocols have same sensitivity to these
   spurious phase shifts. The trap-off technique helps both equally.

Summary: How Different States Acquire Different Phases
------------------------------------------------------
The CZ gate works because |11⟩ acquires a DIFFERENT phase than |01⟩ + |10⟩.

**For |00⟩:** Dark to the laser. Picks up only Zeeman/Stark phases (error terms).

**For |01⟩ and |10⟩:** Simple two-level Rabi dynamics between |01⟩↔|0r⟩.
   Phase determined by (Ω, Δ) via generalized Rabi frequency.

**For |11⟩:** Three-level dynamics |11⟩↔|ψ_sym⟩↔|rr⟩ with blockade.
   Blockade prevents |rr⟩ population, modifying the oscillation pattern.
   This gives a DIFFERENT phase than twice the single-atom phase.

The protocols (LP or JP) choose pulse parameters so that:

    φ₁₁ - φ₀₁ - φ₁₀ = π    (up to single-qubit phases)

This is the CZ gate condition!

References
----------
- Levine et al., PRL 123, 170503 (2019) - Two-pulse CZ gate (LP protocol)
- Jandura & Pupillo, PRX Quantum 3, 010353 (2022) - Time-optimal gates (JP protocol)
- Bluvstein PhD Thesis (Harvard, 2024) - Comprehensive error analysis
- Saffman et al., Rev. Mod. Phys. 82, 2313 (2010) - Rydberg physics review

Author: Quantum Simulation Team
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

# QuTiP imports for quantum objects
try:
    from qutip import Qobj, basis, tensor, qeye, mesolve
except ImportError:
    raise ImportError(
        "QuTiP is required for Hamiltonian construction. "
        "Install with: pip install qutip"
    )

from .constants import HBAR, MU_B


# =============================================================================
# HILBERT SPACE CONSTRUCTION
# =============================================================================

@dataclass
class HilbertSpace:
    """
    Container for Hilbert space objects for one atom.
    
    What is a Hilbert Space?
    ------------------------
    In quantum mechanics, the Hilbert space is the mathematical space where
    quantum states "live." For a 3-level atom, it's a 3-dimensional complex
    vector space with basis states |0⟩, |1⟩, |r⟩.
    
    This class stores pre-computed basis states, projectors, and transition
    operators that we need repeatedly when building Hamiltonians.
    
    Attributes
    ----------
    dim : int
        Hilbert space dimension (3 or 4)
    basis : Dict[str, Qobj]
        Dictionary of basis state kets: {'0': |0⟩, '1': |1⟩, 'r': |r⟩, ...}
    projectors : Dict[str, Qobj]
        Dictionary of projection operators: {'0': |0⟩⟨0|, '1': |1⟩⟨1|, ...}
    transitions : Dict[str, Qobj]
        Dictionary of transition operators: {'r->1': |1⟩⟨r|, '1->r': |r⟩⟨1|, ...}
    identity : Qobj
        Identity operator on this Hilbert space
        
    Examples
    --------
    >>> hs = build_hilbert_space(dim=3)
    >>> hs.basis['r']  # Rydberg state ket
    Quantum object: dims = [[3], [1]], shape = (3, 1), type = ket
    >>> hs.projectors['1']  # Projector onto |1⟩
    Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper
    >>> hs.transitions['1->r']  # Raising operator |r⟩⟨1|
    Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper
    """
    dim: int
    basis: Dict[str, Qobj] = field(default_factory=dict)
    projectors: Dict[str, Qobj] = field(default_factory=dict)
    transitions: Dict[str, Qobj] = field(default_factory=dict)
    identity: Qobj = None


def build_hilbert_space(dim: int = 3) -> HilbertSpace:
    """
    Build single-atom Hilbert space with basis states and operators.
    
    This creates all the quantum operators we need for one atom. The two-atom
    Hilbert space is then built using tensor products.
    
    Parameters
    ----------
    dim : int
        Dimension of single-atom Hilbert space:
        - dim=3: Standard model with |0⟩, |1⟩, |r⟩
        - dim=4: Extended model with |0⟩, |1⟩, |r+⟩, |r-⟩
        
    Returns
    -------
    HilbertSpace
        Container with all basis states and operators
        
    Physics Notes
    -------------
    **Why 3 levels?**
    The ground qubit states |0⟩ and |1⟩ are typically hyperfine sublevels:
    - Rb87: |0⟩ = |5S₁/₂, F=1, mF=0⟩, |1⟩ = |5S₁/₂, F=2, mF=0⟩
    - Splitting: ~6.8 GHz (microwave frequency)
    
    The Rydberg state |r⟩ is a highly excited state (n ~ 50-100) with:
    - Large orbital radius: ~n² a₀ ~ 0.5 μm for n=70
    - Long lifetime: ~100 μs for n=70
    - Strong interactions: C₆ ~ n¹¹
    
    **Why mF=0 states?**
    These are "clock states" that are first-order insensitive to magnetic
    field fluctuations (quadratic Zeeman shift only), giving better coherence.
    
    **4-level extension:**
    Real σ+ polarized light isn't perfectly pure. A small σ- component
    couples to the mJ=-1/2 Rydberg sublevel. The 4-level model captures:
    - Main coupling: |1⟩ → |r+⟩ (mJ=+1/2)
    - Leakage: |1⟩ → |r-⟩ (mJ=-1/2) with ~1-2% probability
    """
    hs = HilbertSpace(dim=dim)
    
    if dim == 3:
        # 3-level: |0⟩, |1⟩, |r⟩
        # QuTiP basis(N, n) creates |n⟩ in N-dimensional space
        hs.basis = {
            '0': basis(3, 0),  # |0⟩ = (1, 0, 0)ᵀ
            '1': basis(3, 1),  # |1⟩ = (0, 1, 0)ᵀ
            'r': basis(3, 2),  # |r⟩ = (0, 0, 1)ᵀ
        }
        
        # Projectors: P_i = |i⟩⟨i|
        for key, ket in hs.basis.items():
            hs.projectors[key] = ket * ket.dag()
        
        # Transition operators (lowering): σ_ij = |j⟩⟨i| takes |i⟩ → |j⟩
        # These are "lowering" operators in the sense that they lower the
        # excitation level when going r→1→0
        hs.transitions = {
            'r->1': hs.basis['1'] * hs.basis['r'].dag(),  # |1⟩⟨r| (de-excitation)
            'r->0': hs.basis['0'] * hs.basis['r'].dag(),  # |0⟩⟨r| (decay to wrong state)
            '1->r': hs.basis['r'] * hs.basis['1'].dag(),  # |r⟩⟨1| (excitation)
            '1->0': hs.basis['0'] * hs.basis['1'].dag(),  # |0⟩⟨1| (qubit flip)
        }
        
    elif dim == 4:
        # 4-level: |0⟩, |1⟩, |r+⟩ (mJ=+1/2), |r-⟩ (mJ=-1/2)
        hs.basis = {
            '0': basis(4, 0),
            '1': basis(4, 1),
            'r+': basis(4, 2),  # Rydberg mJ = +1/2
            'r-': basis(4, 3),  # Rydberg mJ = -1/2
        }
        
        for key, ket in hs.basis.items():
            hs.projectors[key] = ket * ket.dag()
        
        # More transitions in 4-level model
        hs.transitions = {
            # From |r+⟩
            'r+->1': hs.basis['1'] * hs.basis['r+'].dag(),
            'r+->0': hs.basis['0'] * hs.basis['r+'].dag(),
            'r+->r-': hs.basis['r-'] * hs.basis['r+'].dag(),  # mJ mixing
            # From |r-⟩
            'r-->1': hs.basis['1'] * hs.basis['r-'].dag(),
            'r-->0': hs.basis['0'] * hs.basis['r-'].dag(),
            'r-->r+': hs.basis['r+'] * hs.basis['r-'].dag(),  # mJ mixing
            # Excitation
            '1->r+': hs.basis['r+'] * hs.basis['1'].dag(),
            '1->r-': hs.basis['r-'] * hs.basis['1'].dag(),
            '1->0': hs.basis['0'] * hs.basis['1'].dag(),
        }
    else:
        raise ValueError(f"Unsupported Hilbert space dimension: {dim}. Use 3 or 4.")
    
    hs.identity = qeye(dim)
    
    return hs


# Pre-built Hilbert spaces for convenience
# These are used throughout the module to avoid rebuilding
HS3 = build_hilbert_space(3)  # Standard 3-level model
HS4 = build_hilbert_space(4)  # Extended 4-level model

# Shorthand for basis states (3-level, backward compatible)
b0 = HS3.basis['0']
b1 = HS3.basis['1']
br = HS3.basis['r']


# =============================================================================
# TWO-ATOM OPERATOR CONSTRUCTION
# =============================================================================

def op_two_atom(op1: Qobj, op2: Qobj) -> Qobj:
    """
    Construct two-atom operator from single-atom operators.
    
    For two atoms, the full Hilbert space is the tensor product:
        H_total = H_atom1 ⊗ H_atom2
    
    An operator acting on atom 1 with identity on atom 2 is:
        O₁ ⊗ I₂
    
    Parameters
    ----------
    op1 : Qobj
        Operator acting on atom 1
    op2 : Qobj
        Operator acting on atom 2
        
    Returns
    -------
    Qobj
        Tensor product operator acting on two-atom Hilbert space
        
    Examples
    --------
    >>> # Projector onto |r⟩ for atom 1, identity for atom 2
    >>> P_r1 = op_two_atom(HS3.projectors['r'], HS3.identity)
    
    >>> # Projector onto |rr⟩ (both atoms in Rydberg)
    >>> P_rr = op_two_atom(HS3.projectors['r'], HS3.projectors['r'])
    
    Notes
    -----
    The tensor product ⊗ (Kronecker product) combines two operators:
        (A ⊗ B)|ψ₁⟩|ψ₂⟩ = (A|ψ₁⟩)(B|ψ₂⟩)
    
    For matrices, if A is m×m and B is n×n, then A⊗B is (mn)×(mn).
    """
    return tensor(op1, op2)


# =============================================================================
# HAMILTONIAN BUILDING BLOCKS
# =============================================================================

def build_laser_hamiltonian(
    Omega: complex,
    hs: HilbertSpace,
    polarization: str = "sigma+",
    Omega_minus: float = None,
) -> Qobj:
    """
    Build laser coupling Hamiltonian for two atoms.
    
    The laser drives |1⟩ ↔ |r⟩ transitions via a two-photon process through
    an intermediate P state. In the rotating frame, this becomes a simple
    coupling term.
    
    H_laser = (Ω/2)(|r⟩⟨1|₁ ⊗ I₂ + I₁ ⊗ |r⟩⟨1|₂) + h.c.
    
    Parameters
    ----------
    Omega : complex
        Rabi frequency (rad/s). Can be complex to include laser phase:
        Ω = |Ω| e^{iφ} where φ is the laser phase.
    hs : HilbertSpace
        Hilbert space object (HS3 or HS4)
    polarization : str
        Laser polarization for 4-level model:
        - "sigma+": Couples to |r+⟩ only (ideal)
        - "sigma-": Couples to |r-⟩ only
        - "pi": Equal coupling to both (bad!)
        - "mixed": Small σ- leakage (realistic)
    Omega_minus : float, optional
        For 4-level model with polarization impurity: Rabi frequency
        for the σ- component (typically ~1-5% of main Omega)
        
    Returns
    -------
    Qobj
        Two-atom laser Hamiltonian (Hermitian)
        
    Physics Notes
    -------------
    **Two-photon Raman process:**
    The actual transition |1⟩ → |r⟩ uses two lasers via intermediate |P⟩:
    
        |1⟩ --[Ω₁, 780nm]--> |5P₃/₂⟩ --[Ω₂, 480nm]--> |nS₁/₂⟩
    
    The intermediate state is detuned by Δ_e ~ 1-10 GHz to minimize scattering.
    The effective two-photon Rabi frequency is:
    
        Ω_eff = Ω₁ Ω₂ / (2 Δ_e)
    
    **Laser phase:**
    A complex Rabi frequency Ω = |Ω| e^{iφ} includes the laser phase φ.
    Phase control is crucial for the two-pulse CZ gate where the second
    pulse needs phase shift ξ = 3.9 rad.
    
    **Why both atoms have the same coupling?**
    Global laser illumination couples identically to both atoms (assuming
    uniform intensity). Local addressing would have different Ω₁ and Ω₂.
    """
    I = hs.identity
    
    if hs.dim == 3:
        # Standard 3-level model: |1⟩ ↔ |r⟩
        sigma_1r = hs.transitions['1->r']  # |r⟩⟨1| - excitation
        
        # H = (Ω/2)(σ₁ᵣ + σ₁ᵣ†) ⊗ I + I ⊗ (Ω/2)(σ₁ᵣ + σ₁ᵣ†)
        # Using Ω complex: (Ω σ + Ω* σ†)/2 = Re(Ω)(σ + σ†)/2 + Im(Ω)(σ - σ†)/(2i)
        H_atom = 0.5 * (Omega * sigma_1r + np.conj(Omega) * sigma_1r.dag())
        
        # Sum over both atoms
        H = op_two_atom(H_atom, I) + op_two_atom(I, H_atom)
        
    elif hs.dim == 4:
        # 4-level model: |1⟩ ↔ |r+⟩ and |1⟩ ↔ |r-⟩
        sigma_1rp = hs.transitions['1->r+']  # |r+⟩⟨1|
        sigma_1rm = hs.transitions['1->r-']  # |r-⟩⟨1|
        
        # Main coupling (σ+)
        if polarization == "sigma+":
            H_atom_plus = 0.5 * (Omega * sigma_1rp + np.conj(Omega) * sigma_1rp.dag())
            H_atom_minus = 0.0 * I  # No coupling
        elif polarization == "sigma-":
            H_atom_plus = 0.0 * I
            H_atom_minus = 0.5 * (Omega * sigma_1rm + np.conj(Omega) * sigma_1rm.dag())
        elif polarization == "pi":
            # π polarization couples equally to both (ΔmJ = 0 transitions)
            H_atom_plus = 0.5 * (Omega/np.sqrt(2) * sigma_1rp + 
                                 np.conj(Omega/np.sqrt(2)) * sigma_1rp.dag())
            H_atom_minus = 0.5 * (Omega/np.sqrt(2) * sigma_1rm + 
                                  np.conj(Omega/np.sqrt(2)) * sigma_1rm.dag())
        else:  # "mixed" or explicit Omega_minus
            H_atom_plus = 0.5 * (Omega * sigma_1rp + np.conj(Omega) * sigma_1rp.dag())
            Omega_m = Omega_minus if Omega_minus is not None else 0.02 * np.abs(Omega)
            H_atom_minus = 0.5 * (Omega_m * sigma_1rm + np.conj(Omega_m) * sigma_1rm.dag())
        
        H_atom = H_atom_plus + H_atom_minus
        H = op_two_atom(H_atom, I) + op_two_atom(I, H_atom)
    
    return H


def build_detuning_hamiltonian(
    Delta: float,
    hs: HilbertSpace,
    zeeman_splitting: float = 0,
    Delta_minus: float = None,
) -> Qobj:
    """
    Build two-photon detuning Hamiltonian.
    
    The detuning Δ is the frequency difference between the laser and the
    |1⟩ → |r⟩ transition. In the rotating frame:
    
    H_detuning = -Δ (|r⟩⟨r|₁ ⊗ I₂ + I₁ ⊗ |r⟩⟨r|₂)
    
    Parameters
    ----------
    Delta : float
        Two-photon detuning (rad/s).
        - Δ > 0: Red detuned (laser below resonance)
        - Δ < 0: Blue detuned (laser above resonance)
    hs : HilbertSpace
        Hilbert space object
    zeeman_splitting : float
        For 4-level: splitting between |r+⟩ and |r-⟩ due to B-field (rad/s)
    Delta_minus : float, optional
        For 4-level: explicit detuning for |r-⟩ state
        
    Returns
    -------
    Qobj
        Detuning Hamiltonian
        
    Physics Notes
    -------------
    **Why detune?**
    The optimal CZ gate uses Δ/Ω ≈ 0.377 (Levine et al.). This:
    1. Ensures |01⟩ and |10⟩ pick up the correct phases
    2. Creates a destructive interference that returns |1⟩ → |1⟩
    3. Combined with phase shift ξ, gives φ₁₁ = 2φ₀₁ - π (CZ condition)
    
    **Generalized Rabi frequency:**
    With detuning, the effective Rabi frequency becomes:
        Ω_gen = √(Ω² + Δ²)
    
    The oscillation frequency increases, but population transfer is incomplete.
    
    **Sign convention:**
    We use H = -Δ |r⟩⟨r| so that:
    - Δ > 0 (red): Rydberg state energy is LOWER in rotating frame
    - This matches the standard convention in atomic physics
    """
    I = hs.identity
    
    if hs.dim == 3:
        P_r = hs.projectors['r']
        H = -Delta * (op_two_atom(P_r, I) + op_two_atom(I, P_r))
        
    elif hs.dim == 4:
        # Separate detunings for mJ = +1/2 and -1/2
        P_rp = hs.projectors['r+']
        P_rm = hs.projectors['r-']
        
        # |r+⟩ detuning
        H_plus = -Delta * (op_two_atom(P_rp, I) + op_two_atom(I, P_rp))
        
        # |r-⟩ detuning (includes Zeeman shift if present)
        Delta_m = Delta_minus if Delta_minus is not None else Delta + zeeman_splitting
        H_minus = -Delta_m * (op_two_atom(P_rm, I) + op_two_atom(I, P_rm))
        
        H = H_plus + H_minus
    
    return H


def build_interaction_hamiltonian(
    V: float,
    hs: HilbertSpace,
    V_pm: float = None,
    V_mm: float = None,
) -> Qobj:
    """
    Build Rydberg-Rydberg interaction Hamiltonian.
    
    This is the KEY TERM that creates entanglement! When both atoms are
    in Rydberg states, they experience a strong van der Waals interaction:
    
    H_interaction = V |rr⟩⟨rr|
    
    where V = C₆/R⁶ for van der Waals interactions.
    
    Parameters
    ----------
    V : float
        Interaction strength (rad/s) for |r+r+⟩ or |rr⟩ in 3-level.
        V = C₆/R⁶ where C₆ ~ n¹¹ and R is interatomic spacing.
    hs : HilbertSpace
        Hilbert space object
    V_pm : float, optional
        For 4-level: interaction between |r+⟩ and |r-⟩ states.
        Often V_pm ≈ V due to similar orbital structure.
    V_mm : float, optional
        For 4-level: interaction between two |r-⟩ states.
        
    Returns
    -------
    Qobj
        Interaction Hamiltonian
        
    Physics Notes
    -------------
    **Rydberg Blockade Mechanism:**
    Consider driving |11⟩ → |1r⟩ → |rr⟩:
    
    Without interaction:
        Energy of |rr⟩ = 2 × E_r (just twice single-atom energy)
        
    With interaction:
        Energy of |rr⟩ = 2 × E_r + V
        
    If V >> Ω, the |rr⟩ state is shifted far off-resonance and cannot
    be populated. The system is "blocked" at |1r⟩ or |r1⟩.
    
    **Blockade radius:**
    The condition V(R) = C₆/R⁶ ≈ Ω defines the blockade radius:
        R_b = (C₆/Ω)^(1/6)
    
    For R < R_b, double excitation is suppressed ("blockade regime").
    For R > R_b, atoms excite independently ("independent regime").
    
    **Scaling with n:**
    C₆ ∝ n¹¹, so higher n gives stronger interactions:
    - n=50: C₆ ~ 10 GHz·μm⁶
    - n=70: C₆ ~ 1000 GHz·μm⁶
    - n=100: C₆ ~ 100,000 GHz·μm⁶
    
    But higher n also means longer lifetime (bad) and larger orbital (bad
    for dense arrays). The optimal n balances these tradeoffs.
    
    **Why van der Waals?**
    At typical spacings (3-10 μm), the dominant interaction is van der Waals:
        V = -C₆/R⁶   (always attractive for S-state Rydberg)
    
    At closer range (~1-2 μm), resonant dipole-dipole becomes important:
        V = C₃/R³    (can be + or - depending on angle)
    """
    if hs.dim == 3:
        # Single Rydberg-Rydberg interaction term
        P_r = hs.projectors['r']
        P_rr = op_two_atom(P_r, P_r)  # |rr⟩⟨rr|
        H = V * P_rr
        
    elif hs.dim == 4:
        # Multiple interaction terms for different mJ combinations
        P_rp = hs.projectors['r+']
        P_rm = hs.projectors['r-']
        
        # |r+r+⟩ interaction
        P_rprp = op_two_atom(P_rp, P_rp)
        H = V * P_rprp
        
        # |r+r-⟩ and |r-r+⟩ interactions (cross terms)
        V_cross = V_pm if V_pm is not None else V  # Usually similar
        P_rprm = op_two_atom(P_rp, P_rm)
        P_rmrp = op_two_atom(P_rm, P_rp)
        H = H + V_cross * (P_rprm + P_rmrp)
        
        # |r-r-⟩ interaction
        V_minus = V_mm if V_mm is not None else V
        P_rmrm = op_two_atom(P_rm, P_rm)
        H = H + V_minus * P_rmrm
    
    return H


def build_qubit_hamiltonian(omega_qubit: float, hs: HilbertSpace) -> Qobj:
    """
    Build qubit splitting Hamiltonian.
    
    H_qubit = ω_qubit (|1⟩⟨1|₁ ⊗ I₂ + I₁ ⊗ |1⟩⟨1|₂)
    
    Parameters
    ----------
    omega_qubit : float
        Qubit transition frequency (rad/s).
        For Rb87: ω_qubit = 2π × 6.834 GHz (hyperfine splitting)
    hs : HilbertSpace
        Hilbert space object
        
    Returns
    -------
    Qobj
        Qubit energy Hamiltonian
        
    Physics Notes
    -------------
    **Rotating frame convention:**
    In most simulations, we work in a rotating frame where the qubit
    splitting is transformed away (ω_qubit = 0 in rotating frame).
    
    This term is only needed when:
    1. Simulating in the lab frame
    2. Studying effects of finite qubit-laser detuning
    3. Including microwave driving of the |0⟩ ↔ |1⟩ transition
    """
    P1 = hs.projectors['1']
    I = hs.identity
    return omega_qubit * (op_two_atom(P1, I) + op_two_atom(I, P1))


def build_zeeman_hamiltonian(delta_zeeman: float, hs: HilbertSpace) -> Qobj:
    """
    Build differential Zeeman shift Hamiltonian on qubit states.
    
    Magnetic field causes energy shifts that differ between |0⟩ and |1⟩.
    
    H_zeeman = δ_B (|1⟩⟨1|₁ ⊗ I₂ + I₁ ⊗ |1⟩⟨1|₂)
    
    Parameters
    ----------
    delta_zeeman : float
        Differential Zeeman shift ω_1 - ω_0 (rad/s)
    hs : HilbertSpace
        Hilbert space object
        
    Returns
    -------
    Qobj
        Zeeman shift Hamiltonian
        
    Physics Notes
    -------------
    **Clock states:**
    For mF=0 → mF=0 transitions (clock states), the first-order Zeeman
    shift vanishes. The residual shift is quadratic:
    
        δω = β × B²
    
    where β ~ 575 Hz/G² for Rb87. At B = 1 G, this is only ~575 Hz.
    
    **Non-clock states:**
    For mF ≠ 0 states, the linear Zeeman shift is:
    
        δω = g_F × μ_B × B × mF / ℏ
    
    where g_F ≈ 0.5 for Rb87 F=2. At B = 1 G, this is ~700 kHz × mF.
    This is why clock states are strongly preferred for qubit encoding.
    """
    P1 = hs.projectors['1']
    I = hs.identity
    return delta_zeeman * (op_two_atom(P1, I) + op_two_atom(I, P1))


def build_stark_hamiltonian(
    delta_stark: float,
    hs: HilbertSpace,
    trap_laser_on: bool = True,
) -> Qobj:
    """
    Build differential AC Stark shift Hamiltonian from trap laser.
    
    The trap laser causes a differential light shift between F=1 and F=2
    states due to slightly different polarizabilities.
    
    H_stark = δ_AC (|1⟩⟨1|₁ ⊗ I₂ + I₁ ⊗ |1⟩⟨1|₂)
    
    Parameters
    ----------
    delta_stark : float
        Differential Stark shift ω_Stark = Δα × I / (2ε₀c ℏ) (rad/s).
        For Rb87 at 1 mK trap depth: ~2π × 70 kHz
    hs : HilbertSpace
        Hilbert space object
    trap_laser_on : bool
        If False, return zero Hamiltonian (trap turned off during gate)
        
    Returns
    -------
    Qobj
        AC Stark shift Hamiltonian
        
    Physics Notes
    -------------
    **Differential polarizability:**
    The |F=1⟩ and |F=2⟩ states have slightly different polarizabilities
    at the trap wavelength (typically 1064 nm):
    
        Δα = α(F=2) - α(F=1) ~ 0.1-1% of α_ground
    
    This causes |1⟩ to shift more than |0⟩ (or vice versa).
    
    **Trap-off operation:**
    Some experiments turn off the trap laser during the gate to eliminate
    this shift. The tradeoff is reduced confinement during the gate:
    - Atoms can expand thermally
    - Higher motional dephasing
    - Risk of atom loss
    
    For short gates (~100 ns), trap-off operation is often preferred.
    """
    if not trap_laser_on:
        # Return zero operator (same structure as active Hamiltonian)
        return 0 * op_two_atom(hs.identity, hs.identity)
    
    P1 = hs.projectors['1']
    I = hs.identity
    return delta_stark * (op_two_atom(P1, I) + op_two_atom(I, P1))


# =============================================================================
# FULL HAMILTONIAN CONSTRUCTION
# =============================================================================

def check_rwa_validity(
    Omega: float,
    omega_optical: float = 2*np.pi*384e12,
    threshold: float = 0.01,
    verbose: bool = True,
) -> bool:
    """
    Check if the Rotating Wave Approximation (RWA) is valid.
    
    The RWA neglects counter-rotating terms oscillating at 2×ω_optical.
    Valid when Ω << ω_optical (typically Ω/ω < 1%).
    
    Parameters
    ----------
    Omega : float
        Rabi frequency (rad/s)
    omega_optical : float
        Optical transition frequency (rad/s).
        Default: Rb87 D2 line at 384 THz
    threshold : float
        Maximum Ω/ω ratio for RWA validity (default 0.01 = 1%)
    verbose : bool
        If True, print warning when RWA is marginally valid
        
    Returns
    -------
    bool
        True if RWA is valid, False if counter-rotating terms may matter
        
    Physics Notes
    -------------
    **Bloch-Siegert shift:**
    The leading correction from counter-rotating terms is:
    
        Δω_BS = Ω²/(4ω_optical)
    
    For Ω = 2π × 10 MHz and ω = 2π × 384 THz:
        Δω_BS = (2π×10⁷)² / (4 × 2π×384×10¹²) = 0.065 Hz
    
    This is completely negligible compared to other effects (~kHz).
    
    **When RWA fails:**
    For ultrastrong coupling (Ω/ω > 0.1), the RWA breaks down:
    - Bloch-Siegert shift becomes significant
    - Counter-rotating terms cause additional dynamics
    - Need full Jaynes-Cummings or Rabi Hamiltonian
    
    This regime is not relevant for Rydberg gates (Ω/ω ~ 10⁻⁸).
    """
    ratio = abs(Omega) / omega_optical
    is_valid = ratio < threshold
    
    if verbose and not is_valid:
        print(f"⚠️  RWA WARNING: Ω/ω_optical = {ratio:.2e} > {threshold}")
        print(f"    Counter-rotating terms may contribute ~{ratio*100:.1f}% correction.")
        bloch_siegert = Omega**2 / (4 * omega_optical)
        print(f"    Bloch-Siegert shift: Δω ~ Ω²/(4ω) = {bloch_siegert/(2*np.pi)/1e3:.3f} kHz")
    
    return is_valid


def build_full_hamiltonian(
    Omega: complex,
    Delta: float,
    V: float,
    hs: HilbertSpace = None,
    dim: int = 3,
    polarization: str = "sigma+",
    zeeman_splitting: float = 0,
    omega_qubit: float = 0.0,
    delta_zeeman: float = 0.0,
    delta_stark: float = 0.0,
    trap_laser_on: bool = True,
    check_rwa: bool = False,
    **kwargs,
) -> Qobj:
    """
    Build complete two-atom Hamiltonian with all physics terms.
    
    This is the main function for Hamiltonian construction, combining:
    - Laser coupling (with phase)
    - Two-photon detuning
    - Rydberg-Rydberg interaction
    - Optional: qubit splitting, Zeeman, Stark shifts
    
    H = H_laser + H_detuning + H_interaction + [H_qubit] + [H_zeeman] + [H_stark]
    
    Parameters
    ----------
    Omega : complex
        Rabi frequency (rad/s). Complex value includes laser phase.
    Delta : float
        Two-photon detuning (rad/s)
    V : float
        Rydberg interaction strength (rad/s)
    hs : HilbertSpace, optional
        Pre-built Hilbert space. If None, built from dim.
    dim : int
        Hilbert space dimension (3 or 4). Ignored if hs provided.
    polarization : str
        Laser polarization for 4-level model
    zeeman_splitting : float
        Zeeman shift between mJ Rydberg states (rad/s) for 4-level model
    omega_qubit : float
        Qubit frequency splitting (rad/s). Default 0 = rotating frame.
    delta_zeeman : float
        Differential Zeeman shift on qubit (rad/s). Default 0.
    delta_stark : float
        Differential AC Stark shift from trap (rad/s). Default 0.
    trap_laser_on : bool
        Whether trap laser is on during gate. Affects Stark term.
    check_rwa : bool
        If True, verify RWA validity and warn if marginal.
    **kwargs
        Additional parameters for 4-level model:
        - V_pm, V_mm: Cross-mJ interaction strengths
        - Omega_minus: σ- coupling strength
        - Delta_minus: Detuning for |r-⟩
        
    Returns
    -------
    Qobj
        Full two-atom Hamiltonian (Hermitian)
        
    Examples
    --------
    >>> # Basic 3-level Hamiltonian
    >>> Omega = 2*np.pi * 5e6  # 5 MHz
    >>> Delta = 0.377 * Omega   # Optimal detuning
    >>> V = 2*np.pi * 100e6     # 100 MHz blockade
    >>> H = build_full_hamiltonian(Omega, Delta, V)
    
    >>> # With phase for second pulse in CZ gate
    >>> xi = 3.9  # Phase shift
    >>> H2 = build_full_hamiltonian(Omega * np.exp(1j*xi), Delta, V)
    
    >>> # 4-level model with polarization impurity
    >>> H4 = build_full_hamiltonian(Omega, Delta, V, dim=4, 
    ...                              polarization="mixed")
    """
    if hs is None:
        hs = HS3 if dim == 3 else HS4
    
    # Check RWA validity if requested
    if check_rwa and abs(Omega) > 0:
        check_rwa_validity(abs(Omega), verbose=True)
    
    # Core Hamiltonian terms (always included)
    H = (
        build_laser_hamiltonian(
            Omega, hs, 
            polarization=polarization,
            Omega_minus=kwargs.get('Omega_minus')
        ) +
        build_detuning_hamiltonian(
            Delta, hs, 
            zeeman_splitting=zeeman_splitting,
            Delta_minus=kwargs.get('Delta_minus')
        ) +
        build_interaction_hamiltonian(
            V, hs, 
            V_pm=kwargs.get('V_pm'),
            V_mm=kwargs.get('V_mm')
        )
    )
    
    # Optional physics terms (add if nonzero)
    if omega_qubit != 0:
        H = H + build_qubit_hamiltonian(omega_qubit, hs)
    
    if delta_zeeman != 0:
        H = H + build_zeeman_hamiltonian(delta_zeeman, hs)
    
    if delta_stark != 0:
        H = H + build_stark_hamiltonian(delta_stark, hs, trap_laser_on)
    
    return H


# =============================================================================
# PHASE-MODULATED HAMILTONIAN FOR BANG-BANG CONTROL
# =============================================================================

def build_phase_modulated_hamiltonian(
    Omega: float,
    phase: float,
    V: float,
    hs: HilbertSpace,
    Delta: float = 0.0,
    delta_zeeman: float = 0.0,
    delta_stark: float = 0.0,
    trap_laser_on: bool = True,
) -> Qobj:
    """
    Build Hamiltonian with phase-modulated laser coupling and two-photon detuning.
    
    This is used for smooth sinusoidal CZ gates (Bluvstein/Evered protocol)
    where the laser phase varies continuously as φ(t) = A·cos(ωt - ϕ) + δ₀·t.
    
    H = H_laser(Ω·e^{iφ}) + H_detuning(Δ) + H_interaction(V) + [H_zeeman] + [H_stark]
    
    **CRITICAL FOR DARK STATE PHYSICS:**
    The two-photon detuning Δ (Delta parameter) MUST have OPPOSITE SIGN from
    the intermediate-state detuning Δₑ to maximize dark state population and
    suppress scattering. This is the key insight from Bluvstein's thesis.
    
    The phase slope δ₀ in φ(t) = A·cos(ωt) + δ₀·t corresponds to a constant
    two-photon detuning: Δ = dφ/dt|_{DC} = δ₀
    
    Parameters
    ----------
    Omega : float
        Rabi frequency magnitude (rad/s)
    phase : float
        Laser phase φ (radians). For smooth JP: φ(t) = A·cos(ωt - ϕ) + δ₀·t
    V : float
        Rydberg interaction strength (rad/s)
    hs : HilbertSpace
        Hilbert space object
    Delta : float
        Two-photon detuning (rad/s). CRITICAL for dark state physics!
        Sign convention: Δ > 0 means laser is blue-detuned from two-photon resonance.
        For dark state: sign(Δ) should be OPPOSITE to sign(Δₑ).
        Default 0 (on resonance) but recommend nonzero for dark state CZ.
    delta_zeeman : float
        Differential Zeeman shift (rad/s). Default 0.
    delta_stark : float
        Differential AC Stark shift (rad/s). Default 0.
    trap_laser_on : bool
        Whether trap laser is on during this segment.
        
    Returns
    -------
    Qobj
        Phase-modulated Hamiltonian with detuning
        
    Physics Notes
    -------------
    **Bluvstein/Evered Smooth CZ Protocol:**
    
    The smooth sinusoidal protocol uses φ(t) = A·cos(ωt - ϕ) + δ₀·t where:
    - A ≈ π/2: phase amplitude
    - ω ≈ Ω: modulation frequency (resonant condition)
    - δ₀: two-photon detuning (appears as linear phase ramp)
    
    **Dark State Physics:**
    
    For two atoms in the |11⟩ state driven to Rydberg:
    - Bright state |B⟩ = (|r1⟩ + |1r⟩)/√2 couples strongly to |11⟩
    - Dark state |D⟩ = (|r1⟩ - |1r⟩)/√2 is decoupled from |11⟩
    
    Choosing opposite signs for intermediate-state detuning (Δₑ > 0) and
    two-photon detuning (Δ < 0) maximizes population in |D⟩, which:
    - Suppresses intermediate-state scattering
    - Reduces decoherence during the gate
    - Improves fidelity from ~97.5% to ~99.5%
    
    Reference: Evered et al., Nature 622, 268-272 (2023)
               Bluvstein PhD Thesis (Harvard, 2024)
    """
    # Core terms: laser coupling with complex phase + blockade + detuning
    Omega_complex = Omega * np.exp(1j * phase)
    H = (
        build_laser_hamiltonian(Omega_complex, hs) +
        build_interaction_hamiltonian(V, hs)
    )
    
    # Two-photon detuning - CRITICAL for dark state physics
    if Delta != 0:
        H = H + build_detuning_hamiltonian(Delta, hs)
    
    # Optional physics terms (consistent with LP protocol)
    if delta_zeeman != 0:
        H = H + build_zeeman_hamiltonian(delta_zeeman, hs)
    
    if delta_stark != 0:
        H = H + build_stark_hamiltonian(delta_stark, hs, trap_laser_on)
    
    return H


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_mJ_zeeman_splitting(B_field: float, g_J: float = 2.002) -> float:
    """
    Compute Zeeman splitting between mJ=+1/2 and mJ=-1/2 Rydberg states.
    
    ΔE = g_J × μ_B × B    (for ΔmJ = 1)
    
    Parameters
    ----------
    B_field : float
        Magnetic field strength (Tesla)
    g_J : float
        Landé g-factor for the Rydberg state.
        For S-states (L=0): g_J ≈ 2.002 (almost pure spin)
        
    Returns
    -------
    float
        Zeeman splitting (rad/s)
        
    Examples
    --------
    >>> # 1 Gauss = 10^-4 Tesla
    >>> splitting = compute_mJ_zeeman_splitting(1e-4)
    >>> print(f"{splitting/(2*np.pi)/1e6:.2f} MHz")  # ~2.8 MHz
    """
    return g_J * MU_B * B_field / HBAR


def compute_mJ_coupling_ratio(polarization: str) -> Tuple[float, float]:
    """
    Get relative coupling strengths to mJ=+1/2 and mJ=-1/2 states.
    
    Parameters
    ----------
    polarization : str
        Laser polarization:
        - "sigma+": Pure σ+ (couples only to mJ=+1/2)
        - "sigma-": Pure σ- (couples only to mJ=-1/2)
        - "pi": π polarization (equal coupling)
        - "mixed": Typical experimental impurity (~2% σ-)
        
    Returns
    -------
    Tuple[float, float]
        (ratio_plus, ratio_minus) normalized so sum of squares = 1
        
    Examples
    --------
    >>> r_plus, r_minus = compute_mJ_coupling_ratio("sigma+")
    >>> print(r_plus, r_minus)  # 1.0, 0.0
    
    >>> r_plus, r_minus = compute_mJ_coupling_ratio("mixed")
    >>> print(f"{r_minus**2:.1%}")  # ~2% coupling to wrong state
    """
    if polarization == "sigma+":
        return (1.0, 0.0)
    elif polarization == "sigma-":
        return (0.0, 1.0)
    elif polarization == "pi":
        return (1/np.sqrt(2), 1/np.sqrt(2))
    else:  # "mixed" or unspecified
        impurity = 0.02  # Typical 2% polarization impurity
        return (np.sqrt(1 - impurity), np.sqrt(impurity))


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Hilbert space
    'HilbertSpace',
    'build_hilbert_space',
    'HS3', 'HS4',
    'b0', 'b1', 'br',
    'op_two_atom',
    
    # Hamiltonian building blocks
    'build_laser_hamiltonian',
    'build_detuning_hamiltonian',
    'build_interaction_hamiltonian',
    'build_qubit_hamiltonian',
    'build_zeeman_hamiltonian',
    'build_stark_hamiltonian',
    
    # Full Hamiltonians
    'build_full_hamiltonian',
    'build_phase_modulated_hamiltonian',
    'check_rwa_validity',
    
    # Helpers
    'compute_mJ_zeeman_splitting',
    'compute_mJ_coupling_ratio',
]
