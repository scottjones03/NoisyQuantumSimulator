# Neutral Atom Micro-Physics Models
#
# This subpackage contains all micro-physics models specific to
# neutral atom arrays with optical tweezer trapping.
#
# Modules:
#   - rydberg_gates: Rydberg-mediated CZ and entangling gates (QuTiP/Lindblad)
#   - single_qubit_gates: Raman/microwave single-qubit operations
#   - optical_tweezers: Harmonic trap model, trap depth, heating
#   - aod_slm_motion: Atom reconfiguration via AOD/SLM waveforms
#   - measurement: Fluorescence detection, atom loss, readout fidelity
#   - cooling: Optical molasses, sideband cooling models
#
# Hardware parameters captured:
#   - Rydberg principal quantum number (n)
#   - Rabi frequencies (Ω_ge, Ω_er)
#   - Detunings (Δ_e, δ)
#   - Decay rates (γ_e, γ_r, T2*)
#   - Trap depth, waist, laser power
#   - Inter-atom distance
#   - C6 coefficient
