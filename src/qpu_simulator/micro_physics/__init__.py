# Micro-Physics Layer (Level 0)
#
# Hardware-specific physics simulations for 1-3 qubits per site.
# This layer produces calibrated error models and timing information
# that feed into the primitives layer.
#
# Submodules:
#   - base: Abstract base classes for micro-models
#   - neutral_atoms: Rydberg gates, optical tweezers, AOD/SLM motion
#   - trapped_ions: QCCD, Penning, RF Paul trap models
#   - cavity_qed: Cavity-mediated gates and photon interactions
