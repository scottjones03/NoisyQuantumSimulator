# Measurement Primitive
#
# Abstraction for quantum state readout.
#
# Supports:
#   - Single-shot measurement
#   - Mid-circuit measurement
#   - Basis selection (Z, X, Y)
#   - Destructive vs non-destructive readout
#
# API:
#   Measure(qubit_id, basis='Z', destructive=True)
#
# Returns MeasureResult:
#   - outcome: Measurement result (0 or 1, or None if simulated)
#   - fidelity: Readout fidelity
#   - confusion_matrix: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
#   - duration: Measurement time
#   - loss_probability: Chance of losing atom/ion
#   - post_state: State after measurement (if non-destructive)
#
# Platform specifics:
#   - Neutral atoms: Fluorescence imaging, pushout
#   - Ions: Fluorescence, shelving techniques
#   - Cavity: Dispersive readout, transmission
