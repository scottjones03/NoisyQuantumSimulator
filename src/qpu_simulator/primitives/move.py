# Move Primitive
#
# Abstraction for qubit/atom/ion transport operations.
#
# Unified interface for:
#   - Neutral atoms: AOD/SLM tweezer movement
#   - Trapped ions: Shuttling between trap zones
#   - Cavity QED: Atom transport through cavity
#
# API:
#   Move(qubit_id, start, end, duration=None, constraints=None)
#
# Inputs:
#   - qubit_id: Identifier for the qubit to move
#   - start: Starting position/zone
#   - end: Target position/zone
#   - duration: Optional timing constraint
#   - constraints: Platform-specific constraints (max velocity, etc.)
#
# Returns MoveResult:
#   - duration: Actual move time
#   - heating: Motional energy added (ΔT or Δn)
#   - loss_probability: Chance of atom/ion loss
#   - trajectory: Path taken (optional)
#   - error_map: Decoherence during transit (if any)
#
# Platform-specific implementations handle:
#   - Neutral atoms: Waveform generation, acceleration limits
#   - Ions: Electrode waveforms, junction traversal
