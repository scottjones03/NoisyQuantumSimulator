# Micro-Physics Base Classes
#
# Abstract base classes and interfaces for all micro-physics models.
# These define the contract that each hardware-specific model must fulfill.
#
# Key abstractions:
#   - MicroModel: Base class for all physics simulations
#   - GateSimulator: Interface for gate-level Hamiltonian evolution
#   - MotionSimulator: Interface for classical atom/ion motion
#   - MeasurementModel: Interface for readout and detection
#
# All micro-models must output:
#   - CPTP maps (error channels)
#   - Timing information
#   - Loss/heating probabilities
#   - Fidelity estimates
