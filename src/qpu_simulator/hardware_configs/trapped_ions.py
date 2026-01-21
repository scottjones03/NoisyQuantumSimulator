# Trapped Ion Hardware Configuration
#
# Parameters for various trapped ion architectures.
#
# Ion Species:
#   - Ca-40 (optical qubit)
#   - Ca-43 (hyperfine qubit)
#   - Ba-137 (visible transitions)
#   - Yb-171 (hyperfine qubit)
#   - Sr-88 (optical qubit)
#
# Architecture variants:
#   - Linear chain (single zone)
#   - QCCD (segmented trap with shuttling)
#   - 2D array (Penning or surface trap)
#
# Key Parameters:
#
# Qubit properties:
#   - Qubit transition frequency
#   - T1, T2, T2*
#   - Leakage to other states
#
# Motional modes:
#   - Axial and radial frequencies
#   - Mode spacing
#   - Heating rates (quanta/ms)
#
# Gate parameters:
#   - Single-qubit gate time and fidelity
#   - MS gate time vs number of ions
#   - Crosstalk coefficients
#
# Shuttling (QCCD):
#   - Shuttling speed
#   - Heating per shuttling operation
#   - Junction traversal time
#   - Split/merge times
#
# Measurement:
#   - Detection fidelity
#   - Measurement time
#   - Crosstalk in multi-ion detection
#
# Example configurations:
#   - IonQ (Yb-171, linear chains)
#   - Honeywell/Quantinuum (Ba-137, QCCD)
#   - Duke/Maryland (Yb-171, linear)
#   - NIST (Be-9, multi-zone)
