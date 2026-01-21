# Leakage Error Models
#
# Errors involving states outside computational basis.
#
# Leakage types:
#   - Leakage: |0⟩,|1⟩ → |2⟩,|3⟩,...
#   - Seepage: |2⟩,... → |0⟩,|1⟩
#   - Heating: Motional state excitation
#
# Neutral atom specific:
#   - Rydberg state leakage |r⟩
#   - Intermediate state population |e⟩
#   - Wrong hyperfine state
#
# Ion specific:
#   - Metastable state shelving
#   - Motional mode excitation
#   - Off-resonant coupling to other levels
#
# Modeling approaches:
#   - Extended Hilbert space (explicit leakage levels)
#   - Effective leakage rate (trace out leaked population)
#   - Leakage reduction units (LRUs)
#
# Impact on QEC:
#   - Leakage accumulation
#   - Correlated errors from leaked qubits
#   - Leakage detection and reset strategies
