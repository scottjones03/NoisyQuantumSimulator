# Hardware Configuration Definitions
#
# Parameter sets for different quantum hardware platforms.
# These configurations feed into micro-physics simulations
# and primitive instantiation.
#
# Structure:
#   Each hardware type has a configuration class containing:
#   - Physical constants (atomic properties, trap parameters)
#   - Operational parameters (Rabi frequencies, gate times)
#   - Noise parameters (decay rates, heating rates)
#   - Geometric constraints (distances, zone sizes)
#
# Usage:
#   config = NeutralAtomConfig.from_experiment("lukin_2022")
#   primitives = create_primitives(config)
