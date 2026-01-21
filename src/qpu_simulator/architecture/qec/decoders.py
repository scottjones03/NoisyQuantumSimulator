# Decoder Implementations
#
# Syndrome decoding algorithms for error correction.
#
# Algorithms:
#   - MWPM (Minimum Weight Perfect Matching)
#     - PyMatching integration
#     - Handles measurement errors
#   
#   - Union-Find
#     - Near-linear time complexity
#     - Good for real-time decoding
#   
#   - Belief Propagation
#     - For LDPC codes
#     - Iterative message passing
#   
#   - Neural Network
#     - Learned decoders
#     - Can exploit hardware-specific noise
#
# Interface:
#   decode(syndrome_history, code) -> correction
#
# Performance tracking:
#   - Decoding time
#   - Logical error rate
#   - Failure modes
