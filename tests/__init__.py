# Tests for QPU Simulator
#
# Test organization mirrors source structure:
#   - test_micro_physics/: Unit tests for physics simulations
#   - test_primitives/: Tests for primitive operations
#   - test_architecture/: Tests for scheduling, QEC, compilation
#   - test_integration/: End-to-end integration tests
#
# Running tests:
#   pytest tests/
#   pytest tests/test_micro_physics/ -v
#   pytest tests/ -k "neutral_atom"
