# Cooling Primitive
#
# Abstraction for resetting motional/thermal state.
#
# Used to:
#   - Reset after heating from gates/movement
#   - Prepare ground state for high-fidelity gates
#   - Sympathetic cooling in multi-species systems
#
# API:
#   Cool(qubit_id, target_temperature=None, method='doppler')
#
# Methods:
#   - 'doppler': Doppler cooling (fast, limited temperature)
#   - 'sideband': Resolved sideband cooling (slow, ground state)
#   - 'molasses': Optical molasses (atoms)
#   - 'sympathetic': Cooling via auxiliary species (ions)
#
# Returns CoolResult:
#   - duration: Cooling time
#   - final_temperature: Achieved motional state
#   - success_probability: Chance of successful cooling
#   - loss_probability: Atom/ion loss during cooling
#
# Trade-offs:
#   - Speed vs final temperature
#   - Photon scattering vs fidelity
