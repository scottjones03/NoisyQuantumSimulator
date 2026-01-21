#!/usr/bin/env python3
"""Fix the simulate_CZ_gate function to use two-pulse protocol with phase shift."""

import json
import sys

notebook_path = 'examples/neutral_atoms_rydberg_cz_gate.ipynb'

print(f"Reading {notebook_path}...")
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find the cell with simulate_CZ_gate
cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def simulate_CZ_gate' in source:
            cell_idx = i
            break

if cell_idx is None:
    print("ERROR: Could not find simulate_CZ_gate cell!")
    sys.exit(1)

print(f"Found simulate_CZ_gate in cell {cell_idx}")

cell = nb['cells'][cell_idx]
source = cell['source']

# Find the STEP 7 Hamiltonian construction block
start_idx = None
end_idx = None

for i, line in enumerate(source):
    if 'H_coupling = (' in line and start_idx is None:
        start_idx = i
    if 'H = H_coupling - H_detuning + H_blockade' in line and start_idx is not None:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print("ERROR: Could not find H_coupling block!")
    print(f"  start_idx = {start_idx}, end_idx = {end_idx}")
    sys.exit(1)

print(f"Found H_coupling block: lines {start_idx} to {end_idx}")

# New code for the two-pulse protocol
new_code = '''    # =======================================================================
    # TWO-PULSE CZ PROTOCOL (Levine et al. 2019)
    # =======================================================================
    # The CZ gate requires a PHASE SHIFT between the two pulses!
    # First pulse:  H1 uses Omega
    # Second pulse: H2 uses Omega * e^{i*xi} where xi is computed from Delta, Omega, tau
    #
    # This phase shift ensures |01> and |10> return to themselves (not stuck
    # in |0r> or |r0>) while |11> picks up the correct pi phase.
    # =======================================================================

    # Compute the critical phase shift xi
    xi = compute_phase_shift_xi(Delta_gate, Omega, tau_single)
    Omega_rotated = Omega * xi  # Phase-shifted Rabi frequency for pulse 2

    # Coupling Hamiltonian for FIRST pulse (with Omega)
    H_coupling_1 = (
        Omega / 2 * (tensor(sigma_1r, I_single) + tensor(I_single, sigma_1r)) +
        Omega / 2 * (tensor(sigma_r1, I_single) + tensor(I_single, sigma_r1))
    )

    # Coupling Hamiltonian for SECOND pulse (with Omega * e^{i*xi})
    H_coupling_2 = (
        Omega_rotated / 2 * (tensor(sigma_1r, I_single) + tensor(I_single, sigma_1r)) +
        np.conj(Omega_rotated) / 2 * (tensor(sigma_r1, I_single) + tensor(I_single, sigma_r1))
    )

    # Detuning Hamiltonian (same for both pulses)
    H_detuning = Delta_gate * (
        tensor(br * br.dag(), I_single) + tensor(I_single, br * br.dag())
    )

    # Blockade Hamiltonian (same for both pulses)
    H_blockade = V * Pr

    # Total Hamiltonians for each pulse
    H1 = H_coupling_1 - H_detuning + H_blockade  # First pulse
    H2 = H_coupling_2 - H_detuning + H_blockade  # Second pulse (phase-shifted)
'''

new_lines = [line + '\n' for line in new_code.split('\n')]

# Replace the old Hamiltonian construction
source = source[:start_idx] + new_lines + source[end_idx+1:]
print(f"Replaced Hamiltonian construction ({end_idx - start_idx + 1} old lines -> {len(new_lines)} new lines)")

# Now update STEP 10 to use H1 and H2
new_source = []
for line in source:
    if 'result1 = mesolve(H, psi0' in line:
        new_source.append(line.replace('mesolve(H, psi0', 'mesolve(H1, psi0'))
        print("  Updated: mesolve(H, psi0 -> mesolve(H1, psi0")
    elif 'result2 = mesolve(H, psi_intermediate' in line:
        new_source.append(line.replace('mesolve(H, psi_intermediate', 'mesolve(H2, psi_intermediate'))
        print("  Updated: mesolve(H, psi_intermediate -> mesolve(H2, psi_intermediate")
    elif "'H': H," in line:
        new_source.append("        'H1': H1,  # First pulse Hamiltonian\n")
        new_source.append("        'H2': H2,  # Second pulse Hamiltonian (phase-shifted)\n")
        print("  Updated: 'H': H -> 'H1': H1, 'H2': H2")
    else:
        new_source.append(line)

# Add xi_rad and xi_deg to return dict
final_source = []
for line in new_source:
    if "'Delta_over_Omega': delta_over_omega," in line:
        final_source.append(line)
        final_source.append("        \n")
        final_source.append("        # --- Phase shift (critical for two-pulse protocol!) ---\n")
        final_source.append("        'xi_rad': np.angle(xi),\n")
        final_source.append("        'xi_deg': np.degrees(np.angle(xi)),\n")
        print("  Added: xi_rad, xi_deg to return dict")
    else:
        final_source.append(line)

nb['cells'][cell_idx]['source'] = final_source

# Save the modified notebook
print(f"\nWriting modified notebook to {notebook_path}...")
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("\n" + "="*60)
print("SUCCESS! The following changes were made:")
print("="*60)
print("1. Added compute_phase_shift_xi() call to compute xi")
print("2. Created H1 with Omega (first pulse)")
print("3. Created H2 with Omega*xi (second pulse, phase-shifted)")
print("4. Updated mesolve(H, ...) to mesolve(H1, ...) for first pulse")
print("5. Updated mesolve(H, ...) to mesolve(H2, ...) for second pulse")
print("6. Added xi_rad, xi_deg to return dictionary")
print("\nThis implements the correct two-pulse CZ protocol from Levine et al. 2019")
print("Expected result: Ideal fidelity should now be >99.99% instead of ~57%")
