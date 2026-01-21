#!/usr/bin/env python3
"""Fix the Hamiltonian construction in simulate_CZ_gate."""

import json

notebook_path = "/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/NoisyQuantumSimulator/examples/neutral_atoms_rydberg_cz_gate.ipynb"

with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find cell 54 (0-indexed) which contains simulate_CZ_gate
target_cell = notebook['cells'][54]
lines = target_cell['source']

# Find and fix the Hamiltonian construction for H_coupling_1
fixed_1 = False
for i, line in enumerate(lines):
    if "# Coupling Hamiltonian for FIRST pulse (with Omega)" in line:
        print(f"Found H_coupling_1 block at line {i}")
        # Check what follows
        print("Current block:")
        for j in range(i, min(i+6, len(lines))):
            print(f"  {j}: {repr(lines[j])}")
        
        # Replace the block
        new_block = [
            "    # Coupling Hamiltonian for FIRST pulse (with Omega)\n",
            "    # Hermitian form: Omega|r><1| + Omega*|1><r| \n",
            "    # sigma_r1 = |r><1| (raising), sigma_1r = |1><r| (lowering)\n",
            "    H_coupling_1 = (\n",
            "        Omega / 2 * (tensor(sigma_r1, I_single) + tensor(I_single, sigma_r1)) +\n",
            "        np.conj(Omega) / 2 * (tensor(sigma_1r, I_single) + tensor(I_single, sigma_1r))\n",
            "    )\n",
            "\n",
        ]
        
        # Find the end of the old block (the closing parenthesis)
        end_idx = i + 1
        while end_idx < len(lines) and "    )" not in lines[end_idx]:
            end_idx += 1
        if end_idx < len(lines) and "    )" in lines[end_idx]:
            end_idx += 1
        
        print(f"Replacing lines {i} to {end_idx-1}")
        lines[i:end_idx] = new_block
        fixed_1 = True
        break

if not fixed_1:
    print("ERROR: Could not find H_coupling_1 block!")

# Re-scan for H_coupling_2
lines = target_cell['source']  # Refresh
fixed_2 = False
for i, line in enumerate(lines):
    if "# Coupling Hamiltonian for SECOND pulse" in line:
        print(f"\nFound H_coupling_2 block at line {i}")
        print("Current block:")
        for j in range(i, min(i+6, len(lines))):
            print(f"  {j}: {repr(lines[j])}")
        
        new_block = [
            "    # Coupling Hamiltonian for SECOND pulse (with Omega * e^{i*xi})\n",
            "    H_coupling_2 = (\n",
            "        Omega_rotated / 2 * (tensor(sigma_r1, I_single) + tensor(I_single, sigma_r1)) +\n",
            "        np.conj(Omega_rotated) / 2 * (tensor(sigma_1r, I_single) + tensor(I_single, sigma_1r))\n",
            "    )\n",
            "\n",
        ]
        
        # Find the end of the old block
        end_idx = i + 1
        while end_idx < len(lines) and "    )" not in lines[end_idx]:
            end_idx += 1
        if end_idx < len(lines) and "    )" in lines[end_idx]:
            end_idx += 1
        
        print(f"Replacing lines {i} to {end_idx-1}")
        lines[i:end_idx] = new_block
        fixed_2 = True
        break

if not fixed_2:
    print("ERROR: Could not find H_coupling_2 block!")

if fixed_1 and fixed_2:
    target_cell['source'] = lines
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print("\nSUCCESS! Fixed Hamiltonian construction.")
else:
    print("\nFailed to fix one or both Hamiltonians.")
