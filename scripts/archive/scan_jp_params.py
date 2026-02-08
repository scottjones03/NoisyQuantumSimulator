#!/usr/bin/env python3
"""
Scan different JP gate times to find what gives controlled_phase = π
"""
import numpy as np
import qutip as qt

print("=" * 70)
print("JP PARAMETER SCAN - FIND CORRECT Ωτ")
print("=" * 70)

# Physical parameters
Omega = 2*np.pi * 5e6  # 5 MHz Rabi frequency
V_over_Omega = 200      # Strong blockade
V = V_over_Omega * Omega

# Standard JP phases
phases = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]

# 3-level basis
ket_0 = qt.basis(3, 0)
ket_1 = qt.basis(3, 1)
ket_2 = qt.basis(3, 2)
sigma_1r = qt.basis(3, 2) * qt.basis(3, 1).dag()
I = qt.qeye(3)

def build_H(phi):
    Omega_eff = Omega * np.exp(1j * phi)
    H_atom = 0.5 * (Omega_eff * sigma_1r + np.conj(Omega_eff) * sigma_1r.dag())
    H_laser = qt.tensor(H_atom, I) + qt.tensor(I, H_atom)
    proj_rr = qt.tensor(ket_2 * ket_2.dag(), ket_2 * ket_2.dag())
    H_blockade = V * proj_rr
    return H_laser + H_blockade

psi_00 = qt.tensor(ket_0, ket_0)
psi_01 = qt.tensor(ket_0, ket_1)
psi_10 = qt.tensor(ket_1, ket_0)
psi_11 = qt.tensor(ket_1, ket_1)

def get_overlap(psi_target, psi_final):
    overlap = psi_target.dag() * psi_final
    val = overlap.full()[0,0] if hasattr(overlap, 'full') else complex(overlap)
    return val

def simulate_jp(omega_tau, switching_times_normalized):
    """
    Run JP with given Ωτ.
    switching_times_normalized are fractions of omega_tau.
    """
    tau_total = omega_tau / Omega
    switch_times_phys = [t / Omega for t in switching_times_normalized]
    boundaries = [0] + switch_times_phys + [tau_total]
    
    final_states = {}
    for label, psi0 in [("00", psi_00), ("01", psi_01), ("10", psi_10), ("11", psi_11)]:
        psi = psi0
        for i_seg in range(len(phases)):
            t_start = boundaries[i_seg]
            t_end = boundaries[i_seg + 1]
            duration = t_end - t_start
            if duration > 0:
                H = build_H(phases[i_seg])
                tlist = np.linspace(0, duration, 100)
                result = qt.sesolve(H, psi, tlist)
                psi = result.states[-1]
        final_states[label] = psi
    
    c00 = get_overlap(psi_00, final_states["00"])
    c01 = get_overlap(psi_01, final_states["01"])
    c10 = get_overlap(psi_10, final_states["10"])
    c11 = get_overlap(psi_11, final_states["11"])
    
    phi_00 = np.angle(c00)
    phi_01 = np.angle(c01)
    phi_10 = np.angle(c10)
    phi_11 = np.angle(c11)
    
    controlled_phase = phi_11 - phi_01 - phi_10 + phi_00
    controlled_phase_wrapped = (controlled_phase + np.pi) % (2*np.pi) - np.pi
    
    # Fidelity to computational subspace
    pop_00 = abs(c00)**2
    pop_01 = abs(c01)**2
    pop_10 = abs(c10)**2
    pop_11 = abs(c11)**2
    avg_pop = 0.25 * (pop_00 + pop_01 + pop_10 + pop_11)
    
    return controlled_phase_wrapped, avg_pop

# Standard switching times for V/Ω = 200
switching_times_200 = [0.3328, 0.5859, 3.4340, 3.5530, 4.1204, 6.7431]

print(f"\nV/Ω = {V_over_Omega}")
print(f"Standard switching_times: {switching_times_200}")
print(f"\nScanning Ωτ from 5.0 to 10.0:")
print(f"{'Ωτ':>6} {'Ctrl Phase':>12} {'Avg Pop':>10}")
print("-" * 32)

for omega_tau in np.linspace(5.0, 10.0, 21):
    # Scale switching times proportionally
    scale = omega_tau / 7.0
    switching_times_scaled = [t * scale for t in switching_times_200]
    ctrl_phase, avg_pop = simulate_jp(omega_tau, switching_times_scaled)
    # Check if close to ±π
    phase_error = min(abs(ctrl_phase - np.pi), abs(ctrl_phase + np.pi))
    marker = " <-- GOOD" if phase_error < 0.1 else ""
    print(f"{omega_tau:6.2f} {np.degrees(ctrl_phase):12.2f}° {avg_pop*100:10.2f}%{marker}")

print("\n" + "=" * 70)
print("ALTERNATIVE: Check if we need different phases")
print("=" * 70)

# Try with the paper's exact parameters
omega_tau = 7.0
print(f"\nWith exact literature Ωτ = {omega_tau}:")
ctrl_phase, avg_pop = simulate_jp(omega_tau, switching_times_200)
print(f"  Controlled phase: {np.degrees(ctrl_phase):.2f}°")
print(f"  Avg population: {avg_pop*100:.2f}%")

# The issue might be the phase definition. Let me try flipping signs
print("\nTrying different phase sequences:")

phase_variants = [
    ("Original", [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]),
    ("Negated", [-np.pi/2, 0, np.pi/2, np.pi/2, 0, -np.pi/2, 0]),
    ("Shifted", [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0, -np.pi/2]),
]

for name, test_phases in phase_variants:
    phases = test_phases
    ctrl_phase, avg_pop = simulate_jp(omega_tau, switching_times_200)
    print(f"  {name}: ctrl_phase = {np.degrees(ctrl_phase):.2f}°, pop = {avg_pop*100:.2f}%")
