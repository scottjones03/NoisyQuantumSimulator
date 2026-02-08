#!/usr/bin/env python3
"""
Check different interpretations of JP protocol phases.

The paper might define the phase differently - let's try:
1. Phase on Ω: Ω → Ω e^{iφ}
2. Phase on detuning: Δ_inst = dφ/dt
3. Phase as detuning: Δ = Ω tan(φ)
"""
import numpy as np
import qutip as qt

print("=" * 70)
print("TESTING DIFFERENT JP PHASE INTERPRETATIONS")
print("=" * 70)

# Physical parameters
Omega = 2*np.pi * 5e6
V_over_Omega = 200
V = V_over_Omega * Omega
omega_tau = 7.0
tau_total = omega_tau / Omega

# Standard params
switching_times_dimless = [0.3328, 0.5859, 3.4340, 3.5530, 4.1204, 6.7431]
phases_std = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]

# 3-level basis
ket_0 = qt.basis(3, 0)
ket_1 = qt.basis(3, 1)
ket_2 = qt.basis(3, 2)
sigma_1r = qt.basis(3, 2) * qt.basis(3, 1).dag()  # |r⟩⟨1| (excitation)
sigma_r1 = sigma_1r.dag()  # |1⟩⟨r| (de-excitation)
proj_r = ket_2 * ket_2.dag()  # |r⟩⟨r|
I = qt.qeye(3)

psi_00 = qt.tensor(ket_0, ket_0)
psi_01 = qt.tensor(ket_0, ket_1)
psi_10 = qt.tensor(ket_1, ket_0)
psi_11 = qt.tensor(ket_1, ket_1)

def build_H_interp1(phi):
    """Interpretation 1: Phase on complex Ω. 
    H = (Ω e^{iφ}/2)(|r⟩⟨1| + h.c.) + V|rr⟩⟨rr|
    """
    Omega_eff = Omega * np.exp(1j * phi)
    H_atom = 0.5 * (Omega_eff * sigma_1r + np.conj(Omega_eff) * sigma_r1)
    H_laser = qt.tensor(H_atom, I) + qt.tensor(I, H_atom)
    proj_rr = qt.tensor(proj_r, proj_r)
    H_blockade = V * proj_rr
    return H_laser + H_blockade

def build_H_interp2(phi):
    """Interpretation 2: Phase as instantaneous detuning.
    H = (Ω/2)(|r⟩⟨1| + h.c.) - Δ|r⟩⟨r| + V|rr⟩⟨rr|
    where Δ = Ω·tan(φ) (so φ=±π/2 → Δ=±∞, clipped)
    """
    # Δ = Ω tan(φ) but we need to be careful at ±π/2
    if abs(phi - np.pi/2) < 0.01 or abs(phi + np.pi/2) < 0.01:
        Delta = np.sign(phi) * 100 * Omega  # Large but finite
    else:
        Delta = Omega * np.tan(phi)
    
    H_atom_laser = 0.5 * Omega * (sigma_1r + sigma_r1)
    H_atom_det = -Delta * proj_r
    H_atom = H_atom_laser + H_atom_det
    
    H_laser = qt.tensor(H_atom_laser, I) + qt.tensor(I, H_atom_laser)
    H_det = qt.tensor(H_atom_det, I) + qt.tensor(I, H_atom_det)
    proj_rr = qt.tensor(proj_r, proj_r)
    H_blockade = V * proj_rr
    return H_laser + H_det + H_blockade

def build_H_interp3(phi):
    """Interpretation 3: Phase ONLY on excitation operator.
    H = (Ω/2)(e^{iφ}|r⟩⟨1| + e^{-iφ}|1⟩⟨r|) + V|rr⟩⟨rr|
    This is equivalent to interp1.
    """
    return build_H_interp1(phi)

def build_H_interp4(phi):
    """Interpretation 4: DETUNING equals ±Ω when φ=±π/2, zero when φ=0.
    The JP paper says the effective detuning is Δ_eff = ∂φ/∂t, but within
    each constant-phase segment, we might interpret φ as setting a constant Δ.
    
    Let's try: Δ = Ω·sin(φ) (so φ=0 → Δ=0, φ=±π/2 → Δ=±Ω)
    """
    Delta = Omega * np.sin(phi)
    
    H_atom_laser = 0.5 * Omega * (sigma_1r + sigma_r1)
    H_atom_det = -Delta * proj_r
    
    H_laser = qt.tensor(H_atom_laser, I) + qt.tensor(I, H_atom_laser)
    H_det = qt.tensor(H_atom_det, I) + qt.tensor(I, H_atom_det)
    proj_rr = qt.tensor(proj_r, proj_r)
    H_blockade = V * proj_rr
    return H_laser + H_det + H_blockade

def simulate_with_builder(build_H_func, name):
    switch_times_phys = [t / Omega for t in switching_times_dimless]
    boundaries = [0] + switch_times_phys + [tau_total]
    
    final_states = {}
    for label, psi0 in [("00", psi_00), ("01", psi_01), ("10", psi_10), ("11", psi_11)]:
        psi = psi0
        for i_seg in range(len(phases_std)):
            t_start = boundaries[i_seg]
            t_end = boundaries[i_seg + 1]
            duration = t_end - t_start
            if duration > 0:
                H = build_H_func(phases_std[i_seg])
                tlist = np.linspace(0, duration, 100)
                result = qt.sesolve(H, psi, tlist)
                psi = result.states[-1]
        final_states[label] = psi
    
    # Compute controlled phase
    def get_overlap(psi_t, psi_f):
        overlap = psi_t.dag() * psi_f
        val = overlap.full()[0,0] if hasattr(overlap, 'full') else complex(overlap)
        return val
    
    c00 = get_overlap(psi_00, final_states["00"])
    c01 = get_overlap(psi_01, final_states["01"])
    c10 = get_overlap(psi_10, final_states["10"])
    c11 = get_overlap(psi_11, final_states["11"])
    
    phi_00 = np.angle(c00)
    phi_01 = np.angle(c01)
    phi_10 = np.angle(c10)
    phi_11 = np.angle(c11)
    
    ctrl = phi_11 - phi_01 - phi_10 + phi_00
    ctrl = (ctrl + np.pi) % (2*np.pi) - np.pi
    
    avg_pop = 0.25 * (abs(c00)**2 + abs(c01)**2 + abs(c10)**2 + abs(c11)**2)
    
    print(f"\n{name}:")
    print(f"  Controlled phase: {np.degrees(ctrl):.2f}° (target: ±180°)")
    print(f"  Avg population: {avg_pop*100:.2f}%")
    print(f"  Phases: φ_00={np.degrees(phi_00):.1f}°, φ_01={np.degrees(phi_01):.1f}°, "
          f"φ_10={np.degrees(phi_10):.1f}°, φ_11={np.degrees(phi_11):.1f}°")

simulate_with_builder(build_H_interp1, "Interp 1: Phase on complex Ω (current)")
simulate_with_builder(build_H_interp4, "Interp 4: Phase → sinusoidal detuning Δ=Ω·sin(φ)")

# Try interpretation 4 with different omega_tau
print("\n" + "=" * 70)
print("SCANNING Ωτ WITH DETUNING INTERPRETATION")
print("=" * 70)

def simulate_detuning_interp(omega_tau_val):
    tau_total_val = omega_tau_val / Omega
    # Scale switching times proportionally
    switch_times_phys = [t * omega_tau_val / 7.0 / Omega for t in switching_times_dimless]
    boundaries = [0] + switch_times_phys + [tau_total_val]
    
    final_states = {}
    for label, psi0 in [("00", psi_00), ("01", psi_01), ("10", psi_10), ("11", psi_11)]:
        psi = psi0
        for i_seg in range(len(phases_std)):
            t_start = boundaries[i_seg]
            t_end = boundaries[i_seg + 1]
            duration = t_end - t_start
            if duration > 0:
                H = build_H_interp4(phases_std[i_seg])
                tlist = np.linspace(0, duration, 100)
                result = qt.sesolve(H, psi, tlist)
                psi = result.states[-1]
        final_states[label] = psi
    
    def get_overlap(psi_t, psi_f):
        overlap = psi_t.dag() * psi_f
        val = overlap.full()[0,0] if hasattr(overlap, 'full') else complex(overlap)
        return val
    
    c00 = get_overlap(psi_00, final_states["00"])
    c01 = get_overlap(psi_01, final_states["01"])
    c10 = get_overlap(psi_10, final_states["10"])
    c11 = get_overlap(psi_11, final_states["11"])
    
    phi_00 = np.angle(c00)
    phi_01 = np.angle(c01)
    phi_10 = np.angle(c10)
    phi_11 = np.angle(c11)
    
    ctrl = phi_11 - phi_01 - phi_10 + phi_00
    ctrl = (ctrl + np.pi) % (2*np.pi) - np.pi
    avg_pop = 0.25 * (abs(c00)**2 + abs(c01)**2 + abs(c10)**2 + abs(c11)**2)
    return ctrl, avg_pop

print(f"{'Ωτ':>6} {'Ctrl Phase':>12} {'Avg Pop':>10}")
print("-" * 32)
for omega_tau_val in np.linspace(5.0, 10.0, 21):
    ctrl, pop = simulate_detuning_interp(omega_tau_val)
    phase_err = min(abs(ctrl - np.pi), abs(ctrl + np.pi))
    marker = " <-- !" if phase_err < 0.2 else ""
    print(f"{omega_tau_val:6.2f} {np.degrees(ctrl):12.2f}° {pop*100:10.2f}%{marker}")
