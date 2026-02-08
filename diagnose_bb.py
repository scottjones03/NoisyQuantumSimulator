#!/usr/bin/env python3
"""
Diagnose JP bang-bang phase error:
1. What V/Ω do we actually have?
2. Are the phase bounds too restrictive? (only ±π/2 — paper uses exactly ±π/2 and 0)
3. Are the omega_tau bounds too narrow?
4. Does a quick optimisation help?
5. Are there sorting/overlap issues with switching times?
"""
import sys
import numpy as np
sys.path.insert(0, "src")

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.simulation import simulate_CZ_gate
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.configurations import (
    JPSimulationInputs, SmoothJPSimulationInputs,
    TwoPhotonExcitationConfig, NoiseSourceConfig, LaserParameters,
)
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.protocols import (
    JP_OMEGA_TAU_VALIDATED, JP_SWITCHING_TIMES_VALIDATED, JP_PHASES_VALIDATED,
)

excitation = TwoPhotonExcitationConfig(
    laser_1=LaserParameters(power=50e-6, waist=50e-6, polarization="pi"),
    laser_2=LaserParameters(power=0.3, waist=50e-6, polarization="sigma+"),
    Delta_e=2 * np.pi * 1e9,
)
noise_off = NoiseSourceConfig(
    include_spontaneous_emission=False,
    include_intermediate_scattering=False,
    include_motional_dephasing=False,
    include_doppler_dephasing=False,
    include_intensity_noise=False,
    include_laser_dephasing=False,
    include_magnetic_dephasing=False,
)


def run(omega_tau, switching_times, phases, label=""):
    inp = JPSimulationInputs(
        excitation=excitation, noise=noise_off,
        omega_tau=omega_tau,
        switching_times=switching_times,
        phases=phases,
    )
    r = simulate_CZ_gate(
        simulation_inputs=inp, species="Rb87", n_rydberg=70,
        spacing_factor=2.8, temperature=2e-6,
        include_noise=False, verbose=False, return_dataclass=True,
    )
    f = r.avg_fidelity
    phi_err = r.phase_info.get('phase_error_from_pi_deg', np.nan)
    ctrl = r.phase_info.get('controlled_phase_deg', np.nan)
    v_omega = r.V_over_Omega
    return f, phi_err, ctrl, v_omega


# ── 1. Check actual V/Ω ──────────────────────────────────────────────
print("=" * 80)
print("1. What V/Ω do we actually have with default apparatus?")
print("=" * 80)
f0, phi_err0, ctrl0, v_omega = run(
    JP_OMEGA_TAU_VALIDATED,
    JP_SWITCHING_TIMES_VALIDATED,
    JP_PHASES_VALIDATED,
    "5-seg validated"
)
print(f"   V/Ω = {v_omega:.1f}")
print(f"   F = {f0:.6f}, φ_err = {phi_err0:+.2f}°, φ_ctrl = {ctrl0:+.2f}°")
print(f"   The 5-seg params were optimised for V/Ω=200. We have V/Ω={v_omega:.1f}.")
print()

# ── 2. Sweep omega_tau to find better value for this V/Ω ─────────────
print("=" * 80)
print("2. Sweep omega_tau (keeping default switching_times scaled proportionally)")
print("=" * 80)

base_ot = JP_OMEGA_TAU_VALIDATED  # 22.08
base_st = JP_SWITCHING_TIMES_VALIDATED  # [2.214, 8.823, 13.258, 19.867]
base_ph = JP_PHASES_VALIDATED  # [π/2, 0, -π/2, 0, π/2]

best_f = 0
best_ot = base_ot
for ot in np.linspace(5, 40, 71):
    # Scale switching times proportionally
    scale = ot / base_ot
    st = [t * scale for t in base_st]
    f, pe, ctrl, _ = run(ot, st, base_ph)
    tag = " ← BEST" if f > best_f else ""
    if f > best_f or abs(ot - base_ot) < 0.5:
        print(f"   Ωτ = {ot:6.2f}: F = {f:.6f}, φ_err = {pe:+7.2f}°{tag}")
    if f > best_f:
        best_f = f
        best_ot = ot
print(f"\n   Best Ωτ = {best_ot:.2f} (F = {best_f:.6f})")
print()

# ── 3. Phase range test — does allowing phases beyond ±π/2 help? ─────
print("=" * 80)
print("3. Phase range test — scan phase magnitude")
print("=" * 80)

for phi_max in [np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, np.pi]:
    ph = [phi_max, 0, -phi_max, 0, phi_max]
    f, pe, ctrl, _ = run(base_ot, base_st, ph)
    print(f"   φ_max = {phi_max/np.pi:.3f}π ({np.degrees(phi_max):6.1f}°): "
          f"F = {f:.6f}, φ_err = {pe:+7.2f}°")
print()

# ── 4. Full grid search: omega_tau × switching_time offsets ───────────
print("=" * 80)
print("4. Quick local optimisation around validated params")
print("=" * 80)

from scipy.optimize import minimize

def objective(x):
    """x = [omega_tau, t1, t2, t3, t4]"""
    ot = x[0]
    st = sorted(x[1:5])  # ensure sorted
    ph = [np.pi/2, 0, -np.pi/2, 0, np.pi/2]
    
    # Check validity: times must be within (0, ot)
    if st[0] <= 0 or st[-1] >= ot:
        return 1e6
    for i in range(len(st) - 1):
        if st[i+1] - st[i] < 0.1:  # min segment width
            return 1e6
    
    f, pe, ctrl, _ = run(ot, st, ph)
    # Cost: prioritise fidelity, then phase error
    infidelity = (1 - f) * 100
    phase_cost = (1 - np.cos(np.radians(pe)/2)**2) * 100
    return 10 * infidelity**2 + 5 * phase_cost**2

x0 = [base_ot] + list(base_st)
print(f"   Starting from x0 = {[f'{v:.3f}' for v in x0]}")

result = minimize(
    objective, x0, method='Nelder-Mead',
    options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.001, 'adaptive': True}
)

best_x = result.x
best_st = sorted(best_x[1:5])
f_opt, pe_opt, ctrl_opt, _ = run(best_x[0], best_st, base_ph)
print(f"   Optimised: Ωτ = {best_x[0]:.4f}")
print(f"   Switching times: {[f'{t:.4f}' for t in best_st]}")
print(f"   F = {f_opt:.6f}, φ_err = {pe_opt:+.2f}°, φ_ctrl = {ctrl_opt:+.2f}°")
print(f"   ({result.nfev} function evaluations)")
print()

# ── 5. Also try optimising phases jointly ─────────────────────────────
print("=" * 80)
print("5. Full optimisation: omega_tau + switching_times + phases")
print("=" * 80)

def objective_full(x):
    """x = [omega_tau, t1, t2, t3, t4, phi0, phi1, phi2, phi3, phi4]"""
    ot = x[0]
    st = sorted(x[1:5])
    ph = list(x[5:10])
    
    if st[0] <= 0 or st[-1] >= ot:
        return 1e6
    for i in range(len(st) - 1):
        if st[i+1] - st[i] < 0.1:
            return 1e6
    
    f, pe, ctrl, _ = run(ot, st, ph)
    infidelity = (1 - f) * 100
    phase_cost = (1 - np.cos(np.radians(pe)/2)**2) * 100
    return 10 * infidelity**2 + 5 * phase_cost**2

x0_full = [base_ot] + list(base_st) + list(base_ph)
print(f"   Starting from validated 5-seg params")

result_full = minimize(
    objective_full, x0_full, method='Nelder-Mead',
    options={'maxiter': 2000, 'xatol': 0.001, 'fatol': 0.0001, 'adaptive': True}
)

bx = result_full.x
bst = sorted(bx[1:5])
bph = list(bx[5:10])
f_full, pe_full, ctrl_full, _ = run(bx[0], bst, bph)
print(f"   Optimised: Ωτ = {bx[0]:.4f}")
print(f"   Switching times: {[f'{t:.4f}' for t in bst]}")
print(f"   Phases: {[f'{p/np.pi:+.4f}π' for p in bph]}")
print(f"   F = {f_full:.6f}, φ_err = {pe_full:+.2f}°, φ_ctrl = {ctrl_full:+.2f}°")
print(f"   ({result_full.nfev} function evaluations)")
print()

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"   Baseline (validated 5-seg):  F = {f0:.6f}, φ_err = {phi_err0:+.2f}°")
print(f"   Times-only optimised:        F = {f_opt:.6f}, φ_err = {pe_opt:+.2f}°")
print(f"   Full (times+phases) optimised: F = {f_full:.6f}, φ_err = {pe_full:+.2f}°")
