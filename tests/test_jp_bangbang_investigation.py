"""
Test module: Investigate JP bang-bang 20-degree phase error.

ROOT CAUSE: The jp_bangbang optimizer builds JPSimulationInputs with
switching_times and phases, but simulate_CZ_gate() routes ALL
JPSimulationInputs through the smooth JP solver, which IGNORES
switching_times/phases. Only omega_tau passes through.
"""
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    JPSimulationInputs,
    SmoothJPSimulationInputs,
    TwoPhotonExcitationConfig,
    LaserParameters,
    NoiseSourceConfig,
)

exc = TwoPhotonExcitationConfig(
    laser_1=LaserParameters(power=50e-6, waist=40e-6, linewidth_hz=1000),
    laser_2=LaserParameters(power=0.3, waist=25e-6, linewidth_hz=1000),
    Delta_e=2 * np.pi * 1e9,
)
noise = NoiseSourceConfig(
    include_spontaneous_emission=False,
    include_intermediate_scattering=False,
    include_motional_dephasing=False,
    include_doppler_dephasing=False,
)
kw = dict(n_rydberg=70, temperature=2e-6, spacing_factor=2.8,
          include_noise=False, verbose=False)


def run(si, label=""):
    r = simulate_CZ_gate(simulation_inputs=si, **kw)
    F = r.avg_fidelity
    pe = abs(r.phase_info.get("phase_error_from_pi_deg", 0.0))
    if label:
        print(f"  {label}: F={F:.6f}, phase_err={pe:.2f} deg")
    return F, pe


# ==================================================================
# TEST 1: switching_times are ignored by the simulator
# ==================================================================
print("=" * 70)
print("TEST 1: Are switching_times ignored by the simulator?")
print("=" * 70)

jp_a = JPSimulationInputs(excitation=exc, noise=noise, omega_tau=7.0)
Fa, _ = run(jp_a, "JP default (no switch times)")

jp_b = JPSimulationInputs(
    excitation=exc, noise=noise, omega_tau=7.0,
    switching_times=[2.214, 8.823, 13.258, 19.867],
    phases=[np.pi / 2, 0, -np.pi / 2, 0, np.pi / 2],
)
Fb, _ = run(jp_b, "JP 5-seg switching times")

jp_c = JPSimulationInputs(
    excitation=exc, noise=noise, omega_tau=7.0,
    switching_times=[0.3328, 0.5859, 3.434, 3.553, 4.1204, 6.7431],
    phases=[np.pi / 2, 0, -np.pi / 2, -np.pi / 2, 0, np.pi / 2, 0],
)
Fc, _ = run(jp_c, "JP 7-seg switching times")

jp_d = JPSimulationInputs(
    excitation=exc, noise=noise, omega_tau=7.0,
    switching_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    phases=[0, np.pi, 0, np.pi, 0, np.pi, 0],
)
Fd, _ = run(jp_d, "JP garbage switch times")

if abs(Fa - Fb) < 1e-10 and abs(Fa - Fc) < 1e-10 and abs(Fa - Fd) < 1e-10:
    print("\nCONFIRMED: switching_times and phases are 100% IGNORED.")
else:
    print(f"\nUNEXPECTED: diffs B-A={Fb-Fa:.2e}, C-A={Fc-Fa:.2e}, D-A={Fd-Fa:.2e}")

# ==================================================================
# TEST 2: JPSimulationInputs == SmoothJPSimulationInputs (defaults)
# ==================================================================
print()
print("=" * 70)
print("TEST 2: JPSimulationInputs == SmoothJPSimulationInputs (defaults)")
print("=" * 70)

for ot in [5.0, 7.0, 10.09, 15.0, 20.0]:
    rj = simulate_CZ_gate(
        simulation_inputs=JPSimulationInputs(excitation=exc, noise=noise, omega_tau=ot), **kw)
    rs = simulate_CZ_gate(
        simulation_inputs=SmoothJPSimulationInputs(excitation=exc, noise=noise, omega_tau=ot), **kw)
    m = "MATCH" if abs(rj.avg_fidelity - rs.avg_fidelity) < 1e-10 else "DIFFER"
    print(f"  ot={ot:6.2f}  JP_F={rj.avg_fidelity:.6f}  SJP_F={rs.avg_fidelity:.6f}  {m}")

# ==================================================================
# TEST 3: Full 5-param smooth JP vs 1-param "bang-bang"
# ==================================================================
print()
print("=" * 70)
print("TEST 3: Full 5-param smooth JP vs 1-param bang-bang")
print("=" * 70)

run(
    SmoothJPSimulationInputs(
        excitation=exc, noise=noise, omega_tau=10.09,
        A=0.311 * np.pi, omega_mod_ratio=1.242,
        phi_offset=4.696, delta_over_omega=0.0205,
    ),
    "SmoothJP optimized (5 params)",
)
run(
    JPSimulationInputs(excitation=exc, noise=noise, omega_tau=7.0),
    "JP bang-bang (1 effective param)",
)

# ==================================================================
# TEST 4: Best omega_tau alone (1D scan)
# ==================================================================
print()
print("=" * 70)
print("TEST 4: Best omega_tau alone (1D scan)")
print("=" * 70)

from scipy.optimize import minimize_scalar


def neg_f(ot):
    r = simulate_CZ_gate(
        simulation_inputs=SmoothJPSimulationInputs(
            excitation=exc, noise=noise, omega_tau=ot),
        **kw)
    return -r.avg_fidelity


t0 = time.time()
res = minimize_scalar(neg_f, bounds=(3.0, 25.0), method="bounded",
                      options={"maxiter": 30})
dt = time.time() - t0
rb = simulate_CZ_gate(
    simulation_inputs=SmoothJPSimulationInputs(
        excitation=exc, noise=noise, omega_tau=res.x),
    **kw)
pe_best = abs(rb.phase_info.get('phase_error_from_pi_deg', 0.0))
print(f"  Best omega_tau={res.x:.4f}, F={rb.avg_fidelity:.6f}, "
      f"phase_err={pe_best:.2f} deg, time={dt:.1f}s")
print("  vs 5-param optimized: F=0.999971, phase_err=0.33 deg")

# ==================================================================
# DIAGNOSIS
# ==================================================================
print()
print("=" * 70)
print("DIAGNOSIS: JP bang-bang optimizer is broken. It optimizes")
print("switching_times that the simulator ignores. Only omega_tau")
print("passes through. The fix is to remove jp_bangbang as an")
print("optimizer variant (it adds no value over smooth_jp).")
print("=" * 70)
