#!/usr/bin/env python3
"""
Full 3-protocol optimization test at JP-friendly V/Omega ~ 171.

Uses Delta_e = 2*pi * 0.5 GHz (instead of default 1 GHz) to bring
V/Omega from 342.5 down to 171.3 — well within the JP lookup table
validated range [10-200].
"""

import time
import numpy as np
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.optimize_cz_gate import (
    optimize_cz_gate, ApparatusConstraints, run_baseline
)


def main():
    # JP-friendly apparatus: Delta_e = 0.5 GHz gives V/Omega ~ 171
    apparatus = ApparatusConstraints(
        Delta_e=2 * np.pi * 0.5e9,  # 0.5 GHz -> V/Omega ~ 171
    )
    
    print("=" * 70)
    print("CZ GATE OPTIMIZER — FULL 3-PROTOCOL TEST")
    print("=" * 70)
    print(f"Delta_e/2pi = {apparatus.Delta_e / (2*np.pi*1e9):.2f} GHz")
    print(f"Expected V/Omega ~ 171 (JP validated range: 10-200)")
    print()
    
    # ---- Baselines ----
    print("-" * 70)
    print("BASELINES (literature parameters, no optimization)")
    print("-" * 70)
    for proto in ["lp", "smooth_jp", "jp_bangbang"]:
        print(f"\n  {proto}:")
        try:
            bm = run_baseline(proto, apparatus, include_noise=False, verbose=False)
            print(f"    avg_fidelity = {bm['avg_fidelity']*100:.2f}%")
            print(f"    phase_error  = {bm['phase_error_deg']:.1f} deg")
            print(f"    F(|11>)      = {bm['f11']*100:.2f}%")
            print(f"    V/Omega      = {bm.get('V_over_Omega', '?'):.1f}")
            print(f"    gate_time    = {bm.get('gate_time_us', '?'):.3f} us")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # ---- Optimize LP ----
    print()
    print("=" * 70)
    print("OPTIMIZING LP PROTOCOL")
    print("=" * 70)
    t0 = time.time()
    res_lp = optimize_cz_gate(
        protocol="lp",
        apparatus=apparatus,
        include_noise=False,
        maxiter=60,
        verbose=True,
    )
    dt_lp = time.time() - t0
    print(f"\nLP optimization: {dt_lp:.1f}s")
    print(res_lp)
    
    # ---- Optimize Smooth JP ----
    print()
    print("=" * 70)
    print("OPTIMIZING SMOOTH JP PROTOCOL")
    print("=" * 70)
    t0 = time.time()
    res_sjp = optimize_cz_gate(
        protocol="smooth_jp",
        apparatus=apparatus,
        include_noise=False,
        maxiter=60,
        verbose=True,
    )
    dt_sjp = time.time() - t0
    print(f"\nSmooth JP optimization: {dt_sjp:.1f}s")
    print(res_sjp)
    
    # ---- Optimize JP Bang-Bang ----
    print()
    print("=" * 70)
    print("OPTIMIZING JP BANG-BANG PROTOCOL")
    print("=" * 70)
    t0 = time.time()
    res_jp = optimize_cz_gate(
        protocol="jp_bangbang",
        apparatus=apparatus,
        include_noise=False,
        maxiter=60,
        verbose=True,
    )
    dt_jp = time.time() - t0
    print(f"\nJP bang-bang optimization: {dt_jp:.1f}s")
    print(res_jp)
    
    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY — ALL PROTOCOLS")
    print("=" * 70)
    for label, res, dt in [
        ("LP", res_lp, dt_lp),
        ("Smooth JP", res_sjp, dt_sjp),
        ("JP Bang-Bang", res_jp, dt_jp),
    ]:
        m = res.best_metrics
        print(f"  {label:15s}: F={m['avg_fidelity']*100:.4f}%  "
              f"phi_err={m['phase_error_deg']:.2f} deg  "
              f"F11={m['f11']*100:.2f}%  "
              f"t_gate={m.get('gate_time_us', 0):.3f} us  "
              f"({dt:.0f}s, {res.n_evaluations} evals)")
    
    # Check target
    print()
    target = 99.0
    for label, res in [("LP", res_lp), ("Smooth JP", res_sjp), ("JP BB", res_jp)]:
        f = res.best_metrics['avg_fidelity'] * 100
        status = "✅ PASS" if f >= target else "❌ FAIL"
        print(f"  {label:15s}: {f:.4f}% {status} (target >= {target}%)")


if __name__ == "__main__":
    main()
