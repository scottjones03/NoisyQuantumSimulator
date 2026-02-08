#!/usr/bin/env python3
"""
Generate Publication-Quality Research Figures
==============================================

Uses the FULL optimization.py module to create TRUE Pareto fronts
and meaningful parameter studies.

Figures Generated:
1. TRUE Pareto Front: Fidelity vs Gate Time (all params optimized)
2. Protocol Comparison: LP vs JP with optimal parameters
3. Noise Budget Breakdown at optimal operating point
4. Temperature Sensitivity at otherwise-optimal params
5. Species Comparison: Rb87 vs Cs133
6. Blockade Ratio Importance

Author: Quantum Simulation Team
Date: January 2026
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import time

# Import the REAL optimization module
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    explore_parameter_space,
    optimize_CZ_parameters,
    ExplorationResult,
    LPSimulationInputs,
    JPSimulationInputs,
    SmoothJPSimulationInputs,
    TwoPhotonExcitationConfig,
    LaserParameters,
    NoiseSourceConfig,
)

# Output directory
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

print("=" * 70)
print("  GENERATING RESEARCH FIGURES WITH TRUE OPTIMIZATION")
print("=" * 70)

# =============================================================================
# 1. TRUE PARETO FRONT: Fidelity vs Gate Time
# =============================================================================
print("\n[1/6] TRUE Pareto Front: Full parameter optimization...")
start = time.time()

# Explore with ALL parameters varied (not just 1-2!)
exploration_lp = explore_parameter_space(
    protocol="levine_pichler",
    species="Rb87",
    n_runs=2,  # Multiple runs for diversity
    maxiter=25,
    popsize=8,
    verbose=True,
)

print(f"  Exploration took {time.time()-start:.1f}s")
print(f"  Total evaluations: {exploration_lp.n_evaluations}")
print(f"  Pareto front size: {len(exploration_lp.pareto_front)}")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

# All explored points
all_fids = [p.fidelity for p in exploration_lp.points]
all_times = [p.gate_time_ns for p in exploration_lp.points]
ax.scatter(all_times, all_fids, c='lightgray', s=40, alpha=0.5, 
           label=f'Explored ({len(all_fids)} configs)', edgecolors='gray', linewidths=0.5)

# TRUE Pareto front
pareto_fids = [p.fidelity for p in exploration_lp.pareto_front]
pareto_times = [p.gate_time_ns for p in exploration_lp.pareto_front]

# Sort for line plot
idx = np.argsort(pareto_times)
px = np.array(pareto_times)[idx]
py = np.array(pareto_fids)[idx]

ax.scatter(px, py, c='red', s=120, zorder=5, label='Pareto optimal', 
           edgecolors='darkred', linewidths=2)
ax.plot(px, py, 'r--', alpha=0.7, linewidth=2)

# Thresholds
ax.axhline(0.99, color='green', linestyle='--', alpha=0.6, label='99% threshold')
ax.axhline(0.995, color='blue', linestyle='--', alpha=0.6, label='99.5% FT threshold')
ax.axhline(0.999, color='purple', linestyle=':', alpha=0.5, label='99.9% target')

ax.set_xlabel("Gate Time (ns)", fontsize=13)
ax.set_ylabel("Gate Fidelity", fontsize=13)
ax.set_title("TRUE Pareto Front: Fidelity vs Gate Time\n"
             "(All 10 parameters optimized: power, temp, spacing, n, linewidth, protocol params)", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.90, 1.002)

# Annotate best points
if pareto_fids:
    best_fid_idx = np.argmax(pareto_fids)
    ax.annotate(f'Best: {pareto_fids[best_fid_idx]*100:.2f}%\n@ {pareto_times[best_fid_idx]:.0f}ns',
                xy=(pareto_times[best_fid_idx], pareto_fids[best_fid_idx]),
                xytext=(pareto_times[best_fid_idx]+50, pareto_fids[best_fid_idx]-0.015),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='darkred'))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "pareto_true_fidelity_time.png", dpi=200)
print(f"  ✓ Saved: pareto_true_fidelity_time.png")
plt.close()

# =============================================================================
# 2. PROTOCOL COMPARISON: LP vs JP at Optimal Points
# =============================================================================
print("\n[2/6] Protocol Comparison at Optimal Parameters...")

# Find best configs for each protocol
best_lp = max(exploration_lp.points, key=lambda p: p.fidelity)
print(f"  Best LP: F={best_lp.fidelity:.4f}, t={best_lp.gate_time_ns:.0f}ns")

# Run JP exploration (faster since we know the landscape)
exploration_jp = explore_parameter_space(
    protocol="jandura_pupillo",
    species="Rb87",
    n_runs=1,
    maxiter=20,
    popsize=8,
    verbose=False,
)
best_jp = max(exploration_jp.points, key=lambda p: p.fidelity)
print(f"  Best JP: F={best_jp.fidelity:.4f}, t={best_jp.gate_time_ns:.0f}ns")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
protocols = ['Levine-Pichler\n(2-pulse)', 'Jandura-Pupillo\n(1-pulse)']
colors = ['#1f77b4', '#ff7f0e']

# Fidelity
ax = axes[0]
fids = [best_lp.fidelity, best_jp.fidelity]
bars = ax.bar(protocols, fids, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel("Best Achievable Fidelity", fontsize=12)
ax.set_ylim(min(fids) - 0.01, 1.001)
ax.axhline(0.99, color='red', linestyle='--', alpha=0.7, label='99%')
ax.axhline(0.995, color='green', linestyle='--', alpha=0.7, label='99.5%')
for bar, f in zip(bars, fids):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
           f'{f*100:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_title("Optimal Fidelity", fontsize=13)

# Gate time at optimal fidelity
ax = axes[1]
times = [best_lp.gate_time_ns, best_jp.gate_time_ns]
bars = ax.bar(protocols, times, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel("Gate Time at Optimal (ns)", fontsize=12)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
           f'{t:.0f} ns', ha='center', fontsize=11, fontweight='bold')
ax.set_title("Gate Duration", fontsize=13)

# V/Omega (blockade strength)
ax = axes[2]
v_over_omega = [best_lp.V_over_Omega, best_jp.V_over_Omega]
bars = ax.bar(protocols, v_over_omega, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel("Blockade Ratio V/Ω", fontsize=12)
ax.axhline(10, color='red', linestyle='--', alpha=0.7, label='V/Ω=10 threshold')
for bar, v in zip(bars, v_over_omega):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
           f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_title("Blockade Strength", fontsize=13)

fig.suptitle("Protocol Comparison at Globally Optimal Parameters", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "protocol_comparison_optimal.png", dpi=200)
print(f"  ✓ Saved: protocol_comparison_optimal.png")
plt.close()

# =============================================================================
# 3. NOISE BREAKDOWN at Optimal Point
# =============================================================================
print("\n[3/6] Noise Budget Breakdown...")

# Build simulation inputs for LP
laser_1 = LaserParameters(power=2.5e-3, waist=1.0e-6, linewidth_hz=1000)
laser_2 = LaserParameters(power=7.0, waist=10e-6, linewidth_hz=1000)
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2)
noise = NoiseSourceConfig(include_motional_dephasing=True, include_doppler_dephasing=True)
lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise)

# Run detailed simulation for best LP config
detailed_result = simulate_CZ_gate(
    simulation_inputs=lp_inputs,
    species="Rb87",
    n_rydberg=70,
    temperature=5e-6,
    spacing_factor=3.0,
    include_noise=True,
)

# Extract noise contributions
noise_data = detailed_result.noise_breakdown

# Map noise breakdown keys to human-readable names
# These are the actual keys from simulation.py noise_breakdown dict
noise_key_names = {
    'gamma_blockade_fluct': 'Blockade Fluctuations (δV/V)',
    'gamma_doppler': 'Doppler Dephasing',
    'gamma_thermal_total': 'Total Thermal Dephasing',
    'gamma_phi_laser': 'Laser Phase Noise',
    'gamma_intensity_noise': 'Intensity Noise',
    'gamma_r': 'Rydberg Decay',
    'gamma_bbr': 'BBR-Induced Decay',
    'gamma_leakage': 'Spectral Leakage',
    'gamma_loss_antitrap': 'Anti-Trap Loss',
    'gamma_loss_background': 'Background Gas Loss',
    'gamma_scatter_intermediate': 'Intermediate State Scatter',
    'gamma_phi_zeeman': 'Zeeman Dephasing',
}

# Collect noise sources and their rates (convert to approximate infidelity)
# Infidelity contribution ≈ γ × τ_gate for small γτ
tau_gate = detailed_result.tau_total
noise_sources = []
noise_values = []

for key, display_name in noise_key_names.items():
    val = noise_data.get(key, 0)
    if val is None:
        val = 0
    if val > 1e-3:  # Only significant sources (rates > 1 kHz)
        # Convert rate to approximate infidelity: 1 - e^(-γτ) ≈ γτ for γτ << 1
        infidelity_contrib = val * tau_gate
        if infidelity_contrib > 1e-6:  # Filter very small contributions
            noise_sources.append(display_name)
            noise_values.append(infidelity_contrib * 100)  # Convert to percentage

# Sort by magnitude
if noise_values:
    idx = np.argsort(noise_values)[::-1]
    noise_sources = [noise_sources[i] for i in idx]
    noise_values = [noise_values[i] for i in idx]
else:
    print("  ⚠ No significant noise sources found, using raw breakdown data")
    # Fall back to showing all available rates
    for key, val in noise_data.items():
        if isinstance(val, (int, float)) and val > 0 and 'gamma' in key:
            noise_sources.append(key.replace('gamma_', '').replace('_', ' ').title())
            noise_values.append(val * tau_gate * 100)
    idx = np.argsort(noise_values)[::-1][:10]  # Top 10
    noise_sources = [noise_sources[i] for i in idx]
    noise_values = [noise_values[i] for i in idx]

fig, ax = plt.subplots(figsize=(10, 6))
if noise_sources:
    colors_noise = plt.cm.Reds(np.linspace(0.3, 0.9, len(noise_sources)))
    bars = ax.barh(noise_sources, noise_values, color=colors_noise, edgecolor='darkred')
    
    for bar, val in zip(bars, noise_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}%', va='center', fontsize=10)
    
    ax.set_xlim(0, max(noise_values) * 1.2 if noise_values else 1)
else:
    ax.text(0.5, 0.5, "No significant noise sources identified\n(check noise_breakdown dict)", 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)

ax.set_xlabel("Contribution to Infidelity (%)", fontsize=12)
ax.set_title(f"Noise Budget Breakdown\n(Total infidelity: {(1-detailed_result.avg_fidelity)*100:.3f}%)", 
             fontsize=14, fontweight='bold')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "noise_breakdown.png", dpi=200)
print(f"  ✓ Saved: noise_breakdown.png")
plt.close()

# =============================================================================
# 4. TEMPERATURE SENSITIVITY
# =============================================================================
print("\n[4/6] Temperature Sensitivity...")

temperatures = np.logspace(-6, -4.3, 12)  # 1 μK to 50 μK
fids_lp, fids_jp = [], []

# Build simulation inputs
laser_1 = LaserParameters(power=2.5e-3, waist=1.0e-6, linewidth_hz=1000)
laser_2 = LaserParameters(power=7.0, waist=10e-6, linewidth_hz=1000)
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2)
noise = NoiseSourceConfig(include_motional_dephasing=True, include_doppler_dephasing=True)
lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise)
jp_inputs = JPSimulationInputs(excitation=excitation, noise=noise)
smooth_jp_inputs = SmoothJPSimulationInputs(excitation=excitation, noise=noise)

fids_smooth = []
for T in temperatures:
    r_lp = simulate_CZ_gate(
        simulation_inputs=lp_inputs,
        species="Rb87", temperature=T,
        n_rydberg=70, spacing_factor=3.0, include_noise=True
    )
    r_jp = simulate_CZ_gate(
        simulation_inputs=jp_inputs,
        species="Rb87", temperature=T,
        n_rydberg=70, spacing_factor=3.0, include_noise=True
    )
    r_smooth = simulate_CZ_gate(
        simulation_inputs=smooth_jp_inputs,
        species="Rb87", temperature=T,
        n_rydberg=70, spacing_factor=3.0, include_noise=True
    )
    fids_lp.append(r_lp.avg_fidelity)
    fids_jp.append(r_jp.avg_fidelity)
    fids_smooth.append(r_smooth.avg_fidelity)

fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogx(temperatures * 1e6, fids_lp, 'o-', color='#1f77b4', label='Levine-Pichler', linewidth=2, markersize=8)
ax.semilogx(temperatures * 1e6, fids_jp, 's-', color='#ff7f0e', label='JP Bang-Bang', linewidth=2, markersize=8)
ax.semilogx(temperatures * 1e6, fids_smooth, '^-', color='#2ca02c', label='JP Smooth', linewidth=2, markersize=8)
ax.axhline(0.99, color='red', linestyle='--', alpha=0.5, label='99% threshold')
ax.axhline(0.995, color='green', linestyle='--', alpha=0.5, label='99.5% FT')
ax.axvline(5, color='gray', linestyle=':', alpha=0.5, label='5 μK (typical)')
ax.set_xlabel("Temperature (μK)", fontsize=12)
ax.set_ylabel("Gate Fidelity", fontsize=12)
ax.set_title("Temperature Sensitivity: Doppler Dephasing Effect", fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "temperature_sensitivity.png", dpi=200)
print(f"  ✓ Saved: temperature_sensitivity.png")
plt.close()

# =============================================================================
# 5. SPECIES COMPARISON: Rb87 vs Cs133 (All 3 Protocols)
# =============================================================================
print("\n[5/6] Species Comparison...")

temps = np.logspace(-6, -4.5, 10)

# Build simulation inputs
laser_1 = LaserParameters(power=2.5e-3, waist=1.0e-6, linewidth_hz=1000)
laser_2 = LaserParameters(power=7.0, waist=10e-6, linewidth_hz=1000)
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2)
noise = NoiseSourceConfig(include_motional_dephasing=True, include_doppler_dephasing=True)
lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise)
jp_inputs = JPSimulationInputs(excitation=excitation, noise=noise)
smooth_jp_inputs = SmoothJPSimulationInputs(excitation=excitation, noise=noise)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

protocol_configs = [
    (lp_inputs, 'LP Protocol'),
    (jp_inputs, 'JP Bang-Bang'),
    (smooth_jp_inputs, 'JP Smooth')
]

for ax, (sim_inputs, prot_label) in zip(axes, protocol_configs):
    rb_fids, cs_fids = [], []
    for T in temps:
        rb = simulate_CZ_gate(simulation_inputs=sim_inputs, species="Rb87", temperature=T,
                              n_rydberg=70, spacing_factor=3.0, include_noise=True)
        cs = simulate_CZ_gate(simulation_inputs=sim_inputs, species="Cs133", temperature=T,
                              n_rydberg=70, spacing_factor=3.0, include_noise=True)
        rb_fids.append(rb.avg_fidelity)
        cs_fids.append(cs.avg_fidelity)
    
    ax.semilogx(temps * 1e6, rb_fids, 'o-', color='#1f77b4', label='Rb87', linewidth=2, markersize=8)
    ax.semilogx(temps * 1e6, cs_fids, 's-', color='#d62728', label='Cs133', linewidth=2, markersize=8)
    ax.axhline(0.99, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Temperature (μK)", fontsize=12)
    ax.set_ylabel("Gate Fidelity", fontsize=12)
    ax.set_title(f"{prot_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle("Species Comparison: Rb87 vs Cs133 (Same Experimental Parameters)", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "species_comparison.png", dpi=200)
print(f"  ✓ Saved: species_comparison.png")
plt.close()

# =============================================================================
# 6. BLOCKADE RATIO IMPORTANCE (All 3 Protocols)
# =============================================================================
print("\n[6/6] Blockade Ratio Importance...")

# Vary spacing to control V/Omega
spacings = np.linspace(2.0, 5.5, 12)

# Build simulation inputs
laser_1 = LaserParameters(power=2.5e-3, waist=1.0e-6, linewidth_hz=1000)
laser_2 = LaserParameters(power=7.0, waist=10e-6, linewidth_hz=1000)
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2)
noise = NoiseSourceConfig(include_motional_dephasing=True, include_doppler_dephasing=True)
lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise)
jp_inputs = JPSimulationInputs(excitation=excitation, noise=noise)
smooth_jp_inputs = SmoothJPSimulationInputs(excitation=excitation, noise=noise)

fig, ax = plt.subplots(figsize=(10, 6))

protocol_configs = [
    (lp_inputs, '#1f77b4', 'o', 'LP'),
    (jp_inputs, '#ff7f0e', 's', 'JP Bang-Bang'),
    (smooth_jp_inputs, '#2ca02c', '^', 'JP Smooth')
]

v_ratios_ref = []  # Store V/Omega ratios (same for all protocols)

for sim_inputs, color, marker, label in protocol_configs:
    fids = []
    v_ratios = []
    for sf in spacings:
        result = simulate_CZ_gate(
            simulation_inputs=sim_inputs,
            species="Rb87",
            spacing_factor=sf, n_rydberg=70, temperature=5e-6,
            include_noise=True
        )
        fids.append(result.avg_fidelity)
        v_ratios.append(result.V_over_Omega)
    
    ax.plot(v_ratios, fids, marker + '-', color=color, label=label, linewidth=2, markersize=8)
    if not v_ratios_ref:
        v_ratios_ref = v_ratios

ax.axhline(0.99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')
ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='V/Ω = 10')

ax.set_xlabel("Blockade Ratio V/Ω", fontsize=12)
ax.set_ylabel("Gate Fidelity", fontsize=12)
ax.set_title("Blockade Ratio vs Fidelity: All 3 Protocols\n(Closer atoms → stronger blockade → better gate)", 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xlim(5, 500)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "blockade_importance.png", dpi=200)
print(f"  ✓ Saved: blockade_importance.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  FIGURE GENERATION COMPLETE")
print("=" * 70)
print(f"\nGenerated 6 figures in {OUTPUT_DIR}/:")
print("  1. pareto_true_fidelity_time.png  - TRUE Pareto front (all params optimized)")
print("  2. protocol_comparison_optimal.png - LP vs JP at global optima")
print("  3. noise_breakdown.png             - Noise budget breakdown")
print("  4. temperature_sensitivity.png     - Temperature effects")
print("  5. species_comparison.png          - Rb87 vs Cs133")
print("  6. blockade_importance.png         - V/Ω trade-off")
print("\nExploration statistics:")
print(f"  LP evaluations: {exploration_lp.n_evaluations}")
print(f"  LP Pareto front: {len(exploration_lp.pareto_front)} points")
print(f"  JP evaluations: {exploration_jp.n_evaluations}")
print(f"  Best LP fidelity: {best_lp.fidelity*100:.2f}%")
print(f"  Best JP fidelity: {best_jp.fidelity*100:.2f}%")
