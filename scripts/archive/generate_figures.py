#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Neutral Atom CZ Gate Simulation
=========================================================================

This script generates comprehensive figures with proper physics optimization.
Each figure addresses specific aspects of the CZ gate performance.
"""

import sys
sys.path.insert(0, '/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/NoisyQuantumSimulator/src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import warnings
warnings.filterwarnings('ignore')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    LPSimulationInputs,
    JPSimulationInputs,
    SmoothJPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)

# Create figures directory
os.makedirs('figures', exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_simulation(
    protocol: str,
    species: str,
    temperature: float = 5e-6,
    n_rydberg: int = 60,
    spacing_factor: float = 2.0,
    laser_power_1: float = 50e-6,
    laser_power_2: float = 0.3,
    laser_waist: float = 50e-6,
    linewidth_hz: float = 1000,
    Delta_e: float = 5e9,
    include_noise: bool = True,
    verbose: bool = False
) -> dict:
    """Run a single CZ gate simulation."""
    is_lp = protocol.lower() in ["levine_pichler", "lp"]
    is_smooth_jp = protocol.lower() in ["smooth_jp", "dark_state", "sinusoidal_jp"]
    
    laser_1 = LaserParameters(
        power=laser_power_1,
        waist=laser_waist,
        linewidth_hz=linewidth_hz,
        polarization="sigma+",
        polarization_purity=0.99
    )
    
    laser_2 = LaserParameters(
        power=laser_power_2,
        waist=laser_waist,
        linewidth_hz=linewidth_hz,
        polarization="sigma+",
        polarization_purity=0.99
    )
    
    excitation = TwoPhotonExcitationConfig(
        laser_1=laser_1,
        laser_2=laser_2,
        Delta_e=Delta_e
    )
    
    noise_config = NoiseSourceConfig()
    
    if is_lp:
        sim_inputs = LPSimulationInputs(
            excitation=excitation,
            noise=noise_config,
            delta_over_omega=None,
            omega_tau=None
        )
    elif is_smooth_jp:
        sim_inputs = SmoothJPSimulationInputs(
            excitation=excitation,
            noise=noise_config,
            omega_tau=None  # Use validated defaults
        )
    else:
        # JP bang-bang
        sim_inputs = JPSimulationInputs(
            excitation=excitation,
            noise=noise_config,
            omega_tau=None
        )
    
    try:
        result = simulate_CZ_gate(
            simulation_inputs=sim_inputs,
            species=species,
            n_rydberg=n_rydberg,
            temperature=temperature,
            spacing_factor=spacing_factor,
            tweezer_power=0.02,
            tweezer_waist=0.8e-6,
            include_noise=include_noise,
            verbose=verbose
        )
        
        return {
            'fidelity': result.avg_fidelity,
            'gate_time_ns': result.gate_time_us * 1000,
            'noise_breakdown': result.noise_breakdown or {},
            'Omega_MHz': result.Omega_MHz,
            'V_over_Omega': result.V_over_Omega,
            'protocol': protocol,
            'species': species,
            'temperature': temperature,
            'spacing_factor': spacing_factor,
            'n_rydberg': n_rydberg,
            'laser_power_1': laser_power_1,
            'laser_power_2': laser_power_2,
            'Delta_e': Delta_e,
            'success': True
        }
    except Exception as e:
        if verbose:
            print(f"  Simulation failed: {e}")
        return {'success': False, 'error': str(e), 'fidelity': 0, 'gate_time_ns': 0}


# =============================================================================
# FIGURE 1: MULTI-PANEL PARETO FRONTS
# =============================================================================

def generate_pareto_figure():
    """Generate multi-panel figure showing parameter dependencies."""
    print("\n" + "="*60)
    print("FIGURE 1: Multi-Panel Parameter Study")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Common power grid
    power_grid = [
        (100e-6, 0.5), (80e-6, 0.4), (60e-6, 0.35),
        (50e-6, 0.3), (40e-6, 0.25), (30e-6, 0.2),
        (25e-6, 0.18), (20e-6, 0.15), (15e-6, 0.12),
        (12e-6, 0.1), (10e-6, 0.08), (8e-6, 0.06),
        (6e-6, 0.05), (5e-6, 0.04), (4e-6, 0.03),
        (3e-6, 0.025), (2e-6, 0.02),
    ]
    
    # Panel 1: Temperature comparison
    ax = axes[0, 0]
    print("\nPanel 1: Temperature comparison...")
    for temp, color, label in [(1e-6, 'blue', 'T=1μK'), (5e-6, 'green', 'T=5μK'), (20e-6, 'red', 'T=20μK')]:
        results = []
        for p1, p2 in power_grid:
            for delta_e in [4e9, 5e9, 6e9]:
                r = run_simulation('levine_pichler', 'Rb87', temperature=temp,
                                   laser_power_1=p1, laser_power_2=p2, Delta_e=delta_e, spacing_factor=2.0)
                if r['success'] and r['V_over_Omega'] > 5:
                    results.append(r)
        if results:
            times = [r['gate_time_ns'] for r in results]
            fids = [r['fidelity']*100 for r in results]
            ax.scatter(times, fids, c=color, alpha=0.3, s=20)
            # Envelope
            bins = np.arange(0, max(times)+100, 50)
            env_t, env_f = [], []
            for i in range(len(bins)-1):
                in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                if in_bin:
                    env_t.append((bins[i]+bins[i+1])/2)
                    env_f.append(max(in_bin))
            if env_t:
                ax.plot(env_t, env_f, '-', color=color, lw=2, label=label)
    ax.set_xlabel('Gate Time (ns)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Temperature Effect (LP)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(85, 100.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    # Panel 2: Rydberg state comparison
    ax = axes[0, 1]
    print("Panel 2: Rydberg state comparison...")
    for n_ryd, color, label in [(50, 'purple', 'n=50'), (60, 'blue', 'n=60'), (70, 'orange', 'n=70')]:
        results = []
        for p1, p2 in power_grid:
            for delta_e in [4e9, 5e9, 6e9]:
                r = run_simulation('levine_pichler', 'Rb87', n_rydberg=n_ryd,
                                   laser_power_1=p1, laser_power_2=p2, Delta_e=delta_e, spacing_factor=2.0)
                if r['success'] and r['V_over_Omega'] > 5:
                    results.append(r)
        if results:
            times = [r['gate_time_ns'] for r in results]
            fids = [r['fidelity']*100 for r in results]
            ax.scatter(times, fids, c=color, alpha=0.3, s=20)
            bins = np.arange(0, max(times)+100, 50)
            env_t, env_f = [], []
            for i in range(len(bins)-1):
                in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                if in_bin:
                    env_t.append((bins[i]+bins[i+1])/2)
                    env_f.append(max(in_bin))
            if env_t:
                ax.plot(env_t, env_f, '-', color=color, lw=2, label=label)
    ax.set_xlabel('Gate Time (ns)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Rydberg State Effect (LP)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(85, 100.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    # Panel 3: Intermediate detuning comparison
    ax = axes[0, 2]
    print("Panel 3: Intermediate detuning comparison...")
    for delta_e, color, label in [(3e9, 'red', 'Δₑ=3GHz'), (5e9, 'blue', 'Δₑ=5GHz'), (8e9, 'green', 'Δₑ=8GHz')]:
        results = []
        for p1, p2 in power_grid:
            r = run_simulation('levine_pichler', 'Rb87', laser_power_1=p1, laser_power_2=p2,
                               Delta_e=delta_e, spacing_factor=2.0)
            if r['success'] and r['V_over_Omega'] > 5:
                results.append(r)
        if results:
            times = [r['gate_time_ns'] for r in results]
            fids = [r['fidelity']*100 for r in results]
            ax.scatter(times, fids, c=color, alpha=0.3, s=20)
            bins = np.arange(0, max(times)+100, 50)
            env_t, env_f = [], []
            for i in range(len(bins)-1):
                in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                if in_bin:
                    env_t.append((bins[i]+bins[i+1])/2)
                    env_f.append(max(in_bin))
            if env_t:
                ax.plot(env_t, env_f, '-', color=color, lw=2, label=label)
    ax.set_xlabel('Gate Time (ns)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Intermediate Detuning Effect (LP)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(85, 100.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    # Panel 4: Species comparison
    ax = axes[1, 0]
    print("Panel 4: Species comparison...")
    for species, n_ryd, color, label in [('Rb87', 60, 'blue', 'Rb87 n=60'), ('Cs133', 50, 'red', 'Cs133 n=50')]:
        results = []
        for p1, p2 in power_grid:
            for delta_e in [4e9, 5e9, 6e9]:
                r = run_simulation('levine_pichler', species, n_rydberg=n_ryd,
                                   laser_power_1=p1, laser_power_2=p2, Delta_e=delta_e, spacing_factor=2.0)
                if r['success'] and r['V_over_Omega'] > 5:
                    results.append(r)
        if results:
            times = [r['gate_time_ns'] for r in results]
            fids = [r['fidelity']*100 for r in results]
            ax.scatter(times, fids, c=color, alpha=0.3, s=20)
            bins = np.arange(0, max(times)+100, 50)
            env_t, env_f = [], []
            for i in range(len(bins)-1):
                in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                if in_bin:
                    env_t.append((bins[i]+bins[i+1])/2)
                    env_f.append(max(in_bin))
            if env_t:
                ax.plot(env_t, env_f, '-', color=color, lw=2, label=label)
    ax.set_xlabel('Gate Time (ns)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Species Comparison (LP)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(85, 100.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    # Panel 5: Laser linewidth comparison
    ax = axes[1, 1]
    print("Panel 5: Laser linewidth comparison...")
    for lw_hz, color, label in [(100, 'blue', '100 Hz'), (1000, 'green', '1 kHz'), (10000, 'red', '10 kHz')]:
        results = []
        for p1, p2 in power_grid:
            for delta_e in [4e9, 5e9, 6e9]:
                r = run_simulation('levine_pichler', 'Rb87', linewidth_hz=lw_hz,
                                   laser_power_1=p1, laser_power_2=p2, Delta_e=delta_e, spacing_factor=2.0)
                if r['success'] and r['V_over_Omega'] > 5:
                    results.append(r)
        if results:
            times = [r['gate_time_ns'] for r in results]
            fids = [r['fidelity']*100 for r in results]
            ax.scatter(times, fids, c=color, alpha=0.3, s=20)
            bins = np.arange(0, max(times)+100, 50)
            env_t, env_f = [], []
            for i in range(len(bins)-1):
                in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                if in_bin:
                    env_t.append((bins[i]+bins[i+1])/2)
                    env_f.append(max(in_bin))
            if env_t:
                ax.plot(env_t, env_f, '-', color=color, lw=2, label=label)
    ax.set_xlabel('Gate Time (ns)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Laser Linewidth Effect (LP)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(85, 100.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    # Panel 6: Protocol comparison (LP vs JP bang-bang vs JP smooth)
    ax = axes[1, 2]
    print("Panel 6: Protocol comparison (3 protocols)...")
    for protocol, color, label in [
        ('levine_pichler', 'blue', 'LP'),
        ('jandura_pupillo', 'orange', 'JP (bang-bang)'),
        ('smooth_jp', 'green', 'JP (smooth)')
    ]:
        results = []
        for p1, p2 in power_grid:
            for delta_e in [4e9, 5e9, 6e9]:
                r = run_simulation(protocol, 'Rb87', laser_power_1=p1, laser_power_2=p2,
                                   Delta_e=delta_e, spacing_factor=2.0)
                if r['success'] and r['V_over_Omega'] > 5:
                    results.append(r)
        if results:
            times = [r['gate_time_ns'] for r in results]
            fids = [r['fidelity']*100 for r in results]
            ax.scatter(times, fids, c=color, alpha=0.3, s=20)
            bins = np.arange(0, max(times)+100, 50)
            env_t, env_f = [], []
            for i in range(len(bins)-1):
                in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                if in_bin:
                    env_t.append((bins[i]+bins[i+1])/2)
                    env_f.append(max(in_bin))
            if env_t:
                ax.plot(env_t, env_f, '-', color=color, lw=2, label=label)
    ax.set_xlabel('Gate Time (ns)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Protocol Comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(85, 100.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    plt.suptitle('CZ Gate Parameter Study (Rb87, Strong Blockade Regime)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/pareto_true_fidelity_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved figures/pareto_true_fidelity_time.png")


# =============================================================================
# FIGURE 2: BLOCKADE IMPORTANCE (BOTH PROTOCOLS)
# =============================================================================

def generate_blockade_figure():
    """Show how V/Ω ratio affects gate fidelity for all 3 protocols."""
    print("\n" + "="*60)
    print("FIGURE 2: Blockade Importance (All Protocols)")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    protocols = [
        ('levine_pichler', 'LP Protocol'),
        ('jandura_pupillo', 'JP Bang-Bang'),
        ('smooth_jp', 'JP Smooth')
    ]
    
    for ax, (protocol, title) in zip(axes, protocols):
        print(f"\nRunning {title}...")
        
        # Use different spacing factors to vary V/Ω
        spacing_factors = np.linspace(1.5, 6.0, 30)
        
        results = []
        for sf in spacing_factors:
            result = run_simulation(
                protocol, "Rb87",
                spacing_factor=sf,
                laser_power_1=50e-6,
                laser_power_2=0.3,
                Delta_e=5e9,
                include_noise=False  # Pure coherent to see blockade physics
            )
            if result['success']:
                results.append({
                    'spacing_factor': sf,
                    'V_over_Omega': result['V_over_Omega'],
                    'fidelity': result['fidelity'],
                    'gate_time_ns': result['gate_time_ns'],
                })
                print(f"  sf={sf:.2f}: V/Ω={result['V_over_Omega']:.1f}, F={result['fidelity']*100:.2f}%, τ={result['gate_time_ns']:.0f}ns")
        
        if results:
            v_omega = [r['V_over_Omega'] for r in results]
            fidelity = [r['fidelity'] * 100 for r in results]
            gate_times = [r['gate_time_ns'] for r in results]
            
            # Color by gate time
            sc = ax.scatter(v_omega, fidelity, c=gate_times, cmap='viridis', s=80, zorder=5, 
                           edgecolors='black', linewidths=0.5)
            
            # Sort and plot line
            sorted_data = sorted(zip(v_omega, fidelity))
            ax.plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], 
                   'k-', linewidth=1.5, alpha=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Gate Time (ns)')
            
            # Mark regions
            ax.axvspan(10, max(v_omega)*1.2, alpha=0.1, color='green')
            ax.axvspan(3, 10, alpha=0.1, color='yellow')
            ax.axvspan(0.01, 3, alpha=0.1, color='red')
            
            ax.text(50, 72, 'Strong\nblockade', fontsize=9, color='green')
            ax.text(5, 72, 'Marginal', fontsize=9, color='orange')
            ax.text(0.5, 72, 'Weak', fontsize=9, color='red')
            
            ax.set_xlabel('V/Ω (Blockade Strength)', fontsize=12)
            ax.set_ylabel('Fidelity (%)', fontsize=12)
            ax.set_title(f'{title} (No Noise)', fontsize=13)
            ax.set_xscale('log')
            ax.set_xlim(0.05, 1000)
            ax.set_ylim(70, 102)
            ax.grid(True, alpha=0.3)
            ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figures/blockade_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved figures/blockade_importance.png")


# =============================================================================
# FIGURE 3: NOISE BREAKDOWN (SIDE-BY-SIDE + PARAMETER SENSITIVITY)
# =============================================================================

def generate_noise_breakdown_figure():
    """Generate noise analysis with side-by-side protocol comparison."""
    print("\n" + "="*60)
    print("FIGURE 3: Noise Breakdown and Sensitivity")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Top row: Side-by-side bar charts for each gate time
    gate_configs = [
        (100e-6, 0.5, "Fast (~60ns)"),
        (30e-6, 0.2, "Medium (~150ns)"),
        (10e-6, 0.08, "Slow (~400ns)"),
    ]
    
    for idx, (p1, p2, speed_label) in enumerate(gate_configs):
        ax = fig.add_subplot(2, 3, idx + 1)
        
        noise_data = []
        for protocol, prot_label in [
            ('levine_pichler', 'LP'),
            ('jandura_pupillo', 'JP-BB'),
            ('smooth_jp', 'JP-S')
        ]:
            result = run_simulation(
                protocol, "Rb87",
                spacing_factor=2.0,
                laser_power_1=p1,
                laser_power_2=p2,
                Delta_e=5e9,
                include_noise=True
            )
            
            if result['success'] and result['noise_breakdown']:
                nb = result['noise_breakdown']
                noise_data.append({
                    'label': prot_label,
                    'fidelity': result['fidelity'],
                    'gate_time_ns': result['gate_time_ns'],
                    'Rydberg decay': nb.get('gamma_r', 0) / (2*np.pi*1e3),
                    'Laser dephasing': nb.get('gamma_phi', 0) / (2*np.pi*1e3),
                    'Thermal motion': nb.get('gamma_phi_thermal', 0) / (2*np.pi*1e3),
                    'Intermediate scatter': nb.get('gamma_scatter_intermediate', 0) / (2*np.pi*1e3),
                })
                print(f"  {speed_label} {prot_label}: F={result['fidelity']*100:.1f}%, τ={result['gate_time_ns']:.0f}ns")
        
        if noise_data:
            x = np.arange(len(noise_data))
            width = 0.6
            
            noise_sources = ['Rydberg decay', 'Laser dephasing', 'Thermal motion', 'Intermediate scatter']
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
            
            bottom = np.zeros(len(noise_data))
            for source, color in zip(noise_sources, colors):
                values = [max(0, d.get(source, 0)) for d in noise_data]
                if max(values) > 0.001:
                    ax.bar(x, values, width, label=source, bottom=bottom, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                    bottom += np.array(values)
            
            for i, d in enumerate(noise_data):
                ax.text(i, bottom[i] + 0.5, f"F={d['fidelity']*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([d['label'] for d in noise_data])
            ax.set_ylabel('Noise Rate (kHz)')
            actual_time = noise_data[0]['gate_time_ns'] if noise_data else 0
            ax.set_title(f'{speed_label}\n(τ≈{actual_time:.0f}ns)')
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom row: Parameter sensitivity plots
    # Panel 4: Fidelity vs Temperature
    ax = fig.add_subplot(2, 3, 4)
    temps = np.logspace(np.log10(0.5e-6), np.log10(50e-6), 12)
    for protocol, color, label in [
        ('levine_pichler', 'blue', 'LP'),
        ('jandura_pupillo', 'orange', 'JP-BB'),
        ('smooth_jp', 'green', 'JP-S')
    ]:
        fids = []
        for temp in temps:
            r = run_simulation(protocol, 'Rb87', temperature=temp, spacing_factor=2.0,
                              laser_power_1=30e-6, laser_power_2=0.2, Delta_e=5e9)
            fids.append(r['fidelity']*100 if r['success'] else np.nan)
        ax.semilogx(temps*1e6, fids, 'o-', color=color, label=label, lw=2, markersize=6)
    ax.set_xlabel('Temperature (μK)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Temperature Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(90, 100)
    
    # Panel 5: Fidelity vs Laser Linewidth
    ax = fig.add_subplot(2, 3, 5)
    linewidths = np.logspace(1, 5, 12)
    for protocol, color, label in [
        ('levine_pichler', 'blue', 'LP'),
        ('jandura_pupillo', 'orange', 'JP-BB'),
        ('smooth_jp', 'green', 'JP-S')
    ]:
        fids = []
        for lw in linewidths:
            r = run_simulation(protocol, 'Rb87', linewidth_hz=lw, spacing_factor=2.0,
                              laser_power_1=30e-6, laser_power_2=0.2, Delta_e=5e9)
            fids.append(r['fidelity']*100 if r['success'] else np.nan)
        ax.semilogx(linewidths/1e3, fids, 'o-', color=color, label=label, lw=2, markersize=6)
    ax.set_xlabel('Laser Linewidth (kHz)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Laser Linewidth Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(85, 100)
    
    # Panel 6: Fidelity vs Intermediate Detuning
    ax = fig.add_subplot(2, 3, 6)
    delta_es = np.linspace(1e9, 10e9, 12)
    for protocol, color, label in [
        ('levine_pichler', 'blue', 'LP'),
        ('jandura_pupillo', 'orange', 'JP-BB'),
        ('smooth_jp', 'green', 'JP-S')
    ]:
        fids = []
        for de in delta_es:
            r = run_simulation(protocol, 'Rb87', Delta_e=de, spacing_factor=2.0,
                              laser_power_1=30e-6, laser_power_2=0.2)
            fids.append(r['fidelity']*100 if r['success'] else np.nan)
        ax.plot(delta_es/1e9, fids, 'o-', color=color, label=label, lw=2, markersize=6)
    ax.set_xlabel('Intermediate Detuning Δₑ (GHz)')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('Intermediate Detuning Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(85, 100)
    
    plt.tight_layout()
    plt.savefig('figures/noise_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved figures/noise_breakdown.png")


# =============================================================================
# FIGURE 4: SPECIES COMPARISON (DETAILED)
# =============================================================================

def generate_species_comparison_figure():
    """Compare species with detailed parameter optimization."""
    print("\n" + "="*60)
    print("FIGURE 4: Species Comparison")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    power_grid = [
        (100e-6, 0.5), (80e-6, 0.4), (60e-6, 0.35),
        (50e-6, 0.3), (40e-6, 0.25), (30e-6, 0.2),
        (25e-6, 0.18), (20e-6, 0.15), (15e-6, 0.12),
        (12e-6, 0.1), (10e-6, 0.08), (8e-6, 0.06),
    ]
    
    species_configs = [
        ('Rb87', 60, 'blue', 'o', 'Rb87 (n=60)'),
        ('Cs133', 50, 'red', 's', 'Cs133 (n=50)'),
    ]
    
    protocols = [
        ('levine_pichler', 'LP Protocol'),
        ('jandura_pupillo', 'JP Bang-Bang'),
        ('smooth_jp', 'JP Smooth')
    ]
    
    for ax, (protocol, prot_label) in zip(axes, protocols):
        print(f"\n{prot_label}:")
        
        for species, n_ryd, color, marker, label in species_configs:
            results = []
            for p1, p2 in power_grid:
                for delta_e in [3e9, 5e9, 7e9]:
                    for sf in [1.8, 2.0, 2.2]:
                        r = run_simulation(protocol, species, n_rydberg=n_ryd,
                                          laser_power_1=p1, laser_power_2=p2,
                                          Delta_e=delta_e, spacing_factor=sf)
                        if r['success'] and r['V_over_Omega'] > 5:
                            results.append(r)
            
            if results:
                times = [r['gate_time_ns'] for r in results]
                fids = [r['fidelity']*100 for r in results]
                ax.scatter(times, fids, c=color, alpha=0.2, s=20)
                
                # Envelope
                bins = np.arange(0, max(times)+100, 50)
                env_t, env_f = [], []
                for i in range(len(bins)-1):
                    in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                    if in_bin:
                        env_t.append((bins[i]+bins[i+1])/2)
                        env_f.append(max(in_bin))
                if env_t:
                    ax.plot(env_t, env_f, '-', color=color, lw=2.5, label=label)
                    ax.scatter(env_t, env_f, c=color, s=50, marker=marker, zorder=5, edgecolors='black', linewidths=0.5)
                    print(f"  {label}: {len(env_t)} points, best F={max(env_f):.1f}%")
        
        ax.set_xlabel('Gate Time (ns)', fontsize=12)
        ax.set_ylabel('Fidelity (%)', fontsize=12)
        ax.set_title(f'{prot_label}', fontsize=13)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 800)
        ax.set_ylim(85, 100.5)
        ax.grid(True, alpha=0.3)
        ax.axhline(99, color='gray', ls='--', alpha=0.5)
    
    plt.suptitle('CZ Gate Species Comparison (T=5μK, Strong Blockade)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/species_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved figures/species_comparison.png")


# =============================================================================
# FIGURE 5: PROTOCOL COMPARISON (OPTIMIZED)
# =============================================================================

def generate_protocol_comparison_figure():
    """Compare LP and JP protocols with proper optimization."""
    print("\n" + "="*60)
    print("FIGURE 5: Protocol Comparison (Optimized)")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Comprehensive optimization grid
    power_grid = [
        (100e-6, 0.5), (80e-6, 0.4), (60e-6, 0.35),
        (50e-6, 0.3), (40e-6, 0.25), (30e-6, 0.2),
        (25e-6, 0.18), (20e-6, 0.15), (15e-6, 0.12),
        (12e-6, 0.1), (10e-6, 0.08), (8e-6, 0.06),
        (6e-6, 0.05), (5e-6, 0.04), (4e-6, 0.03),
    ]
    
    for protocol, color, marker, label in [
        ('levine_pichler', 'blue', 'o', 'LP (Levine-Pichler)'),
        ('jandura_pupillo', 'orange', 's', 'JP Bang-Bang'),
        ('smooth_jp', 'green', '^', 'JP Smooth (Bluvstein)')
    ]:
        print(f"\nOptimizing {label}...")
        results = []
        
        # Scan over multiple parameters
        for p1, p2 in power_grid:
            for delta_e in [3e9, 4e9, 5e9, 6e9, 7e9]:
                for sf in [1.8, 2.0, 2.2, 2.5]:
                    r = run_simulation(protocol, 'Rb87', 
                                      laser_power_1=p1, laser_power_2=p2,
                                      Delta_e=delta_e, spacing_factor=sf)
                    if r['success'] and r['V_over_Omega'] > 5:
                        results.append(r)
        
        if results:
            times = [r['gate_time_ns'] for r in results]
            fids = [r['fidelity']*100 for r in results]
            
            # Scatter all points
            ax.scatter(times, fids, c=color, alpha=0.15, s=15)
            
            # Envelope (best fidelity in each time bin)
            bins = np.arange(0, min(max(times), 1500)+100, 40)
            env_t, env_f = [], []
            for i in range(len(bins)-1):
                in_bin = [fids[j] for j, t in enumerate(times) if bins[i] <= t < bins[i+1]]
                if in_bin:
                    env_t.append((bins[i]+bins[i+1])/2)
                    env_f.append(max(in_bin))
            
            if env_t:
                ax.plot(env_t, env_f, '-', color=color, lw=2.5, label=label)
                ax.scatter(env_t, env_f, c=color, s=60, marker=marker, zorder=5, 
                          edgecolors='black', linewidths=0.5)
                best_f = max(env_f)
                best_t = env_t[env_f.index(best_f)]
                print(f"  Best: F={best_f:.2f}% at τ={best_t:.0f}ns")
                print(f"  Range: τ={min(env_t):.0f}-{max(env_t):.0f}ns")
    
    ax.axhline(99, color='gray', ls='--', alpha=0.5, label='99% threshold')
    ax.axhline(99.5, color='gray', ls=':', alpha=0.5, label='99.5% threshold')
    
    ax.set_xlabel('Gate Time (ns)', fontsize=14)
    ax.set_ylabel('Fidelity (%)', fontsize=14)
    ax.set_title('CZ Gate Protocol Comparison\n(Rb87, T=5μK, Optimized Parameters)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1200)
    ax.set_ylim(85, 100.5)
    
    plt.tight_layout()
    plt.savefig('figures/protocol_comparison_optimal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved figures/protocol_comparison_optimal.png")


# =============================================================================
# FIGURE 6: TEMPERATURE SENSITIVITY (BOTH PROTOCOLS)
# =============================================================================

def generate_temperature_figure():
    """Show temperature sensitivity for all 3 protocols."""
    print("\n" + "="*60)
    print("FIGURE 6: Temperature Sensitivity (All Protocols)")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    temperatures = np.logspace(np.log10(0.5e-6), np.log10(100e-6), 15)
    
    power_configs = [
        (50e-6, 0.3, 'blue', 'Fast (~100ns)'),
        (20e-6, 0.15, 'green', 'Medium (~200ns)'),
        (10e-6, 0.08, 'orange', 'Slow (~400ns)'),
    ]
    
    protocols = [
        ('levine_pichler', 'LP Protocol'),
        ('jandura_pupillo', 'JP Bang-Bang'),
        ('smooth_jp', 'JP Smooth')
    ]
    
    for ax, (protocol, prot_label) in zip(axes, protocols):
        print(f"\n{prot_label}:")
        
        for p1, p2, color, speed_label in power_configs:
            results = []
            for temp in temperatures:
                # Optimize delta_e for each temperature
                best_fid = 0
                best_result = None
                for delta_e in [4e9, 5e9, 6e9]:
                    r = run_simulation(protocol, 'Rb87', temperature=temp,
                                      spacing_factor=2.0, laser_power_1=p1, laser_power_2=p2,
                                      Delta_e=delta_e)
                    if r['success'] and r['fidelity'] > best_fid:
                        best_fid = r['fidelity']
                        best_result = r
                
                if best_result:
                    results.append({
                        'temperature': temp,
                        'fidelity': best_result['fidelity'],
                        'gate_time_ns': best_result['gate_time_ns']
                    })
            
            if results:
                temps = [r['temperature'] * 1e6 for r in results]
                fids = [r['fidelity'] * 100 for r in results]
                gate_time = results[len(results)//2]['gate_time_ns']
                
                ax.semilogx(temps, fids, 'o-', color=color, lw=2, markersize=6,
                           label=f'{speed_label}, τ≈{gate_time:.0f}ns')
                print(f"  {speed_label}: F={min(fids):.1f}%-{max(fids):.1f}%")
        
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.7, label='Typical (5μK)')
        ax.axvline(x=1, color='gray', linestyle=':', alpha=0.7, label='State-of-art (1μK)')
        
        ax.set_xlabel('Temperature (μK)', fontsize=12)
        ax.set_ylabel('Fidelity (%)', fontsize=12)
        ax.set_title(f'{prot_label}', fontsize=13)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0.3, 150)
        ax.set_ylim(85, 100.5)
        ax.axhline(99, color='gray', ls='--', alpha=0.3)
    
    plt.suptitle('CZ Gate Temperature Sensitivity (Rb87, Strong Blockade)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/temperature_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved figures/temperature_sensitivity.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GENERATING PUBLICATION FIGURES FOR NEUTRAL ATOM CZ GATE SIMULATION")
    print("="*70)
    
    generate_pareto_figure()
    generate_blockade_figure()
    generate_noise_breakdown_figure()
    generate_species_comparison_figure()
    generate_protocol_comparison_figure()
    generate_temperature_figure()
    
    print("\n" + "="*70)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput files:")
    print("  - figures/pareto_true_fidelity_time.png (6-panel parameter study)")
    print("  - figures/blockade_importance.png (both protocols)")
    print("  - figures/noise_breakdown.png (side-by-side + sensitivity)")
    print("  - figures/species_comparison.png (both protocols)")
    print("  - figures/protocol_comparison_optimal.png (optimized)")
    print("  - figures/temperature_sensitivity.png (both protocols)")
