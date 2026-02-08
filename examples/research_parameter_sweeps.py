#!/usr/bin/env python3
"""
Research Parameter Sweeps for Rydberg CZ Gates
===============================================

Comprehensive parameter sweeps comparing Levine-Pichler and Jandura-Pupillo protocols
for teaching physics students about the effects of various experimental parameters
on gate fidelity and gate time.

Generates publication-quality figures showing:
1. Temperature sweep (thermal dephasing)
2. Laser linewidth sweep (laser dephasing)
3. Intermediate detuning sweep (scattering)
4. Spacing factor sweep (blockade strength)
5. Rydberg n-value sweep (lifetime/blockade trade-off)
6. Species comparison (Rb87 vs Cs133)
7. Laser power sweep (gate speed)
8. Tweezer power sweep (trap depth)
9. Numerical aperture sweep
10. Pulse shape comparison
11. Protocol summary comparison
12. Noise breakdown analysis
13. Clock vs non-clock B-field comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Import simulation components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.simulation import simulate_CZ_gate
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.configurations import (
    LPSimulationInputs,
    JPSimulationInputs,
    TwoPhotonExcitationConfig,
    LaserParameters,
    NoiseSourceConfig,
)


@dataclass
class SweepResult:
    """Container for sweep results"""
    param_values: np.ndarray
    fidelities_lp: np.ndarray
    fidelities_jp: np.ndarray
    gate_times_lp: np.ndarray
    gate_times_jp: np.ndarray
    v_over_omega_lp: np.ndarray
    v_over_omega_jp: np.ndarray
    noise_breakdowns_lp: list
    noise_breakdowns_jp: list


# Default experimental parameters (good conditions)
DEFAULT_PARAMS = {
    'species': "Rb87",
    'n_rydberg': 70,
    'temperature': 20e-6,  # 20 μK in Kelvin
    'laser_linewidth_hz': 100.0,
    'Delta_e': 2 * np.pi * 1e9,  # 1 GHz intermediate detuning
    'spacing_factor': 1.5,
    'rydberg_power_1': 2.5e-3,   # First leg laser power (W)
    'rydberg_power_2': 1.0,      # Second leg laser power (W)
    'tweezer_power': 10e-3,  # 10 mW
    'NA': 0.5,
    'B_field': 0.0,  # No magnetic field for clock states
    'pulse_shape': "square",
    'include_noise': True,
    'verbose': False,
}


def run_single_simulation(protocol: str, **kwargs):
    """Run a single simulation and return results"""
    # Merge with defaults
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    
    try:
        # Build laser parameters
        laser_1 = LaserParameters(
            power=params.get('rydberg_power_1', 2.5e-3),
            waist=1.0e-6,
            linewidth_hz=params.get('laser_linewidth_hz', 100.0),
        )
        laser_2 = LaserParameters(
            power=params.get('rydberg_power_2', 1.0),
            waist=10e-6,
            linewidth_hz=params.get('laser_linewidth_hz', 100.0),
        )
        excitation = TwoPhotonExcitationConfig(
            laser_1=laser_1,
            laser_2=laser_2,
            Delta_e=params.get('Delta_e', None),
        )
        noise = NoiseSourceConfig(include_motional_dephasing=True)
        
        # Create protocol-specific inputs
        if protocol.lower() in ("levine_pichler", "lp"):
            simulation_inputs = LPSimulationInputs(
                excitation=excitation,
                noise=noise,
                pulse_shape=params.get('pulse_shape', 'time_optimal'),
            )
        else:
            simulation_inputs = JPSimulationInputs(
                excitation=excitation,
                noise=noise,
            )
        
        result = simulate_CZ_gate(
            simulation_inputs=simulation_inputs,
            species=params.get('species', 'Rb87'),
            n_rydberg=params.get('n_rydberg', 70),
            temperature=params.get('temperature', 5e-6),
            spacing_factor=params.get('spacing_factor', 3.0),
            tweezer_power=params.get('tweezer_power', 30e-3),
            tweezer_waist=params.get('tweezer_waist', 1e-6),
            B_field=params.get('B_field', 1e-4),
            NA=params.get('NA', 0.5),
            include_noise=params.get('include_noise', True),
            verbose=params.get('verbose', False),
        )
        return result
    except Exception as e:
        print(f"  Warning: {protocol} simulation failed: {e}")
        return None


def run_sweep(param_name: str, param_values: np.ndarray, **fixed_kwargs) -> SweepResult:
    """Run parameter sweep for both protocols"""
    
    n = len(param_values)
    fidelities_lp = np.zeros(n)
    fidelities_jp = np.zeros(n)
    gate_times_lp = np.zeros(n)
    gate_times_jp = np.zeros(n)
    v_over_omega_lp = np.zeros(n)
    v_over_omega_jp = np.zeros(n)
    noise_breakdowns_lp = []
    noise_breakdowns_jp = []
    
    print(f"\nSweeping {param_name} ({n} points)...")
    
    for i, val in enumerate(param_values):
        kwargs = fixed_kwargs.copy()
        kwargs[param_name] = val
        
        # Levine-Pichler
        result_lp = run_single_simulation("levine_pichler", **kwargs)
        if result_lp:
            fidelities_lp[i] = result_lp.avg_fidelity
            gate_times_lp[i] = result_lp.gate_time_us
            v_over_omega_lp[i] = result_lp.V_over_Omega
            noise_breakdowns_lp.append(result_lp.noise_breakdown if hasattr(result_lp, 'noise_breakdown') else {})
        else:
            fidelities_lp[i] = np.nan
            gate_times_lp[i] = np.nan
            v_over_omega_lp[i] = np.nan
            noise_breakdowns_lp.append({})
        
        # Jandura-Pupillo
        result_jp = run_single_simulation("jandura_pupillo", **kwargs)
        if result_jp:
            fidelities_jp[i] = result_jp.avg_fidelity
            gate_times_jp[i] = result_jp.gate_time_us
            v_over_omega_jp[i] = result_jp.V_over_Omega
            noise_breakdowns_jp.append(result_jp.noise_breakdown if hasattr(result_jp, 'noise_breakdown') else {})
        else:
            fidelities_jp[i] = np.nan
            gate_times_jp[i] = np.nan
            v_over_omega_jp[i] = np.nan
            noise_breakdowns_jp.append({})
        
        print(f"  [{i+1}/{n}] {param_name}={val:.4g}: LP={fidelities_lp[i]:.4f}, JP={fidelities_jp[i]:.4f}")
    
    return SweepResult(
        param_values=param_values,
        fidelities_lp=fidelities_lp,
        fidelities_jp=fidelities_jp,
        gate_times_lp=gate_times_lp,
        gate_times_jp=gate_times_jp,
        v_over_omega_lp=v_over_omega_lp,
        v_over_omega_jp=v_over_omega_jp,
        noise_breakdowns_lp=noise_breakdowns_lp,
        noise_breakdowns_jp=noise_breakdowns_jp
    )


def plot_combined_sweep(
    result: SweepResult,
    param_name: str,
    param_label: str,
    title: str,
    output_path: Path,
    log_x: bool = False,
    log_y_fid: bool = False,
    infidelity: bool = True,
    x_scale: float = 1.0
):
    """Create 4-panel figure for parameter sweep"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    x = result.param_values * x_scale
    
    # Panel (a): Fidelity vs parameter
    ax = axes[0, 0]
    if infidelity:
        y_lp = 1 - result.fidelities_lp
        y_jp = 1 - result.fidelities_jp
        ylabel = "Infidelity (1 - F)"
    else:
        y_lp = result.fidelities_lp
        y_jp = result.fidelities_jp
        ylabel = "Fidelity"
    
    ax.plot(x, y_lp, 'b-o', label='Levine-Pichler', markersize=6, linewidth=2)
    ax.plot(x, y_jp, 'r-s', label='Jandura-Pupillo', markersize=6, linewidth=2)
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('(a) Gate Fidelity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale('log')
    if log_y_fid or infidelity:
        ax.set_yscale('log')
    
    # Panel (b): Gate time vs parameter
    ax = axes[0, 1]
    ax.plot(x, result.gate_times_lp, 'b-o', label='Levine-Pichler', markersize=6, linewidth=2)
    ax.plot(x, result.gate_times_jp, 'r-s', label='Jandura-Pupillo', markersize=6, linewidth=2)
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel('Gate Time (μs)', fontsize=11)
    ax.set_title('(b) Gate Duration', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale('log')
    
    # Panel (c): V/Ω vs parameter
    ax = axes[1, 0]
    ax.plot(x, result.v_over_omega_lp, 'b-o', label='Levine-Pichler', markersize=6, linewidth=2)
    ax.plot(x, result.v_over_omega_jp, 'r-s', label='Jandura-Pupillo', markersize=6, linewidth=2)
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel('V/Ω (blockade ratio)', fontsize=11)
    ax.set_title('(c) Blockade Strength', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale('log')
    
    # Panel (d): Fidelity vs Gate time trade-off
    ax = axes[1, 1]
    ax.scatter(result.gate_times_lp, result.fidelities_lp, c='blue', s=60, 
               label='Levine-Pichler', marker='o', alpha=0.7)
    ax.scatter(result.gate_times_jp, result.fidelities_jp, c='red', s=60,
               label='Jandura-Pupillo', marker='s', alpha=0.7)
    
    # Add annotations to show parameter value
    for i, val in enumerate(x):
        ax.annotate(f'{val:.2g}', (result.gate_times_lp[i], result.fidelities_lp[i]),
                   fontsize=7, alpha=0.6)
        ax.annotate(f'{val:.2g}', (result.gate_times_jp[i], result.fidelities_jp[i]),
                   fontsize=7, alpha=0.6)
    
    ax.set_xlabel('Gate Time (μs)', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('(d) Fidelity-Time Trade-off', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_noise_breakdown(
    result: SweepResult,
    param_name: str,
    param_label: str,
    title: str,
    output_path: Path,
    log_x: bool = False,
    x_scale: float = 1.0
):
    """Create noise breakdown stacked area plot"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title} - Noise Breakdown", fontsize=14, fontweight='bold')
    
    x = result.param_values * x_scale
    
    # Get all noise types
    all_noise_types = set()
    for nb in result.noise_breakdowns_lp + result.noise_breakdowns_jp:
        all_noise_types.update(nb.keys())
    all_noise_types = sorted(all_noise_types)
    
    if not all_noise_types:
        print(f"  No noise breakdown data available for {title}")
        plt.close()
        return
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_noise_types)))
    
    for ax_idx, (ax, breakdowns, protocol) in enumerate([
        (axes[0], result.noise_breakdowns_lp, "Levine-Pichler"),
        (axes[1], result.noise_breakdowns_jp, "Jandura-Pupillo")
    ]):
        # Build data arrays
        data = {nt: np.zeros(len(x)) for nt in all_noise_types}
        for i, nb in enumerate(breakdowns):
            for nt, val in nb.items():
                data[nt][i] = val
        
        # Stack plot
        bottom = np.zeros(len(x))
        for nt, color in zip(all_noise_types, colors):
            ax.fill_between(x, bottom, bottom + data[nt], label=nt, color=color, alpha=0.8)
            bottom += data[nt]
        
        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel('Infidelity Contribution', fontsize=11)
        ax.set_title(f'{protocol}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        if log_x:
            ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_species_comparison(results_rb: SweepResult, results_cs: SweepResult, 
                           param_name: str, param_label: str, output_path: Path,
                           x_scale: float = 1.0):
    """Compare Rb87 vs Cs133 performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Species Comparison: Rb87 vs Cs133', fontsize=14, fontweight='bold')
    
    x = results_rb.param_values * x_scale
    
    # Fidelity comparison
    ax = axes[0, 0]
    ax.plot(x, 1 - results_rb.fidelities_lp, 'b-o', label='Rb87 LP', markersize=5)
    ax.plot(x, 1 - results_rb.fidelities_jp, 'b--s', label='Rb87 JP', markersize=5)
    ax.plot(x, 1 - results_cs.fidelities_lp, 'r-o', label='Cs133 LP', markersize=5)
    ax.plot(x, 1 - results_cs.fidelities_jp, 'r--s', label='Cs133 JP', markersize=5)
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel('Infidelity', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('(a) Infidelity Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Gate time comparison
    ax = axes[0, 1]
    ax.plot(x, results_rb.gate_times_lp, 'b-o', label='Rb87 LP', markersize=5)
    ax.plot(x, results_rb.gate_times_jp, 'b--s', label='Rb87 JP', markersize=5)
    ax.plot(x, results_cs.gate_times_lp, 'r-o', label='Cs133 LP', markersize=5)
    ax.plot(x, results_cs.gate_times_jp, 'r--s', label='Cs133 JP', markersize=5)
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel('Gate Time (μs)', fontsize=11)
    ax.set_title('(b) Gate Time Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # V/Ω comparison
    ax = axes[1, 0]
    ax.plot(x, results_rb.v_over_omega_lp, 'b-o', label='Rb87 LP', markersize=5)
    ax.plot(x, results_rb.v_over_omega_jp, 'b--s', label='Rb87 JP', markersize=5)
    ax.plot(x, results_cs.v_over_omega_lp, 'r-o', label='Cs133 LP', markersize=5)
    ax.plot(x, results_cs.v_over_omega_jp, 'r--s', label='Cs133 JP', markersize=5)
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel('V/Ω', fontsize=11)
    ax.set_title('(c) Blockade Ratio Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Trade-off comparison
    ax = axes[1, 1]
    ax.scatter(results_rb.gate_times_lp, results_rb.fidelities_lp, c='blue', s=50, 
               label='Rb87 LP', marker='o', alpha=0.7)
    ax.scatter(results_rb.gate_times_jp, results_rb.fidelities_jp, c='blue', s=50,
               label='Rb87 JP', marker='s', alpha=0.7)
    ax.scatter(results_cs.gate_times_lp, results_cs.fidelities_lp, c='red', s=50,
               label='Cs133 LP', marker='o', alpha=0.7)
    ax.scatter(results_cs.gate_times_jp, results_cs.fidelities_jp, c='red', s=50,
               label='Cs133 JP', marker='s', alpha=0.7)
    ax.set_xlabel('Gate Time (μs)', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('(d) Fidelity-Time Trade-off', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pulse_shape_comparison(output_path: Path):
    """Compare different pulse shapes"""
    
    pulse_shapes = ['square', 'gaussian', 'blackman']
    protocols = ['levine_pichler', 'jandura_pupillo']
    
    results = {ps: {} for ps in pulse_shapes}
    
    print("\nPulse shape comparison...")
    for ps in pulse_shapes:
        for protocol in protocols:
            result = run_single_simulation(protocol, pulse_shape=ps)
            if result:
                results[ps][protocol] = {
                    'fidelity': result.avg_fidelity,
                    'gate_time': result.gate_time_us,
                    'v_over_omega': result.V_over_Omega,
                    'noise': result.noise_breakdown if hasattr(result, 'noise_breakdown') else {}
                }
                print(f"  {ps}/{protocol}: F={result.avg_fidelity:.4f}, t={result.gate_time_us:.3f}μs")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Pulse Shape Comparison', fontsize=14, fontweight='bold')
    
    x = np.arange(len(pulse_shapes))
    width = 0.35
    
    # Fidelity
    ax = axes[0]
    fid_lp = [results[ps].get('levine_pichler', {}).get('fidelity', 0) for ps in pulse_shapes]
    fid_jp = [results[ps].get('jandura_pupillo', {}).get('fidelity', 0) for ps in pulse_shapes]
    ax.bar(x - width/2, fid_lp, width, label='Levine-Pichler', color='blue', alpha=0.7)
    ax.bar(x + width/2, fid_jp, width, label='Jandura-Pupillo', color='red', alpha=0.7)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(pulse_shapes)
    ax.set_title('(a) Fidelity', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.90, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Gate time
    ax = axes[1]
    time_lp = [results[ps].get('levine_pichler', {}).get('gate_time', 0) for ps in pulse_shapes]
    time_jp = [results[ps].get('jandura_pupillo', {}).get('gate_time', 0) for ps in pulse_shapes]
    ax.bar(x - width/2, time_lp, width, label='Levine-Pichler', color='blue', alpha=0.7)
    ax.bar(x + width/2, time_jp, width, label='Jandura-Pupillo', color='red', alpha=0.7)
    ax.set_ylabel('Gate Time (μs)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(pulse_shapes)
    ax.set_title('(b) Gate Time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Noise breakdown (stacked bar)
    ax = axes[2]
    all_noise_types = set()
    for ps in pulse_shapes:
        for protocol in protocols:
            if protocol in results[ps] and 'noise' in results[ps][protocol]:
                all_noise_types.update(results[ps][protocol]['noise'].keys())
    all_noise_types = sorted(all_noise_types)
    
    if all_noise_types:
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_noise_types)))
        
        # Just show LP noise breakdown
        bottom = np.zeros(len(pulse_shapes))
        for nt, color in zip(all_noise_types, colors):
            vals = []
            for ps in pulse_shapes:
                nb = results[ps].get('levine_pichler', {}).get('noise', {})
                vals.append(nb.get(nt, 0))
            ax.bar(x, vals, width*2, bottom=bottom, label=nt, color=color, alpha=0.8)
            bottom += np.array(vals)
        
        ax.set_ylabel('Infidelity Contribution', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(pulse_shapes)
        ax.set_title('(c) Noise Breakdown (LP)', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_bfield_comparison(output_path: Path):
    """Compare clock vs non-clock states at different B-fields"""
    
    b_fields = np.array([0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    
    print("\nB-field / clock state comparison...")
    
    results_clock = {'lp': [], 'jp': []}
    results_nonclock = {'lp': [], 'jp': []}
    
    # For clock states, use (1,0) and (2,0)
    # For non-clock states, use (1,1) and (2,1)
    for b in b_fields:
        # Clock states (default)
        res_lp = run_single_simulation("levine_pichler", B_field=b, 
                                       qubit_0=(1, 0), qubit_1=(2, 0))
        res_jp = run_single_simulation("jandura_pupillo", B_field=b,
                                       qubit_0=(1, 0), qubit_1=(2, 0))
        results_clock['lp'].append(res_lp.avg_fidelity if res_lp else np.nan)
        results_clock['jp'].append(res_jp.avg_fidelity if res_jp else np.nan)
        
        # Non-clock states
        res_lp = run_single_simulation("levine_pichler", B_field=b,
                                       qubit_0=(1, 1), qubit_1=(2, 1))
        res_jp = run_single_simulation("jandura_pupillo", B_field=b,
                                       qubit_0=(1, 1), qubit_1=(2, 1))
        results_nonclock['lp'].append(res_lp.avg_fidelity if res_lp else np.nan)
        results_nonclock['jp'].append(res_jp.avg_fidelity if res_jp else np.nan)
        
        print(f"  B={b:.1e}: clock LP={results_clock['lp'][-1]:.4f}, non-clock LP={results_nonclock['lp'][-1]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Clock vs Non-Clock States: B-Field Sensitivity', fontsize=14, fontweight='bold')
    
    b_fields_mG = b_fields * 1e4  # Convert to milliGauss for display
    
    # Fidelity vs B-field
    ax = axes[0]
    ax.plot(b_fields_mG, results_clock['lp'], 'b-o', label='Clock LP', markersize=6)
    ax.plot(b_fields_mG, results_clock['jp'], 'b--s', label='Clock JP', markersize=6)
    ax.plot(b_fields_mG, results_nonclock['lp'], 'r-o', label='Non-clock LP', markersize=6)
    ax.plot(b_fields_mG, results_nonclock['jp'], 'r--s', label='Non-clock JP', markersize=6)
    ax.set_xlabel('B-field (mG)', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('(a) Fidelity vs B-field', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Infidelity (log scale)
    ax = axes[1]
    ax.semilogy(b_fields_mG, 1 - np.array(results_clock['lp']), 'b-o', label='Clock LP', markersize=6)
    ax.semilogy(b_fields_mG, 1 - np.array(results_clock['jp']), 'b--s', label='Clock JP', markersize=6)
    ax.semilogy(b_fields_mG, 1 - np.array(results_nonclock['lp']), 'r-o', label='Non-clock LP', markersize=6)
    ax.semilogy(b_fields_mG, 1 - np.array(results_nonclock['jp']), 'r--s', label='Non-clock JP', markersize=6)
    ax.set_xlabel('B-field (mG)', fontsize=11)
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_title('(b) Infidelity vs B-field (log scale)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_protocol_summary(output_path: Path):
    """Summary comparison of both protocols under various conditions"""
    
    print("\nProtocol summary...")
    
    conditions = [
        ("Ideal", {}),
        ("Hot (100μK)", {"temperature": 100e-6}),
        ("Noisy laser (1kHz)", {"laser_linewidth_hz": 1000}),
        ("Close spacing (1.2x)", {"spacing_factor": 1.2}),
        ("Low n (50)", {"n_rydberg": 50}),
        ("High n (90)", {"n_rydberg": 90}),
        ("Cs133", {"species": "Cs133"}),
    ]
    
    results = []
    for name, kwargs in conditions:
        res_lp = run_single_simulation("levine_pichler", **kwargs)
        res_jp = run_single_simulation("jandura_pupillo", **kwargs)
        results.append({
            'name': name,
            'lp_fid': res_lp.avg_fidelity if res_lp else np.nan,
            'lp_time': res_lp.gate_time_us if res_lp else np.nan,
            'jp_fid': res_jp.avg_fidelity if res_jp else np.nan,
            'jp_time': res_jp.gate_time_us if res_jp else np.nan,
        })
        print(f"  {name}: LP={results[-1]['lp_fid']:.4f}, JP={results[-1]['jp_fid']:.4f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Protocol Comparison Summary', fontsize=14, fontweight='bold')
    
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.35
    
    # Fidelity
    ax = axes[0]
    fid_lp = [r['lp_fid'] for r in results]
    fid_jp = [r['jp_fid'] for r in results]
    ax.bar(x - width/2, fid_lp, width, label='Levine-Pichler', color='blue', alpha=0.7)
    ax.bar(x + width/2, fid_jp, width, label='Jandura-Pupillo', color='red', alpha=0.7)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_title('(a) Fidelity Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='99% threshold')
    
    # Gate time
    ax = axes[1]
    time_lp = [r['lp_time'] for r in results]
    time_jp = [r['jp_time'] for r in results]
    ax.bar(x - width/2, time_lp, width, label='Levine-Pichler', color='blue', alpha=0.7)
    ax.bar(x + width/2, time_jp, width, label='Jandura-Pupillo', color='red', alpha=0.7)
    ax.set_ylabel('Gate Time (μs)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_title('(b) Gate Time Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Run all parameter sweeps and generate figures"""
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "figures" / "parameter_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Research Parameter Sweeps for Rydberg CZ Gates")
    print("=" * 60)
    
    # 1. Temperature sweep (convert μK to K)
    print("\n" + "=" * 40)
    print("1. Temperature Sweep (Thermal Dephasing)")
    print("=" * 40)
    temps_uK = np.array([5, 10, 20, 40, 80, 150])
    temps_K = temps_uK * 1e-6
    result_temp = run_sweep("temperature", temps_K)
    # Scale back to μK for plotting
    result_temp.param_values = temps_uK
    plot_combined_sweep(
        result_temp, "temperature", "Temperature (μK)",
        "Effect of Temperature on CZ Gate Performance",
        output_dir / "01_temperature_sweep.png"
    )
    plot_noise_breakdown(
        result_temp, "temperature", "Temperature (μK)",
        "Temperature Sweep",
        output_dir / "01_temperature_noise.png"
    )
    
    # 2. Laser linewidth sweep
    print("\n" + "=" * 40)
    print("2. Laser Linewidth Sweep (Laser Dephasing)")
    print("=" * 40)
    linewidths = np.array([10, 50, 100, 500, 1000, 5000])
    result_lw = run_sweep("laser_linewidth_hz", linewidths)
    plot_combined_sweep(
        result_lw, "laser_linewidth_hz", "Laser Linewidth (Hz)",
        "Effect of Laser Linewidth on CZ Gate Performance",
        output_dir / "02_laser_linewidth_sweep.png",
        log_x=True
    )
    plot_noise_breakdown(
        result_lw, "laser_linewidth_hz", "Laser Linewidth (Hz)",
        "Laser Linewidth Sweep",
        output_dir / "02_laser_linewidth_noise.png",
        log_x=True
    )
    
    # 3. Intermediate detuning sweep
    print("\n" + "=" * 40)
    print("3. Intermediate Detuning Sweep (Scattering)")
    print("=" * 40)
    detunings_GHz = np.array([0.5, 1, 2, 5, 10])
    detunings = 2 * np.pi * detunings_GHz * 1e9
    result_det = run_sweep("Delta_e", detunings)
    # Scale to GHz for plotting
    result_det.param_values = detunings_GHz
    plot_combined_sweep(
        result_det, "Delta_e", "Δₑ / 2π (GHz)",
        "Effect of Intermediate Detuning on CZ Gate Performance",
        output_dir / "03_detuning_sweep.png"
    )
    
    # 4. Spacing factor sweep
    print("\n" + "=" * 40)
    print("4. Spacing Factor Sweep (Blockade Strength)")
    print("=" * 40)
    spacings = np.array([1.2, 1.5, 2.0, 2.5, 3.0])
    result_space = run_sweep("spacing_factor", spacings)
    plot_combined_sweep(
        result_space, "spacing_factor", "Spacing Factor (r/r_blockade)",
        "Effect of Atom Spacing on CZ Gate Performance",
        output_dir / "04_spacing_sweep.png"
    )
    
    # 5. Rydberg n-value sweep
    print("\n" + "=" * 40)
    print("5. Rydberg n-Value Sweep (Lifetime vs Blockade)")
    print("=" * 40)
    n_values = np.array([50, 60, 70, 80, 90])
    result_n = run_sweep("n_rydberg", n_values)
    plot_combined_sweep(
        result_n, "n_rydberg", "Principal Quantum Number n",
        "Effect of Rydberg State n on CZ Gate Performance",
        output_dir / "05_n_rydberg_sweep.png"
    )
    
    # 6. Laser power sweep
    print("\n" + "=" * 40)
    print("6. Laser Power Sweep (Gate Speed)")
    print("=" * 40)
    powers = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    result_power = run_sweep("rydberg_power_2", powers)
    plot_combined_sweep(
        result_power, "rydberg_power_2", "Laser Power (W)",
        "Effect of Laser Power on CZ Gate Performance",
        output_dir / "06_laser_power_sweep.png",
        log_x=True
    )
    
    # 7. Tweezer power sweep (convert mW to W)
    print("\n" + "=" * 40)
    print("7. Tweezer Power Sweep (Trap Depth)")
    print("=" * 40)
    tweezer_powers_mW = np.array([1, 3, 5, 10, 20, 50])
    tweezer_powers_W = tweezer_powers_mW * 1e-3
    result_tw = run_sweep("tweezer_power", tweezer_powers_W)
    result_tw.param_values = tweezer_powers_mW
    plot_combined_sweep(
        result_tw, "tweezer_power", "Tweezer Power (mW)",
        "Effect of Tweezer Power on CZ Gate Performance",
        output_dir / "07_tweezer_power_sweep.png",
        log_x=True
    )
    
    # 8. Numerical aperture sweep
    print("\n" + "=" * 40)
    print("8. Numerical Aperture Sweep (Beam Waist)")
    print("=" * 40)
    NAs = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    result_NA = run_sweep("NA", NAs)
    plot_combined_sweep(
        result_NA, "NA", "Numerical Aperture",
        "Effect of Numerical Aperture on CZ Gate Performance",
        output_dir / "08_NA_sweep.png"
    )
    
    # 9. Species comparison (temperature sweep for both)
    print("\n" + "=" * 40)
    print("9. Species Comparison (Rb87 vs Cs133)")
    print("=" * 40)
    temps_species_uK = np.array([10, 20, 40, 80])
    temps_species_K = temps_species_uK * 1e-6
    result_rb = run_sweep("temperature", temps_species_K, species="Rb87")
    result_cs = run_sweep("temperature", temps_species_K, species="Cs133")
    result_rb.param_values = temps_species_uK
    result_cs.param_values = temps_species_uK
    plot_species_comparison(
        result_rb, result_cs, "temperature", "Temperature (μK)",
        output_dir / "09_species_comparison.png"
    )
    
    # 10. Pulse shape comparison
    print("\n" + "=" * 40)
    print("10. Pulse Shape Comparison")
    print("=" * 40)
    plot_pulse_shape_comparison(output_dir / "10_pulse_shape_comparison.png")
    
    # 11. B-field / clock state comparison
    print("\n" + "=" * 40)
    print("11. B-Field / Clock State Comparison")
    print("=" * 40)
    plot_bfield_comparison(output_dir / "11_bfield_clock_comparison.png")
    
    # 12. Protocol summary
    print("\n" + "=" * 40)
    print("12. Protocol Summary")
    print("=" * 40)
    plot_protocol_summary(output_dir / "12_protocol_summary.png")
    
    print("\n" + "=" * 60)
    print("All figures saved to:", output_dir)
    print("=" * 60)
    
    # List all generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
