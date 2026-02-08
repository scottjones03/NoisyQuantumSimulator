"""
CZ Gate Visualization Tools
============================

Visualization functions for parameter space exploration and optimization results.

Key Functions
-------------
- plot_exploration_results(): Scatter plot of all evaluated points with Pareto front
- plot_pareto_comparison(): Compare Pareto fronts across protocols/conditions
- plot_parameter_heatmap(): 2D heatmap of fidelity vs parameter combinations
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Optional, Dict, Tuple, Any

from .optimization import ExplorationResult, EvaluatedPoint


def plot_exploration_results(
    result: ExplorationResult,
    ax: Optional[plt.Axes] = None,
    color_by: str = "V_over_Omega",
    show_pareto: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7),
    cmap: str = "viridis",
) -> plt.Axes:
    """
    Plot all evaluated points from parameter space exploration.
    
    Parameters
    ----------
    result : ExplorationResult
        Exploration results to visualize
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    color_by : str
        Property to color points by: "V_over_Omega", "laser_linewidth_kHz", "Omega_MHz"
    show_pareto : bool
        Whether to highlight Pareto frontier
    title : str, optional
        Plot title. Auto-generated if None.
    figsize : tuple
        Figure size (width, height) in inches
    cmap : str
        Matplotlib colormap name
        
    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    if not result.points:
        print("No points to plot!")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    fidelities = np.array([p.fidelity for p in result.points])
    gate_times = np.array([p.gate_time_ns for p in result.points])
    
    # Color values
    if color_by == "V_over_Omega":
        colors = np.array([p.V_over_Omega for p in result.points])
        clabel = "V/Ω"
    elif color_by == "laser_linewidth_kHz":
        colors = np.array([p.laser_linewidth_kHz for p in result.points])
        clabel = "Laser linewidth (kHz)"
    elif color_by == "Omega_MHz":
        colors = np.array([p.Omega_MHz for p in result.points])
        clabel = "Ω (MHz)"
    else:
        colors = np.ones(len(result.points))
        clabel = ""
    
    # Plot all points
    scatter = ax.scatter(
        gate_times, 
        fidelities * 100,
        c=colors,
        cmap=cmap,
        s=30,
        alpha=0.6,
        edgecolors='none',
    )
    
    # Add colorbar
    if color_by and len(np.unique(colors)) > 1:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(clabel, fontsize=11)
    
    # Highlight Pareto front
    if show_pareto and result.pareto_front:
        pareto_times = [p.gate_time_ns for p in result.pareto_front]
        pareto_fids = [p.fidelity * 100 for p in result.pareto_front]
        
        # Sort for line plot
        sorted_idx = np.argsort(pareto_times)
        pareto_times = np.array(pareto_times)[sorted_idx]
        pareto_fids = np.array(pareto_fids)[sorted_idx]
        
        ax.plot(pareto_times, pareto_fids, 'r-', linewidth=2, label='Pareto front')
        ax.scatter(pareto_times, pareto_fids, c='red', s=80, zorder=5, 
                   edgecolors='darkred', linewidths=1.5, marker='o')
    
    # Labels and formatting
    ax.set_xlabel("Gate Time (ns)", fontsize=12)
    ax.set_ylabel("Fidelity (%)", fontsize=12)
    
    if title is None:
        title = f"{result.protocol.upper()} - {result.n_evaluations} evaluations"
    ax.set_title(title, fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set reasonable y limits
    if len(fidelities) > 0:
        ymin = max(50, min(fidelities) * 100 - 5)
        ymax = min(100.5, max(fidelities) * 100 + 1)
        ax.set_ylim(ymin, ymax)
    
    if show_pareto:
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    return ax


def plot_pareto_comparison(
    results: List[ExplorationResult],
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 7),
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Compare Pareto fronts from multiple explorations.
    
    Parameters
    ----------
    results : list of ExplorationResult
        Multiple exploration results to compare
    labels : list of str, optional
        Labels for each result. Uses protocol name if None.
    ax : plt.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    colors : list of str, optional
        Colors for each result
        
    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [r.protocol for r in results]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for result, label, color in zip(results, labels, colors):
        if not result.pareto_front:
            continue
            
        # Sort Pareto front by gate time
        pareto_sorted = sorted(result.pareto_front, key=lambda p: p.gate_time_ns)
        times = [p.gate_time_ns for p in pareto_sorted]
        fids = [p.fidelity * 100 for p in pareto_sorted]
        
        # Plot line and points
        ax.plot(times, fids, '-', color=color, linewidth=2, label=label)
        ax.scatter(times, fids, c=[color], s=60, zorder=5, 
                   edgecolors='white', linewidths=1)
    
    ax.set_xlabel("Gate Time (ns)", fontsize=12)
    ax.set_ylabel("Fidelity (%)", fontsize=12)
    ax.set_title("Pareto Front Comparison", fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return ax


def plot_parameter_heatmap(
    result: ExplorationResult,
    x_param: str = "gate_time_ns",
    y_param: str = "Omega_MHz",
    z_param: str = "fidelity",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "RdYlGn",
    n_bins: int = 20,
) -> plt.Axes:
    """
    Create a 2D heatmap showing how a metric varies across parameter space.
    
    Parameters
    ----------
    result : ExplorationResult
        Exploration results
    x_param : str
        Parameter for x-axis
    y_param : str
        Parameter for y-axis
    z_param : str
        Parameter for color (e.g., "fidelity", "infidelity")
    ax : plt.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    n_bins : int
        Number of bins per axis
        
    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    if not result.points:
        print("No points to plot!")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    def get_param(point, name):
        if hasattr(point, name):
            return getattr(point, name)
        elif name in point.noise_breakdown:
            return point.noise_breakdown[name]
        return None
    
    x_vals = np.array([get_param(p, x_param) for p in result.points if get_param(p, x_param) is not None])
    y_vals = np.array([get_param(p, y_param) for p in result.points if get_param(p, y_param) is not None])
    z_vals = np.array([get_param(p, z_param) for p in result.points if get_param(p, z_param) is not None])
    
    if len(x_vals) == 0:
        print(f"Could not extract parameter '{x_param}'")
        return None
    
    # Create 2D histogram
    # Use the best (max) z value in each bin
    x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    y_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)
    
    z_grid = np.full((n_bins, n_bins), np.nan)
    
    for x, y, z in zip(x_vals, y_vals, z_vals):
        i = min(np.searchsorted(x_edges, x) - 1, n_bins - 1)
        j = min(np.searchsorted(y_edges, y) - 1, n_bins - 1)
        i = max(0, i)
        j = max(0, j)
        
        if np.isnan(z_grid[j, i]) or z > z_grid[j, i]:
            z_grid[j, i] = z
    
    # Plot
    if z_param == "fidelity":
        z_grid = z_grid * 100  # Convert to percentage
        zlabel = "Fidelity (%)"
    elif z_param == "infidelity":
        z_grid = z_grid * 100
        zlabel = "Infidelity (%)"
    else:
        zlabel = z_param
    
    im = ax.imshow(
        z_grid,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='auto',
        cmap=cmap,
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(zlabel, fontsize=11)
    
    # Labels
    param_labels = {
        "gate_time_ns": "Gate Time (ns)",
        "Omega_MHz": "Ω (MHz)",
        "laser_linewidth_kHz": "Laser Linewidth (kHz)",
        "V_over_Omega": "V/Ω",
        "fidelity": "Fidelity",
        "temperature": "Temperature (K)",
    }
    
    ax.set_xlabel(param_labels.get(x_param, x_param), fontsize=12)
    ax.set_ylabel(param_labels.get(y_param, y_param), fontsize=12)
    ax.set_title(f"{result.protocol.upper()}: {zlabel} Heatmap", fontsize=14)
    
    plt.tight_layout()
    return ax


def plot_noise_breakdown(
    points: List[EvaluatedPoint],
    top_n: int = 5,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Axes:
    """
    Show breakdown of noise sources across evaluated points.
    
    Parameters
    ----------
    points : list of EvaluatedPoint
        Points to analyze
    top_n : int
        Show top N points by fidelity
    ax : plt.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N points by fidelity
    sorted_points = sorted(points, key=lambda p: p.fidelity, reverse=True)[:top_n]
    
    # Collect noise breakdown
    all_sources = set()
    for p in sorted_points:
        all_sources.update(p.noise_breakdown.keys())
    
    all_sources = sorted(all_sources)
    
    # Build data matrix
    x = np.arange(len(sorted_points))
    width = 0.8 / len(all_sources) if all_sources else 0.8
    
    for i, source in enumerate(all_sources):
        values = [p.noise_breakdown.get(source, 0) * 100 for p in sorted_points]
        ax.bar(x + i * width, values, width, label=source)
    
    ax.set_xticks(x + width * len(all_sources) / 2)
    ax.set_xticklabels([f"F={p.fidelity*100:.1f}%" for p in sorted_points], rotation=45)
    ax.set_ylabel("Contribution (%)", fontsize=12)
    ax.set_title("Noise Breakdown for Top Points", fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return ax


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "plot_exploration_results",
    "plot_pareto_comparison",
    "plot_parameter_heatmap",
    "plot_noise_breakdown",
]
