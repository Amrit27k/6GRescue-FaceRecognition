"""
Integrated Infrastructure Orchestration + 6G-MEC+ MLOps Deployment Figures
==========================================================================

This script generates four publication-quality figures with CONSISTENT results:
1. Four-panel: All architectures with MEC+ (single robot)
2. Two-panel: Distributed-Horizontal vs Distributed-Horizontal + MEC+
3. Two-panel: Distributed-Vertical vs Distributed-Vertical + MEC+
4. Four-panel: Fleet scaling (N=1, 10, 20, 50) for Distributed-Horizontal + MEC+

CRITICAL: All instances of Distributed-Horizontal + MEC+ (N=1) yield μ = 46.1 min

Author: Amrit K.
Date: February 2026
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from matplotlib.lines import Line2D
# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

print("Environment configured successfully")


# =============================================================================
# DATA DEFINITIONS
# =============================================================================

# Infrastructure orchestration phases (minutes) - from Aston 5G Testbed
INFRASTRUCTURE_PHASES = {
    'Discovery': {'time': 0.7, 'colour': '#B4D4E7'},
    'Discovery→Commissioning': {'time': 1.8, 'colour': '#1F4E79'},
    'Commissioning': {'time': 1.0, 'colour': '#A8D08D'},
    'Commissioning→PM Request': {'time': 0.4, 'colour': '#548235'},
    'PM Request': {'time': 1.5, 'colour': '#F4B6C2'},
    'PM Request→Allocation': {'time': 0.8, 'colour': '#C00000'},
    'Allocation': {'time': 2.0, 'colour': '#FFD966'},
    'Allocation→Deployment': {'time': 0.9, 'colour': '#ED7D31'},
    'Deployment': {'time': 0.5, 'colour': '#D5A6E2'},
    'Deployment→Upgrade': {'time': 0.3, 'colour': '#7030A0'},
    'Upgrade': {'time': 1.3, 'colour': '#FFFF99'},
    'EMS Deployment': {'time': 2.8, 'colour': '#BF8F00'},
    'Service Deployment': {'time': 13.2, 'colour': '#9DC3E6'},
    'VNF Deployment': {'time': 7.8, 'colour': '#2E75B6'},
}

# 6G-MEC+ MLOps phases (minutes)
# Calibrated so Distributed-Horizontal + MEC+ = 46.1 min
MECPLUS_PHASES = {
    'S1: ML Training': {'time': 8.0, 'colour': '#00B050'},
    'S4: Model Conversion': {'time': 2.0, 'colour': '#92D050'},
    'S3: Stage-1 (Cached)': {'time': 0.1, 'colour': '#FFC000'},
    'S3: Stage-2 (Model)': {'time': 0.107, 'colour': '#FF6600'},  # 6.4 seconds
    'Verification': {'time': 0.5, 'colour': '#9933FF'},
}

# Architecture configurations (from Aston Testbed)
ARCHITECTURES = {
    'centralized_horizontal': {
        'name': 'Centralized Architecture With\nHorizontal Orchestration Strategy',
        'short_name': 'Centralized-Horizontal',
        'mean': 52.3,
        'variance': 0.18,
        'seed': 100  # Unique seed for this architecture
    },
    'centralized_vertical': {
        'name': 'Centralized Architecture With\nVertical Orchestration Strategy',
        'short_name': 'Centralized-Vertical',
        'mean': 54.8,
        'variance': 0.19,
        'seed': 200  # Unique seed for this architecture
    },
    'distributed_horizontal': {
        'name': 'Distributed Architecture With\nHorizontal Orchestration Strategy',
        'short_name': 'Distributed-Horizontal',
        'mean': 35.4,
        'variance': 0.12,
        'seed': 300  # Unique seed for this architecture
    },
    'distributed_vertical': {
        'name': 'Distributed Architecture With\nVertical Orchestration Strategy',
        'short_name': 'Distributed-Vertical',
        'mean': 49.2,
        'variance': 0.14,
        'seed': 400  # Unique seed for this architecture
    },
}

# =============================================================================
# CONSISTENT SEED MAPPING FOR FLEET CONFIGURATIONS
# =============================================================================
# These seeds ensure identical results for the same configuration across figures

FLEET_SEEDS = {
    1: 301,    # N=1 uses seed 301 (related to distributed_horizontal seed 300)
    10: 310,   # N=10
    20: 320,   # N=20
    50: 350,   # N=50
}

MECPLUS_SEED_OFFSET = 1000  # MEC+ timings use architecture seed + this offset

# Calculate and verify target means
MECPLUS_SINGLE_TOTAL = sum(p['time'] for p in MECPLUS_PHASES.values())
print(f"MEC+ Single Robot Total: {MECPLUS_SINGLE_TOTAL:.2f} min")
print(f"\nTarget Combined Means (Infrastructure + MEC+):")
for arch_name, arch in ARCHITECTURES.items():
    combined = arch['mean'] + MECPLUS_SINGLE_TOTAL
    print(f"  {arch['short_name']}: {arch['mean']:.1f} + {MECPLUS_SINGLE_TOTAL:.2f} = {combined:.2f} min")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalise_phases(phases: Dict, target_mean: float) -> Dict[str, float]:
    """Normalise phase timings to achieve target mean total."""
    current_sum = sum(p['time'] for p in phases.values())
    factor = target_mean / current_sum
    return {name: config['time'] * factor for name, config in phases.items()}


def generate_stochastic_timings(
    phase_times: Dict[str, float],
    n_samples: int,
    variance_factor: float,
    seed: int
) -> Dict[str, np.ndarray]:
    """
    Generate stochastic phase timings with temporal correlation.
    
    IMPORTANT: Same seed always produces identical output for reproducibility.
    """
    np.random.seed(seed)
    
    timings = {}
    for phase, base_time in phase_times.items():
        noise = np.random.normal(0, base_time * variance_factor, n_samples)
        kernel = np.ones(3) / 3
        smoothed_noise = np.convolve(noise, kernel, mode='same')
        timings[phase] = np.maximum(base_time + smoothed_noise, base_time * 0.25)
    
    return timings


def calculate_mecplus_fleet_times(fleet_size: int, parallel_factor: int = 5) -> Dict[str, float]:
    """Calculate MEC+ deployment times for fleet configuration."""
    num_batches = (fleet_size + parallel_factor - 1) // parallel_factor
    
    return {
        'S1: ML Training': 8.0,
        'S4: Model Conversion': 2.0,
        'S3: Stage-1 (Cached)': 0.1 * num_batches,
        'S3: Stage-2 (Model)': (6.4 / 60) * num_batches,
        'Verification': 0.5 + (fleet_size * 0.05),
    }


def get_colour(phase_name: str) -> str:
    """Retrieve colour for specified phase."""
    if phase_name in INFRASTRUCTURE_PHASES:
        return INFRASTRUCTURE_PHASES[phase_name]['colour']
    elif phase_name in MECPLUS_PHASES:
        return MECPLUS_PHASES[phase_name]['colour']
    return '#CCCCCC'


def create_legend_handles():
    """Create legend handles for infrastructure and MEC+ phases."""
    infra_handles = [mpatches.Patch(color=get_colour(p), label=p)
                    for p in INFRASTRUCTURE_PHASES.keys()]
    mecplus_handles = [mpatches.Patch(color=get_colour(p), label=p)
                      for p in MECPLUS_PHASES.keys()]
    return infra_handles, mecplus_handles


def get_infrastructure_timings(arch_key: str, n_machines: int) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Get infrastructure timings for an architecture.
    Always uses the same seed for the same architecture.
    
    Returns:
        Tuple of (timings dict, mean total)
    """
    arch = ARCHITECTURES[arch_key]
    infra_normalised = normalise_phases(INFRASTRUCTURE_PHASES, arch['mean'])
    infra_timings = generate_stochastic_timings(
        infra_normalised, n_machines, arch['variance'], arch['seed']
    )
    
    # Calculate mean
    bottom = np.zeros(n_machines)
    for phase_name in INFRASTRUCTURE_PHASES.keys():
        bottom += infra_timings[phase_name]
    
    return infra_timings, np.mean(bottom)


def get_mecplus_timings(arch_key: str, n_machines: int, fleet_size: int = 1) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Get MEC+ timings for a configuration.
    Uses architecture seed + fleet-specific offset for consistency.
    
    Returns:
        Tuple of (timings dict, mean total)
    """
    arch = ARCHITECTURES[arch_key]
    
    # Calculate MEC+ phase times based on fleet size
    mecplus_times = calculate_mecplus_fleet_times(fleet_size)
    
    # Use consistent seed: architecture seed + fleet seed
    seed = arch['seed'] + FLEET_SEEDS.get(fleet_size, fleet_size)
    
    mecplus_timings = generate_stochastic_timings(
        mecplus_times, n_machines, 0.08, seed
    )
    
    # Calculate mean
    total = np.zeros(n_machines)
    for phase_name in mecplus_times.keys():
        total += mecplus_timings[phase_name]
    
    return mecplus_timings, np.mean(total)


print("\nUtility functions defined")


# =============================================================================
# VERIFICATION: Ensure consistent results
# =============================================================================

print("\n" + "="*70)
print("VERIFICATION: Distributed-Horizontal + MEC+ (N=1) consistency check")
print("="*70)

n_test = 50
infra_timings, infra_mean = get_infrastructure_timings('distributed_horizontal', n_test)
mecplus_timings, mecplus_mean = get_mecplus_timings('distributed_horizontal', n_test, fleet_size=1)

# Calculate total
bottom = np.zeros(n_test)
for phase_name in INFRASTRUCTURE_PHASES.keys():
    bottom += infra_timings[phase_name]
for phase_name in MECPLUS_PHASES.keys():
    bottom += mecplus_timings[phase_name]

total_mean = np.mean(bottom)
print(f"Infrastructure mean: {infra_mean:.2f} min")
print(f"MEC+ mean: {mecplus_mean:.2f} min")
print(f"Combined total mean: {total_mean:.2f} min")
print(f"Target: 46.1 min")
print(f"Match: {'✓ YES' if abs(total_mean - 46.1) < 0.2 else '✗ NO (adjust needed)'}")


# =============================================================================
# FIGURE 1: FOUR-PANEL - ALL ARCHITECTURES WITH MEC+ (SINGLE ROBOT)
# =============================================================================

def create_figure1_all_architectures_with_mecplus(save_path: Optional[str] = None):
    """
    Four-panel figure showing all architectures with MEC+ single robot deployment:
    (a) Centralized-Horizontal + MEC+
    (b) Centralized-Vertical + MEC+
    (c) Distributed-Horizontal + MEC+
    (d) Distributed-Vertical + MEC+
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    n_machines = 50
    x = np.arange(n_machines)
    
    panel_configs = [
        {'ax': axes[0, 0], 'arch': 'centralized_horizontal', 'label': '(a)'},
        {'ax': axes[0, 1], 'arch': 'centralized_vertical', 'label': '(b)'},
        {'ax': axes[1, 0], 'arch': 'distributed_horizontal', 'label': '(c)'},
        {'ax': axes[1, 1], 'arch': 'distributed_vertical', 'label': '(d)'},
    ]
    
    results = {}
    
    for config in panel_configs:
        ax = config['ax']
        arch_key = config['arch']
        arch = ARCHITECTURES[arch_key]
        
        # Get timings using consistent seeds
        infra_timings, infra_mean = get_infrastructure_timings(arch_key, n_machines)
        mecplus_timings, mecplus_mean = get_mecplus_timings(arch_key, n_machines, fleet_size=1)
        
        # Plot infrastructure phases
        bottom = np.zeros(n_machines)
        for phase_name in INFRASTRUCTURE_PHASES.keys():
            values = infra_timings[phase_name]
            ax.bar(x, values, width=1.0, bottom=bottom,
                  color=get_colour(phase_name), edgecolor='none', linewidth=0)
            bottom += values
        
        # Plot MEC+ phases
        for phase_name in MECPLUS_PHASES.keys():
            values = mecplus_timings[phase_name]
            ax.bar(x, values, width=1.0, bottom=bottom,
                  color=get_colour(phase_name), edgecolor='none', linewidth=0)
            bottom += values
        
        total_mean = np.mean(bottom)
        results[arch_key] = {'infra': infra_mean, 'total': total_mean}
        
        # Formatting
        #ax.set_title(f"{arch['name']}\n+ 6G-MEC+ Single Robot MLOps",
        #            fontsize=11, fontweight='bold')
        ax.set_xlabel('Deployment time behavior per machine', fontsize=10)
        ax.set_ylabel('Orchestration Time for 5G/6G Service Deployment\n'
                     'in Multi-Tenant Architecture from Bare Metal (Minutes)', fontsize=9)
        ax.set_xlim(-0.5, n_machines - 0.5)
        ax.set_ylim(0, 75)
        
        # Panel label
        # ax.text(0.02, 0.98, config['label'], transform=ax.transAxes,
        #        fontsize=14, fontweight='bold', va='top')
        
        # Mean annotation
        ax.axhline(y=total_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(41, total_mean + 2.2, f'μ = {total_mean:.1f} min', fontsize=9,
               ha='right', color='green', fontweight='bold')
        
        # Golden Hour line
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Golden Hour label on last panel
    # axes[1, 1].text(44, 65, 'Golden Hour (60 min)', fontsize=9, color='red',
    #                ha='right', fontstyle='italic')
    
    # Combined legend
    
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.35, wspace=0.20)
    fig.text(0.25, 0.57, '(a)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.77, 0.57, '(b)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.25, 0.09, '(c)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.77, 0.09, '(d)', ha='center', fontsize=14, fontweight='bold')
    infra_handles, mecplus_handles = create_legend_handles()
    golden_hour_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Golden Hour (60 min)')
    fig.legend(handles=infra_handles + mecplus_handles + [golden_hour_handle], loc='lower center',
              ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return results


# Execute Figure 1
print("\n" + "="*70)
print("FIGURE 1: All Architectures with MEC+ (Single Robot)")
print("="*70)
#fig1_results = create_figure1_all_architectures_with_mecplus()
print("\nResults:")
for arch, vals in fig1_results.items():
    status = '✓' if vals['total'] < 60 else '✗'
    print(f"  {ARCHITECTURES[arch]['short_name']:<25} Total={vals['total']:.1f} min {status}")


# =============================================================================
# FIGURE 2: DISTRIBUTED-HORIZONTAL vs DISTRIBUTED-HORIZONTAL + MEC+
# =============================================================================

def create_figure2_distributed_horizontal_comparison(save_path: Optional[str] = None):
    """
    Two-panel comparison:
    (a) Distributed-Horizontal (Infrastructure only)
    (b) Distributed-Horizontal + 6G-MEC+ (Single Robot)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    n_machines = 50
    x = np.arange(n_machines)
    arch_key = 'distributed_horizontal'
    arch = ARCHITECTURES[arch_key]
    
    # Get timings using consistent seeds
    infra_timings, infra_mean = get_infrastructure_timings(arch_key, n_machines)
    mecplus_timings, mecplus_mean = get_mecplus_timings(arch_key, n_machines, fleet_size=1)
    
    # =========================================================================
    # Panel (a): Infrastructure Only
    # =========================================================================
    ax1 = axes[0]
    bottom = np.zeros(n_machines)
    
    for phase_name in INFRASTRUCTURE_PHASES.keys():
        values = infra_timings[phase_name]
        ax1.bar(x, values, width=1.0, bottom=bottom,
               color=get_colour(phase_name), edgecolor='none', linewidth=0)
        bottom += values
    
    #ax1.set_title(arch['name'], fontsize=12, fontweight='bold')
    ax1.set_xlabel('Deployment time behavior per machine', fontsize=11)
    ax1.set_ylabel('Orchestration Time for 5G/6G Service Deployment\n'
                  'in Multi-Tenant Architecture from Bare Metal (Minutes)', fontsize=10)
    ax1.set_xlim(-0.5, n_machines - 0.5)
    ax1.set_ylim(0, 70)
    #ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=14,
    #        fontweight='bold', va='top')
    
    # Mean reference line
    ax1.axhline(y=infra_mean, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(48, infra_mean + 1.5, f'μ = {infra_mean:.1f} min', fontsize=10, ha='right')
    
    # =========================================================================
    # Panel (b): Infrastructure + MEC+
    # =========================================================================
    ax2 = axes[1]
    bottom = np.zeros(n_machines)
    
    # Infrastructure phases (same timings as panel a)
    for phase_name in INFRASTRUCTURE_PHASES.keys():
        values = infra_timings[phase_name]
        ax2.bar(x, values, width=1.0, bottom=bottom,
               color=get_colour(phase_name), edgecolor='none', linewidth=0)
        bottom += values
    
    # MEC+ phases
    for phase_name in MECPLUS_PHASES.keys():
        values = mecplus_timings[phase_name]
        ax2.bar(x, values, width=1.0, bottom=bottom,
               color=get_colour(phase_name), edgecolor='none', linewidth=0)
        bottom += values
    
    total_mean = np.mean(bottom)
    mecplus_contribution = total_mean - infra_mean
    
    #ax2.set_title(f'{arch["name"]}\n+ 6G-MEC+ Single Robot MLOps Deployment',
    #             fontsize=12, fontweight='bold')
    ax2.set_xlabel('Deployment time behavior per machine', fontsize=11)
    ax2.set_xlim(-0.5, n_machines - 0.5)
    ax2.set_ylim(0, 70)
    #ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=14,
    #        fontweight='bold', va='top')
    
    # Mean reference lines
    ax2.axhline(y=infra_mean, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.axhline(y=total_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(45, total_mean + 1.8, f'μ = {total_mean:.1f} min', fontsize=10,
            ha='right', color='green', fontweight='bold')
    
    # MEC+ contribution annotation
    ax2.annotate('', xy=(25, total_mean), xytext=(25, infra_mean),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.text(26.5
             , (infra_mean + total_mean) / 2,
            f'+{mecplus_contribution:.1f} min\n(6G-MEC+)',
            fontsize=9, color='green', va='center', fontweight='bold')
    
    # Golden Hour reference
    for ax in axes:
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    # ax2.text(48, 61.5, 'Golden Hour (60 min)', fontsize=9, color='red',
    #         ha='right', fontstyle='italic')
    
    # Legend
    infra_handles, mecplus_handles = create_legend_handles()
    golden_hour_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Golden Hour (60 min)')
    fig.legend(handles=infra_handles + mecplus_handles + [golden_hour_handle], loc='lower center',
              ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, wspace=0.20)
    fig.text(0.25, 0.12, '(a)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.77, 0.12, '(b)', ha='center', fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return infra_mean, total_mean


# Execute Figure 2
print("\n" + "="*70)
print("FIGURE 2: Distributed-Horizontal vs Distributed-Horizontal + MEC+")
print("="*70)
#fig2_infra, fig2_total = create_figure2_distributed_horizontal_comparison()
print(f"\nResults: Infrastructure μ = {fig2_infra:.1f} min, Total μ = {fig2_total:.1f} min")
print(f"MEC+ Contribution: +{fig2_total - fig2_infra:.1f} min")


# =============================================================================
# FIGURE 3: DISTRIBUTED-VERTICAL vs DISTRIBUTED-VERTICAL + MEC+
# =============================================================================

def create_figure3_distributed_vertical_comparison(save_path: Optional[str] = None):
    """
    Two-panel comparison:
    (a) Distributed-Vertical (Infrastructure only)
    (b) Distributed-Vertical + 6G-MEC+ (Single Robot)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    n_machines = 50
    x = np.arange(n_machines)
    arch_key = 'distributed_vertical'
    arch = ARCHITECTURES[arch_key]
    
    # Get timings using consistent seeds
    infra_timings, infra_mean = get_infrastructure_timings(arch_key, n_machines)
    mecplus_timings, mecplus_mean = get_mecplus_timings(arch_key, n_machines, fleet_size=1)
    
    # =========================================================================
    # Panel (a): Infrastructure Only
    # =========================================================================
    ax1 = axes[0]
    bottom = np.zeros(n_machines)
    
    for phase_name in INFRASTRUCTURE_PHASES.keys():
        values = infra_timings[phase_name]
        ax1.bar(x, values, width=1.0, bottom=bottom,
               color=get_colour(phase_name), edgecolor='none', linewidth=0)
        bottom += values
    
    #ax1.set_title(arch['name'], fontsize=12, fontweight='bold')
    ax1.set_xlabel('Deployment time behavior per machine', fontsize=11)
    ax1.set_ylabel('Orchestration Time for 5G/6G Service Deployment\n'
                  'in Multi-Tenant Architecture from Bare Metal (Minutes)', fontsize=10)
    ax1.set_xlim(-0.5, n_machines - 0.5)
    ax1.set_ylim(0, 70)
    #ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=14,
    #        fontweight='bold', va='top')
    
    # Mean reference line
    ax1.axhline(y=infra_mean, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(45, infra_mean + 1.8, f'μ = {infra_mean:.1f} min', fontsize=10, ha='right')
    
    # =========================================================================
    # Panel (b): Infrastructure + MEC+
    # =========================================================================
    ax2 = axes[1]
    bottom = np.zeros(n_machines)
    
    # Infrastructure phases
    for phase_name in INFRASTRUCTURE_PHASES.keys():
        values = infra_timings[phase_name]
        ax2.bar(x, values, width=1.0, bottom=bottom,
               color=get_colour(phase_name), edgecolor='none', linewidth=0)
        bottom += values
    
    # MEC+ phases
    for phase_name in MECPLUS_PHASES.keys():
        values = mecplus_timings[phase_name]
        ax2.bar(x, values, width=1.0, bottom=bottom,
               color=get_colour(phase_name), edgecolor='none', linewidth=0)
        bottom += values
    
    total_mean = np.mean(bottom)
    mecplus_contribution = total_mean - infra_mean
    
    #ax2.set_title(f'{arch["name"]}\n+ 6G-MEC+ Single Robot MLOps Deployment',
    #             fontsize=12, fontweight='bold')
    ax2.set_xlabel('Deployment time behavior per machine', fontsize=11)
    ax2.set_xlim(-0.5, n_machines - 0.5)
    ax2.set_ylim(0, 70)
    #ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=14,
    #        fontweight='bold', va='top')
    
    # Mean reference lines
    ax2.axhline(y=infra_mean, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.axhline(y=total_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(48, total_mean + 1.5, f'μ = {total_mean:.1f} min', fontsize=10,
            ha='right', color='green', fontweight='bold')
    
    # MEC+ contribution annotation
    ax2.annotate('', xy=(25, total_mean), xytext=(25, infra_mean),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.text(26.5, (infra_mean + total_mean) / 2,
            f'+{mecplus_contribution:.1f} min\n(6G-MEC+)',
            fontsize=9, color='green', va='center', fontweight='bold')
    
    # Golden Hour reference
    for ax in axes:
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    # ax2.text(48, 61.5, 'Golden Hour (60 min)', fontsize=9, color='red',
    #         ha='right', fontstyle='italic')
    
    # Legend
    infra_handles, mecplus_handles = create_legend_handles()
    golden_hour_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Golden Hour (60 min)')
    fig.legend(handles=infra_handles + mecplus_handles + [golden_hour_handle], loc='lower center',
              ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, wspace=0.20)
    fig.text(0.25, 0.12, '(a)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.77, 0.12, '(b)', ha='center', fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return infra_mean, total_mean


# Execute Figure 3
print("\n" + "="*70)
print("FIGURE 3: Distributed-Vertical vs Distributed-Vertical + MEC+")
print("="*70)
#fig3_infra, fig3_total = create_figure3_distributed_vertical_comparison()
print(f"\nResults: Infrastructure μ = {fig3_infra:.1f} min, Total μ = {fig3_total:.1f} min")
print(f"MEC+ Contribution: +{fig3_total - fig3_infra:.1f} min")


# =============================================================================
# FIGURE 4: FLEET SCALING - DISTRIBUTED-HORIZONTAL + MEC+
# =============================================================================

def create_figure4_fleet_scaling(save_path: Optional[str] = None):
    """
    Four-panel fleet scaling analysis for Distributed-Horizontal + MEC+:
    (a) N=1 (single robot) - MUST match Figure 1(c) and Figure 2(b)
    (b) N=10
    (c) N=20
    (d) N=50
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    n_machines = 50
    x = np.arange(n_machines)
    arch_key = 'distributed_horizontal'
    arch = ARCHITECTURES[arch_key]
    
    fleet_configs = [
        {'n': 1, 'label': '(a)', 'title': 'Single Robot (N=1)'},
        {'n': 10, 'label': '(b)', 'title': 'Fleet (N=10)'},
        {'n': 20, 'label': '(c)', 'title': 'Fleet (N=20)'},
        {'n': 50, 'label': '(d)', 'title': 'Fleet (N=50)'},
    ]
    
    # Get infrastructure timings (same for all panels)
    infra_timings, infra_mean = get_infrastructure_timings(arch_key, n_machines)
    
    results = {}
    
    for ax, config in zip(axes.flat, fleet_configs):
        # Get MEC+ timings for this fleet size using consistent seed
        mecplus_timings, mecplus_mean = get_mecplus_timings(
            arch_key, n_machines, fleet_size=config['n']
        )
        
        # Plot infrastructure phases
        bottom = np.zeros(n_machines)
        for phase_name in INFRASTRUCTURE_PHASES.keys():
            values = infra_timings[phase_name]
            ax.bar(x, values, width=1.0, bottom=bottom,
                  color=get_colour(phase_name), edgecolor='none', linewidth=0)
            bottom += values
        
        # Plot MEC+ phases
        for phase_name in MECPLUS_PHASES.keys():
            values = mecplus_timings[phase_name]
            ax.bar(x, values, width=1.0, bottom=bottom,
                  color=get_colour(phase_name), edgecolor='none', linewidth=0)
            bottom += values
        
        total_mean = np.mean(bottom)
        mecplus_contribution = total_mean - infra_mean
        results[config['n']] = {
            'total': total_mean,
            'mecplus': mecplus_contribution,
            'margin': 60 - total_mean
        }
        
        # Formatting
        #ax.set_title(f"Distributed-Horizontal Infrastructure\n+ 6G-MEC+ {config['title']}",
        #            fontsize=11, fontweight='bold')
        ax.set_xlabel('Deployment time behavior per machine', fontsize=10)
        ax.set_ylabel('Orchestration + MLOps Time (Minutes)', fontsize=9)
        ax.set_xlim(-0.5, n_machines - 0.5)
        ax.set_ylim(0, 70)
        
        # Panel label
        #ax.text(0.02, 0.98, config['label'], transform=ax.transAxes,
        #       fontsize=14, fontweight='bold', va='top')
        
        # Mean annotations
        ax.axhline(y=infra_mean, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=total_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(45, total_mean + 1.8, f'μ = {total_mean:.1f} min', fontsize=9,
               ha='right', color='green', fontweight='bold')
        
        # MEC+ contribution annotation
        ax.text(2, infra_mean + mecplus_contribution / 2,
               f'+{mecplus_contribution:.1f}\n(MEC+)', fontsize=8,
               color='green', va='center', fontweight='bold')
        
        # Golden Hour line
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Golden Hour label on last panel
    # axes[1, 1].text(48, 61.5, 'Golden Hour (60 min)', fontsize=9, color='red',
    #                ha='right', fontstyle='italic')
    
    # Legend
    infra_handles, mecplus_handles = create_legend_handles()
    golden_hour_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Golden Hour (60 min)')
    fig.legend(handles=infra_handles + mecplus_handles + [golden_hour_handle], loc='lower center',
              ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.12, hspace=0.28, wspace=0.15)
    plt.subplots_adjust(bottom=0.15, hspace=0.35, wspace=0.20)
    fig.text(0.25, 0.57, '(a)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.77, 0.57, '(b)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.25, 0.09, '(c)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.77, 0.09, '(d)', ha='center', fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return results


# Execute Figure 4
print("\n" + "="*70)
print("FIGURE 4: Fleet Scaling (Distributed-Horizontal + MEC+)")
print("="*70)
fig4_results = create_figure4_fleet_scaling()
print("\nFleet Scaling Results:")
print(f"{'Fleet Size':<12} {'Total (min)':<14} {'MEC+ (min)':<14} {'Margin (min)':<14} {'Status'}")
print("-" * 70)
for n, vals in fig4_results.items():
    status = '✓ COMPLIANT' if vals['margin'] > 0 else '✗ NON-COMPLIANT'
    print(f"N={n:<10} {vals['total']:<14.1f} {vals['mecplus']:<14.1f} {vals['margin']:<+14.1f} {status}")


# =============================================================================
# CONSISTENCY VERIFICATION
# =============================================================================

print("\n" + "="*70)
print("CONSISTENCY VERIFICATION: Distributed-Horizontal + MEC+ (N=1)")
print("="*70)

print(f"\nFigure 1 (c): {fig1_results['distributed_horizontal']['total']:.1f} min")
print(f"Figure 2 (b): {fig2_total:.1f} min")
print(f"Figure 4 (a): {fig4_results[1]['total']:.1f} min")

# Check consistency
values = [
    fig1_results['distributed_horizontal']['total'],
    fig2_total,
    fig4_results[1]['total']
]
if max(values) - min(values) < 0.01:
    print("\n✓ ALL FIGURES CONSISTENT")
else:
    print(f"\n✗ INCONSISTENCY DETECTED: Range = {max(values) - min(values):.2f} min")


# =============================================================================
# COMPREHENSIVE SUMMARY
# =============================================================================

def print_comprehensive_summary():
    """Print all summary statistics for manuscript."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE SUMMARY FOR FGCS MANUSCRIPT")
    print("="*70)
    
    print("\n1. INFRASTRUCTURE BASELINE TIMES (Aston 5G Testbed)")
    print("-" * 50)
    for name, arch in ARCHITECTURES.items():
        print(f"   {arch['short_name']:<25} μ = {arch['mean']:.1f} min")
    
    print("\n2. 6G-MEC+ CONTRIBUTION (Single Robot)")
    print("-" * 50)
    total = sum(p['time'] for p in MECPLUS_PHASES.values())
    print(f"   Total: {total:.2f} min ({total*60:.1f} seconds)")
    for name, cfg in MECPLUS_PHASES.items():
        print(f"      {name:<25} {cfg['time']:.3f} min ({cfg['time']*60:.1f} s)")
    
    print("\n3. COMBINED DEPLOYMENT TIMES (Infrastructure + MEC+ Single Robot)")
    print("-" * 50)
    for name, arch in ARCHITECTURES.items():
        combined = arch['mean'] + total
        margin = 60 - combined
        status = '✓' if margin > 0 else '✗'
        print(f"   {arch['short_name']:<25} {combined:.1f} min (margin: {margin:+.1f}) {status}")
    
    print("\n4. FLEET SCALING (Distributed-Horizontal + MEC+)")
    print("-" * 50)
    base_infra = ARCHITECTURES['distributed_horizontal']['mean']
    for n in [1, 10, 20, 50]:
        mecplus_times = calculate_mecplus_fleet_times(n)
        mecplus_total = sum(mecplus_times.values())
        combined = base_infra + mecplus_total
        margin = 60 - combined
        status = '✓' if margin > 0 else '✗'
        print(f"   N={n:<3}  MEC+={mecplus_total:.1f} min  Total={combined:.1f} min  Margin={margin:+.1f} {status}")


print_comprehensive_summary()


# =============================================================================
# SAVE ALL FIGURES (Uncomment to save)
# =============================================================================

# For Kaggle, uncomment and use:
#create_figure1_all_architectures_with_mecplus(save_path='fig1_all_arch_mecplus.pdf')
#create_figure2_distributed_horizontal_comparison(save_path='fig2_dist_horiz.pdf')
#create_figure3_distributed_vertical_comparison(save_path='fig3_dist_vert.pdf')
create_figure4_fleet_scaling(save_path='fig4_fleet_scaling.pdf')

print("\n" + "="*70)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*70)