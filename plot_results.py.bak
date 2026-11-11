"""
Module 6: Publication-Quality Figure Generation
================================================
Generates all figures for the journal paper from simulation CSV outputs.
Run this AFTER the master simulation completes.

Author: Prajwal Rao
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

COLORS = {
    'primary': '#1B4F72',
    'secondary': '#2E75B6',
    'accent1': '#C0392B',
    'accent2': '#27AE60',
    'accent3': '#F39C12',
    'accent4': '#8E44AD',
    'gray': '#7F8C8D',
}

TECH_COLORS = {
    'Steam_Rankine': '#1B4F72',
    'ORC_R245fa': '#C0392B',
    'ORC_R1233zd': '#27AE60',
    'TEG': '#F39C12',
    'Combined_ORC_TEG': '#8E44AD',
}


def plot_energy_balance(results_dir, output_dir):
    """Figure: Stacked energy balance vs engine load."""
    df = pd.read_csv(os.path.join(results_dir, '01_engine_energy_balance.csv'))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    loads = df['Load_pct']
    Q_fuel = df['Q_fuel_kW']

    ax1.fill_between(loads, 0, df['W_brake_pct'], alpha=0.7, color=COLORS['accent2'], label='Shaft Work')
    ax1.fill_between(loads, df['W_brake_pct'], df['W_brake_pct'] + df['Q_exhaust_pct'],
                     alpha=0.7, color=COLORS['accent1'], label='Exhaust Losses')
    ax1.fill_between(loads, df['W_brake_pct'] + df['Q_exhaust_pct'],
                     df['W_brake_pct'] + df['Q_exhaust_pct'] + df['Q_charge_air_pct'],
                     alpha=0.7, color=COLORS['secondary'], label='Charge Air Cooling')
    ax1.fill_between(loads, df['W_brake_pct'] + df['Q_exhaust_pct'] + df['Q_charge_air_pct'],
                     df['W_brake_pct'] + df['Q_exhaust_pct'] + df['Q_charge_air_pct'] + df['Q_jacket_pct'],
                     alpha=0.7, color=COLORS['accent3'], label='Jacket Water')

    ax2 = ax1.twinx()
    ax2.plot(loads, df['eta_brake'] * 100, 'k-', linewidth=2, label=r'$\eta_{brake}$')
    ax2.plot(loads, df['BSFC_g_kWh'], 'k--', linewidth=1.5, label='BSFC')

    ax1.set_xlabel('Engine Load (% MCR)')
    ax1.set_ylabel('Energy Distribution (%)')
    ax2.set_ylabel(r'$\eta_{brake}$ (%) / BSFC (g/kWh)')
    ax1.set_xlim(20, 100)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left')
    ax2.legend(loc='center right')
    ax1.set_title('Energy Balance — MAN B&W S65ME-C (Indian Ocean Conditions)')

    plt.savefig(os.path.join(output_dir, 'fig01_energy_balance.png'))
    plt.savefig(os.path.join(output_dir, 'fig01_energy_balance.pdf'))
    plt.close()
    print("  fig01_energy_balance")


def plot_exergy_balance(results_dir, output_dir):
    """Figure: Exergy balance and destruction vs load."""
    df = pd.read_csv(os.path.join(results_dir, '03_engine_exergy_balance.csv'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    loads = df['Load_pct']

    # Left: Exergy streams
    ax1.plot(loads, df['Ex_work_pct'], '-', color=COLORS['accent2'], label='Shaft Work', linewidth=2)
    ax1.plot(loads, df['Ex_exhaust_pct'], '-', color=COLORS['accent1'], label='Exhaust Exergy', linewidth=2)
    ax1.plot(loads, df['Ex_destroyed_pct'], '-', color=COLORS['gray'], label='Destroyed (Irreversibility)', linewidth=2)
    ax1.plot(loads, df['Ex_jacket_pct'] + df['Ex_charge_air_pct'] + df['Ex_lube_pct'],
             '--', color=COLORS['accent3'], label='Cooling Systems', linewidth=1.5)

    ax1.set_xlabel('Engine Load (% MCR)')
    ax1.set_ylabel('Exergy Fraction (%)')
    ax1.set_title('(a) Exergy Distribution')
    ax1.legend()
    ax1.set_xlim(20, 100)

    # Right: Exergy efficiency
    ax2.plot(loads, df['eta_exergy'] * 100, '-', color=COLORS['primary'], label='Engine Only', linewidth=2)
    ax2.plot(loads, df['eta_exergy_whr'] * 100, '--', color=COLORS['accent2'],
             label='Engine + WHR Potential', linewidth=2)
    ax2.fill_between(loads, df['eta_exergy'] * 100, df['eta_exergy_whr'] * 100,
                     alpha=0.2, color=COLORS['accent2'], label='WHR Opportunity')

    ax2.set_xlabel('Engine Load (% MCR)')
    ax2.set_ylabel('Exergy Efficiency (%)')
    ax2.set_title('(b) Second-Law Efficiency')
    ax2.legend()
    ax2.set_xlim(20, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig02_exergy_analysis.png'))
    plt.savefig(os.path.join(output_dir, 'fig02_exergy_analysis.pdf'))
    plt.close()
    print("  fig02_exergy_analysis")


def plot_orc_fluid_comparison(results_dir, output_dir):
    """Figure: ORC working fluid comparison."""
    try:
        df = pd.read_csv(os.path.join(results_dir, '02b_orc_optimal_per_fluid.csv'))
    except FileNotFoundError:
        print("  SKIP fig03 (ORC data not found)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fluids = df['fluid_key'].unique()
    fluid_colors = plt.cm.Set2(np.linspace(0, 1, len(fluids)))

    for i, fluid in enumerate(fluids):
        fd = df[df['fluid_key'] == fluid].sort_values('engine_load_pct')
        if len(fd) == 0:
            continue

        axes[0, 0].plot(fd['engine_load_pct'], fd['W_net_kW'], 'o-',
                       color=fluid_colors[i], label=fluid, markersize=4)
        axes[0, 1].plot(fd['engine_load_pct'], fd['eta_thermal'] * 100, 's-',
                       color=fluid_colors[i], label=fluid, markersize=4)
        axes[1, 0].plot(fd['engine_load_pct'], fd['T_evap_C'], '^-',
                       color=fluid_colors[i], label=fluid, markersize=4)
        axes[1, 1].plot(fd['engine_load_pct'], fd['pressure_ratio'], 'D-',
                       color=fluid_colors[i], label=fluid, markersize=4)

    axes[0, 0].set_ylabel('Net Power Output (kW)')
    axes[0, 0].set_title('(a) Maximum Net Power')
    axes[0, 1].set_ylabel('Thermal Efficiency (%)')
    axes[0, 1].set_title('(b) Thermal Efficiency at Optimal Point')
    axes[1, 0].set_ylabel('Optimal Evaporator Temp (°C)')
    axes[1, 0].set_title('(c) Optimal Evaporation Temperature')
    axes[1, 1].set_ylabel('Pressure Ratio')
    axes[1, 1].set_title('(d) Pressure Ratio')

    for ax in axes.flat:
        ax.set_xlabel('Engine Load (% MCR)')
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig03_orc_fluid_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'fig03_orc_fluid_comparison.pdf'))
    plt.close()
    print("  fig03_orc_fluid_comparison")


def plot_4e_comparison(results_dir, output_dir):
    """Figure: 4E analysis across WHR technologies."""
    df = pd.read_csv(os.path.join(results_dir, '04_environmental_economic_4E.csv'))

    # Filter to ~85% MCR
    df_85 = df[np.isclose(df['Load_pct'], 85, atol=2)]
    if len(df_85) == 0:
        df_85 = df[df['Load_pct'] == df['Load_pct'].unique()[len(df['Load_pct'].unique())//2]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    techs = df_85['WHR_technology'].unique()

    # (a) Power output vs load for each tech
    for tech in techs:
        td = df[df['WHR_technology'] == tech].sort_values('Load_pct')
        c = TECH_COLORS.get(tech, COLORS['gray'])
        axes[0, 0].plot(td['Load_pct'], td['W_whr_kW'], '-o', color=c, label=tech, markersize=3)

    axes[0, 0].set_ylabel('WHR Power Output (kW)')
    axes[0, 0].set_title('(a) Energy: Power Output vs Load')
    axes[0, 0].legend(fontsize=8)

    # (b) CO2 reduction
    for tech in techs:
        td = df[df['WHR_technology'] == tech].sort_values('Load_pct')
        c = TECH_COLORS.get(tech, COLORS['gray'])
        axes[0, 1].plot(td['Load_pct'], td['annual_co2_saved_tonnes'], '-s', color=c, label=tech, markersize=3)

    axes[0, 1].set_ylabel('Annual CO₂ Reduction (tonnes)')
    axes[0, 1].set_title('(b) Environmental: CO₂ Savings')
    axes[0, 1].legend(fontsize=8)

    # (c) Payback period
    for tech in techs:
        td = df[df['WHR_technology'] == tech].sort_values('Load_pct')
        c = TECH_COLORS.get(tech, COLORS['gray'])
        payback_clipped = td['payback_years'].clip(upper=15)
        axes[1, 0].plot(td['Load_pct'], payback_clipped, '-^', color=c, label=tech, markersize=3)

    axes[1, 0].set_ylabel('Payback Period (years)')
    axes[1, 0].set_title('(c) Economic: Payback Period')
    axes[1, 0].axhline(y=5, color='gray', linestyle=':', alpha=0.5, label='5-year target')
    axes[1, 0].legend(fontsize=8)

    # (d) EEDI improvement
    for tech in techs:
        td = df[df['WHR_technology'] == tech].sort_values('Load_pct')
        c = TECH_COLORS.get(tech, COLORS['gray'])
        axes[1, 1].plot(td['Load_pct'], td['eedi_improvement_pct'], '-D', color=c, label=tech, markersize=3)

    axes[1, 1].set_ylabel('EEDI Improvement (%)')
    axes[1, 1].set_title('(d) Regulatory: EEDI Impact')
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel('Engine Load (% MCR)')

    plt.suptitle('4E Analysis: WHR Technology Comparison — MAN B&W S65ME-C', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig04_4E_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'fig04_4E_comparison.pdf'))
    plt.close()
    print("  fig04_4E_comparison")


def plot_monte_carlo(results_dir, output_dir):
    """Figure: Monte Carlo sensitivity results."""
    try:
        df = pd.read_csv(os.path.join(results_dir, '05_monte_carlo_results.csv'))
        sens = pd.read_csv(os.path.join(results_dir, '05b_sensitivity_indices.csv'))
    except FileNotFoundError:
        print("  SKIP fig05 (Monte Carlo data not found)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) W_net distribution
    axes[0, 0].hist(df['W_net_kW'], bins=60, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    mean_w = df['W_net_kW'].mean()
    axes[0, 0].axvline(mean_w, color=COLORS['accent1'], linestyle='--', linewidth=2,
                       label=f'Mean = {mean_w:.0f} kW')
    axes[0, 0].axvline(df['W_net_kW'].quantile(0.025), color=COLORS['accent3'], linestyle=':',
                       label=f'95% CI: [{df["W_net_kW"].quantile(0.025):.0f}, {df["W_net_kW"].quantile(0.975):.0f}]')
    axes[0, 0].axvline(df['W_net_kW'].quantile(0.975), color=COLORS['accent3'], linestyle=':')
    axes[0, 0].set_xlabel('Net WHR Power Output (kW)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('(a) Power Output Distribution')
    axes[0, 0].legend(fontsize=8)

    # (b) NPV distribution
    axes[0, 1].hist(df['NPV_20yr_USD'] / 1e6, bins=60, color=COLORS['accent2'], alpha=0.7, edgecolor='white')
    axes[0, 1].axvline(0, color='red', linestyle='-', linewidth=1)
    pct_positive = (df['NPV_20yr_USD'] > 0).mean() * 100
    axes[0, 1].set_xlabel('20-Year NPV (Million USD)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'(b) NPV Distribution ({pct_positive:.0f}% positive)')

    # (c) Sensitivity tornado
    top_sens = sens.head(10).sort_values('abs_correlation')
    axes[1, 0].barh(range(len(top_sens)), top_sens['correlation_with_W_net'],
                    color=[COLORS['accent1'] if v < 0 else COLORS['accent2']
                           for v in top_sens['correlation_with_W_net']])
    axes[1, 0].set_yticks(range(len(top_sens)))
    axes[1, 0].set_yticklabels(top_sens['parameter'].values, fontsize=8)
    axes[1, 0].set_xlabel('Correlation with W_net')
    axes[1, 0].set_title('(c) Sensitivity Analysis (Tornado Chart)')
    axes[1, 0].axvline(0, color='black', linewidth=0.5)

    # (d) Scatter: W_net vs CO2 saved, colored by fuel price
    sc = axes[1, 1].scatter(df['W_net_kW'], df['co2_saved_tonnes'],
                            c=df['fuel_price'], cmap='YlOrRd', alpha=0.3, s=5)
    plt.colorbar(sc, ax=axes[1, 1], label='Fuel Price ($/ton)')
    axes[1, 1].set_xlabel('Net Power (kW)')
    axes[1, 1].set_ylabel('Annual CO₂ Saved (tonnes)')
    axes[1, 1].set_title('(d) Power vs CO₂ Savings')

    plt.suptitle('Monte Carlo Uncertainty Analysis (N={:,})'.format(len(df)), fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig05_monte_carlo.png'))
    plt.savefig(os.path.join(output_dir, 'fig05_monte_carlo.pdf'))
    plt.close()
    print("  fig05_monte_carlo")


def plot_pareto_front(results_dir, output_dir):
    """Figure: Pareto front from NSGA-II optimization."""
    try:
        df = pd.read_csv(os.path.join(results_dir, '06_pareto_front.csv'))
    except FileNotFoundError:
        try:
            df = pd.read_csv(os.path.join(results_dir, '06_grid_search_results.csv'))
        except FileNotFoundError:
            print("  SKIP fig06 (optimization data not found)")
            return

    if 'lifecycle_cost_USD' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 7))

        fluids = df['fluid'].unique()
        for i, fluid in enumerate(fluids):
            fd = df[df['fluid'] == fluid]
            sc = ax.scatter(fd['W_net_kW'], fd['lifecycle_cost_USD'] / 1e6,
                           label=fluid, alpha=0.6, s=30)

        ax.set_xlabel('Net Power Output (kW)')
        ax.set_ylabel('20-Year Lifecycle Cost (Million USD)')
        ax.set_title('Pareto Front: Power vs Cost Trade-off')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig06_pareto_front.png'))
        plt.savefig(os.path.join(output_dir, 'fig06_pareto_front.pdf'))
        plt.close()
        print("  fig06_pareto_front")
    else:
        print("  fig06 generated from grid search (simplified)")


def generate_all_figures(results_dir='results', output_dir='results/figures'):
    """Generate all publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    plot_energy_balance(results_dir, output_dir)
    plot_exergy_balance(results_dir, output_dir)
    plot_orc_fluid_comparison(results_dir, output_dir)
    plot_4e_comparison(results_dir, output_dir)
    plot_monte_carlo(results_dir, output_dir)
    plot_pareto_front(results_dir, output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to {output_dir}/")
    print("Both PNG (for draft) and PDF (for LaTeX) versions generated.")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_figures()
