"""
Generate missing figures: fig07 (advanced exergy) and fig08 (seawater sweep).
Run from the simulations/ directory:
    python generate_missing_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'primary':   '#1B4F72',
    'secondary': '#2E75B6',
    'accent1':   '#C0392B',
    'accent2':   '#27AE60',
    'accent3':   '#F39C12',
    'accent4':   '#8E44AD',
    'gray':      '#7F8C8D',
}

COMP_LABELS = {
    'engine_cylinder': 'Engine\nCylinders',
    'turbocharger':    'Turbocharger',
    'exhaust_boiler':  'Exhaust\nBoiler',
    'orc_evaporator':  'ORC\nEvaporator',
    'orc_turbine':     'ORC\nTurbine',
    'orc_condenser':   'ORC\nCondenser',
    'orc_pump':        'ORC\nPump',
}

COMP_ORDER = [
    'engine_cylinder', 'turbocharger', 'exhaust_boiler',
    'orc_evaporator', 'orc_turbine', 'orc_condenser', 'orc_pump',
]

FLUID_STYLES = {
    'Toluene':     {'color': COLORS['primary'],   'marker': 'o', 'ls': '-'},
    'Cyclohexane': {'color': COLORS['accent1'],   'marker': 's', 'ls': '-'},
    'n-Pentane':   {'color': COLORS['accent2'],   'marker': '^', 'ls': '-'},
    'R1233zd(E)':  {'color': COLORS['accent3'],   'marker': 'D', 'ls': '-'},
}


def plot_advanced_exergy():
    df_all = pd.read_csv(os.path.join(RESULTS_DIR, '07_advanced_exergy_decomposition.csv'))
    df = df_all[np.isclose(df_all['Load_pct'], 85)].copy()
    df['Component'] = df['Component'].str.strip()
    df = df.set_index('Component').reindex(COMP_ORDER)

    labels = [COMP_LABELS[c] for c in COMP_ORDER]
    x = np.arange(len(COMP_ORDER))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel (a): total vs avoidable destruction ---
    bars_total = ax1.bar(x - w/2, df['E_D_total_kW'], w,
                         label='Total destruction', color=COLORS['secondary'], alpha=0.85)
    bars_avail = ax1.bar(x + w/2, df['E_D_avoidable_kW'], w,
                         label='Avoidable (AV)', color=COLORS['accent1'], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Exergy destruction (kW)')
    ax1.set_title('(a) Total vs avoidable destruction\nat 85% MCR, $T_{sw}$ = 29 °C')
    ax1.legend()
    ax1.set_ylim(0, max(df['E_D_total_kW'].max(), df['E_D_avoidable_kW'].max()) * 1.15)

    # Annotate the zero-avoidable ORC evaporator bar
    evap_idx = COMP_ORDER.index('orc_evaporator')
    ax1.annotate('Zero avoidable\n(2 K pinch floor)',
                 xy=(evap_idx + w/2, 5),
                 xytext=(evap_idx + w/2 + 0.6, 200),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                 fontsize=8, ha='left')

    # --- Panel (b): four-quadrant stacked bars ---
    av_en = df['E_D_AV_EN_kW'].values
    av_ex = df['E_D_AV_EX_kW'].values
    un_en = df['E_D_UN_EN_kW'].values
    un_ex = df['E_D_UN_EX_kW'].values

    ax2.bar(x, av_en, label='AV-EN (improvable, own component)',
            color=COLORS['accent1'], alpha=0.9)
    ax2.bar(x, av_ex, bottom=av_en, label='AV-EX (improvable, other components)',
            color=COLORS['accent3'], alpha=0.9)
    ax2.bar(x, un_en, bottom=av_en + av_ex, label='UN-EN (structural floor, own)',
            color=COLORS['secondary'], alpha=0.6)
    ax2.bar(x, un_ex, bottom=av_en + av_ex + un_en, label='UN-EX (structural floor, system)',
            color=COLORS['gray'], alpha=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Exergy destruction (kW)')
    ax2.set_title('(b) Four-quadrant decomposition\n(AV/UN × EN/EX)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(0, df['E_D_total_kW'].max() * 1.15)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUTPUT_DIR, f'fig07_advanced_exergy.{ext}'))
    plt.close()
    print('  fig07_advanced_exergy')


def plot_seawater_sweep():
    df = pd.read_csv(os.path.join(RESULTS_DIR, '09_seawater_sensitivity.csv'))

    top_fluids = ['Toluene', 'Cyclohexane', 'n-Pentane', 'R1233zd(E)']
    df = df[df['fluid'].isin(top_fluids)].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for fluid in top_fluids:
        s = FLUID_STYLES[fluid]
        fd = df[df['fluid'] == fluid].sort_values('T_sw_C')
        ax1.plot(fd['T_sw_C'], fd['W_net_kW'],
                 color=s['color'], marker=s['marker'], ls=s['ls'],
                 markersize=4, label=fluid)
        ax2.plot(fd['T_sw_C'], fd['eta_th'] * 100,
                 color=s['color'], marker=s['marker'], ls=s['ls'],
                 markersize=4, label=fluid)

    # Mark the Indian Ocean baseline (29 °C)
    for ax in (ax1, ax2):
        ax.axvline(29, color='black', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.text(29.3, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1,
                'Indian Ocean\nbaseline (29 °C)',
                fontsize=8, va='top', color='black')

    ax1.set_ylabel('Net power output (kW)')
    ax1.set_title('Seawater temperature sensitivity — top four working fluids\n'
                  '(85% MCR; slope ≈ 0.76%/K; fluid ranking invariant over 15–32 °C)')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Seawater temperature (°C)')
    ax2.set_ylabel('Thermal efficiency (%)')
    ax2.legend(loc='upper right')

    ax1.set_xlim(15, 32)
    ax2.set_xlim(15, 32)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUTPUT_DIR, f'fig08_sw_sweep.{ext}'))
    plt.close()
    print('  fig08_sw_sweep')


if __name__ == '__main__':
    print('Generating missing figures...')
    plot_advanced_exergy()
    plot_seawater_sweep()
    print('Done. Saved to', OUTPUT_DIR)
