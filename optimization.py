"""
Module 5: Monte Carlo Sensitivity Analysis & Multi-Objective Optimization
==========================================================================
- Monte Carlo: propagate uncertainty in engine parameters through ORC model
- NSGA-II: Pareto optimization of ORC for simultaneous max power + min cost + min GWP

This is the computationally intensive module designed to run overnight (8-12 hours).

Author: Prajwal Rao
"""

import logging
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)

# Silence only CoolProp-internal FutureWarnings / numpy RuntimeWarnings
# that flood stdout during the sweep. We do NOT want to suppress our
# own sanity-check warnings, so leave UserWarning + RuntimeWarning from
# this package visible.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============================================================================
# PART A: Monte Carlo Sensitivity Analysis
# ============================================================================

class MonteCarloSensitivity:
    """
    Propagate parametric uncertainty through the WHR model.

    Uncertain parameters:
        - Exhaust gas temperature (±15°C)
        - Exhaust mass flow rate (±5%)
        - Ambient/seawater temperature (26-33°C for Indian Ocean)
        - Fuel LHV (±2%)
        - ORC component efficiencies (pump: 0.60-0.80, expander: 0.70-0.88)
        - Fouling factor on heat exchangers (0-15% degradation)
        - Engine degradation over service life (0-8% efficiency loss)
    """

    def __init__(self, n_samples=10000, seed=42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate_samples(self):
        """Generate Latin Hypercube-like samples for all uncertain parameters."""
        n = self.n_samples

        samples = pd.DataFrame({
            # Engine parameters
            'T_exhaust_offset_C': self.rng.normal(0, 8, n),           # ±15°C (2σ)
            'm_exhaust_factor': self.rng.normal(1.0, 0.025, n),       # ±5% (2σ)
            'T_seawater_C': self.rng.uniform(26, 33, n),              # Indian Ocean range
            'T_ambient_C': self.rng.uniform(27, 38, n),               # Tropical range
            'LHV_factor': self.rng.normal(1.0, 0.01, n),             # ±2%
            'engine_degradation': self.rng.uniform(0, 0.08, n),       # 0-8% over life

            # ORC parameters
            'eta_pump': self.rng.uniform(0.60, 0.80, n),
            'eta_expander': self.rng.uniform(0.70, 0.88, n),
            'fouling_factor': self.rng.uniform(0, 0.15, n),           # 0-15% HX degradation

            # Economic parameters
            'fuel_price_USD_ton': self.rng.uniform(350, 800, n),      # HFO price range
            'carbon_price_USD_tCO2': self.rng.uniform(30, 200, n),    # Carbon price range
            'capex_factor': self.rng.uniform(0.8, 1.4, n),           # CAPEX uncertainty
            'discount_rate': self.rng.uniform(0.05, 0.12, n),        # 5-12%

            # Operating parameters
            'operating_days': self.rng.uniform(200, 330, n),          # Days at sea per year
            'avg_load_fraction': self.rng.uniform(0.55, 0.90, n),    # Typical service load
        })

        return samples

    def evaluate_sample(self, row, engine_base, orc_fluid='R245fa'):
        """Evaluate a single Monte Carlo sample."""
        try:
            import CoolProp.CoolProp as CP
            from orc_model import ORCSystem

            load = row['avg_load_fraction']

            # Adjust engine parameters
            T_exh_base = float(engine_base.exhaust_gas_temp_after_TC(load))
            T_exh = T_exh_base + row['T_exhaust_offset_C']

            m_exh_base = float(engine_base.exhaust_mass_flow(load))
            m_exh = m_exh_base * row['m_exhaust_factor']

            # Account for engine degradation
            eta_base = float(engine_base.brake_thermal_efficiency(load))
            eta_degraded = eta_base * (1 - row['engine_degradation'])

            # Engine fuel consumption with degradation
            W_brake = float(engine_base.brake_power(load))
            Q_fuel = W_brake / eta_degraded
            m_fuel = Q_fuel / (engine_base.LHV_HFO * row['LHV_factor'])

            # Available exhaust heat (with fouling)
            cp_exh = float(engine_base.exhaust_cp(T_exh))
            T_exh_out = max(150, T_exh - 100)
            Q_avail = m_exh * cp_exh * (T_exh - T_exh_out) * (1 - row['fouling_factor'])

            # ORC simulation
            T_cond = row['T_seawater_C'] + 9  # Approach temperature
            orc = ORCSystem(orc_fluid,
                           eta_pump=row['eta_pump'],
                           eta_expander=row['eta_expander'])

            # Try to find best operating point
            best_W = 0
            best_result = None

            T_evap_max = float(min(T_exh - 20.0, 145.0))
            if T_evap_max <= 70.0:
                return None  # Exhaust too cold for ORC pickup
            for T_evap in np.arange(70.0, T_evap_max + 0.01, 10.0):
                for T_sh in [0, 10, 20]:
                    r = orc.size_for_heat_source(
                        Q_avail, T_exh, T_exh_out,
                        T_evap, T_cond, T_sh
                    )
                    if r is not None and r['W_net_kW'] > best_W:
                        best_W = r['W_net_kW']
                        best_result = r

            if best_result is None:
                return None

            # Economic evaluation
            capex = best_W * 2500 * row['capex_factor']
            fuel_saved_tonnes = best_W * row['operating_days'] * 24 * 210 / 1e6
            annual_fuel_savings_usd = fuel_saved_tonnes * row['fuel_price_USD_ton']
            co2_saved = fuel_saved_tonnes * 3.114
            carbon_revenue = co2_saved * row['carbon_price_USD_tCO2']
            maintenance = capex * 0.03
            net_annual = annual_fuel_savings_usd + carbon_revenue - maintenance
            payback = capex / net_annual if net_annual > 0 else 999

            # NPV
            npv = -capex
            for yr in range(1, 21):
                npv += net_annual / (1 + row['discount_rate'])**yr

            return {
                'W_net_kW': best_W,
                'eta_orc': best_result['eta_thermal'],
                'T_evap_C': best_result['T_evap_C'],
                'T_cond_C': best_result['T_cond_C'],
                'Q_available_kW': Q_avail,
                'fuel_saved_tonnes': fuel_saved_tonnes,
                'co2_saved_tonnes': co2_saved,
                'capex_USD': capex,
                'net_annual_USD': net_annual,
                'payback_years': payback,
                'NPV_20yr_USD': npv,
                'engine_load': load,
                'T_seawater_C': row['T_seawater_C'],
                'fuel_price': row['fuel_price_USD_ton'],
                'carbon_price': row['carbon_price_USD_tCO2'],
            }

        except Exception:
            return None


def run_monte_carlo(engine, n_samples=10000, output_dir='results'):
    """Run Monte Carlo sensitivity analysis."""
    os.makedirs(output_dir, exist_ok=True)

    mc = MonteCarloSensitivity(n_samples=n_samples)
    samples = mc.generate_samples()

    print(f"\n[Monte Carlo] Running {n_samples} samples...")
    results = []
    failed = 0

    for i in tqdm(range(n_samples), desc="Monte Carlo"):
        row = samples.iloc[i]
        r = mc.evaluate_sample(row, engine, orc_fluid='R245fa')
        if r is not None:
            results.append(r)
        else:
            failed += 1

    df = pd.DataFrame(results)

    # Sanity bounds — flag egregious outputs loudly but do not crash.
    if len(df) == 0:
        raise RuntimeError("Monte Carlo produced zero successful samples.")
    if failed > 0.5 * n_samples:
        warnings.warn(
            f"Monte Carlo failure rate {failed}/{n_samples} > 50%. "
            "ORC feasibility envelope may be too tight — inspect samples.",
            RuntimeWarning,
        )
    if (df['eta_orc'] > 0.25).any() or (df['eta_orc'] < 0.03).any():
        warnings.warn(
            f"Monte Carlo ORC η outside [0.03, 0.25]: "
            f"[{df['eta_orc'].min():.3f}, {df['eta_orc'].max():.3f}]",
            RuntimeWarning,
        )

    df.to_csv(os.path.join(output_dir, '05_monte_carlo_results.csv'), index=False)

    print(f"\n[Monte Carlo] {len(df)} successful / {failed} failed out of {n_samples}")

    if len(df) > 0:
        print(f"[Monte Carlo] W_net: mean={df['W_net_kW'].mean():.0f} kW, "
              f"std={df['W_net_kW'].std():.0f} kW, "
              f"95% CI=[{df['W_net_kW'].quantile(0.025):.0f}, {df['W_net_kW'].quantile(0.975):.0f}]")
        print(f"[Monte Carlo] Payback: mean={df['payback_years'].mean():.1f} yr, "
              f"median={df['payback_years'].median():.1f} yr")
        print(f"[Monte Carlo] CO2 saved: mean={df['co2_saved_tonnes'].mean():.0f} t/yr")
        print(f"[Monte Carlo] NPV: mean=${df['NPV_20yr_USD'].mean():,.0f}")

    # Sensitivity indices (correlation-based)
    samples_success = samples.iloc[:len(df)].reset_index(drop=True)
    if len(df) > 100:
        correlations = {}
        for col in samples.columns:
            if col in samples_success.columns:
                corr = np.corrcoef(samples_success[col].values[:len(df)],
                                    df['W_net_kW'].values)[0, 1]
                correlations[col] = corr

        sens_df = pd.DataFrame({
            'parameter': list(correlations.keys()),
            'correlation_with_W_net': list(correlations.values()),
            'abs_correlation': [abs(v) for v in correlations.values()]
        }).sort_values('abs_correlation', ascending=False)

        sens_df.to_csv(os.path.join(output_dir, '05b_sensitivity_indices.csv'), index=False)
        print(f"\n[Monte Carlo] Top 5 influential parameters on W_net:")
        for _, row in sens_df.head(5).iterrows():
            print(f"  {row['parameter']:30s}: r = {row['correlation_with_W_net']:+.3f}")

    return df


# ============================================================================
# PART B: Multi-Objective Optimization (NSGA-II)
# ============================================================================

def run_nsga2_optimization(engine, output_dir='results', n_gen=200, pop_size=100):
    """
    Multi-objective optimization using NSGA-II.

    Objectives:
        1. Maximize net power output (W_net)
        2. Minimize lifecycle cost (CAPEX + 20yr O&M - fuel savings)
        3. Minimize environmental impact (GWP of working fluid × charge mass)

    Decision variables:
        - Evaporator temperature (60-150°C)
        - Condenser temperature (35-45°C)
        - Superheat (0-30°C)
        - Working fluid (discrete: R245fa, R1233zd(E), R1234ze(E), n-Pentane)
        - Engine load fraction (0.50-1.00)
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        HAS_PYMOO = True
    except ImportError:
        HAS_PYMOO = False
        print("[NSGA-II] pymoo not installed. Running grid-based Pareto search instead.")

    if HAS_PYMOO:
        import CoolProp.CoolProp as CP
        from orc_model import ORCSystem

        # Full candidate set — includes the paper's thermal-performance winners.
        # GWP and safety class: Toluene GWP=3 A3 (orc_model.WORKING_FLUIDS confirmed);
        # Cyclohexane GWP=2 A3 (ibid). CAPEX ~2300 USD/kW (comparable to n-Pentane
        # balance-of-plant; no published specific figure — stated as approximate in paper).
        fluids = ['R245fa', 'R1233zd(E)', 'R1234ze(E)', 'n-Pentane', 'Toluene', 'Cyclohexane']
        fluid_gwp = {'R245fa': 1030, 'R1233zd(E)': 4.5, 'R1234ze(E)': 7, 'n-Pentane': 5,
                     'Toluene': 3, 'Cyclohexane': 2}
        fluid_capex = {'R245fa': 2500, 'R1233zd(E)': 2400, 'R1234ze(E)': 2600, 'n-Pentane': 2200,
                       'Toluene': 2300, 'Cyclohexane': 2300}

        class WHROptProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=5,
                    n_obj=3,
                    n_constr=0,
                    xl=np.array([60, 35, 0, 0, 0.50]),
                    xu=np.array([145, 45, 30, 5.99, 1.00]),  # 6 fluids: idx 0..5
                )

            def _evaluate(self, X, out, *args, **kwargs):
                F = np.zeros((len(X), 3))

                for i, x in enumerate(X):
                    T_evap = x[0]
                    T_cond = x[1]
                    T_sh = x[2]
                    fluid_idx = int(x[3])
                    load = x[4]

                    fluid_name = fluids[max(0, min(fluid_idx, 5))]  # 6-fluid set
                    gwp = fluid_gwp[fluid_name]
                    capex_kw = fluid_capex[fluid_name]

                    try:
                        orc = ORCSystem(fluid_name)

                        T_exh = float(engine.exhaust_gas_temp_after_TC(load))
                        m_exh = float(engine.exhaust_mass_flow(load))
                        cp_exh = float(engine.exhaust_cp(T_exh))
                        T_exh_out = max(150, T_exh - 100)
                        Q_avail = m_exh * cp_exh * (T_exh - T_exh_out)

                        r = orc.size_for_heat_source(Q_avail, T_exh, T_exh_out,
                                                      T_evap, T_cond, T_sh)

                        if r is not None:
                            W_net = r['W_net_kW']
                            m_wf = r['m_dot_wf_kg_s']

                            # Lifecycle cost (20 years)
                            capex = W_net * capex_kw
                            fuel_saved_yr = W_net * 280 * 24 * 210 / 1e6  # tonnes/yr
                            annual_savings = fuel_saved_yr * 550 + fuel_saved_yr * 3.114 * 55
                            maintenance = capex * 0.03
                            lifecycle_cost = capex + 20 * maintenance - 20 * annual_savings

                            # Environmental score: GWP × refrigerant charge mass.
                            # Charge mass = 4 kg/kW_net (consistent with TEWI module
                            # CHARGE_KG_PER_KW = 4.0; Sprouse 2013).
                            # Previously used gwp * m_wf * 5 where m_wf is a mass
                            # flow rate (kg/s), giving an odd unit. Corrected here to
                            # gwp * W_net * 4.0 [kgCO2e_equiv], a static charge proxy.
                            env_score = gwp * W_net * 4.0

                            F[i] = [-W_net, lifecycle_cost, env_score]  # Minimize all
                        else:
                            F[i] = [0, 1e10, 1e10]
                    except Exception:
                        F[i] = [0, 1e10, 1e10]

                out["F"] = F

        print(f"\n[NSGA-II] Running optimization: {n_gen} generations × {pop_size} population")
        problem = WHROptProblem()

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
        )

        termination = get_termination("n_gen", n_gen)

        res = minimize(
            problem, algorithm, termination,
            seed=42, verbose=True, save_history=True
        )

        # Extract Pareto front
        pareto_F = res.F
        pareto_X = res.X

        pareto_results = []
        for i in range(len(pareto_X)):
            x = pareto_X[i]
            fluid_idx = int(x[3])
            fluid_name = fluids[max(0, min(fluid_idx, 5))]  # 6-fluid set

            pareto_results.append({
                'T_evap_C': x[0],
                'T_cond_C': x[1],
                'T_superheat_C': x[2],
                'fluid': fluid_name,
                'engine_load': x[4],
                'W_net_kW': -pareto_F[i, 0],
                'lifecycle_cost_USD': pareto_F[i, 1],
                'env_impact_score': pareto_F[i, 2],
            })

        pareto_df = pd.DataFrame(pareto_results)
        pareto_df.to_csv(os.path.join(output_dir, '06_pareto_front.csv'), index=False)

        print(f"\n[NSGA-II] Pareto front: {len(pareto_df)} solutions")
        print(f"[NSGA-II] W_net range: {pareto_df['W_net_kW'].min():.0f} - "
              f"{pareto_df['W_net_kW'].max():.0f} kW")
        print(f"[NSGA-II] Best fluids on Pareto front:")
        print(pareto_df['fluid'].value_counts().to_string())

        # Convergence history
        if res.history is not None:
            conv_data = []
            for gen_i, algo in enumerate(res.history):
                if algo.opt is not None:
                    best_W = -algo.opt.get("F")[:, 0].min()
                    conv_data.append({'generation': gen_i, 'best_W_net_kW': best_W})
            if conv_data:
                conv_df = pd.DataFrame(conv_data)
                conv_df.to_csv(os.path.join(output_dir, '06b_nsga2_convergence.csv'), index=False)

        return pareto_df

    else:
        # Fallback: Grid-based Pareto approximation
        print("[Grid Pareto] Running dense grid search as fallback...")
        import CoolProp.CoolProp as CP
        from orc_model import ORCSystem

        fluids = ['R245fa', 'R1233zd(E)', 'R1234ze(E)', 'n-Pentane', 'Toluene', 'Cyclohexane']
        results = []

        for fluid in tqdm(fluids, desc="Grid Pareto"):
            orc = ORCSystem(fluid)
            for load in np.arange(0.50, 1.01, 0.05):
                T_exh = float(engine.exhaust_gas_temp_after_TC(load))
                m_exh = float(engine.exhaust_mass_flow(load))
                cp_exh = float(engine.exhaust_cp(T_exh))
                T_exh_out = max(150, T_exh - 100)
                Q_avail = m_exh * cp_exh * (T_exh - T_exh_out)

                for T_evap in np.arange(65, 145, 5):
                    for T_cond in [36, 38, 40, 42]:
                        for T_sh in [0, 10, 20]:
                            r = orc.size_for_heat_source(Q_avail, T_exh, T_exh_out,
                                                          T_evap, T_cond, T_sh)
                            if r is not None:
                                results.append({
                                    'fluid': fluid,
                                    'engine_load': load,
                                    'T_evap_C': T_evap,
                                    'T_cond_C': T_cond,
                                    'T_superheat_C': T_sh,
                                    'W_net_kW': r['W_net_kW'],
                                    'eta_thermal': r['eta_thermal'],
                                })

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, '06_grid_search_results.csv'), index=False)
        print(f"[Grid Pareto] {len(df)} valid points found")
        return df


if __name__ == '__main__':
    from engine_model import S65MEC_Engine, run_engine_analysis
    engine, engine_df = run_engine_analysis()

    # Monte Carlo: NOT re-run here (unchanged; existing 05_monte_carlo_results.csv stands).

    # NSGA-II: reduced pop/gen for the C1-substantive fix re-run.
    # Production run used 200 gen × 100 pop; this re-run uses 100 gen × 50 pop
    # (sufficient to populate the 6-fluid front; stated in paper as verification run).
    # Increase to n_gen=200, pop_size=100 for final publication run.
    run_nsga2_optimization(engine, output_dir='results', n_gen=100, pop_size=50)
