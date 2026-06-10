"""
Module 2: Organic Rankine Cycle (ORC) Simulation
=================================================
Models ORC waste heat recovery with multiple working fluids using CoolProp.
Performs parametric sweeps over evaporator pressure, superheat, and condenser
temperature for each fluid.

Author: Prajwal Rao
"""

import logging
import numpy as np
import pandas as pd
from itertools import product
import CoolProp.CoolProp as CP

logger = logging.getLogger(__name__)


# Working fluids to evaluate. Keys are user-facing identifiers; 'name'
# is the EXACT CoolProp fluid string (case-sensitive). Validated at
# import time — see _validate_fluid_names() below.
WORKING_FLUIDS = {
    'R245fa':     {'name': 'R245fa',     'GWP': 1030, 'ODP': 0, 'safety': 'B1',  'T_crit_C': 154.0},
    'R1234ze(E)': {'name': 'R1234ze(E)', 'GWP': 7,    'ODP': 0, 'safety': 'A2L', 'T_crit_C': 109.4},
    'R1233zd(E)': {'name': 'R1233zd(E)', 'GWP': 4.5,  'ODP': 0, 'safety': 'A1',  'T_crit_C': 166.5},
    'R134a':      {'name': 'R134a',      'GWP': 1430, 'ODP': 0, 'safety': 'A1',  'T_crit_C': 101.1},
    'R601':       {'name': 'n-Pentane',  'GWP': 5,    'ODP': 0, 'safety': 'A3',  'T_crit_C': 196.6},
    'R600a':      {'name': 'Isobutane',  'GWP': 3,    'ODP': 0, 'safety': 'A3',  'T_crit_C': 134.7},
    'Cyclohexane':{'name': 'Cyclohexane','GWP': 2,    'ODP': 0, 'safety': 'A3',  'T_crit_C': 280.5},
    'Toluene':    {'name': 'Toluene',    'GWP': 3,    'ODP': 0, 'safety': 'A3',  'T_crit_C': 318.6},
}


def _validate_fluid_names():
    """Confirm every CoolProp fluid name is resolvable. Raises on failure."""
    bad = []
    for key, info in WORKING_FLUIDS.items():
        try:
            CP.PropsSI('Tcrit', info['name'])
        except Exception as exc:
            bad.append(f"  {key} -> CoolProp name '{info['name']}': {exc}")
    if bad:
        raise RuntimeError(
            "CoolProp cannot resolve the following working-fluid names:\n"
            + "\n".join(bad)
            + "\nFix WORKING_FLUIDS before running the overnight sweep."
        )


_validate_fluid_names()


class ORCSystem:
    """
    Subcritical Organic Rankine Cycle with optional regenerator.

    Cycle state points:
        1 -> 2 : Pump (isentropic with eta_pump)
        2 -> 3 : Preheater + Evaporator (+ optional superheat)
        3 -> 4 : Expander (isentropic with eta_expander)
        4 -> 1 : Condenser (+ optional regenerator between 4->1 and 2->3)
    """

    def __init__(self, fluid_name, eta_pump=0.70, eta_expander=0.80,
                 eta_regenerator=0.0, pinch_point_K=10.0):
        """
        Parameters
        ----------
        fluid_name : str
            CoolProp fluid identifier
        eta_pump : float
            Isentropic pump efficiency
        eta_expander : float
            Isentropic expander efficiency
        eta_regenerator : float
            Regenerator effectiveness (0 = no regenerator)
        pinch_point_K : float
            Minimum pinch point temperature difference (K)
        """
        self.fluid = fluid_name
        self.eta_pump = eta_pump
        self.eta_exp = eta_expander
        self.eta_regen = eta_regenerator
        self.pinch_K = pinch_point_K

        # Cache critical properties
        self.T_crit = CP.PropsSI('Tcrit', fluid_name)  # K
        self.P_crit = CP.PropsSI('pcrit', fluid_name)   # Pa

    def solve_cycle(self, T_evap_C, T_cond_C, T_superheat_C=0.0):
        """
        Solve the ORC thermodynamic cycle.

        Parameters
        ----------
        T_evap_C : float
            Evaporation temperature (°C)
        T_cond_C : float
            Condensation temperature (°C)
        T_superheat_C : float
            Superheat above saturation at evaporator exit (°C)

        Returns
        -------
        dict : Cycle performance metrics, or None if infeasible
        """
        try:
            T_evap = T_evap_C + 273.15  # K
            T_cond = T_cond_C + 273.15  # K

            # Feasibility checks
            if T_evap >= self.T_crit - 5.0:
                return None  # Too close to critical point
            if T_cond >= T_evap - 10.0:
                return None  # Insufficient cycle temperature difference
            if T_cond < 273.15:
                return None  # Sub-zero condenser: ignore
            # Refuse supercritical / near-critical condenser
            if T_cond >= self.T_crit - 20.0:
                return None

            # State 1: Condenser outlet (saturated liquid)
            P_cond = CP.PropsSI('P', 'T', T_cond, 'Q', 0, self.fluid)
            h1 = CP.PropsSI('H', 'T', T_cond, 'Q', 0, self.fluid)
            s1 = CP.PropsSI('S', 'T', T_cond, 'Q', 0, self.fluid)

            # Evaporator pressure (saturation at T_evap)
            P_evap = CP.PropsSI('P', 'T', T_evap, 'Q', 1, self.fluid)

            if P_evap <= P_cond or not np.isfinite(P_evap):
                return None
            if P_evap >= 0.95 * self.P_crit:
                return None  # Subcritical cycle only

            # State 2: Pump outlet (isentropic + efficiency)
            h2s = CP.PropsSI('H', 'P', P_evap, 'S', s1, self.fluid)
            h2 = h1 + (h2s - h1) / self.eta_pump
            T2 = CP.PropsSI('T', 'P', P_evap, 'H', h2, self.fluid)

            # State 3: Expander inlet (saturated or superheated vapor)
            # NOTE: T_evap is already in Kelvin. Superheat in K == °C magnitude.
            if T_superheat_C > 0:
                T3_K = T_evap + T_superheat_C
                # Clamp below critical to avoid CoolProp failure
                T3_K = min(T3_K, self.T_crit - 1.0)
                h3 = CP.PropsSI('H', 'T', T3_K, 'P', P_evap, self.fluid)
                s3 = CP.PropsSI('S', 'T', T3_K, 'P', P_evap, self.fluid)
            else:
                h3 = CP.PropsSI('H', 'T', T_evap, 'Q', 1, self.fluid)
                s3 = CP.PropsSI('S', 'T', T_evap, 'Q', 1, self.fluid)
                T3_K = T_evap

            # State 4: Expander outlet (isentropic + efficiency)
            h4s = CP.PropsSI('H', 'P', P_cond, 'S', s3, self.fluid)
            h4 = h3 - self.eta_exp * (h3 - h4s)
            if not np.isfinite(h4):
                return None
            T4 = CP.PropsSI('T', 'P', P_cond, 'H', h4, self.fluid)
            s4 = CP.PropsSI('S', 'P', P_cond, 'H', h4, self.fluid)

            # Reject heavily wet expansion (quality < 0.85 at exit)
            try:
                x4 = CP.PropsSI('Q', 'P', P_cond, 'H', h4, self.fluid)
                if np.isfinite(x4) and 0.0 <= x4 < 0.85:
                    return None
            except Exception:
                pass  # Superheated outlet — Q undefined, that's fine

            # Specific work and heat
            w_pump = h2 - h1           # J/kg (positive, work input)
            w_expander = h3 - h4       # J/kg (positive, work output)
            w_net = w_expander - w_pump
            q_in = h3 - h2             # J/kg heat input from exhaust

            if w_net <= 0 or q_in <= 0:
                return None

            eta_thermal = w_net / q_in
            back_work_ratio = w_pump / w_expander

            # Sanity clamp: reject non-physical thermal efficiencies.
            # Subcritical ORC with finite source temperatures cannot
            # exceed Carnot_max(T_evap, T_cond), which is always <1.
            if not np.isfinite(eta_thermal) or eta_thermal <= 0 or eta_thermal > 0.40:
                return None

            # 2nd-law efficiency vs Carnot between working-fluid peak
            # (evaporator saturation) and the condenser sink. For the
            # exhaust-source comparison, see exergy_model.orc_exergy_analysis.
            carnot_eta = 1.0 - T_cond / T_evap
            eta_exergy = eta_thermal / carnot_eta if carnot_eta > 1e-6 else 0.0

            # Pressure ratio
            pressure_ratio = P_evap / P_cond

            # Volumetric expansion ratio (important for expander sizing)
            rho3 = CP.PropsSI('D', 'T', T3_K, 'P', P_evap, self.fluid)
            rho4 = CP.PropsSI('D', 'P', P_cond, 'H', h4, self.fluid)
            vol_ratio = rho3 / rho4 if rho4 > 0 else float('inf')

            return {
                'fluid': self.fluid,
                'T_evap_C': T_evap_C,
                'T_cond_C': T_cond_C,
                'T_superheat_C': T_superheat_C,
                'P_evap_kPa': P_evap / 1000,
                'P_cond_kPa': P_cond / 1000,
                'pressure_ratio': pressure_ratio,
                'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4,
                's1': s1, 's3': s3, 's4': s4,
                'T2_C': T2 - 273.15,
                'T3_C': T3_K - 273.15,
                'T4_C': T4 - 273.15,
                'w_pump_kJ_kg': w_pump / 1000,
                'w_expander_kJ_kg': w_expander / 1000,
                'w_net_kJ_kg': w_net / 1000,
                'q_in_kJ_kg': q_in / 1000,
                'eta_thermal': eta_thermal,
                'eta_exergy': eta_exergy,
                'back_work_ratio': back_work_ratio,
                'vol_expansion_ratio': vol_ratio,
            }

        except Exception as exc:
            logger.debug(
                "ORC cycle solve failed (%s, T_evap=%.1f, T_cond=%.1f, "
                "T_sh=%.1f): %s",
                self.fluid, T_evap_C, T_cond_C, T_superheat_C, exc,
            )
            return None

    def size_for_heat_source(self, Q_available_kW, T_source_in_C, T_source_out_C,
                             T_evap_C, T_cond_C, T_superheat_C=0.0):
        """
        Size ORC for a given heat source (exhaust gas stream).

        Parameters
        ----------
        Q_available_kW : float
            Available thermal power from heat source (kW)
        T_source_in_C : float
            Heat source inlet temperature (°C) — exhaust gas temp
        T_source_out_C : float
            Heat source outlet temperature (°C) — after heat exchange
        T_evap_C, T_cond_C, T_superheat_C : float
            ORC operating conditions

        Returns
        -------
        dict : Sized system performance, or None if infeasible
        """
        cycle = self.solve_cycle(T_evap_C, T_cond_C, T_superheat_C)
        if cycle is None:
            return None

        # Pinch point check
        if T_source_out_C < T_evap_C + self.pinch_K:
            return None
        if T_source_in_C < cycle['T3_C'] + self.pinch_K:
            return None

        # Mass flow rate of working fluid
        q_in = cycle['q_in_kJ_kg']  # kJ/kg
        if q_in <= 0:
            return None
        m_dot_wf = Q_available_kW / q_in  # kg/s

        # Power output
        W_net_kW = m_dot_wf * cycle['w_net_kJ_kg']
        W_pump_kW = m_dot_wf * cycle['w_pump_kJ_kg']
        W_exp_kW = m_dot_wf * cycle['w_expander_kJ_kg']

        cycle.update({
            'Q_in_kW': Q_available_kW,
            'T_source_in_C': T_source_in_C,
            'T_source_out_C': T_source_out_C,
            'm_dot_wf_kg_s': m_dot_wf,
            'W_net_kW': W_net_kW,
            'W_pump_kW': W_pump_kW,
            'W_expander_kW': W_exp_kW,
        })

        return cycle


def parametric_sweep_single_fluid(fluid_name, T_source_C, T_cond_C,
                                   T_evap_range, superheat_range,
                                   Q_available_kW, T_source_out_C):
    """
    Parametric sweep over evaporator temperature and superheat for one fluid.
    """
    results = []
    orc = ORCSystem(fluid_name)

    for T_evap, T_sh in product(T_evap_range, superheat_range):
        r = orc.size_for_heat_source(
            Q_available_kW, T_source_C, T_source_out_C,
            T_evap, T_cond_C, T_sh
        )
        if r is not None:
            results.append(r)

    return results


def run_orc_analysis(engine_df, output_dir='results'):
    """
    Run comprehensive ORC parametric study across multiple fluids and loads.

    This is the computationally intensive part — designed to run for hours
    with fine parameter resolution.
    """
    import os
    from tqdm import tqdm
    os.makedirs(output_dir, exist_ok=True)

    # Condenser temperature depends on seawater temp (~29°C Indian Ocean)
    T_cond_C = 38.0  # 29°C seawater + ~9°C approach

    # Evaporator temperature sweep (°C)
    T_evap_range = np.arange(60, 155, 2.5)  # Fine sweep

    # Superheat sweep (°C)
    superheat_range = np.arange(0, 35, 5)

    # Load conditions to analyze
    load_indices = [0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 1.00]

    all_results = []
    fluids_to_run = list(WORKING_FLUIDS.keys())

    from engine_model import S65MEC_Engine
    engine = S65MEC_Engine(ambient_temp_C=30, seawater_temp_C=29)

    total_combos = len(fluids_to_run) * len(load_indices)
    print(f"\n[ORC Model] Starting parametric sweep: {len(fluids_to_run)} fluids × "
          f"{len(load_indices)} loads × {len(T_evap_range)} evap temps × "
          f"{len(superheat_range)} superheat = "
          f"{len(fluids_to_run)*len(load_indices)*len(T_evap_range)*len(superheat_range)} cases")

    pbar = tqdm(total=total_combos, desc="ORC sweep")

    for fluid_key in fluids_to_run:
        fluid_info = WORKING_FLUIDS[fluid_key]
        cp_name = fluid_info['name']

        for load in load_indices:
            T_exh = float(engine.exhaust_gas_temp_after_TC(load))
            m_exh = float(engine.exhaust_mass_flow(load))
            cp_exh = float(engine.exhaust_cp(T_exh))

            # Minimum exhaust outlet temp: acid dew point (~150°C for HFO with sulfur)
            T_exh_out = max(150.0, T_exh - 120)

            # Available heat
            Q_avail = m_exh * cp_exh * (T_exh - T_exh_out)

            try:
                results = parametric_sweep_single_fluid(
                    cp_name, T_exh, T_cond_C,
                    T_evap_range, superheat_range,
                    Q_avail, T_exh_out
                )

                for r in results:
                    r['engine_load_pct'] = load * 100
                    r['fluid_key'] = fluid_key
                    r['GWP'] = fluid_info['GWP']
                    r['ODP'] = fluid_info['ODP']
                    r['safety_class'] = fluid_info['safety']
                    all_results.append(r)
            except Exception as e:
                print(f"  Warning: {fluid_key} at {load*100}% failed: {e}")

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(all_results)

    # Sanity checks — a publication-grade subcritical ORC with
    # moderate-grade exhaust heat should sit in η_thermal ∈ [0.03, 0.25].
    # Fail loudly (warning, not crash) if the sweep produced garbage so
    # the overnight run can still finish and be triaged in the morning.
    if len(df) == 0:
        raise RuntimeError(
            "ORC parametric sweep produced NO valid operating points. "
            "Check fluid names, temperature ranges, and pinch constraints."
        )
    eta_min, eta_max = df['eta_thermal'].min(), df['eta_thermal'].max()
    if eta_max > 0.30 or eta_min < 0.0 or eta_max < 0.03:
        import warnings
        warnings.warn(
            f"ORC thermal efficiency outside expected range [0.03, 0.30]: "
            f"[{eta_min:.3f}, {eta_max:.3f}]. Check cycle solver.",
            RuntimeWarning,
        )
    if (df['W_net_kW'] <= 0).any():
        import warnings
        warnings.warn("Some ORC points have non-positive W_net — inspect.", RuntimeWarning)

    df.to_csv(os.path.join(output_dir, '02_orc_parametric_results.csv'), index=False)

    print(f"\n[ORC Model] {len(df)} valid ORC operating points found")
    print(f"[ORC Model] η_thermal range: {eta_min:.3f} - {eta_max:.3f}")
    print(f"[ORC Model] Saved to {output_dir}/02_orc_parametric_results.csv")

    # Find optimal conditions per fluid per load
    if len(df) > 0:
        optimal = df.loc[df.groupby(['fluid_key', 'engine_load_pct'])['W_net_kW'].idxmax()]
        optimal.to_csv(os.path.join(output_dir, '02b_orc_optimal_per_fluid.csv'), index=False)
        print(f"[ORC Model] Optimal conditions saved to {output_dir}/02b_orc_optimal_per_fluid.csv")

        print("\n[ORC Model] Best net power per fluid at 85% MCR:")
        at_85 = optimal[optimal['engine_load_pct'] == 85].sort_values('W_net_kW', ascending=False)
        for _, row in at_85.head(5).iterrows():
            print(f"  {row['fluid_key']:15s}: W_net={row['W_net_kW']:.1f} kW, "
                  f"η_th={row['eta_thermal']:.3f}, T_evap={row['T_evap_C']:.0f}°C")

    return df


if __name__ == '__main__':
    from engine_model import run_engine_analysis
    _, engine_df = run_engine_analysis()
    run_orc_analysis(engine_df)
