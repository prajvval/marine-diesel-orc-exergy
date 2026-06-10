"""
Module 3: Exergy (2nd Law) Analysis
====================================
Performs exergy analysis on the engine system and each WHR technology.
Quantifies irreversibilities, exergy destruction, and 2nd-law efficiency.

This is the key physics contribution that separates journal papers from
textbook summaries.

Author: Prajwal Rao
"""

import numpy as np
import pandas as pd


class ExergyAnalyzer:
    """
    Exergy (availability) analysis for marine diesel engine and WHR systems.

    Dead state: seawater temperature and atmospheric pressure.
    """

    def __init__(self, T0_C=29.0, P0_kPa=101.325):
        """
        Parameters
        ----------
        T0_C : float
            Dead-state temperature (°C). Indian Ocean seawater ~29°C.
        P0_kPa : float
            Dead-state pressure (kPa). Atmospheric.
        """
        self.T0 = T0_C + 273.15  # K
        self.P0 = P0_kPa * 1000  # Pa

    def heat_exergy(self, Q_kW, T_source_C):
        """
        Exergy of a heat stream at constant temperature.
        Ex = Q * (1 - T0/T_source)

        For a stream with varying temperature, use heat_exergy_stream instead.
        """
        T_source = np.asarray(T_source_C) + 273.15
        carnot = 1.0 - self.T0 / T_source
        return np.asarray(Q_kW) * carnot

    def heat_exergy_stream(self, Q_kW, T_in_C, T_out_C):
        """
        Flow exergy of a sensible-heat stream cooling from T_in to T_out.

        For a constant-cp stream, rigorous specific exergy is
            ε = cp·[(T_in - T_out) - T0·ln(T_in/T_out)]
        and with Q = m·cp·(T_in - T_out) this becomes
            Ex = Q · [1 - T0·ln(T_in/T_out) / (T_in - T_out)]
        i.e. the Carnot factor evaluated at the log-mean temperature.
        This is exact for constant cp and more accurate than a simple
        arithmetic-mean approximation used previously.
        """
        T_in = np.asarray(T_in_C, dtype=float) + 273.15
        T_out = np.asarray(T_out_C, dtype=float) + 273.15
        Q = np.asarray(Q_kW, dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            dT = T_in - T_out
            carnot = np.where(
                np.abs(dT) > 1e-3,
                1.0 - self.T0 * np.log(T_in / T_out) / dT,
                1.0 - self.T0 / (0.5 * (T_in + T_out)),
            )
        # Non-physical if stream at/below dead state
        carnot = np.clip(carnot, 0.0, 1.0)
        return Q * carnot

    def fuel_exergy(self, Q_fuel_kW, fuel_type='HFO'):
        """
        Chemical exergy of fuel input.
        For hydrocarbon fuels, exergy ≈ 1.04-1.06 × LHV.
        """
        if fuel_type == 'HFO':
            phi = 1.06  # Exergy-to-energy ratio for heavy fuel oil
        elif fuel_type == 'MDO':
            phi = 1.04
        else:
            phi = 1.05
        return np.asarray(Q_fuel_kW) * phi

    def work_exergy(self, W_kW):
        """Work is pure exergy."""
        return np.asarray(W_kW)

    def engine_exergy_balance(self, engine, load_fractions):
        """
        Complete exergy balance for the engine across load conditions.

        Returns DataFrame with exergy streams and destruction.
        """
        loads = np.asarray(load_fractions)

        Q_fuel = engine.fuel_energy_input(loads)
        W_brake = engine.brake_power(loads)
        T_exh = engine.exhaust_gas_temp_after_TC(loads)
        T0_C = self.T0 - 273.15

        # Exhaust heat referenced to the dead state (seawater), so the
        # subsequent exergy calculation is thermodynamically consistent.
        Q_exhaust = engine.exhaust_heat(loads, T_ref_C=T0_C)

        # Cooling system temperatures (representative)
        T_jacket_in_C = 80.0   # Jacket water outlet (high temp side)
        T_jacket_out_C = 72.0  # Jacket water inlet
        T_charge_in_C = 180.0  # Charge air before cooler
        T_charge_out_C = 40.0  # Charge air after cooler
        T_lube_C = 55.0        # Lube oil average

        Q_jacket = engine.jacket_water_heat(loads)
        Q_charge = engine.charge_air_heat(loads)
        Q_lube = engine.lube_oil_heat(loads)
        Q_radiation = engine.radiation_loss(loads)

        # Exergy calculations (all heat streams referenced to dead state T0)
        Ex_fuel = self.fuel_exergy(Q_fuel)
        Ex_work = self.work_exergy(W_brake)
        Ex_exhaust = self.heat_exergy_stream(Q_exhaust, T_exh, T0_C)
        Ex_jacket = self.heat_exergy(Q_jacket, (T_jacket_in_C + T_jacket_out_C) / 2)
        Ex_charge = self.heat_exergy_stream(Q_charge, T_charge_in_C, T_charge_out_C)
        Ex_lube = self.heat_exergy(Q_lube, T_lube_C)
        Ex_radiation = self.heat_exergy(Q_radiation, 200)  # Approx surface temp

        # Exergy destruction (by difference)
        Ex_destroyed = Ex_fuel - Ex_work - Ex_exhaust - Ex_jacket - Ex_charge - Ex_lube - Ex_radiation

        # Efficiencies
        eta_exergy = Ex_work / Ex_fuel
        eta_exergy_with_whr_potential = (Ex_work + Ex_exhaust) / Ex_fuel

        df = pd.DataFrame({
            'Load_pct': loads * 100,
            'Ex_fuel_kW': Ex_fuel,
            'Ex_work_kW': Ex_work,
            'Ex_exhaust_kW': Ex_exhaust,
            'Ex_jacket_kW': Ex_jacket,
            'Ex_charge_air_kW': Ex_charge,
            'Ex_lube_kW': Ex_lube,
            'Ex_radiation_kW': Ex_radiation,
            'Ex_destroyed_kW': Ex_destroyed,
            'eta_exergy': eta_exergy,
            'eta_exergy_whr': eta_exergy_with_whr_potential,
            'irreversibility_ratio': Ex_destroyed / Ex_fuel,
        })

        # Percentage columns
        for col in ['Ex_work_kW', 'Ex_exhaust_kW', 'Ex_jacket_kW',
                     'Ex_charge_air_kW', 'Ex_lube_kW', 'Ex_radiation_kW', 'Ex_destroyed_kW']:
            pct_col = col.replace('_kW', '_pct')
            df[pct_col] = df[col] / df['Ex_fuel_kW'] * 100

        return df

    def orc_exergy_analysis(self, orc_results_df):
        """
        Exergy analysis for each ORC operating point.

        Adds exergy destruction in each component (pump, evaporator, expander, condenser).
        """
        if len(orc_results_df) == 0:
            return orc_results_df

        df = orc_results_df.copy()

        T0 = self.T0

        # Exergy at each state point: ex = (h - h0) - T0*(s - s0)
        # We approximate h0 and s0 as condenser outlet conditions
        # (close to dead state for properly designed system)

        # Exergy input from heat source
        T_source_avg = (df['T_source_in_C'] + df['T_source_out_C']) / 2 + 273.15
        df['Ex_heat_input_kW'] = df['Q_in_kW'] * (1 - T0 / T_source_avg)

        # Exergy of net work output
        df['Ex_work_output_kW'] = df['W_net_kW']

        # Exergy destruction in the ORC cycle
        df['Ex_destroyed_orc_kW'] = df['Ex_heat_input_kW'] - df['Ex_work_output_kW']

        # Second-law efficiency of ORC
        df['eta_2nd_law_orc'] = np.where(
            df['Ex_heat_input_kW'] > 0,
            df['Ex_work_output_kW'] / df['Ex_heat_input_kW'],
            0
        )

        # Component-level breakdown.
        # NOTE: This is a *literature-typical allocation* for subcritical
        # ORC with moderate-grade heat sources (see e.g. Quoilin et al.,
        # Appl. Therm. Eng. 51 (2013); Lecompte et al., Appl. Energy 150
        # (2015)). Evaporator irreversibility dominates because of the
        # large finite-ΔT between exhaust gas and boiling organic fluid.
        # For state-point-level exergy destruction, a full component
        # second-law balance (not used here for speed) should replace
        # this split. [TODO: add component-level audit for final paper.]
        FRAC_EVAP, FRAC_EXP, FRAC_PUMP = 0.55, 0.30, 0.03  # sum 0.88
        df['Ex_destroyed_evaporator_kW'] = df['Ex_destroyed_orc_kW'] * FRAC_EVAP
        df['Ex_destroyed_expander_kW']   = df['Ex_destroyed_orc_kW'] * FRAC_EXP
        df['Ex_destroyed_pump_kW']       = df['Ex_destroyed_orc_kW'] * FRAC_PUMP
        df['Ex_destroyed_condenser_kW']  = (df['Ex_destroyed_orc_kW']
                                            - df['Ex_destroyed_evaporator_kW']
                                            - df['Ex_destroyed_expander_kW']
                                            - df['Ex_destroyed_pump_kW'])

        return df


def run_exergy_analysis(engine, engine_df, orc_df=None, output_dir='results'):
    """Run complete exergy analysis and save results."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    analyzer = ExergyAnalyzer(T0_C=29.0)

    loads = np.linspace(0.20, 1.00, 81)
    exergy_df = analyzer.engine_exergy_balance(engine, loads)
    exergy_df.to_csv(os.path.join(output_dir, '03_engine_exergy_balance.csv'), index=False)

    print(f"\n[Exergy Model] Engine exergy balance computed for {len(loads)} load points")
    print(f"[Exergy Model] Exergy efficiency range: "
          f"{exergy_df['eta_exergy'].min():.3f} - {exergy_df['eta_exergy'].max():.3f}")
    print(f"[Exergy Model] Irreversibility ratio range: "
          f"{exergy_df['irreversibility_ratio'].min():.3f} - {exergy_df['irreversibility_ratio'].max():.3f}")

    if orc_df is not None and len(orc_df) > 0:
        orc_exergy_df = analyzer.orc_exergy_analysis(orc_df)
        orc_exergy_df.to_csv(os.path.join(output_dir, '03b_orc_exergy_analysis.csv'), index=False)
        print(f"[Exergy Model] ORC exergy analysis: {len(orc_exergy_df)} points")
        print(f"[Exergy Model] ORC 2nd law efficiency range: "
              f"{orc_exergy_df['eta_2nd_law_orc'].min():.3f} - "
              f"{orc_exergy_df['eta_2nd_law_orc'].max():.3f}")
    else:
        orc_exergy_df = None

    return exergy_df, orc_exergy_df


if __name__ == '__main__':
    from engine_model import S65MEC_Engine, run_engine_analysis
    engine, engine_df = run_engine_analysis()
    run_exergy_analysis(engine, engine_df)
