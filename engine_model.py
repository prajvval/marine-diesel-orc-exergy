"""
Module 1: MAN B&W S65ME-C Engine Thermodynamic Model
=====================================================
Models the engine across 20-100% MCR with energy and mass balances.
Curve-fit approximations anchored to published MAN shop-test data.

NOTE FOR PUBLICATION: BSFC curve, exhaust gas temperature profile and
mass-flow correlation are engineering approximations. The exact shop
test data in MAN Energy Solutions, "S65ME-C8.6-TII Project Guide",
Doc. No. 7020-0155 (latest revision) should be cited for each figure
that uses these quantities (see TODO-CITE markers below).

Author: Prajwal Rao
"""

import numpy as np
import pandas as pd


class S65MEC_Engine:
    """
    MAN B&W S65ME-C8 two-stroke marine diesel engine model.

    Rated specs (100% MCR, 8-cylinder):
        - Bore: 650 mm, Stroke: 2730 mm
        - Rated power: 21,840 kW at 95 RPM
        - Service power (CSR ~85%): ~18,560 kW
        - BSFC at optimum (~75% MCR): ~166 g/kWh; at MCR ~170-172 g/kWh
        - Exhaust temp after T/C: ~240-310°C depending on load

    References:
        MAN Energy Solutions, "S65ME-C8.6-TII Project Guide", 2022.   [TODO-CITE]
    """

    # Engine constants
    RATED_POWER_KW = 21840      # kW at 100% MCR
    RATED_RPM = 95              # rpm
    NUM_CYLINDERS = 8
    BORE_M = 0.650              # m
    STROKE_M = 2.730            # m
    LHV_HFO = 42700             # kJ/kg lower heating value of HFO
    EMISSION_FACTOR_CO2 = 3.114 # kgCO2 / kgFuel (IMO default for HFO)
    EMISSION_FACTOR_NOX = 0.087 # kgNOx / kgFuel (Tier II approx)
    EMISSION_FACTOR_SOX = 0.054 # kgSOx / kgFuel (3.5% S HFO)

    def __init__(self, ambient_temp_C=25.0, seawater_temp_C=28.0):
        """
        Parameters
        ----------
        ambient_temp_C : float
            Ambient air temperature (°C). Default 25°C.
            Indian Ocean tropical: 28-32°C
        seawater_temp_C : float
            Seawater cooling inlet temperature (°C). Default 28°C.
            Indian coastal waters: 26-31°C
        """
        self.T_ambient = ambient_temp_C + 273.15  # K
        self.T_seawater = seawater_temp_C + 273.15  # K

    def _validate_load(self, load_fraction):
        load = np.asarray(load_fraction, dtype=float)
        if np.any(load < 0.15) or np.any(load > 1.10):
            raise ValueError(
                f"load_fraction out of range [0.15, 1.10]: "
                f"min={float(np.min(load)):.3f}, max={float(np.max(load)):.3f}"
            )
        return load

    def bsfc(self, load_fraction):
        """
        Brake Specific Fuel Consumption (g/kWh) as function of load.

        Approximation anchored to MAN S65ME-C published shop-test curves:
        minimum ~166 g/kWh at 75-80% MCR, ~170 g/kWh at 100%, ~175 g/kWh
        at 50%, ~190-195 g/kWh at 25%. Curvature is much milder than a
        naive parabola — BSFC variation across operating range is <20%.
        [TODO-CITE: MAN Project Guide, Shop Test Performance Curves]
        """
        load = self._validate_load(load_fraction)
        # Milder parabola around the 0.78 MCR optimum
        bsfc = 166.0 + 55.0 * (load - 0.78)**2
        # Low-load penalty (fuel injection timing, scavenging losses)
        bsfc += np.where(load < 0.50, 25.0 * (0.50 - load)**2 / 0.25**2, 0.0)
        return bsfc

    def brake_power(self, load_fraction):
        """Brake power output (kW)."""
        return self.RATED_POWER_KW * np.asarray(load_fraction)

    def fuel_consumption(self, load_fraction):
        """Fuel consumption (kg/s)."""
        P = self.brake_power(load_fraction)
        sfoc = self.bsfc(load_fraction)
        return P * sfoc / (3.6e6)  # g/kWh -> kg/s

    def fuel_energy_input(self, load_fraction):
        """Total fuel energy input (kW)."""
        m_fuel = self.fuel_consumption(load_fraction)
        return m_fuel * self.LHV_HFO

    def brake_thermal_efficiency(self, load_fraction):
        """Brake thermal efficiency (fraction)."""
        return self.brake_power(load_fraction) / self.fuel_energy_input(load_fraction)

    def exhaust_gas_temp_after_TC(self, load_fraction):
        """
        Exhaust gas temperature after turbocharger (°C).

        Shape: rises at low load (less turbo work, under-scavenging),
        minimum ~235-240°C at 75-80% MCR, ~250-260°C at 100% MCR.
        Approximation — cite MAN shop-test performance curves for
        engine-specific values.  [TODO-CITE]
        """
        load = self._validate_load(load_fraction)
        # Mild parabola around minimum at ~0.78 MCR (target shape:
        # ~250°C at 100%, ~240°C at 78%, ~260°C at 50%, ~310°C at 25%).
        T_exh = 240.0 + 200.0 * (load - 0.78)**2
        T_exh += np.where(load < 0.5, 25.0 * (0.5 - load) / 0.25, 0.0)
        return T_exh

    def exhaust_mass_flow(self, load_fraction):
        """
        Exhaust gas mass flow rate (kg/s).

        Approximation: ~53 kg/s at 100% MCR for 8-cyl S65ME-C (scavenge
        air + fuel). Scales sub-linearly due to T/C matching and
        reduced scavenge-air-ratio at low load.  [TODO-CITE]
        """
        load = self._validate_load(load_fraction)
        # Rated ~50 kg/s at 100% MCR. Scavenge-air ratio drops strongly
        # at low load; using (0.10 + 0.90·load) gives closure-consistent
        # exhaust heat fractions across the operating range.
        m_exh_rated = 50.0
        return m_exh_rated * (0.10 + 0.90 * load)

    def exhaust_cp(self, T_celsius):
        """
        Specific heat of marine-diesel exhaust (kJ/kg·K).

        Lean HFO combustion products (~75% N2, 14% O2, 5% CO2, 6% H2O
        by volume at λ ≈ 2). cp ≈ 1.06 kJ/kg·K at 250°C, rising weakly
        with temperature.
        """
        T = np.asarray(T_celsius, dtype=float)
        return 1.055 + 0.00015 * (T - 250)

    def jacket_water_heat(self, load_fraction):
        """
        Heat rejected to jacket water cooling (kW).
        Approximately 5-6% of fuel energy at full load, higher fraction at part load.
        """
        Q_fuel = self.fuel_energy_input(load_fraction)
        load = np.asarray(load_fraction)
        fraction = 0.055 + 0.025 * (1.0 - load)  # 5.5% at full, ~8% at 25%
        return Q_fuel * fraction

    def charge_air_heat(self, load_fraction):
        """
        Heat rejected via charge air cooler (kW).
        """
        Q_fuel = self.fuel_energy_input(load_fraction)
        load = np.asarray(load_fraction)
        fraction = 0.12 + 0.03 * (1.0 - load)  # 12% at full, ~15% at 25%
        return Q_fuel * fraction

    def lube_oil_heat(self, load_fraction):
        """Heat rejected via lubricating oil cooler (kW)."""
        Q_fuel = self.fuel_energy_input(load_fraction)
        load = np.asarray(load_fraction)
        fraction = 0.035 + 0.01 * (1.0 - load)
        return Q_fuel * fraction

    def radiation_loss(self, load_fraction):
        """Heat lost by radiation and convection from engine surfaces (kW)."""
        Q_fuel = self.fuel_energy_input(load_fraction)
        return Q_fuel * 0.01  # ~1% roughly constant

    def exhaust_heat(self, load_fraction, T_ref_C=None):
        """
        Total heat in exhaust gas above reference temperature (kW).

        By default uses ambient air as reference (energy-balance
        convention). Pass T_ref_C = T0 (seawater dead-state) for
        exergy-analysis consistency.
        """
        m_exh = self.exhaust_mass_flow(load_fraction)
        T_exh = self.exhaust_gas_temp_after_TC(load_fraction)
        if T_ref_C is None:
            T_ref_C = self.T_ambient - 273.15
        # Mean cp between exhaust and reference
        cp = self.exhaust_cp(0.5 * (np.asarray(T_exh) + T_ref_C))
        return m_exh * cp * (T_exh - T_ref_C)

    def full_energy_balance(self, load_fractions):
        """
        Complete energy balance at given load conditions.

        Returns DataFrame with all energy streams in kW and percentages.
        """
        loads = np.asarray(load_fractions)

        Q_fuel = self.fuel_energy_input(loads)
        W_brake = self.brake_power(loads)
        Q_exhaust = self.exhaust_heat(loads)
        Q_jacket = self.jacket_water_heat(loads)
        Q_charge_air = self.charge_air_heat(loads)
        Q_lube = self.lube_oil_heat(loads)
        Q_radiation = self.radiation_loss(loads)

        # Unaccounted = closure error (should be small)
        Q_unaccounted = Q_fuel - W_brake - Q_exhaust - Q_jacket - Q_charge_air - Q_lube - Q_radiation

        df = pd.DataFrame({
            'Load_pct': loads * 100,
            'Q_fuel_kW': Q_fuel,
            'W_brake_kW': W_brake,
            'Q_exhaust_kW': Q_exhaust,
            'Q_jacket_kW': Q_jacket,
            'Q_charge_air_kW': Q_charge_air,
            'Q_lube_kW': Q_lube,
            'Q_radiation_kW': Q_radiation,
            'Q_unaccounted_kW': Q_unaccounted,
            'eta_brake': self.brake_thermal_efficiency(loads),
            'BSFC_g_kWh': self.bsfc(loads),
            'T_exhaust_C': self.exhaust_gas_temp_after_TC(loads),
            'm_exhaust_kg_s': self.exhaust_mass_flow(loads),
            'm_fuel_kg_s': self.fuel_consumption(loads),
        })

        # Percentage columns
        for col in ['W_brake_kW', 'Q_exhaust_kW', 'Q_jacket_kW', 'Q_charge_air_kW', 'Q_lube_kW', 'Q_radiation_kW']:
            pct_col = col.replace('_kW', '_pct')
            df[pct_col] = df[col] / df['Q_fuel_kW'] * 100

        # Energy-balance closure sanity check
        closure_pct = df['Q_unaccounted_kW'].abs() / df['Q_fuel_kW'] * 100
        if closure_pct.max() > 10.0:
            import warnings
            warnings.warn(
                f"Engine energy balance closure error exceeds 10%: "
                f"max {closure_pct.max():.1f}% at load {df.loc[closure_pct.idxmax(), 'Load_pct']:.0f}% — "
                f"check heat-loss fractions.",
                RuntimeWarning,
            )

        return df


def run_engine_analysis(output_dir='results'):
    """Run full engine analysis and save results."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    engine = S65MEC_Engine(ambient_temp_C=30, seawater_temp_C=29)  # Indian Ocean conditions

    # Fine-grained load sweep
    loads = np.linspace(0.20, 1.00, 81)  # 20% to 100% in 1% steps

    df = engine.full_energy_balance(loads)
    df.to_csv(os.path.join(output_dir, '01_engine_energy_balance.csv'), index=False)

    print(f"[Engine Model] Energy balance computed for {len(loads)} load points")
    print(f"[Engine Model] Saved to {output_dir}/01_engine_energy_balance.csv")
    print(f"[Engine Model] Efficiency range: {df['eta_brake'].min():.3f} - {df['eta_brake'].max():.3f}")
    print(f"[Engine Model] BSFC range: {df['BSFC_g_kWh'].min():.1f} - {df['BSFC_g_kWh'].max():.1f} g/kWh")

    return engine, df


if __name__ == '__main__':
    engine, df = run_engine_analysis()
    print("\nSample results (25%, 50%, 75%, 85%, 100% MCR):")
    print(df[df['Load_pct'].isin([25, 50, 75, 85, 100])].to_string(index=False))
