"""
Module 4: Environmental & Regulatory Impact Model
==================================================
Quantifies CO2 reduction, fuel savings, EEDI improvement, and CII rating
impact from waste heat recovery systems.

Provides the "Environmental Engineering" dimension of the paper.

Author: Prajwal Rao
"""

import numpy as np
import pandas as pd


class EnvironmentalModel:
    """
    Environmental impact and IMO regulatory compliance analysis.
    """

    # IMO emission factors (Fourth IMO GHG Study, 2020)
    EF_CO2_HFO = 3.114      # kgCO2 / kgFuel (HFO / residual fuel)
    EF_CO2_MDO = 3.206       # kgCO2 / kgFuel (MDO)
    EF_NOX_TIER2 = 0.087     # kgNOx / kgFuel (approximate Tier II)
    EF_SOX_HFO_35S = 0.054   # kgSOx / kgFuel (3.5% S HFO)
    EF_SOX_VLSFO = 0.010     # kgSOx / kgFuel (0.5% S VLSFO)
    EF_PM_HFO = 0.007        # kgPM / kgFuel (HFO)

    # Carbon pricing scenarios ($/tCO2)
    CARBON_PRICES = {
        'EU_ETS_2025': 55,
        'EU_ETS_2030_proj': 90,
        'IMO_levy_proposed': 150,
        'social_cost_carbon': 200,
    }

    def __init__(self, operating_days_per_year=280, hours_per_day=24,
                 fuel_price_usd_per_ton=550, fuel_type='HFO'):
        self.op_days = operating_days_per_year
        self.hours_day = hours_per_day
        self.annual_hours = operating_days_per_year * hours_per_day
        self.fuel_price = fuel_price_usd_per_ton
        self.fuel_type = fuel_type

        if fuel_type == 'HFO':
            self.ef_co2 = self.EF_CO2_HFO
            self.ef_sox = self.EF_SOX_HFO_35S
        else:
            self.ef_co2 = self.EF_CO2_MDO
            self.ef_sox = self.EF_SOX_VLSFO

    def annual_fuel_savings(self, W_whr_kW, bsfc_g_kWh=210.0):
        """
        Annual fuel saved (tonnes) by WHR displacing auxiliary load.

        Fuel displaced = W_whr [kW] × hours [h] × BSFC_aux [g/kWh]
        Returned in tonnes/year.

        Parameters
        ----------
        W_whr_kW : float or array
            Net WHR electrical output (kW).
        bsfc_g_kWh : float or array, default 210 g/kWh
            BSFC of displaced power source (auxiliary diesel genset).
        """
        bsfc_aux = np.where(np.asarray(bsfc_g_kWh) > 0, bsfc_g_kWh, 210.0)
        fuel_saved_g = np.asarray(W_whr_kW) * self.annual_hours * bsfc_aux  # g/yr
        return fuel_saved_g / 1e6  # tonnes/yr

    def annual_co2_reduction(self, fuel_saved_tonnes):
        """Annual CO2 reduction (tonnes)."""
        return np.asarray(fuel_saved_tonnes) * self.ef_co2

    def annual_nox_reduction(self, fuel_saved_tonnes):
        """Annual NOx reduction (tonnes)."""
        return np.asarray(fuel_saved_tonnes) * self.EF_NOX_TIER2

    def annual_sox_reduction(self, fuel_saved_tonnes):
        """Annual SOx reduction (tonnes)."""
        return np.asarray(fuel_saved_tonnes) * self.ef_sox

    def annual_pm_reduction(self, fuel_saved_tonnes):
        """Annual PM reduction (tonnes)."""
        return np.asarray(fuel_saved_tonnes) * self.EF_PM_HFO

    def eedi_improvement(self, engine_power_kW, W_whr_kW, vessel_dwt=80000,
                          Vref_knots=14.5):
        """
        Estimate EEDI improvement from WHR.

        Attained EEDI = (P_ME * SFC_ME * C_F) / (DWT * V_ref)
        WHR reduces effective P_ME by displacing auxiliary power.

        Parameters
        ----------
        engine_power_kW : float
            Main engine MCR power
        W_whr_kW : float or array
            WHR net power output
        vessel_dwt : float
            Vessel deadweight tonnage
        Vref_knots : float
            Reference speed (knots)

        Returns
        -------
        dict with baseline EEDI, improved EEDI, and % improvement
        """
        # EEDI [gCO2 / (ton·nm)]:
        #     (P_ME·SFC_ME·CF + P_AE·SFC_AE·CF) / (DWT · V_ref)
        # with P in kW, SFC in g/kWh, CF in gCO2/gFuel, V in knots.
        SFC_ME = 175.0       # g/kWh, main engine
        SFC_AE = 210.0       # g/kWh, auxiliary genset
        P_AE = 0.05 * engine_power_kW     # typical bulk carrier
        CF = self.ef_co2                  # gCO2 per gFuel (≈3.114 for HFO)

        denom = vessel_dwt * Vref_knots   # ton · knots(=nm/h)

        numerator_base = engine_power_kW * SFC_ME * CF + P_AE * SFC_AE * CF
        eedi_base = numerator_base / denom

        # With WHR: auxiliary power demand reduced (cannot go negative)
        W_whr = np.asarray(W_whr_kW, dtype=float)
        P_AE_reduced = np.maximum(P_AE - W_whr, 0.0)

        numerator_whr = engine_power_kW * SFC_ME * CF + P_AE_reduced * SFC_AE * CF
        eedi_whr = numerator_whr / denom

        improvement_pct = (eedi_base - eedi_whr) / eedi_base * 100.0

        return {
            'eedi_baseline': eedi_base,
            'eedi_with_whr': eedi_whr,
            'eedi_improvement_pct': improvement_pct,
        }

    def cii_rating(self, annual_co2_tonnes, vessel_dwt=80000,
                    annual_distance_nm=100000):
        """
        Estimate CII (Carbon Intensity Indicator) rating.

        CII = Annual CO2 / (DWT × Distance)
        Ratings: A (much better), B (slightly better), C (moderate),
                 D (slightly worse), E (much worse)

        Parameters for bulk carrier reference lines (IMO MEPC.364(79)):
            Required CII = a × DWT^(-c)
        """
        cii_actual = np.asarray(annual_co2_tonnes) * 1e6 / (vessel_dwt * annual_distance_nm)

        # Reference line for bulk carriers (approximate coefficients)
        # CII_ref = 4745 * DWT^(-0.622) for 2023
        cii_ref_2023 = 4745 * vessel_dwt**(-0.622)

        # Annual reduction factors
        # 2024: 5%, 2025: 7%, 2026: 9%, 2027: 11%
        year = 2026
        reduction_factor = 1 - (0.05 + 0.02 * (year - 2024))  # 9% for 2026
        cii_required = cii_ref_2023 * reduction_factor

        # Rating boundaries (approximate)
        ratio = cii_actual / cii_required
        rating = np.where(ratio < 0.83, 'A',
                 np.where(ratio < 0.94, 'B',
                 np.where(ratio < 1.06, 'C',
                 np.where(ratio < 1.18, 'D', 'E'))))

        return {
            'cii_actual': cii_actual,
            'cii_required_2026': cii_required,
            'cii_ratio': ratio,
            'cii_rating': rating,
        }

    def economic_analysis(self, W_whr_kW, capex_per_kW=2500):
        """
        Simplified techno-economic analysis.

        Parameters
        ----------
        W_whr_kW : float or array
            Net WHR power output
        capex_per_kW : float
            Capital cost per kW of WHR system ($)
        """
        W = np.asarray(W_whr_kW)

        capex = W * capex_per_kW
        fuel_saved_tonnes = self.annual_fuel_savings(W, 210)
        annual_savings_fuel = fuel_saved_tonnes * self.fuel_price

        # Carbon credit revenue
        co2_saved = self.annual_co2_reduction(fuel_saved_tonnes)
        carbon_revenue = co2_saved * self.CARBON_PRICES['EU_ETS_2025']

        total_annual_savings = annual_savings_fuel + carbon_revenue

        # Maintenance cost (3% of CAPEX annually)
        annual_maintenance = capex * 0.03

        net_annual_savings = total_annual_savings - annual_maintenance
        payback_years = np.where(net_annual_savings > 0, capex / net_annual_savings, float('inf'))

        # NPV over 20 years at 8% discount rate
        discount_rate = 0.08
        npv = -capex
        for year in range(1, 21):
            npv += net_annual_savings / (1 + discount_rate)**year

        return {
            'CAPEX_USD': capex,
            'annual_fuel_savings_tonnes': fuel_saved_tonnes,
            'annual_fuel_cost_savings_USD': annual_savings_fuel,
            'annual_co2_reduction_tonnes': co2_saved,
            'annual_carbon_credit_USD': carbon_revenue,
            'annual_maintenance_USD': annual_maintenance,
            'net_annual_savings_USD': net_annual_savings,
            'payback_years': payback_years,
            'NPV_20yr_USD': npv,
        }


def run_environmental_analysis(engine, orc_optimal_df=None, output_dir='results'):
    """Run complete 4E environmental and economic analysis."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    env = EnvironmentalModel(
        operating_days_per_year=280,  # Indian coastal route
        fuel_price_usd_per_ton=550,   # HFO price
        fuel_type='HFO'
    )

    loads = np.linspace(0.20, 1.00, 81)

    results = []
    for load in loads:
        # Baseline engine emissions (no WHR).
        # m_fuel [kg/s] × 3600 [s/h] × annual_hours [h/yr] / 1000 [kg/t]
        m_fuel = float(engine.fuel_consumption(load))
        annual_fuel_tonnes = m_fuel * 3600.0 * env.annual_hours / 1000.0
        annual_co2_base = annual_fuel_tonnes * env.ef_co2

        # Assume WHR power scales with load (using a representative ORC η)
        T_exh = float(engine.exhaust_gas_temp_after_TC(load))
        m_exh = float(engine.exhaust_mass_flow(load))
        cp_exh = float(engine.exhaust_cp(T_exh))
        T_exh_out = max(150, T_exh - 100)
        Q_avail = m_exh * cp_exh * (T_exh - T_exh_out)

        # Representative ORC efficiencies for different systems
        whr_scenarios = {
            'Steam_Rankine': {'eta': 0.12, 'capex_per_kW': 2000},
            'ORC_R245fa': {'eta': 0.11, 'capex_per_kW': 2500},
            'ORC_R1233zd': {'eta': 0.105, 'capex_per_kW': 2400},
            'TEG': {'eta': 0.04, 'capex_per_kW': 5000},
            'Combined_ORC_TEG': {'eta': 0.13, 'capex_per_kW': 3500},
        }

        for tech_name, params in whr_scenarios.items():
            W_whr = Q_avail * params['eta']

            fuel_saved = env.annual_fuel_savings(W_whr, 210)
            co2_saved = env.annual_co2_reduction(fuel_saved)
            nox_saved = env.annual_nox_reduction(fuel_saved)
            sox_saved = env.annual_sox_reduction(fuel_saved)
            pm_saved = env.annual_pm_reduction(fuel_saved)

            eedi = env.eedi_improvement(engine.RATED_POWER_KW, W_whr)
            cii = env.cii_rating(annual_co2_base - co2_saved)
            econ = env.economic_analysis(W_whr, params['capex_per_kW'])

            results.append({
                'Load_pct': load * 100,
                'WHR_technology': tech_name,
                'Q_available_kW': Q_avail,
                'W_whr_kW': W_whr,
                'eta_whr': params['eta'],
                'annual_fuel_saved_tonnes': float(fuel_saved),
                'annual_co2_saved_tonnes': float(co2_saved),
                'annual_nox_saved_tonnes': float(nox_saved),
                'annual_sox_saved_tonnes': float(sox_saved),
                'annual_pm_saved_tonnes': float(pm_saved),
                'eedi_improvement_pct': float(eedi['eedi_improvement_pct']),
                'cii_ratio': float(cii['cii_ratio']),
                'CAPEX_USD': float(econ['CAPEX_USD']),
                'annual_savings_USD': float(econ['net_annual_savings_USD']),
                'payback_years': float(econ['payback_years']),
                'NPV_20yr_USD': float(econ['NPV_20yr_USD']),
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, '04_environmental_economic_4E.csv'), index=False)

    print(f"\n[Environmental Model] 4E analysis: {len(df)} data points")
    print(f"[Environmental Model] Saved to {output_dir}/04_environmental_economic_4E.csv")

    # Summary at 85% MCR
    at_85 = df[np.isclose(df['Load_pct'], 85, atol=1)]
    if len(at_85) > 0:
        print("\n[Environmental Model] Summary at 85% MCR:")
        for _, row in at_85.iterrows():
            print(f"  {row['WHR_technology']:20s}: W={row['W_whr_kW']:.0f} kW, "
                  f"CO2={row['annual_co2_saved_tonnes']:.0f} t/yr, "
                  f"Payback={row['payback_years']:.1f} yr, "
                  f"EEDI +{row['eedi_improvement_pct']:.2f}%")

    return df


if __name__ == '__main__':
    from engine_model import S65MEC_Engine, run_engine_analysis
    engine, engine_df = run_engine_analysis()
    run_environmental_analysis(engine)
