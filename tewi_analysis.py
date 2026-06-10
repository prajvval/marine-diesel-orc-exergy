"""
Module 10: Total Equivalent Warming Impact (TEWI) per ORC Working Fluid
========================================================================
Life-cycle climate impact over a 20-year system life:

    TEWI = Direct  +  Indirect
         = GWP * m_charge * (leak_rate * life + (1 - eol_recovery))
           -  annual_co2_saved * life           (sign convention: savings
                                                 are a credit, reducing TEWI)

Direct is the refrigerant release emission; indirect is the net
atmospheric balance from displaced genset combustion. A fluid with a low
GWP can be net-negative TEWI if it enables comparable WHR output.

Assumptions (per EN378 / UNEP-RTOC 2022 practice):
  - Charge mass: 4 kg per kW_net ORC output
  - Annual leak rate: 5%
  - Lifetime: 20 yr
  - End-of-life recovery: 90%
  - Net power reference: per-fluid best operating point at 85% MCR from
    the existing 02b_orc_optimal_per_fluid.csv
  - Annual CO2 savings: from 04_environmental_economic_4E.csv at 85% MCR
    using the auxiliary-genset displacement pathway

Shows that R1233zd(E) (GWP=4.5) can beat R245fa (GWP=1030) on lifecycle
climate impact even with comparable instantaneous cycle performance.

Author: Prajwal Rao
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# GWP100 values (IPCC AR6 where available; AR5 for older fluids)
FLUID_GWP = {
    "R245fa":      1030.0,
    "R1234ze(E)":     7.0,
    "R1233zd(E)":     4.5,
    "R134a":       1430.0,
    "R601":           5.0,    # n-Pentane
    "R600a":          3.0,    # Isobutane
    "Cyclohexane":    2.0,
    "Toluene":        3.0,
}

# Design parameters
CHARGE_KG_PER_KW = 4.0
LEAK_RATE_ANNUAL = 0.05
LIFETIME_YEARS = 20
EOL_RECOVERY = 0.90
LOAD_PCT_REF = 85.0
TARGET_WHR_TECH = "ORC_R245fa"  # ORC family entry in environmental CSV
LEAK_SENSITIVITY = [0.02, 0.05, 0.10]


def _load_inputs(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    orc = pd.read_csv(os.path.join(results_dir, "02b_orc_optimal_per_fluid.csv"))
    env = pd.read_csv(os.path.join(results_dir, "04_environmental_economic_4E.csv"))
    return orc, env


def _net_power_per_fluid(orc_df: pd.DataFrame) -> pd.DataFrame:
    sub = orc_df[np.isclose(orc_df["engine_load_pct"], LOAD_PCT_REF)]
    if sub.empty:
        raise RuntimeError("No 85% MCR rows in 02b_orc_optimal_per_fluid.csv")
    return sub.groupby("fluid_key", as_index=False).agg(
        W_net_kW=("W_net_kW", "max"),
    )


def _co2_savings_baseline(env_df: pd.DataFrame) -> float:
    """Representative annual CO2 savings (t/yr) per kW_net at 85% MCR, from
    the ORC_R245fa row (any ORC row works since the pathway is identical -
    only per-kW CO2 matters here)."""
    row = env_df[(env_df["WHR_technology"] == TARGET_WHR_TECH) &
                 (np.isclose(env_df["Load_pct"], LOAD_PCT_REF))]
    if row.empty:
        raise RuntimeError("Missing ORC_R245fa at 85% MCR in 04 CSV")
    r = row.iloc[0]
    return float(r["annual_co2_saved_tonnes"]) / float(r["W_whr_kW"])  # t/(kW*yr)


def tewi_rows(orc_df: pd.DataFrame, env_df: pd.DataFrame,
              leak_rate: float = LEAK_RATE_ANNUAL) -> pd.DataFrame:
    """TEWI per fluid at the reference leak rate."""
    power = _net_power_per_fluid(orc_df)
    co2_per_kW_yr = _co2_savings_baseline(env_df)

    rows = []
    for _, r in power.iterrows():
        fluid = r["fluid_key"]
        W = float(r["W_net_kW"])
        gwp = FLUID_GWP.get(fluid, np.nan)
        if not np.isfinite(gwp):
            continue
        m_charge = CHARGE_KG_PER_KW * W
        direct = gwp * m_charge * (leak_rate * LIFETIME_YEARS + (1.0 - EOL_RECOVERY))
        annual_co2 = co2_per_kW_yr * W
        indirect_savings = annual_co2 * LIFETIME_YEARS
        net_tewi = (direct - indirect_savings * 1000.0) / 1000.0  # kg -> tCO2e
        # direct is in kg CO2-eq (GWP * kg); indirect_savings is in t CO2.
        # Convert direct to tonnes then subtract savings:
        direct_t = direct / 1000.0
        net_tewi_t = direct_t - indirect_savings
        rows.append({
            "fluid":                     fluid,
            "GWP":                       gwp,
            "W_net_kW":                  W,
            "charge_mass_kg":            m_charge,
            "leak_rate":                 leak_rate,
            "direct_emissions_tCO2eq":   direct_t,
            "indirect_savings_tCO2":     indirect_savings,
            "net_TEWI_tCO2eq_per_lifetime": net_tewi_t,
        })
    out = pd.DataFrame(rows).sort_values("net_TEWI_tCO2eq_per_lifetime")
    out["TEWI_rank"] = np.arange(1, len(out) + 1)
    return out


def tewi_sensitivity(orc_df: pd.DataFrame,
                      env_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rate in LEAK_SENSITIVITY:
        df = tewi_rows(orc_df, env_df, leak_rate=rate)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def main(output_dir: str = "results") -> None:
    orc_df, env_df = _load_inputs(output_dir)
    main_df = tewi_rows(orc_df, env_df, leak_rate=LEAK_RATE_ANNUAL)
    sens_df = tewi_sensitivity(orc_df, env_df)

    main_path = os.path.join(output_dir, "10_tewi_per_fluid.csv")
    sens_path = os.path.join(output_dir, "10b_tewi_with_sensitivity.csv")
    main_df.to_csv(main_path, index=False)
    sens_df.to_csv(sens_path, index=False)

    print(f"[TEWI] Main table saved to {main_path} ({len(main_df)} fluids)")
    print(f"[TEWI] Sensitivity saved to {sens_path} "
          f"({len(LEAK_SENSITIVITY)} leak rates x {len(main_df)} fluids)")
    print("\n[TEWI] Ranking (lowest net TEWI = best climate impact):")
    for _, r in main_df.iterrows():
        print(f"  {int(r['TEWI_rank'])}. {r['fluid']:12s}  GWP={r['GWP']:7.1f}  "
              f"direct={r['direct_emissions_tCO2eq']:8.1f} tCO2e  "
              f"netTEWI={r['net_TEWI_tCO2eq_per_lifetime']:10.0f} tCO2e")

    # Headline comparison
    try:
        r245 = main_df[main_df["fluid"] == "R245fa"].iloc[0]
        r1233 = main_df[main_df["fluid"] == "R1233zd(E)"].iloc[0]
        diff = r245["net_TEWI_tCO2eq_per_lifetime"] - r1233["net_TEWI_tCO2eq_per_lifetime"]
        print(f"\n[TEWI] R1233zd(E) beats R245fa by {diff:,.0f} tCO2e over 20-yr lifetime.")
    except IndexError:
        pass


if __name__ == "__main__":
    main()
