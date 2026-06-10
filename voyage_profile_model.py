"""
Module 8: Voyage-Profile Weighted WHR Performance
==================================================
Weights steady-state WHR results (CSV 04_environmental_economic_4E.csv)
by time-in-load fractions from two representative vessel schedules used
by Indian coastal bulkers:

    A. "Indian coastal" (~60% open ocean):
         48 h/wk in port (aux only, 0% ME)
          8 h/wk maneuvering (30% ME)
         72 h/wk coastal transit Mumbai-Kandla-Cochin (65% ME)
         40 h/wk open-ocean long-haul (80% ME)
       -> total 168 h/wk

    B. "Long-haul Persian Gulf-Singapore" (~80% open ocean):
         16 h/wk in port
          4 h/wk maneuvering (30% ME)
         13.6 h/wk coastal transit (65% ME)
         134.4 h/wk open ocean (80% ME)

For each WHR technology and each working fluid we compute
voyage-averaged:
  - <W_WHR>            time-weighted net electrical power (kW)
  - annual fuel saved  [tonnes / yr]
  - annual CO2 / NOx / SOx saved [tonnes / yr]
  - voyage payback & 20-year NPV

WHR output is assumed zero when ME load < 20% MCR (engine off / aux only),
consistent with ORC turbine minimum-flow limits and exhaust pinch failure.

Author: Prajwal Rao
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


SCHEDULES = {
    "Indian_coastal": [
        {"phase": "port",        "hours_wk":  48.0, "load_pct":   0.0, "whr_active": False},
        {"phase": "maneuvering", "hours_wk":   8.0, "load_pct":  30.0, "whr_active": True},
        {"phase": "coastal",     "hours_wk":  72.0, "load_pct":  65.0, "whr_active": True},
        {"phase": "open_ocean",  "hours_wk":  40.0, "load_pct":  80.0, "whr_active": True},
    ],
    "Long_haul_PG_SGP": [
        {"phase": "port",        "hours_wk":  16.0, "load_pct":   0.0, "whr_active": False},
        {"phase": "maneuvering", "hours_wk":   4.0, "load_pct":  30.0, "whr_active": True},
        {"phase": "coastal",     "hours_wk":  13.6, "load_pct":  65.0, "whr_active": True},
        {"phase": "open_ocean",  "hours_wk": 134.4, "load_pct":  80.0, "whr_active": True},
    ],
}

HOURS_PER_WEEK = 168.0
OPERATING_DAYS = 280
ANNUAL_HOURS_AT_SEA = OPERATING_DAYS * 24.0  # consistent with environmental_model
DISCOUNT_RATE = 0.08
LIFETIME_YEARS = 20
FUEL_PRICE_USD_PER_T = 550.0
CARBON_PRICE_USD_PER_T = 55.0  # EU ETS 2025 reference


def _nearest_load_row(df: pd.DataFrame, tech: str, load_pct: float) -> pd.Series:
    """Return the row closest to the target load for a given WHR tech."""
    sub = df[df["WHR_technology"] == tech]
    idx = (sub["Load_pct"] - load_pct).abs().idxmin()
    return sub.loc[idx]


def _write_schedule_table(path: str) -> pd.DataFrame:
    """Write the phase definitions (08) and return combined DataFrame."""
    rows = []
    for name, phases in SCHEDULES.items():
        total = sum(p["hours_wk"] for p in phases)
        for p in phases:
            rows.append({
                "profile":       name,
                "phase":         p["phase"],
                "hours_per_week": p["hours_wk"],
                "weight_frac":   p["hours_wk"] / total,
                "engine_load_pct": p["load_pct"],
                "whr_active":    p["whr_active"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def _voyage_weight(env_df: pd.DataFrame, tech: str,
                   schedule: list) -> dict:
    """Time-weight WHR performance columns for one tech under one schedule.

    Port phase contributes zero WHR output but still counts in annual
    operating hours (aux load covered by genset in harbour).
    """
    total_hours = sum(p["hours_wk"] for p in schedule)
    W_weighted = 0.0
    fuel_saved_weighted = 0.0  # kg/h equivalent
    co2_weighted = 0.0
    nox_weighted = 0.0
    sox_weighted = 0.0
    eedi_weighted = 0.0
    capex_ref = None

    for p in schedule:
        w = p["hours_wk"] / total_hours
        if not p["whr_active"]:
            # zero contribution this phase
            continue
        row = _nearest_load_row(env_df, tech, p["load_pct"])
        W_weighted += w * float(row["W_whr_kW"])
        # 04 CSV stores annual values assuming always-on operation;
        # convert to hourly rate and re-integrate with phase weight.
        fuel_saved_weighted += w * float(row["annual_fuel_saved_tonnes"]) / ANNUAL_HOURS_AT_SEA
        co2_weighted        += w * float(row["annual_co2_saved_tonnes"])  / ANNUAL_HOURS_AT_SEA
        nox_weighted        += w * float(row["annual_nox_saved_tonnes"])  / ANNUAL_HOURS_AT_SEA
        sox_weighted        += w * float(row["annual_sox_saved_tonnes"])  / ANNUAL_HOURS_AT_SEA
        eedi_weighted       += w * float(row["eedi_improvement_pct"])
        capex_ref = float(row["CAPEX_USD"])  # size fixed by peak design; use last seen

    # Scale hourly rates up to annual (using operating days at sea)
    fuel_t_yr = fuel_saved_weighted * ANNUAL_HOURS_AT_SEA
    co2_t_yr  = co2_weighted        * ANNUAL_HOURS_AT_SEA
    nox_t_yr  = nox_weighted        * ANNUAL_HOURS_AT_SEA
    sox_t_yr  = sox_weighted        * ANNUAL_HOURS_AT_SEA

    fuel_savings_usd = fuel_t_yr * FUEL_PRICE_USD_PER_T
    carbon_rev_usd = co2_t_yr * CARBON_PRICE_USD_PER_T
    maintenance_usd = (capex_ref or 0.0) * 0.03
    net_annual_usd = fuel_savings_usd + carbon_rev_usd - maintenance_usd

    payback = (capex_ref / net_annual_usd) if (capex_ref and net_annual_usd > 0) else np.inf
    npv = -(capex_ref or 0.0)
    for y in range(1, LIFETIME_YEARS + 1):
        npv += net_annual_usd / (1.0 + DISCOUNT_RATE) ** y

    return {
        "W_whr_avg_kW":             W_weighted,
        "annual_fuel_saved_t":      fuel_t_yr,
        "annual_co2_saved_t":       co2_t_yr,
        "annual_nox_saved_t":       nox_t_yr,
        "annual_sox_saved_t":       sox_t_yr,
        "avg_eedi_improvement_pct": eedi_weighted,
        "CAPEX_USD":                capex_ref or 0.0,
        "net_annual_savings_USD":   net_annual_usd,
        "payback_years":            payback,
        "NPV_20yr_USD":             npv,
    }


def main(output_dir: str = "results") -> None:
    os.makedirs(output_dir, exist_ok=True)
    env_path = os.path.join(output_dir, "04_environmental_economic_4E.csv")
    env_df = pd.read_csv(env_path)

    sched_path = os.path.join(output_dir, "08_voyage_profile_schedules.csv")
    _write_schedule_table(sched_path)

    rows = []
    whr_techs = env_df["WHR_technology"].unique().tolist()
    for profile_name, schedule in SCHEDULES.items():
        for tech in whr_techs:
            res = _voyage_weight(env_df, tech, schedule)
            rows.append({
                "profile":         profile_name,
                "WHR_technology":  tech,
                **res,
            })
    out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "08b_voyage_weighted_performance.csv")
    out.to_csv(out_path, index=False)

    print(f"[Voyage Profile] Schedules saved to {sched_path}")
    print(f"[Voyage Profile] Weighted performance saved to {out_path}")
    print("\n[Voyage Profile] Best WHR per profile by NPV:")
    for profile in SCHEDULES:
        best = out[out["profile"] == profile].sort_values("NPV_20yr_USD",
                                                          ascending=False).iloc[0]
        print(f"  {profile:20s}: {best['WHR_technology']:18s} "
              f"<W>={best['W_whr_avg_kW']:.0f} kW  "
              f"CO2={best['annual_co2_saved_t']:.0f} t/yr  "
              f"payback={best['payback_years']:.1f} yr  "
              f"NPV={best['NPV_20yr_USD']/1e6:.2f} M$")


if __name__ == "__main__":
    main()
