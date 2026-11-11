"""
Module 9: Seawater Temperature Sensitivity of ORC Performance
==============================================================
Sweeps seawater inlet temperature from 15 C (North Atlantic winter) to
32 C (Bay of Bengal summer) in 1 C increments. For each T_sw, the ORC
cycle is resolved for the four top-performing fluids
    - toluene
    - n-pentane
    - R1233zd(E)
    - cyclohexane
at a fixed 85% MCR engine operating point.

Key coupling: T_cond = T_sw + 9 K (condenser approach temperature),
P_cond = P_sat(T_cond). As T_sw rises into tropical conditions, P_cond
rises exponentially, cutting pressure ratio and net power.

A "North Sea reference" is taken at T_sw = 10 C (marginal extrapolation
of the sweep) for publication context; the 15 C point is used for
conservative reporting.

Annual CO2 savings assume 280 operating days/yr against HFO auxiliary
displacement at 210 g/kWh.

Author: Prajwal Rao
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from engine_model import S65MEC_Engine
from orc_model import ORCSystem          # canonical solver — same as headline 684 kW cycle


FLUIDS = ["Toluene", "n-Pentane", "R1233zd(E)", "Cyclohexane"]
T_SW_RANGE_C = np.arange(15.0, 32.1, 1.0)
ENGINE_LOAD = 0.85
APPROACH_K = 9.0               # T_cond - T_sw
SUPERHEAT_C = 5.0
ETA_PUMP = 0.70
ETA_TURB = 0.80
PINCH_EVAP_K = 10.0
OPERATING_DAYS = 280
EF_CO2_HFO = 3.114
BSFC_AUX = 210.0               # g/kWh - displaced genset

# Evaporator temperature sweep — mirrors orc_model.run_orc_analysis exactly.
# The acid-dew-point floor (150 C) constrains T_evap <= 140 C via the
# ORCSystem.size_for_heat_source pinch check (T_source_out - pinch = 140 C).
T_EVAP_RANGE_C = np.arange(60.0, 155.0, 2.5)
T_ACID_DEW_C = 150.0           # ZareNezhad 2009, HFO @ 3.5% S


def _solve_orc_best(fluid: str, T_exh_C: float, Q_exh_kW: float,
                    T_cond_C: float) -> dict:
    """
    Find the ORC point that maximises W_net using orc_model.ORCSystem.

    Sweeps T_EVAP_RANGE_C at fixed 5 K superheat, 10 K evaporator pinch,
    and acid-dew-point floor as the exhaust outlet temperature.  This
    exactly mirrors orc_model.run_orc_analysis, so results are
    self-consistent with the headline toluene 684 kW / 16.5% cycle.

    Root-cause note: the previous _solve_orc set T_evap = 150 C (no
    pinch guard), while ORCSystem.size_for_heat_source rejects T_evap >
    140 C (150 - 10 K pinch = 140 C ceiling).  That 10 C gap produced
    the 722 vs 684 kW discrepancy (C4).

    Raises ValueError if no feasible point exists for any T_evap in the
    sweep range.
    """
    orc = ORCSystem(
        fluid_name=fluid,
        eta_pump=ETA_PUMP,
        eta_expander=ETA_TURB,
        pinch_point_K=PINCH_EVAP_K,
    )
    best = None
    for T_evap_C in T_EVAP_RANGE_C:
        r = orc.size_for_heat_source(
            Q_exh_kW, T_exh_C, T_ACID_DEW_C,
            T_evap_C, T_cond_C,
            T_superheat_C=SUPERHEAT_C,
        )
        if r is None:
            continue
        if best is None or r["W_net_kW"] > best["W_net_kW"]:
            best = r
    if best is None:
        raise ValueError(
            f"No feasible ORC point: fluid={fluid}, T_cond={T_cond_C:.1f} C"
        )
    return {
        "T_cond_C":   T_cond_C,
        "P_cond_kPa": best["P_cond_kPa"],
        "P_evap_kPa": best["P_evap_kPa"],
        "eta_th":     best["eta_thermal"],
        "W_net_kW":   best["W_net_kW"],
        "T_evap_C":   best["T_evap_C"],
    }


def _annual_co2_saved(W_net_kW: float) -> float:
    """Convert net WHR power to annual tCO2 saved via genset displacement."""
    hours = OPERATING_DAYS * 24.0
    fuel_t_yr = W_net_kW * hours * BSFC_AUX / 1.0e6
    return fuel_t_yr * EF_CO2_HFO


def sweep() -> pd.DataFrame:
    engine = S65MEC_Engine(ambient_temp_C=30, seawater_temp_C=29)
    T_exh_C = float(engine.exhaust_gas_temp_after_TC(ENGINE_LOAD))
    m_exh = float(engine.exhaust_mass_flow(ENGINE_LOAD))
    cp_exh = float(engine.exhaust_cp(T_exh_C))
    T_exh_out_C = max(150.0, T_exh_C - 120.0)
    Q_exh_kW = m_exh * cp_exh * (T_exh_C - T_exh_out_C)

    rows = []
    # Reference W_net at T_sw = 15 C for each fluid (derating baseline)
    ref_W = {}
    for fluid in FLUIDS:
        try:
            ref_W[fluid] = _solve_orc_best(fluid, T_exh_C, Q_exh_kW,
                                           15.0 + APPROACH_K)["W_net_kW"]
        except Exception:
            ref_W[fluid] = np.nan

    for T_sw in T_SW_RANGE_C:
        T_cond = T_sw + APPROACH_K
        for fluid in FLUIDS:
            try:
                r = _solve_orc_best(fluid, T_exh_C, Q_exh_kW, T_cond)
            except Exception:
                continue
            derating = 100.0 * (1.0 - r["W_net_kW"] / ref_W[fluid]) \
                       if ref_W[fluid] and not np.isnan(ref_W[fluid]) else np.nan
            rows.append({
                "T_sw_C":                 float(T_sw),
                "fluid":                  fluid,
                "T_cond_C":               r["T_cond_C"],
                "P_cond_kPa":             r["P_cond_kPa"],
                "T_evap_C":               r["T_evap_C"],
                "eta_th":                 r["eta_th"],
                "W_net_kW":               r["W_net_kW"],
                "power_derating_pct_vs_15C": derating,
                "annual_CO2_saved_t":     _annual_co2_saved(r["W_net_kW"]),
            })
    return pd.DataFrame(rows)


def _summary_tropical_vs_temperate(df: pd.DataFrame) -> pd.DataFrame:
    targets = [15.0, 29.0, 32.0]
    rows = []
    for fluid in FLUIDS:
        sub = df[df["fluid"] == fluid]
        row = {"fluid": fluid}
        for T in targets:
            m = sub[np.isclose(sub["T_sw_C"], T)]
            if m.empty:
                continue
            row[f"W_net_kW_{int(T)}C"] = float(m["W_net_kW"].iloc[0])
            row[f"eta_th_{int(T)}C"]   = float(m["eta_th"].iloc[0])
            row[f"CO2_t_yr_{int(T)}C"] = float(m["annual_CO2_saved_t"].iloc[0])
        if "W_net_kW_15C" in row and "W_net_kW_32C" in row:
            row["tropical_penalty_pct_15to32"] = \
                100.0 * (1.0 - row["W_net_kW_32C"] / row["W_net_kW_15C"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(output_dir: str = "results") -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    df = sweep()
    path_sweep = os.path.join(output_dir, "09_seawater_sensitivity.csv")
    df.to_csv(path_sweep, index=False)

    summary = _summary_tropical_vs_temperate(df)
    path_summary = os.path.join(output_dir, "09b_tropical_vs_temperate_summary.csv")
    summary.to_csv(path_summary, index=False)

    print(f"[Seawater Sensitivity] Full sweep saved to {path_sweep} "
          f"({len(df)} rows, {len(FLUIDS)} fluids × {len(T_SW_RANGE_C)} T_sw)")
    print(f"[Seawater Sensitivity] Tropical-vs-temperate summary saved to {path_summary}")
    print("\n[Seawater Sensitivity] Tropical penalty (15 C -> 32 C):")
    for _, row in summary.iterrows():
        pct = row.get("tropical_penalty_pct_15to32", float("nan"))
        w15 = row.get("W_net_kW_15C", float("nan"))
        w32 = row.get("W_net_kW_32C", float("nan"))
        print(f"  {row['fluid']:12s}  W(15C)={w15:6.1f} kW  W(32C)={w32:6.1f} kW  "
              f"penalty={pct:5.1f}%")


if __name__ == "__main__":
    main()
