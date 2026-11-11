"""D4 sensitivity: does the AV-EN ranking survive ETA_CYL_UNAV in {0.51, 0.54, 0.57}?

`advanced_exergy_model.ETA_CYL_UNAV` is the assumed best-realistically-attainable bare
two-stroke brake efficiency. It is not sourced, and the entire "engine cylinders dominate
avoidable improvement potential" headline is computed from the gap between it and the
engine's actual brake efficiency (0.5071 at 85% MCR). The module comment has asked for this
sweep since 2026-06-13.

Result (2026-08-17): the ranking FLIPS at 0.51 — cylinder avoidable destruction collapses to
zero and the turbocharger takes rank 1. See REFINEMENT_VERIFIED_NUMBERS.md section 2.

Usage:
    cd simulations && python d4_eta_cyl_sensitivity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import advanced_exergy_model as aem
from engine_model import S65MEC_Engine

LOAD = 85.0
ETAS = (0.51, 0.54, 0.57)
CANONICAL = 0.54


def main() -> None:
    engine = S65MEC_Engine(ambient_temp_C=30, seawater_temp_C=29)
    eta_actual = engine.brake_thermal_efficiency(LOAD / 100.0)

    rows = []
    for eta in ETAS:
        # Rebind both the constant and the dict entry that captured it at import time.
        aem.ETA_CYL_UNAV = eta
        aem.COMPONENT_PARAMS["engine_cylinder"]["eta_unav"] = eta
        aem._CANONICAL_ORC_CACHE.clear()

        np.random.seed(42)
        df = aem.advanced_decomposition(engine, aem.TARGET_LOADS, results_dir="results")

        d = df[df.Load_pct == LOAD].copy()
        total = d.E_D_AV_EN_kW.sum()
        d["share_pct"] = 100 * d.E_D_AV_EN_kW / total
        d = d.sort_values("E_D_AV_EN_kW", ascending=False)
        for rank, (_, r) in enumerate(d.iterrows(), 1):
            rows.append({
                "eta_cyl_unav": eta,
                "rank": rank,
                "component": r.Component,
                "AV_EN_kW": r.E_D_AV_EN_kW,
                "share_pct": r.share_pct,
                "total_AV_EN_kW": total,
            })

    out = pd.DataFrame(rows)

    print(f"\n=== D4 SENSITIVITY: AV-EN ranking at {LOAD:.0f}% MCR ===")
    print(f"    engine actual brake efficiency = {eta_actual:.4f}\n")
    for eta in ETAS:
        s = out[out.eta_cyl_unav == eta]
        tag = "  <- canonical" if eta == CANONICAL else ""
        print(f"--- eta_cyl_unav = {eta:.2f}   "
              f"(total AV-EN = {s.total_AV_EN_kW.iloc[0]:.1f} kW){tag}")
        print(s[["rank", "component", "AV_EN_kW", "share_pct"]].to_string(
            index=False, float_format=lambda x: f"{x:8.1f}"))
        print()

    orders = {eta: tuple(out[out.eta_cyl_unav == eta].sort_values("rank").component)
              for eta in ETAS}
    base = orders[CANONICAL]
    print("=== RANKING STABILITY ===")
    for eta, order in orders.items():
        verdict = "IDENTICAL to canonical" if order == base else "*** CHANGED ***"
        print(f"  eta={eta:.2f}: {verdict}  ->  {' > '.join(order[:4])}")

    cyl = out[out.component == "engine_cylinder"].set_index("eta_cyl_unav")
    span = cyl.AV_EN_kW.max() - cyl.AV_EN_kW.min()
    print("\n=== engine_cylinder AV-EN sensitivity ===")
    for eta in ETAS:
        tag = "  <- canonical" if eta == CANONICAL else ""
        print(f"  {eta:.2f}: {cyl.loc[eta, 'AV_EN_kW']:8.1f} kW  "
              f"({cyl.loc[eta, 'share_pct']:5.1f}% of AV-EN){tag}")
    print(f"  span across the sweep: {span:.1f} kW "
          f"({100 * span / cyl.loc[CANONICAL, 'AV_EN_kW']:.0f}% of the canonical value)")

    path = "results/07c_d4_eta_cyl_unav_sensitivity.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
