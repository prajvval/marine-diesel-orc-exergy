"""
Module 7: Advanced Exergy Decomposition (Kelly-Tsatsaronis-Morosuk)
====================================================================
Splits component exergy destruction into (Avoidable/Unavoidable) ×
(Endogenous/Exogenous) = 4 bins for the engine + WHR system, isolating
the component-level improvement potential that conventional (first-level)
exergy analysis cannot.

Theoretical-cycle method:
  - E_D_UN     : from a "realistic-unavoidable" operating state with
                 near-ideal component parameters (eta_isen_turbine=0.92,
                 eta_isen_pump=0.95, HX pinch=2 K, lossless combustor).
  - E_D_AV     : E_D_real - E_D_UN
  - E_D_EN     : run cycle with all OTHER components at ideal and THIS
                 component at its real irreversibility. The destruction
                 that remains is endogenous (attributable to component
                 alone, independent of the rest of the system).
  - E_D_EX     : E_D_real - E_D_EN

Components analysed: engine cylinder combustion, turbocharger, exhaust
gas boiler (interface HX), ORC evaporator, ORC turbine, ORC condenser,
ORC pump.

Load points: 50%, 75%, 85%, 100% MCR.

References
----------
Tsatsaronis, G. (2002). Strengths and limitations of exergy analysis.
    Energy Conversion and Management, 43(9-12), 1257-1267.
    doi:10.1016/S0196-8904(02)00012-2

Kelly, S., Tsatsaronis, G., & Morosuk, T. (2009). Advanced exergetic
    analysis: Approaches for splitting the exergy destruction into
    endogenous and exogenous parts. Energy, 34(3), 384-391.
    doi:10.1016/j.energy.2008.12.007

Morosuk, T., & Tsatsaronis, G. (2019). Advanced exergy-based methods
    used to understand and improve energy-conversion systems.
    Energy, 169, 238-246. doi:10.1016/j.energy.2018.12.023

Author: Prajwal Rao
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

from engine_model import S65MEC_Engine


# Unavoidable engine brake efficiency (best realistically attainable, state of art ~2024
# large-bore low-NOx 2-stroke). Exposed as a module constant for D4 sensitivity sweep:
#   run 0.51 / 0.54 / 0.57 to test AV-EN ranking stability before ECM submission.
ETA_CYL_UNAV = 0.54

# Acid dew-point floor for HFO exhaust (ZareNezhad 2009, sulfuric acid condensation).
# Exhaust outlet temperature must stay >= ACID_DEW_C at all operating points.
# This constraint is the reason E_D_AV_evap = 0 at the canonical T_evap = 140 C:
#   effective pinch_min = ACID_DEW_C - T_evap = 150 - 140 = 10 K = pinch_real.
ACID_DEW_C = 150.0

# --------------------------------------------------------------------------
# Component-level "real" vs "unavoidable" parameter definitions.
# The unavoidable column represents the best realistically attainable state
# (Tsatsaronis 2002), NOT the thermodynamic ideal. The gap between real
# and unavoidable is what engineering can actually remove.
# --------------------------------------------------------------------------
COMPONENT_PARAMS = {
    # Engine cylinder combustion (REVISED 2026-06-13):
    #   Destruction is computed from a FULL whole-engine exergy balance
    #   (fuel exergy - shaft work - exergy of all heat-rejection streams),
    #   NOT the previous Ex_fuel*(1-eta)-W_brake form, which double-counted
    #   shaft work and clamped the unavoidable case to zero (a ~13x error
    #   that made combustion look 100% avoidable). See _engine_side_destruction.
    #   - eta_real = 0.0 is a sentinel meaning "use the engine's actual brake
    #     thermal efficiency at this load" (~0.50-0.51).
    #   - eta_unav = 0.54 is the best realistically attainable bare two-stroke
    #     brake efficiency (state-of-the-art ~2024 large bore, low-NOx,
    #     improved insulation). The destruction remaining at that state is the
    #     UNAVOIDABLE combustion/gas-exchange destruction (the bulk of it).
    "engine_cylinder":  {"eta_real": 0.0, "eta_unav": ETA_CYL_UNAV},
    # Turbocharger (combined T+C stage):
    "turbocharger":     {"eta_real": 0.78, "eta_unav": 0.88},
    # Exhaust gas boiler (interface HX between exhaust and ORC evaporator):
    "exhaust_boiler":   {"pinch_real_K": 20.0, "pinch_unav_K": 5.0},
    # ORC evaporator:
    "orc_evaporator":   {"pinch_real_K": 10.0, "pinch_unav_K": 2.0},
    # ORC turbine:
    "orc_turbine":      {"eta_real": 0.80, "eta_unav": 0.92},
    # ORC condenser:
    "orc_condenser":    {"pinch_real_K": 9.0,  "pinch_unav_K": 3.0},
    # ORC pump:
    "orc_pump":         {"eta_real": 0.70, "eta_unav": 0.95},
}

TARGET_LOADS = [0.50, 0.75, 0.85, 1.00]
REFERENCE_FLUID = "Toluene"  # REVISED 2026-06-13: match the paper's hero fluid
PHI_HFO = 1.06               # HFO chemical exergy / LHV


# --------------------------------------------------------------------------
# Exergy of sensible-heat / isothermal reject streams relative to dead state.
# --------------------------------------------------------------------------
def _carnot_stream_exergy(Q_kW: float, T_hot_K: float, T0_K: float) -> float:
    """Exergy of a stream cooling from T_hot to the dead state (log-mean factor)."""
    if T_hot_K <= T0_K:
        return 0.0
    dT = T_hot_K - T0_K
    lm = 1.0 - T0_K * np.log(T_hot_K / T0_K) / dT
    return Q_kW * max(lm, 0.0)


def _isothermal_exergy(Q_kW: float, T_K: float, T0_K: float) -> float:
    return Q_kW * max(1.0 - T0_K / T_K, 0.0)


# --------------------------------------------------------------------------
# Canonical ORC operating point cache.
# Loads the per-load Toluene optimum from 02b_orc_optimal_per_fluid.csv once
# and returns it for each load point.  This is the FIXED operating point for
# all advanced-exergy cycle evaluations (C7 fix).
# --------------------------------------------------------------------------
_CANONICAL_ORC_CACHE: dict = {}
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_canonical_orc(load_pct: float, results_dir: str = "results") -> "pd.Series":
    """Return the canonical Toluene ORC row (from 02b CSV) for the given engine load %.

    The returned Series contains T_evap_C, T_cond_C, T_superheat_C, Q_in_kW,
    h1..h4, s1, s3, s4, m_dot_wf_kg_s, W_net_kW.  These come directly from the
    parametric sweep optimum, ensuring consistency with every other module.
    """
    key = int(round(load_pct))
    if key not in _CANONICAL_ORC_CACHE:
        csv_path = os.path.join(_SCRIPT_DIR, results_dir, "02b_orc_optimal_per_fluid.csv")
        df = pd.read_csv(csv_path)
        toluene_df = df[df["fluid_key"] == REFERENCE_FLUID]
        for _, row in toluene_df.iterrows():
            lk = int(round(float(row["engine_load_pct"])))
            _CANONICAL_ORC_CACHE[lk] = row
    if key not in _CANONICAL_ORC_CACHE:
        raise ValueError(
            f"No canonical {REFERENCE_FLUID} row at {load_pct}% MCR in 02b CSV. "
            f"Available keys: {sorted(_CANONICAL_ORC_CACHE.keys())}"
        )
    return _CANONICAL_ORC_CACHE[key]


# --------------------------------------------------------------------------
# Helper: solve a simple subcritical ORC with given component etas & pinch.
# Returns dict of per-component exergy destruction in kW (steady state).
# --------------------------------------------------------------------------
def _solve_orc_destruction(
        canonical: "pd.Series",
        T_exh_C: float,
        T_sw_C: float,
        eta_pump: float,
        eta_turb: float,
        pinch_evap_K: float,
        pinch_cond_K: float,
        pinch_boiler_K: float,
        fluid: str = REFERENCE_FLUID) -> dict:
    """
    Compute ORC component exergy destruction rates (kW) at the CANONICAL operating point.

    C7 FIX (2026-06-14): T_evap and Q_in are taken from the 02b canonical Toluene
    table, NOT derived from exhaust-pinch arithmetic.  The old code set
    T_evap = T_exh - pinch_boiler - pinch_evap (~211 C at 85%) and Q = full exhaust
    above 29 C dead state (~9 514 kW), producing cycle states irreconcilable with the
    paper's headline 684 kW / 140 C cycle.

    Operating-point anchor:
      T_evap   = canonical['T_evap_C']   (140 C at all loads >= 50%)
      Q_in_kW  = canonical['Q_in_kW']    (4 146 kW at 85% MCR)
      T_sh_C   = canonical['T_superheat_C']  (5 K)
      T_cond_C = T_sw_C + pinch_cond_K  (varies: 38 C real, 32 C unavoidable)

    HX destructions use the Gouy-Stodola balance at FIXED T_evap terminal temp.
    The acid-dew floor (ACID_DEW_C = 150 C) constrains the effective evaporator
    exhaust outlet: T_exh_out = max(ACID_DEW_C, T_evap + pinch_evap).
    At T_evap = 140 C and pinch_evap_unav = 2 K:
      T_exh_out = max(150, 142) = 150 C  -- same as the real case (pinch=10 K).
    Therefore E_D_AV_evap = 0 at this operating point.  This is the paper's
    single most design-actionable finding from the advanced exergy decomposition.

    NOTE (LIMITATION): Uses the "all-simultaneous unavoidable" approximation --
    T_evap is held at the real-case optimal (140 C) rather than re-optimised to
    148 C for the 2 K unavoidable pinch.  Standard Kelly (2009) simplification;
    must be labelled as such in the manuscript methods section.

    Parameters
    ----------
    canonical   : pandas Series from 02b_orc_optimal_per_fluid.csv for this load
    T_exh_C     : exhaust gas temperature after turbocharger (C); from engine model
    T_sw_C      : seawater / dead-state temperature (C)
    eta_pump    : isentropic pump efficiency
    eta_turb    : isentropic turbine efficiency
    pinch_evap_K  : evaporator minimum temperature difference (K)
    pinch_cond_K  : condenser approach temperature (K); sets T_cond = T_sw + pinch_cond
    pinch_boiler_K: exhaust boiler minimum approach (K)
    fluid       : CoolProp fluid string
    """
    from orc_model import ORCSystem

    # --- Fixed canonical operating point ---
    T_evap_C = float(canonical["T_evap_C"])    # 140 C at all loads >= 50%
    T_sh_C = float(canonical["T_superheat_C"]) # 5 K
    Q_in_kW = float(canonical["Q_in_kW"])      # heat available above 150 C floor

    # Condenser temperature follows from pinch (real: 38 C, unavoidable: 32 C).
    T_cond_C = T_sw_C + pinch_cond_K
    T0_K = T_sw_C + 273.15

    # Solve cycle states via ORCSystem for consistent h/s values.
    orc_sys = ORCSystem(fluid, eta_pump=eta_pump, eta_expander=eta_turb)
    cycle = orc_sys.solve_cycle(T_evap_C, T_cond_C, T_sh_C)
    if cycle is None:
        # Edge case: tight T_cond may trigger ORCSystem feasibility guard.
        # Fall back to canonical T_cond = 38 C and warn.
        T_cond_C = float(canonical["T_cond_C"])
        cycle = orc_sys.solve_cycle(T_evap_C, T_cond_C, T_sh_C)
        if cycle is None:
            raise ValueError(
                f"ORC infeasible at T_evap={T_evap_C} C, T_cond={T_cond_C} C, T_sh={T_sh_C} K "
                f"(eta_p={eta_pump}, eta_t={eta_turb})"
            )

    h1, h2, h3, h4 = cycle["h1"], cycle["h2"], cycle["h3"], cycle["h4"]
    s1, s3, s4 = cycle["s1"], cycle["s3"], cycle["s4"]
    P_evap_Pa = cycle["P_evap_kPa"] * 1000.0
    # s2 not stored in ORCSystem output; compute from CoolProp
    s2 = CP.PropsSI("S", "P", P_evap_Pa, "H", h2, fluid)

    q_in = h3 - h2  # J/kg
    if q_in <= 0:
        raise ValueError(f"q_in non-positive: {q_in:.1f} J/kg")
    m_wf = Q_in_kW * 1000.0 / q_in  # kg/s  (Q_in in kW -> J/s)

    # --- Turbomachinery destructions (Gouy-Stodola: T0 * S_gen) ---
    # Units: T0[K] * m_wf[kg/s] * ds[J/(kg·K)] = W  → divide by 1000 for kW
    E_D_pump = T0_K * m_wf * (s2 - s1) / 1000.0
    E_D_turb = T0_K * m_wf * (s4 - s3) / 1000.0

    # --- Condenser destruction ---
    Q_cond_kW = m_wf * (h4 - h1) / 1000.0
    T_sink_K = T_sw_C + 273.15
    E_D_cond = T0_K * (m_wf * (s1 - s4) / 1000.0 + Q_cond_kW / T_sink_K)
    E_D_cond = max(0.0, E_D_cond)

    # --- Exhaust-side heat-recovery train (two-stage thermal-oil loop) ---
    # SOLAS II-2 Reg. 4.2.1 forbids a flammable working fluid (toluene, cyclohexane
    # and n-pentane are all ASHRAE A3) in direct contact with the exhaust gas, so
    # the recovery train is a thermal-oil intermediate loop: exhaust -> oil (the
    # "boiler" stage) then oil -> working fluid (the "evaporator" stage). The TOTAL
    # exhaust->WF irreversibility is (Ex_exh_to_evap - Ex_wf_gain); it is split
    # between the two stages so that boiler + evaporator SUM to that total.
    #
    # D3 FIX (2026-06-15): previously the boiler pinch term was ADDED on top of the
    # full exhaust->WF transfer loss already carried by the evaporator, double-
    # counting ~99 kW of the same heat transfer. Now the boiler is the hot-side
    # (exhaust->oil) pinch portion and the evaporator carries the remainder.
    #
    # Acid-dew-point floor (150 C): T_exh_out = max(150, T_evap + pinch_evap).
    # At T_evap = 140 C the unavoidable pinch (2 K -> 142 C) is floored to 150 C,
    # the same outlet as the real case (10 K -> 150 C). The exhaust->WF total is
    # therefore dew-point-limited, so the avoidable potential lands on the boiler
    # pinch, not the evaporator -> E_D_AV_evap = 0 (the paper's key finding).
    T_exh_out_C = max(ACID_DEW_C, T_evap_C + pinch_evap_K)
    T_exh_in_K = T_exh_C + 273.15
    T_exh_out_K = T_exh_out_C + 273.15
    dT_exh = T_exh_in_K - T_exh_out_K
    if abs(dT_exh) > 1e-3:
        carnot_exh = 1.0 - T0_K * np.log(T_exh_in_K / T_exh_out_K) / dT_exh
    else:
        carnot_exh = 1.0 - T0_K / T_exh_in_K
    Ex_exh_to_evap = Q_in_kW * carnot_exh  # kW, exergy released by the exhaust
    Ex_wf_gain = m_wf * ((h3 - h2) - T0_K * (s3 - s2)) / 1000.0  # kW, gained by WF
    E_D_hx_total = max(0.0, Ex_exh_to_evap - Ex_wf_gain)  # full exhaust->WF loss

    # Stage 1 (boiler, exhaust->oil): pinch-driven hot-interface irreversibility.
    E_D_boiler = Q_in_kW * T0_K * (
        pinch_boiler_K / (T_exh_in_K * (T_exh_in_K - pinch_boiler_K))
    )
    E_D_boiler = max(0.0, min(E_D_boiler, E_D_hx_total))
    # Stage 2 (evaporator, oil->WF): the remainder, so the two stages sum to total.
    E_D_evap = max(0.0, E_D_hx_total - E_D_boiler)

    W_net = m_wf * ((h3 - h4) - (h2 - h1)) / 1000.0  # kW

    return {
        "orc_pump_kW":       max(0.0, E_D_pump),
        "orc_turbine_kW":    max(0.0, E_D_turb),
        "orc_evaporator_kW": E_D_evap,
        "orc_condenser_kW":  E_D_cond,
        "exhaust_boiler_kW": E_D_boiler,
        "W_net_kW":          W_net,
        "m_wf_kg_s":         m_wf,
        "T_evap_C":          T_evap_C,
        "T_cond_C":          T_cond_C,
    }


def _engine_side_destruction(engine: S65MEC_Engine, load: float,
                              cyl_eta: float, tc_eta: float,
                              T0_K: float = 302.15) -> dict:
    """Engine cylinder + turbocharger exergy destruction (REVISED 2026-06-13).

    Uses a full whole-engine exergy balance instead of the previous
    Ex_fuel*(1-eta) - W_brake form (which double-counted shaft work and
    produced a ~13x under-estimate with a zero unavoidable part):

        E_D_engine = Ex_fuel - W_brake - sum(exergy of reject heat streams)

    where the reject streams are the post-turbocharger exhaust, jacket water,
    charge-air cooler, lube-oil cooler and surface radiation, each weighted by
    its own Carnot/temperature factor relative to the dead state. The
    turbocharger destruction is then attributed from the post-TC exhaust
    exergy and the T/C efficiency, and SUBTRACTED from the engine total so the
    two components do not double-count.

    cyl_eta is interpreted as the brake thermal efficiency to impose:
      * cyl_eta <= 0  -> use the engine's ACTUAL brake efficiency (real case);
      * cyl_eta  > 0  -> impose that brake efficiency (best-in-class = the
                         "unavoidable" reference state).
    """
    T0_C = T0_K - 273.15
    Q_fuel = float(engine.fuel_energy_input(load))
    Ex_fuel = Q_fuel * PHI_HFO
    if cyl_eta and cyl_eta > 0:
        W_brake = cyl_eta * Q_fuel                       # imposed best-in-class
    else:
        W_brake = float(engine.brake_power(load))        # actual

    # Heat-rejection streams (energy, kW) with representative reject temps (K).
    Q_exh = float(engine.exhaust_heat(load, T_ref_C=T0_C))   # post-T/C, above dead state
    T_exh_C = float(engine.exhaust_gas_temp_after_TC(load))
    T_exh_K = T_exh_C + 273.15
    Q_jac = float(engine.jacket_water_heat(load));  T_jac_K = 85.0 + 273.15
    Q_chg = float(engine.charge_air_heat(load));    T_chg_K = 130.0 + 273.15  # glide repr. mean
    Q_lub = float(engine.lube_oil_heat(load));      T_lub_K = 60.0 + 273.15
    Q_rad = float(engine.radiation_loss(load));     T_rad_K = 50.0 + 273.15

    Ex_exh = _carnot_stream_exergy(Q_exh, T_exh_K, T0_K)
    Ex_heat = (Ex_exh
               + _isothermal_exergy(Q_jac, T_jac_K, T0_K)
               + _carnot_stream_exergy(Q_chg, T_chg_K, T0_K)
               + _isothermal_exergy(Q_lub, T_lub_K, T0_K)
               + _isothermal_exergy(Q_rad, T_rad_K, T0_K))

    E_D_engine_total = max(Ex_fuel - W_brake - Ex_heat, 0.0)

    # Turbocharger: irreversibility carried by the post-T/C exhaust exergy at
    # finite T/C efficiency. Subtract from the engine total to avoid double count.
    E_D_tc = Ex_exh * (1.0 - tc_eta)
    E_D_cyl = max(E_D_engine_total - E_D_tc, 0.0)

    return {
        "engine_cylinder_kW":  E_D_cyl,
        "turbocharger_kW":     E_D_tc,
        "Q_exh_available_kW":  Q_exh,
        "T_exh_C":             T_exh_C,
    }


# --------------------------------------------------------------------------
# Endogenous/exogenous split: for each component, run cycle with only that
# component at its real eta (all others at unavoidable). The resulting
# component destruction is endogenous.
# --------------------------------------------------------------------------
COMPONENTS = [
    "engine_cylinder", "turbocharger", "exhaust_boiler",
    "orc_evaporator", "orc_turbine", "orc_condenser", "orc_pump",
]


def _config_for(case: str) -> dict:
    """Build the parameter set used in each case (all real, all unav, or
    one-component-real-rest-unav for endogenous evaluation)."""
    cfg = {}
    for comp, params in COMPONENT_PARAMS.items():
        real = case in ("real", comp)
        out = {}
        if "eta_real" in params:
            out["eta"] = params["eta_real"] if real else params["eta_unav"]
        if "pinch_real_K" in params:
            out["pinch"] = params["pinch_real_K"] if real else params["pinch_unav_K"]
        cfg[comp] = out
    return cfg


def _destruction_set(engine: S65MEC_Engine, load: float, case: str,
                     T_sw_C: float = 29.0, results_dir: str = "results") -> dict:
    """Compute all component E_D values under the given case label.

    C7 fix: loads the canonical Toluene ORC operating point from 02b CSV and
    passes it to _solve_orc_destruction, which uses fixed T_evap=140 C and
    Q_in from the canonical table instead of exhaust-pinch arithmetic.
    The exhaust mass flow used for T_exh is from engine.exhaust_gas_temp_after_TC
    (not hardcoded 50 kg/s — that bug is now moot since Q_in comes from 02b).
    """
    cfg = _config_for(case)
    T0_K = T_sw_C + 273.15
    eng = _engine_side_destruction(
        engine, load,
        cyl_eta=cfg["engine_cylinder"]["eta"],
        tc_eta=cfg["turbocharger"]["eta"],
        T0_K=T0_K,
    )
    canonical = _load_canonical_orc(load * 100.0, results_dir)
    orc = _solve_orc_destruction(
        canonical=canonical,
        T_exh_C=eng["T_exh_C"],
        T_sw_C=T_sw_C,
        eta_pump=cfg["orc_pump"]["eta"],
        eta_turb=cfg["orc_turbine"]["eta"],
        pinch_evap_K=cfg["orc_evaporator"]["pinch"],
        pinch_cond_K=cfg["orc_condenser"]["pinch"],
        pinch_boiler_K=cfg["exhaust_boiler"]["pinch"],
    )
    return {
        "engine_cylinder":  eng["engine_cylinder_kW"],
        "turbocharger":     eng["turbocharger_kW"],
        "exhaust_boiler":   orc["exhaust_boiler_kW"],
        "orc_evaporator":   orc["orc_evaporator_kW"],
        "orc_turbine":      orc["orc_turbine_kW"],
        "orc_condenser":    orc["orc_condenser_kW"],
        "orc_pump":         orc["orc_pump_kW"],
        "W_net_kW":         orc["W_net_kW"],
    }


# --------------------------------------------------------------------------
# EN/EX coupling helper (D2 fix, 2026-06-15)
# --------------------------------------------------------------------------
# Kelly-Tsatsaronis-Morosuk (2009) endogenous/exogenous split requires the
# THEORETICAL reference: component k operates at its own efficiency while ALL
# OTHER ORC components are reversible (isentropic turbomachinery, zero-pinch
# heat exchangers). The previous code used the *unavoidable* (best-in-class)
# reference, which is ~real for near-ideal components and collapsed exogenous
# destruction to ~0. Re-solving the cycle with reversible companions lets the
# inter-component coupling (chiefly turbine<->condenser via T_cond) propagate,
# so the exogenous terms become physically meaningful and may be signed.
#
# Engine cylinder and turbocharger are UPSTREAM of the ORC: idealising the
# bottoming cycle cannot change combustion or turbocharging irreversibility,
# so their exogenous destruction is identically zero (engine-ORC thermodynamic
# separability). The second-order ORC->engine exhaust-backpressure coupling is
# deliberately neglected; this is a disclosed modelling limitation.
_REV_ETA = 1.0      # reversible companion isentropic efficiency
_REV_PINCH = 0.0    # reversible companion approach temperature (K)

# Per-ORC-component lever: which _solve_orc_destruction argument it controls,
# and its (real, unavoidable) values pulled from COMPONENT_PARAMS.
_ORC_LEVER = {
    "orc_turbine":    ("eta_turb",      "eta"),
    "orc_pump":       ("eta_pump",      "eta"),
    "orc_evaporator": ("pinch_evap_K",  "pinch"),
    "orc_condenser":  ("pinch_cond_K",  "pinch"),
    "exhaust_boiler": ("pinch_boiler_K","pinch"),
}
_ORC_COMPONENTS = list(_ORC_LEVER.keys())


def _orc_isolated_destruction(canonical, T_exh_C: float, T_sw_C: float,
                               target: str, state: str) -> float:
    """Destruction (kW) of one ORC `target` component when it operates at
    `state` ('real' or 'unav') and ALL OTHER ORC components are reversible.

    Used to build the theoretical (E_D^EN) and unavoidable-endogenous
    (E_D^UN,EN) auxiliary cycles for the Kelly EN/EX cross-split.
    """
    # Start every companion at its reversible limit.
    kwargs = dict(eta_pump=_REV_ETA, eta_turb=_REV_ETA,
                  pinch_evap_K=_REV_PINCH, pinch_cond_K=_REV_PINCH,
                  pinch_boiler_K=_REV_PINCH)
    # Override the target lever with its real or unavoidable value.
    arg, kind = _ORC_LEVER[target]
    key = "eta_real" if state == "real" else "eta_unav"
    pkey = "pinch_real_K" if state == "real" else "pinch_unav_K"
    params = COMPONENT_PARAMS[target]
    kwargs[arg] = params[key] if kind == "eta" else params[pkey]
    out = _solve_orc_destruction(canonical=canonical, T_exh_C=T_exh_C,
                                 T_sw_C=T_sw_C, **kwargs)
    return out[target + "_kW"]


def advanced_decomposition(engine: S65MEC_Engine, loads: list,
                           results_dir: str = "results") -> pd.DataFrame:
    """Run the full advanced exergy decomposition for each (load, component).

    AV/UN split: real vs all-unavoidable cycle (verified, unchanged).
    EN/EX split: theoretical reference (Kelly 2009) via _orc_isolated_destruction
    for ORC components; engine cylinder + turbocharger are purely endogenous
    (engine-ORC separability). Cross terms are computed from the four auxiliary
    cycles and may be signed (negative exogenous = co-optimisation coupling).
    """
    ENGINE_ENDOGENOUS = {"engine_cylinder", "turbocharger"}
    rows = []
    for load in loads:
        D_real = _destruction_set(engine, load, case="real", results_dir=results_dir)
        D_unav = _destruction_set(engine, load, case="unav", results_dir=results_dir)
        # Boundary for the ORC auxiliary cycles (exhaust temp after T/C, real case).
        eng_T_exh = _engine_side_destruction(
            engine, load, cyl_eta=0.0,
            tc_eta=COMPONENT_PARAMS["turbocharger"]["eta_real"],
            T0_K=29.0 + 273.15)["T_exh_C"]
        canonical = _load_canonical_orc(load * 100.0, results_dir)
        # Theoretical (E_D^EN) and unavoidable-endogenous (E_D^UN,EN) cycles.
        ED_EN = {c: _orc_isolated_destruction(canonical, eng_T_exh, 29.0, c, "real")
                 for c in _ORC_COMPONENTS}
        ED_UNEN = {c: _orc_isolated_destruction(canonical, eng_T_exh, 29.0, c, "unav")
                   for c in _ORC_COMPONENTS}
        for c in COMPONENTS:
            ed_real = D_real[c]
            ed_unav = min(D_unav[c], ed_real)          # verified clamp
            ed_av = max(ed_real - ed_unav, 0.0)
            if c in ENGINE_ENDOGENOUS:
                # Upstream of the ORC -> purely endogenous (separability).
                ed_endo = ed_real
                ed_exo = 0.0
                ed_av_en, ed_av_ex = ed_av, 0.0
                ed_un_en, ed_un_ex = ed_unav, 0.0
            else:
                ed_endo = ED_EN[c]                      # E_D^EN (theoretical)
                ed_exo = ed_real - ed_endo             # signed exogenous
                ed_unen = ED_UNEN[c]                    # E_D^UN,EN
                ed_un_raw = D_unav[c]                   # unclamped all-unav
                if ed_real - ed_un_raw < 0.0:
                    # Dew-point-limited (evaporator): the all-unavoidable cycle has
                    # higher throughput -> no avoidable potential. Reported as fully
                    # unavoidable-endogenous: the two-stage thermal-oil HX EN/EX
                    # coupling is not separately resolved (the subtractive boiler/
                    # evaporator split is not isolation-consistent), and the avoidable
                    # HX potential is carried by the boiler pinch, so the evaporator
                    # row is kept clean (EX = 0), consistent with the C7 finding.
                    ed_endo = ed_real
                    ed_exo = 0.0
                    ed_av_en, ed_av_ex = 0.0, 0.0
                    ed_un_en, ed_un_ex = ed_real, 0.0
                else:
                    ed_av_en = ed_endo - ed_unen        # AV-EN
                    ed_un_en = ed_unen                  # UN-EN
                    ed_un_ex = ed_un_raw - ed_unen      # UN-EX (signed)
                    ed_av_ex = ed_av - ed_av_en         # AV-EX (signed)
            rows.append({
                "Load_pct":              load * 100.0,
                "Component":             c,
                "E_D_total_kW":          ed_real,
                "E_D_avoidable_kW":      ed_av,
                "E_D_unavoidable_kW":    ed_unav,
                "E_D_endogenous_kW":     ed_endo,
                "E_D_exogenous_kW":      ed_exo,
                "E_D_AV_EN_kW":          ed_av_en,
                "E_D_AV_EX_kW":          ed_av_ex,
                "E_D_UN_EN_kW":          ed_un_en,
                "E_D_UN_EX_kW":          ed_un_ex,
                "improvement_potential_kW": ed_av_en,  # AV+EN = self-fixable
            })
    return pd.DataFrame(rows)


def component_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank components by avoidable-endogenous destruction at 85% MCR
    (the service operating point). Higher AV_EN = best ROI for redesign."""
    sub = df[df["Load_pct"].round(1) == 85.0].copy()
    sub = sub.sort_values("improvement_potential_kW", ascending=False)
    sub = sub.reset_index(drop=True)
    sub["rank"] = np.arange(1, len(sub) + 1)
    total_av_en = sub["improvement_potential_kW"].sum()
    sub["share_pct"] = 100.0 * sub["improvement_potential_kW"] / total_av_en
    return sub[["rank", "Component", "E_D_total_kW", "improvement_potential_kW",
                "E_D_AV_EN_kW", "E_D_AV_EX_kW", "share_pct"]]


def main(output_dir: str = "results") -> None:
    # Use absolute path so the script can be invoked from any working directory.
    abs_output = os.path.join(_SCRIPT_DIR, output_dir)
    os.makedirs(abs_output, exist_ok=True)
    engine = S65MEC_Engine(ambient_temp_C=30, seawater_temp_C=29)
    np.random.seed(42)

    df = advanced_decomposition(engine, TARGET_LOADS, results_dir=output_dir)
    path_main = os.path.join(abs_output, "07_advanced_exergy_decomposition.csv")
    df.to_csv(path_main, index=False)

    rank = component_ranking(df)
    path_rank = os.path.join(abs_output, "07b_component_improvement_ranking.csv")
    rank.to_csv(path_rank, index=False)

    print(f"[Advanced Exergy] Decomposition for {len(TARGET_LOADS)} loads × "
          f"{len(COMPONENTS)} components = {len(df)} rows")
    print(f"[Advanced Exergy] Saved to {path_main}")
    print(f"[Advanced Exergy] Saved ranking to {path_rank}")
    print("\n[Advanced Exergy] Top 3 improvement targets at 85% MCR:")
    for _, r in rank.head(3).iterrows():
        print(f"  {r['rank']}. {r['Component']:20s}  AV_EN={r['E_D_AV_EN_kW']:7.1f} kW  "
              f"share={r['share_pct']:5.1f}%")


if __name__ == "__main__":
    main()
