#!/usr/bin/env python3
"""
Master Simulation Runner
========================
Runs all simulation modules sequentially and generates figures.
Designed to run overnight (8-12 hours total).

Usage:
    python run_all.py                    # Full run (overnight)
    python run_all.py --quick            # Quick test (~15 min)
    python run_all.py --plots-only       # Just regenerate figures from existing CSVs

Setup:
    pip install -r requirements.txt

Author: Prajwal Rao
"""

import sys
import os
import time
import traceback
import warnings
import logging
from datetime import datetime, timedelta

# Force UTF-8 output on Windows (default cp1252 cannot print η, °, ×, …).
# Must happen before any logging / print calls.
for _stream_name in ('stdout', 'stderr'):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# Ensure we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')


def setup_logging():
    """Configure logging to both console and file, and route warnings
    through the logger so overnight runs capture every RuntimeWarning."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = os.path.join(RESULTS_DIR, f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logging.captureWarnings(True)
    # Ensure our sanity-check warnings are always shown, even if a
    # downstream library added a blanket ignore filter.
    warnings.simplefilter('default', RuntimeWarning)
    warnings.simplefilter('default', UserWarning)
    return log_file


def check_dependencies():
    """Verify all required packages are installed."""
    missing = []
    for pkg in ['numpy', 'pandas', 'matplotlib', 'scipy', 'tqdm']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    try:
        import CoolProp
    except ImportError:
        missing.append('CoolProp')

    try:
        import pymoo
    except ImportError:
        logging.warning("pymoo not installed — NSGA-II will use grid search fallback")
        logging.warning("Install with: pip install pymoo")

    if missing:
        logging.error(f"Missing packages: {', '.join(missing)}")
        logging.error(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    logging.info("All core dependencies verified.")


def _validate_outputs():
    """Final cross-file sanity checks on CSVs before declaring success."""
    import pandas as pd
    problems = []

    def _load(name):
        path = os.path.join(RESULTS_DIR, name)
        if not os.path.exists(path):
            problems.append(f"{name} is missing")
            return None
        return pd.read_csv(path)

    eng = _load('01_engine_energy_balance.csv')
    if eng is not None:
        if not (0.40 <= eng['eta_brake'].max() <= 0.55):
            problems.append(
                f"Engine η_brake peak {eng['eta_brake'].max():.3f} outside plausible [0.40, 0.55]"
            )
        closure = (eng['Q_unaccounted_kW'].abs() / eng['Q_fuel_kW']).max() * 100
        if closure > 10:
            problems.append(f"Energy balance closure error {closure:.1f}% > 10%")

    orc_opt = _load('02b_orc_optimal_per_fluid.csv')
    if orc_opt is not None and len(orc_opt) > 0:
        if orc_opt['eta_thermal'].max() > 0.25 or orc_opt['eta_thermal'].min() < 0.03:
            problems.append(
                f"ORC η_thermal range [{orc_opt['eta_thermal'].min():.3f}, "
                f"{orc_opt['eta_thermal'].max():.3f}] outside [0.03, 0.25]"
            )

    ex = _load('03_engine_exergy_balance.csv')
    if ex is not None:
        if (ex['Ex_destroyed_kW'] < 0).any():
            problems.append("Negative engine exergy destruction detected (2nd-law violation)")
        if ex['eta_exergy'].max() > 0.55:
            problems.append(
                f"Engine exergy efficiency {ex['eta_exergy'].max():.3f} > 0.55 — implausible"
            )

    env = _load('04_environmental_economic_4E.csv')
    if env is not None:
        if (env['annual_co2_saved_tonnes'] < 0).any():
            problems.append("Negative annual CO2 savings — check WHR scenario.")

    if problems:
        logging.error("OUTPUT SANITY CHECK FAILED:")
        for p in problems:
            logging.error(f"  - {p}")
        raise RuntimeError(
            "Output validation failed — results are suspect. "
            "Inspect warnings above before using for publication."
        )
    logging.info("Output sanity checks PASSED.")


def run_full_simulation(quick=False):
    """Run complete simulation pipeline."""
    start_total = time.time()

    # ==========================================
    # PHASE 1: Engine Model (~1 minute)
    # ==========================================
    logging.info("=" * 60)
    logging.info("PHASE 1: Engine Thermodynamic Model")
    logging.info("=" * 60)
    t0 = time.time()

    from engine_model import S65MEC_Engine, run_engine_analysis
    engine, engine_df = run_engine_analysis(output_dir=RESULTS_DIR)

    logging.info(f"Phase 1 complete in {time.time()-t0:.1f}s")

    # ==========================================
    # PHASE 2: ORC Parametric Study (~1-3 hours)
    # ==========================================
    logging.info("=" * 60)
    logging.info("PHASE 2: ORC Working Fluid Parametric Study")
    logging.info("=" * 60)
    t0 = time.time()

    from orc_model import run_orc_analysis
    orc_df = run_orc_analysis(engine_df, output_dir=RESULTS_DIR)

    logging.info(f"Phase 2 complete in {time.time()-t0:.1f}s")

    # ==========================================
    # PHASE 3: Exergy Analysis (~2 minutes)
    # ==========================================
    logging.info("=" * 60)
    logging.info("PHASE 3: Exergy (Second-Law) Analysis")
    logging.info("=" * 60)
    t0 = time.time()

    from exergy_model import run_exergy_analysis
    exergy_df, orc_exergy_df = run_exergy_analysis(engine, engine_df, orc_df,
                                                     output_dir=RESULTS_DIR)

    logging.info(f"Phase 3 complete in {time.time()-t0:.1f}s")

    # ==========================================
    # PHASE 4: Environmental & Economic (4E) (~5 minutes)
    # ==========================================
    logging.info("=" * 60)
    logging.info("PHASE 4: Environmental & Economic 4E Analysis")
    logging.info("=" * 60)
    t0 = time.time()

    from environmental_model import run_environmental_analysis
    env_df = run_environmental_analysis(engine, output_dir=RESULTS_DIR)

    logging.info(f"Phase 4 complete in {time.time()-t0:.1f}s")

    # ==========================================
    # PHASE 5: Monte Carlo (~30 min - 2 hours)
    # ==========================================
    logging.info("=" * 60)
    logging.info("PHASE 5: Monte Carlo Sensitivity Analysis")
    logging.info("=" * 60)
    t0 = time.time()

    from optimization import run_monte_carlo
    n_mc = 2000 if quick else 15000
    mc_df = run_monte_carlo(engine, n_samples=n_mc, output_dir=RESULTS_DIR)

    logging.info(f"Phase 5 complete in {time.time()-t0:.1f}s")

    # ==========================================
    # PHASE 6: NSGA-II Optimization (~2-4 hours)
    # ==========================================
    logging.info("=" * 60)
    logging.info("PHASE 6: Multi-Objective Optimization (NSGA-II)")
    logging.info("=" * 60)
    t0 = time.time()

    from optimization import run_nsga2_optimization
    n_gen = 50 if quick else 200
    pop = 40 if quick else 100
    pareto_df = run_nsga2_optimization(engine, output_dir=RESULTS_DIR,
                                        n_gen=n_gen, pop_size=pop)

    logging.info(f"Phase 6 complete in {time.time()-t0:.1f}s")

    # ==========================================
    # PHASE 7: Generate All Figures (~2 minutes)
    # ==========================================
    logging.info("=" * 60)
    logging.info("PHASE 7: Generating Publication Figures")
    logging.info("=" * 60)
    t0 = time.time()

    from plot_results import generate_all_figures
    generate_all_figures(results_dir=RESULTS_DIR, output_dir=FIGURES_DIR)

    logging.info(f"Phase 7 complete in {time.time()-t0:.1f}s")

    # ==========================================
    # FINAL SANITY CHECKS (fail loudly if outputs are off)
    # ==========================================
    logging.info("Validating output CSVs against sanity bounds...")
    _validate_outputs()

    # ==========================================
    # SUMMARY
    # ==========================================
    total_time = time.time() - start_total
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    secs = int(total_time % 60)

    logging.info("")
    logging.info("=" * 60)
    logging.info("SIMULATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total runtime: {hours}h {mins}m {secs}s")
    logging.info(f"Results directory: {RESULTS_DIR}")
    logging.info(f"Figures directory: {FIGURES_DIR}")
    logging.info("")
    logging.info("Output CSV files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith('.csv'):
            size = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1024
            logging.info(f"  {f:50s} ({size:.0f} KB)")
    logging.info("")
    logging.info("Figures:")
    if os.path.exists(FIGURES_DIR):
        for f in sorted(os.listdir(FIGURES_DIR)):
            logging.info(f"  {f}")
    logging.info("")
    logging.info("Next steps:")
    logging.info("  1. Review CSV data in results/")
    logging.info("  2. Include figures from results/figures/ in your LaTeX paper")
    logging.info("  3. Update paper sections with quantitative results from CSVs")


def run_plots_only():
    """Regenerate figures from existing CSV results."""
    logging.info("Regenerating figures from existing data...")
    from plot_results import generate_all_figures
    generate_all_figures(results_dir=RESULTS_DIR, output_dir=FIGURES_DIR)


if __name__ == '__main__':
    log_file = setup_logging()
    logging.info(f"Log file: {log_file}")

    if '--plots-only' in sys.argv:
        run_plots_only()
    else:
        quick = '--quick' in sys.argv
        if quick:
            logging.info("*** QUICK MODE: reduced samples for testing ***")
        else:
            logging.info("*** FULL MODE: overnight run (~8-12 hours) ***")

        check_dependencies()
        try:
            run_full_simulation(quick=quick)
        except Exception:
            logging.error("SIMULATION ABORTED — full traceback follows:")
            logging.error(traceback.format_exc())
            sys.exit(1)
