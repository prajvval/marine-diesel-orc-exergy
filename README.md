# Marine Diesel ORC — Advanced Exergy & 4E Simulation Suite

## Waste Heat Recovery Analysis for MAN B&W S65ME-C Marine Diesel Engine

This simulation suite performs a comprehensive **4E analysis** (Energy, Exergy, Economic, Environmental) plus an **advanced exergy decomposition** (Tsatsaronis–Morosuk avoidable/unavoidable × endogenous/exogenous splitting) of waste heat recovery systems for the MAN B&W S65ME-C two-stroke marine diesel engine operating in Indian Ocean tropical conditions.

It is the computational backbone for the paper *"Advanced exergy decomposition of a low-speed two-stroke marine diesel with an ORC bottoming cycle in tropical seawater"* (Prajwal Rao, in submission). This repository is released to support the manuscript's Data Availability Statement.

- **Author:** Prajwal Rao, Independent researcher, Chennai, India
- **ORCID:** [0009-0004-2470-3656](https://orcid.org/0009-0004-2470-3656)
- **License:** MIT (see `LICENSE`)

### How to cite

If you use this code or the released CSV outputs, please cite the software via the metadata in `CITATION.cff` (a DOI is minted on Zenodo for each release) and the associated paper once published.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick test run (~15 minutes)
python run_all.py --quick

# 3. Full overnight run (~8-12 hours)
python run_all.py

# 4. Regenerate figures only (if CSVs already exist)
python run_all.py --plots-only
```

## What It Does

| Phase | Module | Runtime | Output |
|-------|--------|---------|--------|
| 1 | `engine_model.py` | ~1 min | Engine energy balance (20-100% MCR) |
| 2 | `orc_model.py` | ~1-3 hrs | ORC parametric sweep (8 fluids × 7 loads × 38 temps × 7 superheat) |
| 3 | `exergy_model.py` | ~2 min | Second-law analysis for engine + ORC |
| 4 | `environmental_model.py` | ~5 min | CO2, NOx, SOx, EEDI, CII, payback, NPV |
| 5 | `optimization.py` (MC) | ~30 min-2 hrs | Monte Carlo with 15,000 samples |
| 6 | `optimization.py` (NSGA-II) | ~2-4 hrs | Multi-objective Pareto optimization |
| 7 | `plot_results.py` | ~2 min | Publication-quality figures (PNG + PDF) |

## Output Files

All results go into `results/`:

- `01_engine_energy_balance.csv` — Energy balance at 81 load points
- `02_orc_parametric_results.csv` — Full ORC sweep (thousands of points)
- `02b_orc_optimal_per_fluid.csv` — Best ORC conditions per fluid per load
- `03_engine_exergy_balance.csv` — Exergy streams and destruction
- `03b_orc_exergy_analysis.csv` — ORC component-level exergy
- `04_environmental_economic_4E.csv` — Full 4E comparison across technologies
- `05_monte_carlo_results.csv` — Monte Carlo uncertainty propagation
- `05b_sensitivity_indices.csv` — Parameter sensitivity ranking
- `06_pareto_front.csv` — NSGA-II Pareto-optimal solutions
- `07_advanced_exergy_decomposition.csv` / `07b_component_improvement_ranking.csv` — Tsatsaronis–Morosuk avoidable/unavoidable × endogenous/exogenous split across 7 components
- `08_voyage_profile_schedules.csv` / `08b_voyage_weighted_performance.csv` — Indian coastal + long-haul voyage load profiles × 5 WHR technologies
- `09_seawater_sensitivity.csv` / `09b_tropical_vs_temperate_summary.csv` — 15–32 °C seawater sweep × 4 fluids
- `10_tewi_per_fluid.csv` / `10b_tewi_with_sensitivity.csv` — 20-year TEWI lifecycle at 2/5/10% leak rates

Additional analysis modules: `advanced_exergy_model.py`, `voyage_profile_model.py`, `seawater_sensitivity.py`, `tewi_analysis.py`. Figures `fig07`–`fig08` are produced by `generate_missing_figures.py`.

Figures go into `results/figures/` in both PNG and PDF formats.

## Dependencies

- Python 3.8+
- CoolProp (thermodynamic properties)
- NumPy, SciPy, Pandas
- Matplotlib (figures)
- tqdm (progress bars)
- pymoo (optional, for NSGA-II; falls back to grid search)
