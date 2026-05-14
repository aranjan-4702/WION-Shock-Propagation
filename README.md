# WION Shock Model

Modeling shock propagation in the World Input-Output Network (WION) with damping and substitution effects.

## What this repo does
- Builds IO matrices (Z, F, X, A, B) from ICIO-style SML CSVs
- Computes vulnerability and damping metrics (Vj, phi, d, D)
- Simulates shock scenarios and aggregates country/sector impacts
- Produces figures and tables for the thesis and presentation

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the scripts or notebooks.

```bash
python -m venv .venv
pip install -r requirements.txt
python main.py
```

Other entry points:
- `python debug_damping.py`
- `python robustness_checks.py --year 2018 --n-iter 10`

## Project layout
- [data/README.md](data/README.md) - raw and processed datasets
- [docs/README.md](docs/README.md) - thesis and presentation sources
- [notebooks/README.md](notebooks/README.md) - exploratory and results notebooks
- [outputs/README.md](outputs/README.md) - generated figures and tables
- [src/README.md](src/README.md) - core model code
- [tests/README.md](tests/README.md) - test scaffolding

## Notes
- Base year is set in [config.py](config.py).
- Outputs are generated under [outputs/README.md](outputs/README.md) and can be regenerated.