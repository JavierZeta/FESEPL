# FESEPL

FESEPL stands for Free-Energy State Estimator with Precision Learning.

This repository is a standalone extraction of the state-space estimator from the
larger project, with the estimator module renamed from `sfec.py` to
`FESEPL.py`, and the estimator class renamed from `SFEC` to `FESEPL`.

## Included files

- `FESEPL.py`: the renamed estimator implementation
- `PlantClass.py`: plant and simulation dynamics
- `runner.py`: single-run experiment script
- `run_FESEPL.py`: experiment helpers used by `runner.py`
- `KalmanFilterClass.py`: Kalman baseline used by the experiment scripts

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the standalone runner:

```bash
python runner.py
```
