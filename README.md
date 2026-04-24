# FESEPL

FESEPL stands for Free-Energy State Estimator with Precision Learning.



## Included files

- `FESEPL.py`: the estimator implementation
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
