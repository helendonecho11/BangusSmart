# Synthetic DO benchmark

Run from this folder after installing requirements:

```powershell
python bangussmart_ml.py all
python bangussmart_ml.py demo
python do_experiment.py
python -m pytest -q
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

This is pipeline evaluation on synthetic data, not real farm accuracy.

`do_experiment.py` uses 365 days per seed (42–46), identical generator assumptions,
and horizons of 1, 3 and 6 hours. Each horizon predicts the measured DO at that
exact future hour, not the minimum DO anywhere in the interval. Features use only
readings available at the forecast origin: 12 lags, per-hour changes over 1/3/6
hours, 6/12-hour rolling means and minima, current temperature/pH, and cyclic hour.

The first 60% of hours fit models, the next 20% choose the model and margin, and
the last 20% evaluate. Training/validation labels reaching the next split are
purged. Observed history before a split may be used as features after it; that
history would be available operationally. Hyperparameters are fixed in code;
there is no test-set tuning. Growth retains its separate batch split.

Persistence, standardized Ridge, Random Forest and scikit-learn Gradient Boosting
share the same rows. No extra boosting library is needed. Model selection includes
persistence and minimizes validation MAE. Among margins 0, .25, .5, .75, 1 mg/L,
choose the one catching most validation crossings with <=1 false alarm per day;
ties prefer the smaller margin. If none meet the budget, use zero margin and
report the observed false-alarm rate. The budget is an explicit prototype policy,
not an aquaculture threshold or an approved cooperative operating policy.

A new crossing opportunity means measured DO >=4 now and measured DO <4 at the
forecast horizon. Each hourly origin is counted separately. These are synthetic
measured crossings, not independent biological events: longer horizons can count
multiple opportunities for one dip, and sensor noise can cause recrossings. An
alert is a forecast <4+margin while DO is fine now. A false alarm is such an alert
with DO >=4 at the target hour. False alarms/day = false alert-hours divided by
all evaluated hours/24, including currently-low hours. No cooldown or deduplication
is applied. These definitions make the comparison reproducible but do not measure
whether an alert arrived early enough for a farmer to act.

CSV files retain every seed, model, horizon, and margin. Aggregates report the
arithmetic mean, sample SD, min and max across five seeds. Selected-policy rows
are chosen independently using each seed's validation period. Test sweeps are
descriptive only; they must not be used to choose another margin.

The saved enhanced one-hour model is the seed-42 validation choice, fitted on the
first 60% only, with its validation-selected margin; it is not refitted on the test
period. Its feature-importance chart shows impurity importance for a tree model
(or absolute standardized coefficients if Ridge wins). Correlated lags can share
importance, and importance does not imply causality. Other seeds/horizons are
evaluation replications, not a deployment ensemble.

`advise` and `/advise` accept 3 or 12 historical values, oldest first. Three values
use the original one-hour model and zero margin. Twelve use the enhanced model
when its artifact exists; otherwise the original model uses the last three.
Responses identify the forecast model and margin. Forecast alerts can reduce
feeding under the existing FEED_FACTOR assumptions. The warning/critical rules,
FEED_TABLE and FEED_FACTOR have not been changed.

Synthetic daily patterns and gradual simulated dips favor models that learn those
patterns. Weather, feeding, equipment failures, sensor calibration, action lead
time, alert fatigue and real fish outcomes remain unvalidated. More synthetic
events reduce Monte Carlo uncertainty, not this domain gap. Numeric version
differences can change forecasts; `requirements-lock.txt` records this run's
environment (Python 3.14 on Windows). The original 150-day rerun is retained in
`models/phase1_metrics.json`; expanded results are separate CSVs.
