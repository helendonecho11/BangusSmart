"""Reproducible SYNTHETIC DO benchmark; no farm-accuracy claims.

Each seed uses 365 days. First 60% fit, next 20% validation, last 20%
test. Purge horizon hours at both boundaries so labels cannot cross splits.
Model choice minimizes validation MAE; margin maximizes validation detection
subject to <= 1 false alert per day (prototype policy, not farm guidance).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import bangussmart_ml as core

FEATURES = (["do_mgl", "temp_c", "ph", "hour_sin", "hour_cos"]
            + [f"do_lag{k}" for k in range(1, 13)]
            + ["change_1", "change_3", "change_6", "mean_6", "min_6", "mean_12", "min_12"])
MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0)


def features(sensor, horizon=None):
    """Past/current inputs only; optionally attach a future training target."""
    df = core.add_cls_features(sensor)
    for k in range(1, 13):
        df[f"do_lag{k}"] = df.do_mgl.shift(k)
    for k in (1, 3, 6):
        df[f"change_{k}"] = (df.do_mgl - df[f"do_lag{k}"]) / k
    for k in (6, 12):
        df[f"mean_{k}"] = df.do_mgl.rolling(k).mean()
        df[f"min_{k}"] = df.do_mgl.rolling(k).min()
    if horizon is not None:
        df["target"] = df.do_mgl.shift(-horizon)
    df["time"] = df.day * 24 + df.hour
    return df.dropna(subset=FEATURES + (["target"] if horizon is not None else []))


def candidates(seed):
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                               max_features=0.8, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                                       learning_rate=0.05, random_state=seed),
    }


def alerts(df, pred, margin):
    fine = df.do_mgl.to_numpy() >= 4.0
    crossing = fine & (df.target.to_numpy() < 4.0)
    flagged = fine & (pred < 4.0 + margin)
    caught = int((flagged & crossing).sum())
    total = int(crossing.sum())
    false = int((flagged & ~crossing).sum())
    return dict(caught=caught, crossings=total, false_alarms=false,
                detection_rate=caught / total if total else float("nan"),
                false_alarms_per_day=false / (len(df) / 24))


def run():
    rows, warning_rows, choices = [], [], []
    for seed in (42, 43, 44, 45, 46):
        sensor = core.make_sensor_data(days=365, seed=seed)
        sensor.to_csv(core.DATA / f"synthetic_sensor_365_seed{seed}.csv", index=False)
        for horizon in (1, 3, 6):
            df = features(sensor, horizon)
            a, b = int(len(sensor) * .6), int(len(sensor) * .8)
            train = df[df.time + horizon < a]
            val = df[(df.time >= a) & (df.time + horizon < b)]
            test = df[df.time >= b]
            models = candidates(seed)
            val_preds = {"Persistence": val.do_mgl.to_numpy()}
            test_preds = {"Persistence": test.do_mgl.to_numpy()}
            for name, model in models.items():
                model.fit(train[FEATURES], train.target)
                val_preds[name] = model.predict(val[FEATURES])
                test_preds[name] = model.predict(test[FEATURES])
            best = min(val_preds, key=lambda n: mean_absolute_error(val.target, val_preds[n]))
            feasible = [m for m in MARGINS if alerts(val, val_preds[best], m)["false_alarms_per_day"] <= 1.0]
            margin = max(feasible, key=lambda m: (alerts(val, val_preds[best], m)["caught"], -m)) if feasible else 0.0
            choices.append(dict(seed=seed, horizon=horizon, model=best, margin=margin))
            for name, pred in test_preds.items():
                rows.append(dict(seed=seed, horizon=horizon, model=name,
                                 mae=mean_absolute_error(test.target, pred), r2=r2_score(test.target, pred),
                                 validation_mae=mean_absolute_error(val.target, val_preds[name])))
                for m in MARGINS:
                    warning_rows.append(dict(seed=seed, horizon=horizon, model=name, margin=m,
                                             selected=name == best and m == margin, **alerts(test, pred, m)))
            if seed == 42 and horizon == 1:
                # Save exactly the evaluated fit; do not refit after margin selection.
                joblib.dump(dict(model=models.get(best), features=FEATURES, margin=margin,
                                 name=best), core.MODELS / "do_early_warning.joblib")
                importance = (models[best].feature_importances_ if hasattr(models.get(best), "feature_importances_")
                              else np.abs(models[best][-1].coef_) if best != "Persistence" else np.array([1.] + [0.] * (len(FEATURES)-1)))
                imp = pd.Series(importance, index=FEATURES).sort_values().tail(12)
                fig, ax = core.new_fig(8, 5)
                core.style_axes(ax, f"{best}: feature importance (synthetic, seed 42, 1 h)")
                ax.barh(imp.index, imp.values, color=core.BLUE)
                ax.set_xlabel("Importance" if best != "Ridge" else "Absolute standardized coefficient")
                fig.tight_layout(); fig.savefig(core.PLOTS / "4_do_feature_importance.png", dpi=150)
                core.plt.close(fig)
            print(f"seed {seed}, horizon {horizon}: validation chose {best}, margin {margin}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(core.DATA / "do_comparison_runs.csv", index=False)
    summary = raw.groupby(["horizon", "model"])[["mae", "r2"]].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(core.DATA / "do_model_comparison.csv")
    warnings = pd.DataFrame(warning_rows)
    warnings.to_csv(core.DATA / "do_alert_sweep.csv", index=False)
    alert_summary = warnings.groupby(["horizon", "model", "margin"])[["detection_rate", "false_alarms_per_day"]].agg(["mean", "std", "min", "max"])
    alert_summary.columns = ["_".join(c) for c in alert_summary.columns]
    alert_summary.to_csv(core.DATA / "do_alert_summary.csv")
    selected = pd.DataFrame(warning_rows).query("selected")
    selected.to_csv(core.DATA / "do_selected_alerts.csv", index=False)
    (core.MODELS / "do_selection.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    fig, axes = core.plt.subplots(1, 2, figsize=(12, 4.5), facecolor=core.SURFACE)
    labels = [f"{h}h {n}" for h, n in summary.index]
    for ax, metric, color in zip(axes, ("mae", "r2"), (core.BLUE, core.ORANGE)):
        core.style_axes(ax, f"Synthetic DO comparison: {metric.upper()} (mean +/- SD)")
        ax.bar(np.arange(len(summary)), summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], color=color, capsize=3)
        ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=8)
        ax.set_ylabel("mg/L" if metric == "mae" else "R2")
    fig.tight_layout(); fig.savefig(core.PLOTS / "5_do_model_comparison.png", dpi=150); core.plt.close(fig)
    print(summary.to_string())
    print("Selected validation policies:\n", selected.to_string(index=False))


if __name__ == "__main__":
    run()
