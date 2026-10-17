"""
BangusSmart ML prototype
========================
Three small machine-learning modules for the BangusSmart proposal:

  1. Water-quality STATUS classifier   (Good / Warning / Critical)
  2. Dissolved-oxygen FORECAST         (predict DO one hour ahead -> early warning)
  3. Fish GROWTH model + FEED advice   (predict average body weight, then daily feed)

IMPORTANT - READ THIS BEFORE PRESENTING RESULTS
-----------------------------------------------
No real Hayuma Cooperative data exists yet, so this script generates SYNTHETIC
data that imitates a bangus farm (daily DO cycle, sensor noise, occasional
low-oxygen events, logistic growth). The model scores you see are therefore a
test of the PIPELINE, not evidence of real-world accuracy. When real sensor and
weighing data are collected, replace the synthetic data with it
(see the `--sensor-csv` / `--growth-csv` options) and re-train.

Reference values taken from SEAFDEC/AQD (milkfish cage culture): pH 7.5-8.3,
DO 4-8 mg/L, temperature 26-32 C, ~400 g market size in ~120-150 days.
Everything marked ASSUMPTION is NOT from SEAFDEC and must be validated with
the cooperative / an aquaculture adviser.

Usage
-----
  python bangussmart_ml.py all          # generate data, train, evaluate, save plots
  python bangussmart_ml.py demo         # run example scenarios through the trained models
  python bangussmart_ml.py predict --temp 30 --ph 7.9 --do 3.4 --hour 5 \
         --do-history 3.9 4.2 4.6 --day 60 --mean-temp 29.5 --fish-count 5000
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Keep Matplotlib's cache writable in restricted Windows workspaces.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"
for d in (DATA, MODELS, PLOTS):
    d.mkdir(exist_ok=True)

SEED = 42
STATUSES = ["Good", "Warning", "Critical"]          # order = severity

# --------------------------------------------------------------------------
# Rules (the "knowledge" the labels come from).  Edit these, then re-train.
# --------------------------------------------------------------------------
# Good ranges are from SEAFDEC/AQD (cage culture). The Warning band width is an
# ASSUMPTION: anything outside Good but inside the Warning limits is "Warning",
# anything beyond is "Critical".
RULES = {
    #            good_low, good_high, warn_low, warn_high
    "temp_c": (26.0, 32.0, 24.0, 34.0),   # ASSUMPTION for warn limits
    "ph":     (7.5, 8.3, 7.0, 8.8),       # ASSUMPTION for warn limits
    "do_mgl": (4.0, 99.0, 3.0, 99.0),     # DO >= 4 good (SEAFDEC 4-8); <3 critical = ASSUMPTION
}

# Feed rate (% of body weight per day) by fish size.  PLACEHOLDER values:
# SEAFDEC says supplemental feeding in ponds should not exceed 4 % of body weight,
# so all values stay at or below 4 %.  Replace with the feeding table the
# cooperative actually uses / the SEAFDEC/AQD table for the culture system.
FEED_TABLE = [(50, 4.0), (150, 3.5), (300, 3.0), (10_000, 2.5)]   # (max weight g, % BW/day)
# Feeding multiplier by water status (ASSUMPTION - common practice is to cut
# feed when oxygen is low, and SEAFDEC trials link low DO with heavy feeding).
FEED_FACTOR = {"Good": 1.0, "Warning": 0.5, "Critical": 0.0}

DO_ALERT_MGL = 4.0     # forecast below this triggers an early warning


def rule_status(temp, ph, do):
    """Rule-based status (0 Good, 1 Warning, 2 Critical) for arrays or scalars."""
    temp, ph, do = np.asarray(temp), np.asarray(ph), np.asarray(do)
    sev = np.zeros(np.broadcast(temp, ph, do).shape, dtype=int)
    for name, x in (("temp_c", temp), ("ph", ph), ("do_mgl", do)):
        gl, gh, wl, wh = RULES[name]
        s = np.where((x < wl) | (x > wh), 2, np.where((x < gl) | (x > gh), 1, 0))
        sev = np.maximum(sev, s)
    return sev


# --------------------------------------------------------------------------
# 1. Synthetic data generators
# --------------------------------------------------------------------------
def make_sensor_data(days=150, seed=SEED):
    """Hourly temperature / pH / DO with a daily cycle, weather offsets and
    low-oxygen events. 'true_*' columns are what the water really was; the
    measured columns add sensor noise (what the IoT device would report)."""
    rng = np.random.default_rng(seed)
    n = days * 24
    t = np.arange(n)
    hour = t % 24
    day = t // 24

    day_temp = rng.normal(0, 1.2, days)           # warm / cool days
    day_ph = rng.normal(0, 0.12, days)
    temp = 29 + day_temp[day] + 2.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    do = 6.0 + 1.6 * np.sin(2 * np.pi * (hour - 10) / 24) - 0.25 * (temp - 29)
    ph = 7.9 + day_ph[day] + 0.22 * np.sin(2 * np.pi * (hour - 10) / 24)

    # Low-oxygen events (cloudy spell, overfeeding, aerator failure): DO sags for
    # 6-16 hours, pH drifts down a little.
    dip = np.zeros(n)
    for _ in range(int(days * 0.22)):
        start = rng.integers(0, n - 20)
        length = rng.integers(6, 17)
        depth = rng.uniform(1.5, 3.5)
        shape = np.sin(np.linspace(0, np.pi, length)) * depth
        dip[start:start + length] += shape[: n - start]
    do = do - dip
    ph = ph - 0.04 * dip
    # occasional heat waves / cold snaps
    for _ in range(int(days * 0.05)):
        start = rng.integers(0, n - 30)
        temp[start:start + 30] += rng.choice([-3.0, 3.0])

    # smooth short-term randomness
    do += pd.Series(rng.normal(0, 0.25, n)).rolling(3, min_periods=1).mean().to_numpy()
    do = np.clip(do, 0.3, 12)

    df = pd.DataFrame({"day": day, "hour": hour,
                       "true_temp_c": temp, "true_ph": ph, "true_do_mgl": do})
    # measured = true + sensor noise (ASSUMPTION: typical low-cost sensor error)
    df["temp_c"] = temp + rng.normal(0, 0.3, n)
    df["ph"] = ph + rng.normal(0, 0.10, n)
    df["do_mgl"] = np.clip(do + rng.normal(0, 0.25, n), 0.1, None)
    # label comes from the TRUE water condition -> model must learn to see through noise
    df["status"] = [STATUSES[i] for i in rule_status(df.true_temp_c, df.true_ph, df.true_do_mgl)]
    return df


def make_growth_data(batches=30, seed=SEED):
    """Average body weight sampled weekly for several batches of fish.
    ASSUMPTIONS: stocking weight 20 g, logistic growth towards ~600 g with
    ~400 g at day ~135 (the only anchor from SEAFDEC's cage-culture figures),
    growth speed shifts +/-4 % per degree C away from 29 C."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for b in range(batches):
        mean_temp = rng.uniform(26.5, 31.5)
        k = 0.0301 * (1 + 0.04 * (mean_temp - 29)) * rng.normal(1, 0.05)
        w0, wmax = rng.normal(20, 3), 600.0
        for d in range(0, 154, 7):
            w = wmax / (1 + (wmax / w0 - 1) * np.exp(-k * d))
            rows.append({"batch": b, "day": d, "mean_temp_c": mean_temp,
                         "avg_weight_g": max(w * rng.normal(1, 0.06), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------
def hour_feats(hour):
    return np.sin(2 * np.pi * np.asarray(hour) / 24), np.cos(2 * np.pi * np.asarray(hour) / 24)


CLS_FEATURES = ["temp_c", "ph", "do_mgl", "hour_sin", "hour_cos"]
DO_FEATURES = ["do_mgl", "do_lag1", "do_lag2", "do_lag3", "do_trend",
               "temp_c", "ph", "hour_sin", "hour_cos"]
GROWTH_FEATURES = ["day", "mean_temp_c"]


def add_cls_features(df):
    df = df.copy()
    df["hour_sin"], df["hour_cos"] = hour_feats(df["hour"])
    return df


def add_do_features(df):
    df = add_cls_features(df)
    for k in (1, 2, 3):
        df[f"do_lag{k}"] = df["do_mgl"].shift(k)
    df["do_trend"] = df["do_mgl"] - df["do_lag3"]
    df["do_next"] = df["do_mgl"].shift(-1)                # target: measured DO one hour ahead
    return df.dropna().reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Training
# --------------------------------------------------------------------------
# Plot style (validated 2-series palette: blue + orange; neutral ink for text)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def new_fig(w=7.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    return fig, ax


def load_or_make(csv, maker):
    if csv:
        print(f"  using your data: {csv}")
        return pd.read_csv(csv)
    return maker()


def train_status(sensor_df, report):
    df = add_cls_features(sensor_df)
    cut = int(df.day.max() * 0.8)                          # time-ordered split: no peeking at the future
    tr, te = df[df.day <= cut], df[df.day > cut]
    clf = GradientBoostingClassifier(random_state=SEED)
    # Warning/Critical hours are rare, so weight them up (otherwise the model just says "Good")
    clf.fit(tr[CLS_FEATURES], tr["status"], sample_weight=compute_sample_weight("balanced", tr["status"]))
    pred = clf.predict(te[CLS_FEATURES])
    rule_on_noisy = [STATUSES[i] for i in rule_status(te.temp_c, te.ph, te.do_mgl)]
    acc = (pred == te.status).mean()
    rule_acc = (np.array(rule_on_noisy) == te.status.to_numpy()).mean()
    print("\n[1] Water-quality status classifier (GradientBoosting)")
    print(f"    train rows {len(tr)}, test rows {len(te)} (last 20 % of days)")
    print(f"    accuracy on noisy sensor readings: model {acc:.3f} | plain threshold rule {rule_acc:.3f}")
    print(classification_report(te.status, pred, labels=STATUSES, zero_division=0))
    report["status"] = {"model_accuracy": round(float(acc), 4), "rule_on_noisy_accuracy": round(float(rule_acc), 4),
                        "test_rows": int(len(te)),
                        "class_counts_test": te.status.value_counts().to_dict()}
    joblib.dump(clf, MODELS / "status_classifier.joblib")

    cm = confusion_matrix(te.status, pred, labels=STATUSES)
    share = cm / cm.sum(axis=1, keepdims=True)              # colour = share of each ACTUAL class
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#2a78d6"])
    fig, ax = new_fig(5.8, 4.8)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(STATUSES); ax.set_yticks(range(3)); ax.set_yticklabels(STATUSES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="#fcfcfb" if share[i, j] > 0.6 else INK)
    ax.set_title("Status classifier: actual vs predicted\n(test period, synthetic data; shade = % of actual class)",
                 loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlabel("Predicted", color=INK2); ax.set_ylabel("Actual", color=INK2)
    ax.tick_params(colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(PLOTS / "1_status_confusion_matrix.png", dpi=150); plt.close(fig)


def train_do_forecast(sensor_df, report):
    df = add_do_features(sensor_df)
    cut = int(df.day.max() * 0.8)
    tr, te = df[df.day <= cut], df[df.day > cut]
    reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED)
    reg.fit(tr[DO_FEATURES], tr["do_next"])
    pred = reg.predict(te[DO_FEATURES])
    mae = mean_absolute_error(te.do_next, pred)
    mae_base = mean_absolute_error(te.do_next, te.do_mgl)   # "next hour = this hour"
    print("\n[2] Dissolved-oxygen forecast, 1 hour ahead (GradientBoosting)")
    print(f"    MAE: model {mae:.3f} mg/L | 'same as now' baseline {mae_base:.3f} mg/L | R2 {r2_score(te.do_next, pred):.3f}")
    # Early-warning check. Hours that are ALREADY below the alert level are easy (any rule that reads
    # the sensor catches them), so the honest test is "new crossings": DO is fine now (>= alert)
    # but really drops below the alert level next hour.
    already_low = te.do_mgl < DO_ALERT_MGL
    cross = (~already_low) & (te.do_next < DO_ALERT_MGL)
    fine_now = ~already_low
    print(f"    early warning of NEW crossings below {DO_ALERT_MGL} mg/L "
          f"({int(cross.sum())} real crossings among {int(fine_now.sum())} fine hours in test):")
    sweep = {}
    for margin in (0.0, 0.5, 1.0):
        flagged = pd.Series(pred < DO_ALERT_MGL + margin, index=te.index)
        caught = int((flagged & cross).sum())
        false_alarms = int((flagged & fine_now & ~cross).sum())
        print(f"      alert when forecast < {DO_ALERT_MGL + margin:.1f}: caught {caught}/{int(cross.sum())}, "
              f"false alarms {false_alarms} of {int((fine_now & ~cross).sum())} fine hours")
        sweep[f"margin_{margin}"] = {"caught": caught, "crossings": int(cross.sum()), "false_alarms": false_alarms}
    report["do_forecast"] = {"mae_model": round(float(mae), 4), "mae_baseline": round(float(mae_base), 4),
                             "r2": round(float(r2_score(te.do_next, pred)), 4), "new_crossing_alerts": sweep}
    joblib.dump(reg, MODELS / "do_forecaster.joblib")

    w = te.iloc[:24 * 7]                                    # first test week
    p = pred[: len(w)]
    x = np.arange(len(w)) / 24
    fig, ax = new_fig(8.5, 4.2)
    style_axes(ax, "Dissolved oxygen one hour ahead: forecast vs measured (one test week)")
    ax.plot(x, w.do_next, color=BLUE, linewidth=2, label="Measured")
    ax.plot(x, p, color=ORANGE, linewidth=2, label="Model forecast")
    ax.axhline(DO_ALERT_MGL, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.02, DO_ALERT_MGL - 0.28, "4 mg/L alert level", ha="left", fontsize=8.5, color=INK2)
    ax.set_xlabel("Days"); ax.set_ylabel("DO (mg/L)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(PLOTS / "2_do_forecast.png", dpi=150); plt.close(fig)


def train_growth(growth_df, report):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_i, te_i = next(gss.split(growth_df, groups=growth_df.batch))
    tr, te = growth_df.iloc[tr_i], growth_df.iloc[te_i]
    reg = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=SEED)
    reg.fit(tr[GROWTH_FEATURES], tr["avg_weight_g"])
    pred = reg.predict(te[GROWTH_FEATURES])
    mae = mean_absolute_error(te.avg_weight_g, pred)
    print("\n[3] Fish growth model (GradientBoosting) - tested on batches it never saw")
    print(f"    MAE {mae:.1f} g | R2 {r2_score(te.avg_weight_g, pred):.3f} | test batches {te.batch.nunique()}")
    report["growth"] = {"mae_g": round(float(mae), 2), "r2": round(float(r2_score(te.avg_weight_g, pred)), 4)}
    joblib.dump(reg, MODELS / "growth_model.joblib")

    fig, ax = new_fig(7.5, 4.4)
    style_axes(ax, "Predicted average weight by day of culture (synthetic batches)")
    ax.scatter(te.day, te.avg_weight_g, s=22, color=BLUE, edgecolor=SURFACE, linewidth=1, label="Weighed samples (test batches)", zorder=3)
    days = np.arange(0, 155)
    ax.plot(days, reg.predict(pd.DataFrame({"day": days, "mean_temp_c": 29.0})), color=ORANGE, linewidth=2,
            label="Model, mean water temp 29 C")
    ax.axhline(400, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(2, 410, "400 g market size", fontsize=8.5, color=INK2)
    ax.set_xlabel("Day of culture"); ax.set_ylabel("Average weight (g)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout(); fig.savefig(PLOTS / "3_growth_curve.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# 4. Using the trained models (this is what the dashboard backend would call)
# --------------------------------------------------------------------------
def feed_rate_pct(weight_g):
    for max_w, rate in FEED_TABLE:
        if weight_g <= max_w:
            return rate
    return FEED_TABLE[-1][1]


def calculate_feed(weight_g, fish_count, status, forecast_alert=False, rate_override=None):
    """Shared biomass logic. FEED_TABLE/FEED_FACTOR remain PLACEHOLDER/ASSUMPTION.

    User-supplied rates are also capped at 4% biomass per day. A forecast alert
    raises Good water to Warning, as in the original advise() behavior.
    """
    if not np.isfinite(weight_g) or weight_g < 0 or not np.isfinite(fish_count) or fish_count < 0 or int(fish_count) != fish_count:
        raise ValueError("Weight and fish count must be finite and nonnegative")
    if status not in FEED_FACTOR:
        raise ValueError("Unknown water status")
    requested_rate = feed_rate_pct(weight_g) if rate_override is None else rate_override
    if not np.isfinite(requested_rate) or requested_rate < 0:
        raise ValueError("Feed rate must be finite and nonnegative")
    rate = min(float(requested_rate), 4.0)
    effective = "Warning" if forecast_alert and status == "Good" else status
    biomass = float(weight_g) * fish_count / 1000
    return {"biomass_kg": biomass, "rate_pct": rate, "rate_was_capped": requested_rate > 4,
            "effective_status": effective, "water_multiplier": FEED_FACTOR[effective],
            "daily_feed_kg": biomass * rate / 100 * FEED_FACTOR[effective]}


def advise(temp, ph, do, hour, do_history, day, mean_temp, fish_count):
    """Previous 3 or 12 hourly DO readings, oldest first.

    Three readings use the original model; twelve use the benchmark-selected
    one-hour model and its validation-selected alert margin, when available.
    """
    if len(do_history) not in (3, 12):
        raise ValueError("do_history must contain 3 or 12 hourly readings, oldest first")
    clf = joblib.load(MODELS / "status_classifier.joblib")
    dof = joblib.load(MODELS / "do_forecaster.joblib")
    gm = joblib.load(MODELS / "growth_model.joblib")
    hs, hc = hour_feats(hour)

    # Status used for decisions = threshold rule (on these synthetic tests it was at least as good as the ML
    # classifier, see README). The ML classifier is still reported alongside it for comparison.
    status = STATUSES[int(rule_status(temp, ph, do))]
    row = pd.DataFrame([[temp, ph, do, hs, hc]], columns=CLS_FEATURES)
    ml_status = clf.predict(row)[0]
    probs = {k: round(float(v), 2) for k, v in zip(clf.classes_, clf.predict_proba(row)[0])}

    lag3, lag2, lag1 = do_history[-3:]
    x = pd.DataFrame([[do, lag1, lag2, lag3, do - lag3, temp, ph, hs, hc]], columns=DO_FEATURES)
    do_next = float(dof.predict(x)[0])
    margin = 0.0
    forecast_model = "Original Gradient Boosting (3-hour history)"
    if len(do_history) == 12 and (MODELS / "do_early_warning.joblib").exists():
        from do_experiment import features
        bundle = joblib.load(MODELS / "do_early_warning.joblib")
        values = list(do_history) + [do]
        hours = [(hour - 12 + k) % 24 for k in range(13)]
        frame = features(pd.DataFrame(dict(do_mgl=values, hour=hours,
                                          day=np.arange(13) // 24, temp_c=temp, ph=ph)))
        do_next = float(bundle["model"].predict(frame[bundle["features"]])[0]) if bundle["model"] is not None else float(do)
        margin = bundle["margin"]
        forecast_model = bundle["name"]
    forecast_alert = do_next < DO_ALERT_MGL + margin

    weight = float(gm.predict(pd.DataFrame([[day, mean_temp]], columns=GROWTH_FEATURES))[0])
    feeding = calculate_feed(weight, fish_count, status, forecast_alert)
    rate, factor = feeding['rate_pct'], feeding['water_multiplier']
    biomass_kg, feed_kg = feeding['biomass_kg'], feeding['daily_feed_kg']

    alerts = []
    if status != "Good":
        alerts.append(f"Water status is {status} (rule)")
    if forecast_alert:
        alerts.append(f"DO forecast {do_next:.1f} mg/L next hour (alert threshold {DO_ALERT_MGL + margin})")
    return {"status_rule": status, "status_ml_classifier": ml_status, "ml_status_probabilities": probs, "do_forecast_next_hour_mgl": round(do_next, 2),
            "do_forecast_model": forecast_model, "do_alert_margin_mgl": margin,
            "synthetic_data_warning": "Synthetic-trained prototype; not validated for real farm decisions.",
            "predicted_avg_weight_g": round(weight, 1), "estimated_biomass_kg": round(biomass_kg, 1),
            "base_feed_rate_pct": rate, "feed_factor": factor, "recommended_feed_today_kg": round(feed_kg, 2),
            "alerts": alerts or ["None"]}


def print_advice(title, r):
    print(f"\n--- {title}")
    for k, v in r.items():
        print(f"    {k:30s} {v}")


def cmd_demo():
    scenarios = [
        ("Normal afternoon, mid-growth", dict(temp=30.2, ph=8.0, do=6.4, hour=15, do_history=[6.1, 6.3, 6.4], day=60, mean_temp=29.5, fish_count=5000)),
        ("Sudden DO drop (an observed event in the synthetic test period: the forecaster MISSES it)", dict(temp=27.91, ph=8.05, do=4.69, hour=23, do_history=[6.23, 5.96, 6.00], day=121, mean_temp=29.0, fish_count=5000)),
        ("Hot day, DO already low", dict(temp=33.1, ph=7.4, do=2.8, hour=14, do_history=[3.6, 3.2, 3.0], day=120, mean_temp=30.5, fish_count=5000)),
    ]
    for name, kw in scenarios:
        print_advice(name, advise(**kw))
    print("\nNOTE: demo values are synthetic-data-trained. Not for real feeding decisions.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["all", "generate", "train", "demo", "predict"])
    ap.add_argument("--sensor-csv", help="real sensor CSV: day,hour,temp_c,ph,do_mgl[,status]")
    ap.add_argument("--growth-csv", help="real weighing CSV: batch,day,mean_temp_c,avg_weight_g")
    ap.add_argument("--temp", type=float); ap.add_argument("--ph", type=float); ap.add_argument("--do", type=float)
    ap.add_argument("--hour", type=int); ap.add_argument("--do-history", type=float, nargs="+", metavar="DO", help="3 or 12 previous hourly readings, oldest first")
    ap.add_argument("--day", type=int); ap.add_argument("--mean-temp", type=float); ap.add_argument("--fish-count", type=int)
    a = ap.parse_args()

    if a.command in ("all", "generate"):
        print("Generating SYNTHETIC data ...")
        s, g = make_sensor_data(), make_growth_data()
        s.to_csv(DATA / "synthetic_sensor_data.csv", index=False)
        g.to_csv(DATA / "synthetic_growth_data.csv", index=False)
        print(f"  sensor rows {len(s)}, growth rows {len(g)} -> {DATA}")
        print("  status mix:", s.status.value_counts().to_dict())

    if a.command in ("all", "train"):
        sensor = load_or_make(a.sensor_csv, lambda: pd.read_csv(DATA / "synthetic_sensor_data.csv"))
        growth = load_or_make(a.growth_csv, lambda: pd.read_csv(DATA / "synthetic_growth_data.csv"))
        if "status" not in sensor:                              # real data without labels -> label with the rules
            sensor["status"] = [STATUSES[i] for i in rule_status(sensor.temp_c, sensor.ph, sensor.do_mgl)]
        report = {}
        train_status(sensor, report)
        train_do_forecast(sensor, report)
        train_growth(growth, report)
        (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
        print(f"\nModels saved in {MODELS}, plots in {PLOTS}")

    if a.command == "demo":
        cmd_demo()
    if a.command == "predict":
        need = ["temp", "ph", "do", "hour", "do_history", "day", "mean_temp", "fish_count"]
        if any(getattr(a, n) is None for n in need):
            sys.exit("predict needs: --temp --ph --do --hour --do-history H-3 H-2 H-1 --day --mean-temp --fish-count")
        print_advice("Result", advise(a.temp, a.ph, a.do, a.hour, a.do_history, a.day, a.mean_temp, a.fish_count))


if __name__ == "__main__":
    main()
