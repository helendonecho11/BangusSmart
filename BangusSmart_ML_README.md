# BangusSmart – Machine Learning Prototype

Prototype of the machine-learning component described in the BangusSmart topic proposal
(IoT + ML smart aquaculture management for bangus farming, Hayuma Cooperative).

> **Read this first:** there is no real farm data yet, so the prototype trains on **synthetic data**
> that imitates a bangus farm. The scores below test that the *pipeline works*. They are **not**
> evidence of real-world accuracy. Real accuracy can only be reported after the models are trained
> and tested on data collected at the cooperative.

## 1. What is inside

| File / folder | Purpose |
|---|---|
| `bangussmart_ml.py` | The whole prototype (data generator, training, evaluation, advice function, demo) |
| `requirements.txt` | Python packages |
| `data/synthetic_*.csv` | Generated practice data (hourly sensor data, weekly fish weights) |
| `data/template_*.csv` | Column layout your **real** data must follow |
| `models/` | Saved trained models + `metrics.json` |
| `plots/` | 3 result charts for your paper/defense |

### The three modules

| # | Module | Input → output | Algorithm |
|---|---|---|---|
| 1 | Water-quality status | temperature, pH, DO, hour → Good / Warning / Critical | Gradient Boosting classifier |
| 2 | Dissolved-oxygen forecast | current DO, previous 3 hours, temp, pH, hour → DO one hour ahead | Gradient Boosting regressor |
| 3 | Growth + feeding advice | day of culture, mean water temp → average fish weight → daily feed (kg) | Gradient Boosting regressor + feeding table |

Water-quality ranges come from SEAFDEC/AQD milkfish cage-culture guidance: pH 7.5–8.3, DO 4–8 mg/L,
temperature 26–32 °C. SEAFDEC also reports that supplemental pond feeding should not exceed 4 % of body
weight. **Everything else (Warning/Critical limits, the feeding table, the feed reduction when water is
poor, the starting fish size) is an assumption** and is marked `ASSUMPTION` / `PLACEHOLDER` in the code.

## 2. Results on the synthetic test data (what they do and do not show)

| Module | Result | Honest reading |
|---|---|---|
| 1 Status | Model accuracy 0.893 vs **plain threshold rule 0.919** | With labels created from thresholds, ML cannot beat the thresholds. The model catches 93 % of Critical hours but raises more false alarms. **The prototype therefore uses the rule for the status decision** and shows the ML result alongside it. ML becomes worthwhile only when labels come from real outcomes (fish stress, mortality, expert judgement). |
| 2 DO forecast | MAE 0.289 mg/L vs 0.362 for "next hour = this hour"; R² 0.885 | Better than the naive baseline on average. |
| 2 Early warning | Of 6 real drops below 4 mg/L (from a fine reading), forecast alert at 4.0 caught **0**; at 4.5 caught 1 (9 false alarms); at 5.0 caught 5 (106 false alarms of 683 fine hours) | **The forecaster does not yet give reliable advance warning of sudden drops.** Only 6 events, so this is indicative, not conclusive. This is the main thing to improve with real data. |
| 3 Growth | MAE 28 g, R² 0.922 on 8 batches the model never saw | Mostly shows the model recovers the growth curve we simulated. |

## 3. Step-by-step: from this prototype to a finished capstone ML component

### Step 1 – Set up (15 min)
1. Install Python 3.10 or newer (python.org). Tick "Add to PATH".
2. Open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   ```

### Step 2 – Run the prototype (5 min)
```
python bangussmart_ml.py all      # generate synthetic data, train 3 models, print metrics, save plots
python bangussmart_ml.py demo     # three example situations through the trained models
python bangussmart_ml.py predict --temp 30 --ph 7.9 --do 3.4 --hour 5 --do-history 3.9 4.2 4.6 --day 60 --mean-temp 29.5 --fish-count 5000
```
Open `plots/` and `models/metrics.json`. Make sure every member can run it and explain each number.

### Step 3 – Understand and defend the design (1–2 days)
Be ready to explain: why a time-ordered split is used (no peeking at the future), why the growth model is
tested on unseen batches, why the rule beat the classifier, and which numbers are assumptions.
Put these in your methodology chapter.

### Step 4 – Confirm the assumptions with the cooperative (1 week)
Ask Hayuma Cooperative: culture system (pond, cage or tank), species stage and stocking size, number of
fish, current feeding table and feeding times, what DO/pH/temperature they consider dangerous, and
whether they keep any past records. Then edit `RULES`, `FEED_TABLE`, `FEED_FACTOR` at the top of
`bangussmart_ml.py` to match. If possible, have an aquaculture adviser or the BFAR/SEAFDEC contact
review them.

### Step 5 – Build the data-collection hardware (1–3 weeks) – see materials list
Wire the sensors to the microcontroller, **calibrate** them, and log one reading per hour (or more often)
with a timestamp. Export to CSV in the layout of `data/template_sensor_data.csv`.
Compare sensor readings with a reference instrument or test kit and record the difference – the proposal
promises sensor-reliability testing.

### Step 6 – Collect real data (as long as possible; at least several weeks)
- Sensor CSV: `day, hour, temp_c, ph, do_mgl` (optional `status` column if experts label hours).
- Growth CSV: sample and weigh fish every 1–2 weeks (`batch, day, mean_temp_c, avg_weight_g`).
- Record events (aerator failure, heavy rain, fish kill, feeding changes) in a notebook – they become labels.
Models need more than a few weeks of data to learn daily and weather patterns, so start collecting early.

### Step 7 – Train on the real data
```
python bangussmart_ml.py train --sensor-csv my_sensor_data.csv --growth-csv my_growth_data.csv
```
Report the new metrics instead of the synthetic ones. Keep the synthetic results only as a "pipeline
validation" section.

### Step 8 – Improve the weak parts (see section 5)

### Step 9 – Connect to the dashboard
`advise(...)` in the script returns status, DO forecast, predicted weight, feed amount and alerts as one
dictionary. Wrap it in a small web API (Flask or FastAPI) and call it from the dashboard each time new
sensor data arrives.

### Step 10 – Test and document
Functional tests, sensor accuracy, response time, and user acceptance with the cooperative (as promised in
your objectives). Keep screenshots, logs, and the metrics file.

## 4. Materials needed

**Software (all free):** Python 3.10+, VS Code or Jupyter Notebook, the packages in `requirements.txt`,
Git/GitHub for version control, a spreadsheet program for cleaning CSVs. Later: Flask or FastAPI and a
database (SQLite/MySQL/PostgreSQL) for the dashboard.

**Data collection hardware (for real data):**
- Microcontroller with Wi-Fi, such as an ESP32 board, plus USB cable and a stable power supply
- Waterproof temperature sensor (DS18B20 type)
- pH sensor and interface board (analog pH kit)
- Dissolved-oxygen sensor and interface board (analog or lab-grade probe)
- pH calibration buffer solutions (pH 4, 7, 10) and a DO calibration method per the sensor manual
- Weatherproof enclosure, cables, connectors, waterproof sealing
- Reliable internet or a local logging option (SD card module) at the farm
- A reference instrument or water test kit to check sensor accuracy

**For growth data:** a hand net, a bucket, a weighing scale (grams), a notebook or phone for recording.

**People/permissions:** a contact person at Hayuma Cooperative, permission to install and sample, and an
adviser who can validate the rules.

Check current prices and availability yourself; they change.

## 5. What else you can do (ordered by value)

1. **Fix early warning (highest value).** Add more history (6–12 hours, rate of change, hours since last
   feeding), weather (rain, cloud cover, wind), and feeding time as features. Try forecasting 3–6 hours
   ahead, and try quantile regression so the model gives a "worst plausible" DO. Tune the alert margin
   by weighing missed events against false alarms with the cooperative.
2. **Compare models** (linear regression, Random Forest, XGBoost/LightGBM, LSTM) with the same split and
   report a table. This is good thesis content.
3. **Anomaly detection** for sensor faults (stuck or impossible readings) so bad sensors do not trigger
   alarms or feeding changes.
4. **Better growth model:** use real weights plus feed, temperature and survival; compare with a standard
   growth curve (logistic/von Bertalanffy) as a baseline.
5. **Feed optimisation:** track feed conversion ratio (FCR); the ML target could become "feed amount that
   gave the best growth without a DO drop".
6. **Explainability:** feature-importance chart so the panel and farmers see why an alert fired.
7. **Computer-vision extension (optional, bigger scope):** fish counting or size estimation from camera
   images. Only if time allows; it needs its own dataset.
8. **Update the proposal** with the specific algorithms, data sources, split strategy, and evaluation
   metrics (MAE, R², precision/recall, early-warning detection rate).

## 6. Known limitations

- Synthetic data: results are not transferable to real farms.
- Early-warning ability is weak (see table above).
- Gradient-boosting growth curves look step-shaped; fine for a prototype, smooth it later if needed.
- Rules and feeding values are placeholders until validated.
- Saved model files depend on the library versions; retrain if you change versions.
- This is a decision-support prototype; it must not replace the judgment of farm personnel.

## 7. References used for the rules

SEAFDEC/AQD. Milkfish culture and cage-culture water-quality guidance: https://www.seafdec.org.ph/milkfish/
Milkfish production and water quality in brackishwater ponds at different feeding levels and frequencies
(SEAFDEC/AQD repository): https://repository.seafdec.org.ph/handle/10862/1755
