"""Length -> weight -> uncertain culture age. Appearance never reveals exact age."""
import argparse
import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bangussmart_ml import MODELS, calculate_feed, rule_status, STATUSES
from media_vision import load_config


class AnalysisOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    square_size_cm: float = Field(gt=0, le=20)
    mean_temp_c: float = Field(default=29, ge=0, le=45)
    capture_date: date = Field(default_factory=lambda: datetime.now(ZoneInfo('Asia/Manila')).date())
    stocking_date: date | None = None
    known_age_days: int | None = Field(default=None, ge=0, le=3650, description='Known batch days SINCE STOCKING, not biological age.')
    fish_count: int | None = Field(default=None, ge=0, le=10000000)
    temp_c: float | None = Field(default=None, ge=-5, le=50)
    ph: float | None = Field(default=None, ge=0, le=14)
    do_mgl: float | None = Field(default=None, ge=0, le=50)
    hour: int | None = Field(default=None, ge=0, le=23)
    do_history: list[float] | None = None
    meals_per_day: int = Field(default=2, ge=1, le=12)
    feed_rate_pct: float | None = Field(default=None, gt=0, le=10)
    sample_fps: float = Field(default_factory=lambda: load_config()['video']['sample_fps'], ge=.1, le=5)

    @model_validator(mode='after')
    def validate_records(self):
        if self.stocking_date:
            days = (self.capture_date - self.stocking_date).days
            if days < 0:
                raise ValueError('Stocking date must be on or before capture date')
            if self.known_age_days is not None and self.known_age_days != days:
                raise ValueError('Known batch age conflicts with stocking and capture dates')
        if self.do_history is not None:
            if len(self.do_history) not in (3, 12) or not all(np.isfinite(v) and v >= 0 for v in self.do_history):
                raise ValueError('do_history needs 3 or 12 finite, nonnegative hourly readings, oldest first')
            if self.hour is None:
                raise ValueError('hour is required with do_history')
        return self

    def recorded_age(self):
        return (self.capture_date - self.stocking_date).days if self.stocking_date else self.known_age_days


def length_to_weight(length_cm, config=None):
    config = config or load_config()
    relation = config['length_weight']
    if not np.isfinite(length_cm) or length_cm <= 0:
        raise ValueError('Length must be finite and positive')
    lo, hi = relation['valid_length_cm']
    if not lo <= length_cm <= hi:
        return {'point_g': None, 'confidence': 'unavailable', 'status': 'outside_length_calibration',
                'warnings': [f'Length {length_cm:.3f} cm is outside the configured {lo}–{hi} cm calibration range. Measure weight or refit with training data; no extrapolation.']}
    return {'point_g': float(relation['a'] * length_cm ** relation['b']), 'confidence': 'low',
            'source': relation['source'], 'status': relation['status'],
            'warnings': ['Length–weight relation is not validated for this cooperative; fork-length landmarks require review.']}


@lru_cache(maxsize=1)
def _growth_model():
    return joblib.load(MODELS / 'growth_model.joblib')


def weight_to_age(weight_g, mean_temp_c, config=None):
    config = config or load_config()
    params = config['age']
    base = {'point_days': None, 'range_days': None, 'confidence': 'unavailable', 'basis': 'days_since_stocking',
            'range_type': 'ASSUMPTION sensitivity envelope, not a confidence interval',
            'warnings': ['Growth model is synthetic; no real-fish age accuracy is known. A photo cannot reveal exact age.']}
    if weight_g is None:
        base['warnings'].append('Weight unavailable: age could not be estimated.')
        return base
    if not np.isfinite(weight_g) or weight_g <= 0 or not np.isfinite(mean_temp_c):
        raise ValueError('Weight must be positive and weight/temperature must be finite')
    if not params['mean_temp_range_c'][0] <= mean_temp_c <= params['mean_temp_range_c'][1]:
        base['warnings'].append('Mean temperature is outside the synthetic growth training range (26.5–31.5 °C).')
        return base
    days = np.arange(params['min_day'], params['max_day'] + 1)
    predictions = _growth_model().predict(pd.DataFrame({'day': days, 'mean_temp_c': mean_temp_c}))
    if not predictions.min() <= weight_g <= predictions.max():
        base['warnings'].append(f'Weight is outside the growth model domain at this temperature ({predictions.min():.3f}–{predictions.max():.3f} g). No boundary age substituted.')
        return base
    differences = np.abs(predictions - weight_g)
    closest = days[np.isclose(differences, differences.min())]
    point = int(np.rint(np.median(closest)))
    matching = days[(predictions >= weight_g * (1 - params['weight_tolerance_fraction'])) & (predictions <= weight_g * (1 + params['weight_tolerance_fraction']))]
    low = min(point - params['minimum_half_width_days'], int(matching.min()) if len(matching) else point)
    high = max(point + params['minimum_half_width_days'], int(matching.max()) if len(matching) else point)
    base.update(point_days=point, range_days=[max(params['min_day'], low), min(params['max_day'], high)], confidence='low',
                weight_tolerance_fraction=params['weight_tolerance_fraction'], minimum_half_width_days=params['minimum_half_width_days'])
    return base


def resolve_age(weight_g, options, config=None):
    config = config or load_config()
    estimated = weight_to_age(weight_g, options.mean_temp_c, config)
    known = options.recorded_age()
    primary = estimated.copy() if known is None else {'point_days': known, 'range_days': [known, known],
               'confidence': 'recorded_unverified', 'basis': 'days_since_stocking', 'source': 'stocking_date' if options.stocking_date else 'known_batch_age',
               'warnings': ['Taken from user records, not inferred from media; biological age at stocking is unknown.']}
    difference = None
    disagreement = False
    if known is not None and estimated['point_days'] is not None:
        # At recorded day zero, any nonzero discrepancy exceeds the relative rule.
        difference = abs(estimated['point_days'] - known) / known if known else (None if estimated['point_days'] else 0.)
        disagreement = estimated['point_days'] != 0 if known == 0 else difference > config['age']['disagreement_fraction']
    return {'primary': primary, 'media_or_weight_cross_check': estimated,
            'disagreement_over_20_percent': disagreement, 'relative_difference': difference,
            'comparison_note': 'Absolute point difference / recorded days; at day zero, any nonzero point flags disagreement. Compare days of culture only, never birth age.'}


def recommend_feed(weight_g, options, count_estimate=None):
    count = options.fish_count if options.fish_count is not None else count_estimate
    missing = [name for name, value in (('average_weight_g', weight_g), ('fish_count', count), ('temp_c', options.temp_c), ('ph', options.ph), ('do_mgl', options.do_mgl)) if value is None]
    if missing:
        return {'available': False, 'daily_feed_kg': None, 'missing': missing,
                'warnings': ['Feed is unavailable without weight, a stock count and complete water readings. No good-water default is assumed.']}
    status = STATUSES[int(rule_status(options.temp_c, options.ph, options.do_mgl))]
    forecast = None
    forecast_alert = False
    if options.do_history is not None:
        # Reuse the original forecast policy. Its predicted weight is not used:
        # the media-measured average weight supplies the biomass here.
        from bangussmart_ml import advise
        forecast = advise(options.temp_c, options.ph, options.do_mgl, options.hour,
                          options.do_history, options.recorded_age() or 0, options.mean_temp_c, count)
        forecast_alert = forecast['do_forecast_next_hour_mgl'] < 4 + forecast['do_alert_margin_mgl']
    feeding = calculate_feed(weight_g, count, status, forecast_alert, options.feed_rate_pct)
    feeding.update(available=True, average_weight_g=weight_g, fish_count=count,
                   count_source='user_supplied' if options.fish_count is not None else 'video_distinct_track_estimate',
                   water_status=status, meals_per_day=options.meals_per_day,
                   per_meal_kg=feeding['daily_feed_kg'] / options.meals_per_day,
                   rate_source='user_supplied_capped_at_4_percent' if options.feed_rate_pct is not None else 'PLACEHOLDER_FEED_TABLE',
                   warnings=['ASSUMPTION: feeding table and water multipliers are not validated for this cooperative.',
                             'Estimated media weight and video count may be unrepresentative of the batch. Confirm with farm personnel.'])
    if forecast is not None:
        feeding['do_forecast'] = {key: forecast[key] for key in ('do_forecast_next_hour_mgl', 'do_forecast_model', 'do_alert_margin_mgl')}
    return feeding


def summarize_fish(fish, options, config=None, video_count=None):
    config = config or load_config()
    for item in fish:
        if 'estimated_weight' not in item:
            item['estimated_weight'] = length_to_weight(item['length_cm'], config)
        item['age'] = resolve_age(item['estimated_weight']['point_g'], options, config)
    weights = [item['estimated_weight']['point_g'] for item in fish if item['estimated_weight']['point_g'] is not None]
    lengths = [item['length_cm'] for item in fish]
    # Do not silently exclude uncalibrated fish and extrapolate their batch biomass.
    average_weight = float(np.mean(weights)) if weights and len(weights) == len(fish) else None
    age = resolve_age(average_weight, options, config)
    warnings = ['Decision support only: assists farm personnel and does not replace their judgment.',
                'Photo/video-derived age is an estimate of days since stocking, never exact biological age.']
    for item in fish:
        warnings.extend(item['estimated_weight'].get('warnings', []))
    warnings = list(dict.fromkeys(warnings))
    if age['disagreement_over_20_percent']:
        warnings.append('Age disagreement exceeds 20%: verify records, scale, measurements and growth assumptions.')
    if video_count is None and options.fish_count is None:
        warnings.append('Image candidates are a sample, not a batch stock count. Enter your actual fish count for feeding.')
    return {'size': {'average_length_cm': float(np.mean(lengths)) if lengths else None, 'length_type': 'fork_length_proxy'},
            'weight': {'average_g': average_weight, 'source': 'estimated_from_length', 'confidence': 'low' if average_weight is not None else 'unavailable'},
            'age': age, 'feed': recommend_feed(average_weight, options, video_count),
            'confidence': 'low', 'warnings': warnings, 'fish': fish,
            'options': options.model_dump(mode='json')}


def refit_length_weight(csv_path, output_path, config_path=None):
    """Fit on TRAINING CSV only; separate function never invoked by evaluation."""
    frame = pd.read_csv(csv_path)
    needed = ['measured_length_cm', 'measured_weight_g']
    if not set(needed) <= set(frame.columns):
        raise ValueError('Training CSV needs measured_length_cm and measured_weight_g')
    if 'split' in frame and not frame['split'].eq('train').all():
        raise ValueError('Refitting accepts only rows labelled train; never use validation/test rows')
    data = frame[needed].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if len(data) < 3 or not np.isfinite(data).all() or (data <= 0).any() or np.unique(data[:, 0]).size < 3:
        raise ValueError('Need at least three distinct, finite positive measured lengths and weights')
    b, intercept = np.polyfit(np.log(data[:, 0]), np.log(data[:, 1]), 1)
    if b <= 0:
        raise ValueError('Nonpositive length-weight exponent: inspect training measurements')
    config = load_config(config_path)
    config['length_weight'].update(a=float(np.exp(intercept)), b=float(b), valid_length_cm=[float(data[:, 0].min()), float(data[:, 0].max())],
                                  source=f'User training measurements: {Path(csv_path).name}; n={len(data)}; log-space least squares, no bias correction',
                                  source_url=None, status='FITTED_TRAINING_ONLY_NOT_VALIDATED')
    Path(output_path).write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config['length_weight']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refit length-weight parameters using TRAINING measurements only.')
    parser.add_argument('training_csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refit_length_weight(args.training_csv, args.output), indent=2))
