"""Read-only held-out evaluation. Never fits constants or tunes on evaluation images."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from media_estimation import length_to_weight, weight_to_age
from media_vision import MeasurementError, decode_image, load_config, measure_image


def error_summary(values):
    if not values:
        return {'n': 0, 'mean_signed_error': None, 'sample_sd_signed_error': None, 'mae': None, 'rmse': None, 'min_error': None, 'max_error': None}
    values = np.asarray(values, float)
    return {'n': len(values), 'mean_signed_error': float(values.mean()),
            'sample_sd_signed_error': float(values.std(ddof=1)) if len(values) > 1 else None,
            'mae': float(np.abs(values).mean()), 'rmse': float(np.sqrt(np.mean(values**2))),
            'min_error': float(values.min()), 'max_error': float(values.max())}


def evaluate(labels_csv, image_folder, config_path=None, mean_temp_c=29, synthetic=False):
    config = load_config(config_path)
    labels = pd.read_csv(labels_csv)
    required = {'image_file', 'known_age_days', 'measured_length_cm', 'measured_weight_g', 'reference_object'}
    if not required <= set(labels):
        raise ValueError('Labels missing required columns: '+', '.join(sorted(required - set(labels))))
    if 'split' in labels and not labels['split'].eq('test').all():
        raise ValueError('Evaluation requires test-only labels when a split column is present. No tuning or training occurs here.')
    report = {'status': 'no_real_labelled_images' if labels.empty else 'evaluated', 'labelled_rows': len(labels),
              'data_kind': 'synthetic' if synthetic else 'user_supplied_not_independently_verified',
              'configuration_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              'configuration': config, 'mean_temp_c_default': mean_temp_c,
              'note': 'No real labelled images supplied; no accuracy scores.' if labels.empty else 'Held-out evaluation only. No parameter fitting or threshold tuning. Errors are prediction minus independent label.',
              'length_error_cm': error_summary([]), 'weight_error_g': error_summary([]), 'age_error_days': error_summary([]),
              'quality_rejection_count': 0, 'quality_rejection_fraction': None, 'unscorable_count': 0,
              'age_range_coverage': None, 'rows': []}
    if labels.empty:
        return report
    if 'notes' in labels and labels['notes'].fillna('').astype(str).str.contains('synthetic', case=False).any():
        report['data_kind'] = 'synthetic_or_mixed'
        report['note'] += ' Contains synthetic labels: do not claim real-fish accuracy.'
    image_folder = Path(image_folder).resolve()
    errors = {'length': [], 'weight': [], 'age': []}
    coverage = []
    for index, label in labels.iterrows():
        row = {'row': int(index) + 2, 'image_file': str(label['image_file'])}
        try:
            path = (image_folder / str(label['image_file'])).resolve()
            if not path.is_relative_to(image_folder) or not path.is_file():
                raise ValueError('Image path must be an existing file inside the image folder')
            # Require an explicit, measured physical scale; never default to 1cm.
            match = re.fullmatch(r'checkerboard:(\d+(?:\.\d+)?)cm', str(label['reference_object']).strip())
            if not match or float(match.group(1)) <= 0:
                raise ValueError('reference_object must be checkerboard:<measured square size>cm')
            ground = {}
            for key in ('measured_length_cm', 'measured_weight_g', 'known_age_days'):
                if pd.isna(label[key]):
                    ground[key] = None
                    continue
                value = float(label[key])
                if not np.isfinite(value) or value < 0 or (key != 'known_age_days' and value == 0):
                    raise ValueError(f'Invalid independent label: {key}')
                ground[key] = value
            temperature = float(label['mean_temp_c']) if 'mean_temp_c' in label and pd.notna(label['mean_temp_c']) else mean_temp_c
            measurement, _ = measure_image(decode_image(path.read_bytes(), config), float(match.group(1)), config)
            if len(measurement['fish']) != 1:
                raise ValueError('Evaluation labels require exactly one measured fish per image; identity matching is not assumed')
            length = measurement['fish'][0]['length_cm']
            weight = length_to_weight(length, config)['point_g']
            # Known age label is NEVER passed as an input to prediction.
            age = weight_to_age(weight, temperature, config)
            row.update(status='measured', length_cm=length, weight_g=weight, age=age, ground_truth=ground)
            for metric, prediction, truth in [('length', length, ground['measured_length_cm']), ('weight', weight, ground['measured_weight_g']), ('age', age['point_days'], ground['known_age_days'])]:
                if prediction is not None and truth is not None:
                    errors[metric].append(prediction - truth)
            if age['range_days'] is not None and ground['known_age_days'] is not None:
                coverage.append(age['range_days'][0] <= ground['known_age_days'] <= age['range_days'][1])
        except MeasurementError as exc:
            row.update(status='quality_rejected', code=exc.code, message=str(exc))
            report['quality_rejection_count'] += 1
        except (ValueError, OSError) as exc:
            row.update(status='unscorable', message=str(exc))
            report['unscorable_count'] += 1
        report['rows'].append(row)
    report.update(length_error_cm=error_summary(errors['length']), weight_error_g=error_summary(errors['weight']), age_error_days=error_summary(errors['age']),
                  quality_rejection_fraction=report['quality_rejection_count'] / len(labels),
                  age_range_coverage=float(np.mean(coverage)) if coverage else None)
    if not any(row['status'] == 'measured' for row in report['rows']):
        report['status'] = 'no_measurable_labelled_images'
        report['note'] += ' No measurable labelled images; accuracy scores unavailable.'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels_csv', type=Path)
    parser.add_argument('image_folder', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--mean-temp-c', type=float, default=29)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.labels_csv, args.image_folder, args.config, args.mean_temp_c, args.synthetic)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded)
