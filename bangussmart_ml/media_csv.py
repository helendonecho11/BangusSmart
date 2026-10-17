"""Validate complete existing sensor/growth schemas. No silent partial-row import."""
import io

import numpy as np
import pandas as pd

from media_estimation import AnalysisOptions, recommend_feed, resolve_age
from media_vision import MeasurementError
from bangussmart_ml import rule_status, STATUSES


def analyze_csv(content, options, config=None):
    import csv
    reader = csv.reader(io.StringIO(content.lstrip('\ufeff')), strict=True)
    try:
        header = next(reader, [])
    except csv.Error as exc:
        raise MeasurementError('invalid_columns', 'Malformed CSV header') from exc
    if not header or len(header) != len(set(header)):
        raise MeasurementError('invalid_columns', 'CSV requires a unique header row.')
    try:
        for line, values in enumerate(reader, 2):
            if values and len(values) != len(header):
                raise MeasurementError('invalid_csv', f'Row {line}: expected {len(header)} values, received {len(values)}')
    except csv.Error as exc:
        raise MeasurementError('invalid_csv', f'CSV quoting error: {exc}') from exc
    try:
        frame = pd.read_csv(io.StringIO(content.lstrip('\ufeff')))
    except (ValueError, pd.errors.ParserError) as exc:
        raise MeasurementError('invalid_csv', str(exc)) from exc
    if frame.empty or len(frame) > 10000:
        raise MeasurementError('csv_row_limit', 'CSV needs 1–10,000 data rows.')
    sensor_columns = {'day', 'hour', 'temp_c', 'ph', 'do_mgl'}
    growth_columns = {'batch', 'day', 'mean_temp_c', 'avg_weight_g'}
    sensor, growth = sensor_columns <= set(frame), growth_columns <= set(frame)
    if not sensor and not growth:
        raise MeasurementError('invalid_columns', 'Supply sensor columns day,hour,temp_c,ph,do_mgl or growth columns batch,day,mean_temp_c,avg_weight_g.',
                               {'missing_sensor': sorted(sensor_columns - set(frame)), 'missing_growth': sorted(growth_columns - set(frame))})
    ranges = {'day': (0, 3650), 'hour': (0, 23), 'temp_c': (-5, 50), 'ph': (0, 14), 'do_mgl': (0, 50),
              'mean_temp_c': (0, 45), 'avg_weight_g': (.000001, 10000)}
    numeric_columns = set(ranges) & set(frame)
    problems = []
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors='coerce')
        lo, hi = ranges[column]
        valid = np.isfinite(values) & values.between(lo, hi)
        if column in ('day', 'hour'):
            valid &= values.eq(np.floor(values))
        for index in frame.index[~valid]:
            problems.append({'row': int(index) + 2, 'column': column, 'message': f'Expected finite {"integer " if column in ("day", "hour") else ""}value in [{lo}, {hi}]'})
        frame[column] = values
    if growth:
        for index in frame.index[frame['batch'].isna() | frame['batch'].astype(str).str.strip().eq('')]:
            problems.append({'row': int(index) + 2, 'column': 'batch', 'message': 'Batch is required'})
    if problems:
        raise MeasurementError('invalid_rows', 'CSV contains invalid data. Correct all reported rows and upload again.', {'error_count': len(problems), 'errors': problems[:100]})
    # Avoid silently mixing multiple ponds/batches into a common temperature/weight.
    if 'batch' in frame and frame['batch'].nunique() > 1:
        groups = [(str(batch), group) for batch, group in frame.groupby('batch', sort=False)]
    else:
        groups = [(str(frame.iloc[-1]['batch']) if 'batch' in frame else 'Unassigned', frame)]
    batches = []
    for batch, group in groups:
        # Latest is explicit chronological day/hour, not arbitrary file order.
        ordered = group.sort_values(['day', 'hour'] if sensor else ['day'], kind='stable')
        last = ordered.iloc[-1]
        updates = options.model_dump()
        updates['known_age_days'] = int(last['day'])
        # Preserve supplied stocking date for the same capture_date; inconsistent records reject.
        if sensor:
            updates.update(mean_temp_c=float(group['temp_c'].mean()), temp_c=float(last['temp_c']), ph=float(last['ph']), do_mgl=float(last['do_mgl']), hour=int(last['hour']))
        elif growth:
            updates['mean_temp_c'] = float(last['mean_temp_c'])
        if options.known_age_days is not None and options.known_age_days != updates['known_age_days']:
            raise MeasurementError('age_conflict', f'Batch {batch}: CSV day conflicts with supplied known batch age.')
        try:
            current = AnalysisOptions(**updates)
        except ValueError as exc:
            raise MeasurementError('record_conflict', f'Batch {batch}: {exc}') from exc
        weight = float(last['avg_weight_g']) if growth else None
        water_status = STATUSES[int(rule_status(current.temp_c, current.ph, current.do_mgl))] if all(v is not None for v in (current.temp_c, current.ph, current.do_mgl)) else None
        batches.append({'batch': batch, 'rows': len(group), 'options_for_media': current.model_dump(mode='json'),
                        'water_status': water_status, 'mean_temp_c': current.mean_temp_c,
                        'weight': {'average_g': weight, 'source': 'measured_csv' if growth else 'unavailable'},
                        'size': {'average_length_cm': None}, 'age': resolve_age(weight, current, config),
                        'feed': recommend_feed(weight, current), 'confidence': 'recorded_unverified',
                        'warnings': ['CSV measurements are user records, not automatically verified.']})
    return {'media_type': 'csv', 'row_count': len(frame), 'schemas': [key for key, present in [('sensor', sensor), ('growth', growth)] if present],
            'batches': batches, 'confidence': 'recorded_unverified', 'annotated_output': [],
            'warnings': ['Latest day/hour is used per batch. Mean temperature for sensor CSV is the arithmetic mean of supplied readings (sampling frequency is not verified).',
                         'Age is days since stocking, not biological age. Decision support assists farm personnel and does not replace their judgment.']}
