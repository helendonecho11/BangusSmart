"""Reproduce SYNTHETIC pipeline checks. These are not real-fish accuracy scores."""
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_media import error_summary
from media_estimation import AnalysisOptions, length_to_weight, weight_to_age, recommend_feed
from media_video import analyze_video
from media_vision import MeasurementError, measure_image
from synthetic_media import make_scene


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases, errors = [], []
    for index, (length, angle, scale) in enumerate([(10, 0, 24), (15, 0, 24), (18, 15, 24), (15, -10, 30)]):
        image, _ = make_scene((length,), scale, angle_degrees=angle)
        measured, annotated = measure_image(image, 1)
        value = measured['fish'][0]['length_cm']
        errors.append(value - length)
        cv2.imwrite(str(output / f'annotated_synthetic_{index}.png'), annotated)
        cases.append({'true_length_cm': length, 'angle_degrees': angle, 'true_pixels_per_cm': scale,
                      'measured_length_cm': value, 'measured_pixels_per_cm': measured['reference']['pixels_per_cm'],
                      'length_error_cm': value - length})
    rejections = []
    for name in ('missing_reference', 'blurry_fish'):
        image, _ = make_scene(reference=name != 'missing_reference')
        if name == 'blurry_fish':
            image[260:] = cv2.GaussianBlur(image[260:], (71, 71), 0)
        try:
            measure_image(image, 1)
            rejections.append({'case': name, 'rejected': False})
        except MeasurementError as exc:
            rejections.append({'case': name, 'rejected': True, 'code': exc.code})
    video_path = output / 'synthetic_tracking.avi'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (1000, 650))
    if not writer.isOpened():
        raise RuntimeError('Could not create synthetic MJPEG video')
    for index in range(15):
        writer.write(make_scene(reference=index != 5)[0])
    writer.release()
    video = analyze_video(video_path, AnalysisOptions(square_size_cm=1), output / 'video_frames', 'video_frames')
    feed = recommend_feed(110, AnalysisOptions(square_size_cm=1, fish_count=5000, temp_c=29, ph=8, do_mgl=6, feed_rate_pct=10))
    report = {'synthetic': True, 'note': 'Drawn silhouettes and a synthetic growth model only. No real-fish accuracy claim.',
              'geometry_cases': cases, 'length_error_cm': error_summary(errors), 'rejection_checks': rejections,
              'video_check': {'sampled': video['video']['sampled_count'], 'accepted': video['video']['accepted_count'],
                              'rejected_fraction': video['video']['rejected_fraction'], 'distinct_tracks': video['estimated_distinct_fish_seen'],
                              'median_length_cm': video['fish'][0]['length_cm'], 'track_samples': video['fish'][0]['samples']},
              'weight_at_15cm': length_to_weight(15), 'synthetic_growth_age_at_110g_29c': weight_to_age(110, 29),
              'feed_cap_check': feed}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    result = run(Path('data') / 'synthetic_media_checks')
    print(json.dumps({key: result[key] for key in ('synthetic', 'length_error_cm', 'video_check', 'rejection_checks')}, indent=2))
