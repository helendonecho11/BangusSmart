"""Synthetic geometry checks only. No claim of real fish accuracy."""
import cv2
import numpy as np
import pytest

from media_vision import MeasurementError, load_config, measure_image, detect_reference
from synthetic_media import make_scene


@pytest.mark.parametrize('length,angle,scale', [(10, 0, 24), (15, 0, 24), (18, 15, 24), (15, -10, 30)])
def test_synthetic_scale_and_fork_length(length, angle, scale):
    image, truth = make_scene((length,), scale, angle_degrees=angle)
    result, annotation = measure_image(image, truth['square_size_cm'])
    assert result['reference']['pixels_per_cm'] == pytest.approx(scale, abs=.15)
    assert result['detected_fish_count'] == 1
    assert result['fish'][0]['length_cm'] == pytest.approx(length, abs=.3)
    assert annotation.shape == image.shape


def test_reference_is_required_and_never_guessed():
    image, _ = make_scene(reference=False)
    with pytest.raises(MeasurementError, match='Cannot measure scale'):
        measure_image(image, 1)


def test_multiple_separated_synthetic_fish():
    image, _ = make_scene((10, 15))
    result, _ = measure_image(image, 1)
    assert result['detected_fish_count'] == 2
    assert sorted(f['length_cm'] for f in result['fish']) == pytest.approx([10, 15], abs=.3)


def test_no_fish_and_blurry_frame_rejected():
    image, _ = make_scene()
    image[260:] = 245
    with pytest.raises(MeasurementError):
        measure_image(image, 1)
    image, _ = make_scene()
    image[260:] = cv2.GaussianBlur(image[260:], (71, 71), 0)
    with pytest.raises(MeasurementError) as error:
        measure_image(image, 1)
    assert error.value.code == 'blurry_image'


def test_distorted_reference_is_rejected():
    image, _ = make_scene()
    distorted = cv2.resize(image, None, fx=1.5, fy=1)
    with pytest.raises(MeasurementError) as error:
        measure_image(distorted, 1)
    assert error.value.code == 'perspective_reference'


def test_published_fork_relation_and_no_extrapolation():
    from media_estimation import length_to_weight
    result = length_to_weight(15)
    assert result['point_g'] == pytest.approx(10 ** (-5.6083 + 3.2598 * np.log10(150)))
    assert length_to_weight(25)['point_g'] is None
    with pytest.raises(ValueError):
        length_to_weight(float('nan'))


def test_age_range_records_precedence_and_disagreement():
    from media_estimation import weight_to_age, resolve_age, AnalysisOptions
    age = weight_to_age(110, 29)
    assert age['range_days'][0] <= age['point_days'] <= age['range_days'][1]
    assert age['range_days'][1] - age['range_days'][0] >= 28
    assert age['confidence'] == 'low'
    options = AnalysisOptions(square_size_cm=1, known_age_days=130)
    result = resolve_age(110, options)
    assert result['primary']['point_days'] == 130
    assert result['primary']['range_days'] == [130, 130]
    assert result['disagreement_over_20_percent']
    assert weight_to_age(10000, 29)['point_days'] is None
    assert weight_to_age(110, 20)['point_days'] is None


def test_shared_feed_cap_and_no_good_water_default():
    from media_estimation import AnalysisOptions, recommend_feed
    options = AnalysisOptions(square_size_cm=1, fish_count=5000, temp_c=29, ph=8, do_mgl=6, feed_rate_pct=10)
    feed = recommend_feed(110, options)
    assert feed['daily_feed_kg'] == 22
    assert feed['rate_was_capped']
    options.do_mgl = 2
    assert recommend_feed(110, options)['daily_feed_kg'] == 0
    options.do_mgl = None
    assert not recommend_feed(110, options)['available']


def test_refit_is_separate_and_rejects_test_rows(tmp_path):
    from media_estimation import refit_length_weight
    import pandas as pd
    lengths = np.array([10., 12., 15., 18.])
    frame = pd.DataFrame({'measured_length_cm': lengths, 'measured_weight_g': .01 * lengths**3, 'split': 'train'})
    path = tmp_path / 'train.csv'
    frame.to_csv(path, index=False)
    parameters = refit_length_weight(path, tmp_path / 'new_config.json')
    assert parameters['a'] == pytest.approx(.01)
    assert parameters['b'] == pytest.approx(3)
    frame['split'] = 'test'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='never use'):
        refit_length_weight(path, tmp_path / 'bad.json')


def test_sensor_and_growth_csv_updates_and_validation():
    from media_csv import analyze_csv
    from media_estimation import AnalysisOptions
    options = AnalysisOptions(square_size_cm=1, fish_count=5000)
    sensor = analyze_csv('day,hour,temp_c,ph,do_mgl\n60,15,30,8,6\n60,14,28,8,3\n', options)
    batch = sensor['batches'][0]
    assert batch['water_status'] == 'Good'
    assert batch['mean_temp_c'] == 29
    assert batch['options_for_media']['hour'] == 15
    assert batch['options_for_media']['do_mgl'] == 6
    assert batch['age']['primary']['point_days'] == 60
    assert not batch['feed']['available']
    good = options.model_copy(update={'temp_c':29, 'ph':8, 'do_mgl':6})
    growth = analyze_csv('batch,day,mean_temp_c,avg_weight_g\nA,60,29,110\n', good)
    assert growth['batches'][0]['feed']['daily_feed_kg'] == 19.25
    with pytest.raises(MeasurementError) as error:
        analyze_csv('day,hour,temp_c,ph,do_mgl\n60,24,30,8,6\n', options)
    assert error.value.code == 'invalid_rows'
    with pytest.raises(MeasurementError):
        analyze_csv('day,temp_c\n60,30\n', options)
    with pytest.raises(MeasurementError):
        analyze_csv('day,hour,temp_c,ph,do_mgl\n60,15,30,8,6,extra\n', options)


def test_synthetic_video_tracking_and_quality(tmp_path):
    from media_video import analyze_video
    from media_estimation import AnalysisOptions
    path = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (1000, 650))
    assert writer.isOpened()
    for index in range(15):
        image, _ = make_scene((15,), reference=index != 5)
        writer.write(image)
    writer.release()
    result = analyze_video(path, AnalysisOptions(square_size_cm=1, temp_c=29, ph=8, do_mgl=6), tmp_path / 'frames', '/frames')
    assert result['video']['sampled_count'] == 3
    assert result['video']['accepted_count'] == 2
    assert result['estimated_distinct_fish_seen'] == 1
    assert result['fish'][0]['samples'] == 2
    assert result['fish'][0]['length_cm'] == pytest.approx(15, abs=.3)
    assert result['feed']['count_source'] == 'video_distinct_track_estimate'
    assert result['video']['frames'][1]['code'] == 'cannot_measure_scale'
