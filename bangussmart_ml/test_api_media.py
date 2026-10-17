"""End-to-end media tests on generated silhouettes, not real-fish evaluation."""
import json
from pathlib import Path

import cv2
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api_media
from media_estimation import AnalysisOptions
from synthetic_media import make_scene


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_media, 'OUTPUT_ROOT', tmp_path / 'results')
    return TestClient(api_media.app)


def image_bytes(reference=True):
    image, _ = make_scene((15,), reference=reference)
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return encoded.tobytes()


def settings(**kwargs):
    return AnalysisOptions(square_size_cm=1, **kwargs).model_dump_json()


def test_image_api_geometry_estimation_feed_and_artifact(client):
    response = client.post('/analyze/image', params={'options': settings(fish_count=5000, temp_c=29, ph=8, do_mgl=6)}, content=image_bytes())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['size']['average_length_cm'] == pytest.approx(15, abs=.3)
    assert result['weight']['average_g'] > 0
    primary = result['age']['primary']
    assert primary['range_days'][0] <= primary['point_days'] <= primary['range_days'][1]
    assert primary['confidence'] == 'low'
    assert result['feed']['available']
    assert result['feed']['daily_feed_kg'] <= result['feed']['biomass_kg'] * .04
    assert client.get(result['annotated_output'][0]['path']).status_code == 200
    assert client.get(result['result_path']).json() == result
    assert result['real_fish_accuracy'] == 'not_evaluated_no_real_labels'


def test_image_api_errors_and_known_age_precedence(client):
    missing = client.post('/analyze/image', params={'options': settings()}, content=image_bytes(False))
    assert missing.status_code == 422
    assert missing.json()['detail']['code'] == 'cannot_measure_scale'
    assert missing.json()['detail']['age']['point_days'] is None
    assert client.post('/analyze/image', params={'options': settings()}, content=b'bad').status_code == 422
    assert client.post('/analyze/image', params={'options': '{"square_size_cm":0}'}, content=image_bytes()).status_code == 422
    response = client.post('/analyze/image', params={'options': settings(known_age_days=100)}, content=image_bytes())
    result = response.json()
    assert result['age']['primary']['point_days'] == 100
    assert result['age']['disagreement_over_20_percent']
    assert result['age']['media_or_weight_cross_check']['point_days'] != 100
    assert not result['feed']['available']  # no stock count or water guessed


def test_csv_api_success_error_and_batch_options(client):
    response = client.post('/analyze/csv', content=b'day,hour,temp_c,ph,do_mgl\n60,15,29,8,6\n')
    assert response.status_code == 200
    result = response.json()
    assert result['batches'][0]['water_status'] == 'Good'
    assert result['batches'][0]['options_for_media']['mean_temp_c'] == 29
    invalid = client.post('/analyze/csv', content=b'day,hour,temp_c,ph,do_mgl\n60,15,29,15,6\n')
    assert invalid.status_code == 422
    assert invalid.json()['detail']['quality']['errors'][0]['column'] == 'ph'


def test_video_api_tracks_and_annotations(client, tmp_path):
    path = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (1000, 650))
    assert writer.isOpened()
    for _ in range(10):
        image, _ = make_scene((15,))
        writer.write(image)
    writer.release()
    response = client.post('/analyze/video', params={'filename': 'synthetic.avi', 'options': settings(fish_count=50, temp_c=29, ph=8, do_mgl=6)}, content=path.read_bytes())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['estimated_distinct_fish_seen'] == 1
    assert result['feed']['fish_count'] == 50
    assert result['feed']['count_source'] == 'user_supplied'
    assert client.get(result['annotated_output'][0]['path']).status_code == 200
    invalid = client.post('/analyze/video', params={'filename': 'bad.mp4', 'options': settings()}, content=b'bad')
    assert invalid.status_code == 422


def test_no_labels_evaluation_and_no_label_leakage(tmp_path):
    from evaluate_media import evaluate
    template = Path(__file__).parent / 'data' / 'template_image_labels.csv'
    empty = evaluate(template, tmp_path)
    assert empty['status'] == 'no_real_labelled_images'
    assert empty['length_error_cm']['n'] == 0
    assert empty['length_error_cm']['mae'] is None
    (tmp_path / 'synthetic.png').write_bytes(image_bytes())
    labels = pd.DataFrame([{'image_file':'synthetic.png', 'known_age_days':1000, 'measured_length_cm':15,
                            'measured_weight_g':30, 'reference_object':'checkerboard:1.0cm', 'notes':'SYNTHETIC', 'split':'test'}])
    path = tmp_path / 'labels.csv'
    labels.to_csv(path, index=False)
    report = evaluate(path, tmp_path, synthetic=True)
    assert report['length_error_cm']['mae'] < .3
    assert report['age_error_days']['mean_signed_error'] < -800  # does not echo the supplied ground-truth age
    assert report['quality_rejection_fraction'] == 0
    assert 'synthetic' in report['data_kind']


def test_upload_page_reference_and_artifact_access(client):
    assert client.get('/').status_code == 200
    assert client.get('/media').status_code == 200
    assert b'8 x 6 squares' in client.get('/media/reference.svg').content
    assert client.get('/static/media.js').status_code == 200
    assert client.get('/media-results/not-an-id/result.json').status_code == 404


def test_optional_model_missing_uses_explicit_fallback():
    from media_vision import load_config, measure_image
    config = load_config()
    config['detector'].update(backend='local_yolo', local_yolo_weights='does-not-exist.pt')
    result, _ = measure_image(make_scene()[0], 1, config)
    assert result['detector_used'] == 'classical_contours'
    assert any('No weights downloaded' in warning for warning in result['warnings'])
