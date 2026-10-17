import base64
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import dashboard
from api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    return TestClient(app)


def test_dates_and_unknown_photo_age(client):
    payload = {"observation_date": "2026-10-17", "stocking_date": "2026-08-18", "hatch_date": "2026-07-01"}
    result = client.post('/api/age', json=payload).json()
    assert result['culture_days'] == 60
    assert result['age_days'] == 108
    assert client.post('/api/age', json={"observation_date": "2026-10-17"}).json()['method'] == 'unavailable'
    payload['stocking_date'] = '2026-10-18'
    assert client.post('/api/age', json=payload).status_code == 422


def test_weight_estimate_is_explicit_and_bounded(client):
    payload = {"observation_date": "2026-10-17", "avg_weight_g": 110}
    result = client.post('/api/age', json=payload).json()
    assert result['method'] == 'synthetic_weight_estimate'
    assert 0 <= result['culture_days'] <= 147
    assert result['age_days'] is None
    payload['avg_weight_g'] = 9999
    assert client.post('/api/age', json=payload).json()['method'] == 'unavailable'


def test_csv_partial_rejection_and_persistence(client):
    result = client.post('/api/uploads', json={"name": "fish.csv", "observation_date": "2026-10-17", "content": "batch,day,avg_weight_g,temp_c,ph,do_mgl\nA,60,110,29,8,6\nB,60,110,29,99,6\n"})
    assert result.status_code == 200
    data = result.json()
    assert len(data['rows']) == 1 and data['rejected_count'] == 1
    assert data['rows'][0]['culture_days'] == 60
    assert data['rows'][0]['feed_kg'] == 19.25
    assert client.get('/api/uploads').json()[0]['id'] == data['id']
    assert client.get(f"/api/uploads/{data['id']}").json()['rows'] == data['rows']


def test_photo_validation_and_saved_record(client):
    image = io.BytesIO()
    Image.new('RGB', (20, 20)).save(image, format='PNG')
    body = {"name": "fish.png", "image": base64.b64encode(image.getvalue()).decode(), "observation": {"observation_date": "2026-10-17", "stocking_date": "2026-08-18"}}
    response = client.post('/api/photos', json=body)
    assert response.status_code == 200
    result = response.json()
    assert result['result']['culture_days'] == 60
    assert client.get(result['url']).content == image.getvalue()
    assert len(client.get('/api/photos').json()) == 1
    body['image'] = base64.b64encode(b'not an image').decode()
    assert client.post('/api/photos', json=body).status_code == 422


def test_dashboard_and_templates(client):
    assert client.get('/').status_code == 200
    assert client.get('/static/app.js').status_code == 200
    assert 'stocking_date' in client.get('/api/template/growth').text


@pytest.mark.parametrize("oxygen,expected", [(6, 19.25), (3.5, 9.625), (2, 0)])
def test_feeding_plan_uses_measured_biomass_and_water(client, oxygen, expected):
    payload = {"observation_date": "2026-10-17", "avg_weight_g": 110, "fish_count": 5000,
               "temp_c": 29, "ph": 8, "do_mgl": oxygen, "meals_per_day": 2}
    feed = client.post('/api/age', json=payload).json()['feeding']
    assert feed['daily_feed_kg'] == expected
    assert feed['biomass_kg'] == 550
    assert feed['feed_per_meal_kg'] == round(expected / 2, 3)
    payload['feed_rate_pct'] = 2
    if oxygen == 6:
        assert client.post('/api/age', json=payload).json()['feeding']['daily_feed_kg'] == 11


def test_no_feed_or_vision_guess_from_dates_alone(client):
    result = client.post('/api/age', json={"observation_date": "2026-10-17", "stocking_date": "2026-08-18"}).json()
    assert not result['feeding']['available']
    assert 'measured average weight' in result['feeding']['missing']
    assert result['vision']['age_from_media'] is None
    assert result['vision']['weight_from_media'] is None


def test_video_reads_frames_and_persists_analysis(client, tmp_path):
    import cv2
    import numpy as np
    import json
    path = tmp_path / 'sample.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 48))
    assert writer.isOpened()
    for value in range(10):
        writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
    writer.release()
    observation = {"observation_date": "2026-10-17", "avg_weight_g": 110, "fish_count": 5000,
                   "temp_c": 29, "ph": 8, "do_mgl": 6}
    response = client.post('/api/videos', params={"name": "sample.avi", "observation": json.dumps(observation)}, content=path.read_bytes(), headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['media_analysis']['width'] == 64
    assert len(data['media_analysis']['sampled_frames']) == 5
    assert data['result']['feeding']['daily_feed_kg'] == 19.25
    assert data['result']['vision']['age_from_media'] is None
    assert client.get(data['media_analysis']['sampled_frames'][0]['url']).status_code == 200
    assert client.get(data['url']).content == path.read_bytes()
    assert len(client.get('/api/videos').json()) == 1
    bad = client.post('/api/videos', params={"name": "bad.mp4", "observation": json.dumps(observation)}, content=b'not a video')
    assert bad.status_code == 422
    assert len(list((tmp_path / 'data' / 'videos').glob('*.mp4'))) == 0
