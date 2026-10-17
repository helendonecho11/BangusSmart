import numpy as np
import pytest
import bangussmart_ml as core
from do_experiment import features, FEATURES


@pytest.mark.parametrize("temp,ph,do,expected", [
    (26, 7.5, 4, 0), (32, 8.3, 8, 0), (25, 7.9, 5, 1),
    (29, 7.9, 3, 1), (29, 7.9, 2.99, 2), (34.01, 7.9, 5, 2),
    (29, 6.99, 5, 2), (29, 8.81, 5, 2)])
def test_rule_status(temp, ph, do, expected):
    assert core.rule_status(temp, ph, do) == expected


def test_rule_arrays():
    np.testing.assert_array_equal(core.rule_status(29, 7.9, [5, 3.5, 2]), [0, 1, 2])


@pytest.mark.parametrize("weight,expected", [(0,4), (50,4), (50.1,3.5), (150,3.5),
                                          (150.1,3), (300,3), (300.1,2.5), (20000,2.5)])
def test_feed_rate(weight, expected):
    assert core.feed_rate_pct(weight) == expected


@pytest.mark.parametrize("history", [[6.1,6.3,6.4], [6.0] * 12])
def test_advise_smoke(history):
    result = core.advise(30, 8, 6.4, 15, history, 60, 29.5, 5000)
    assert result["status_rule"] == "Good"
    assert 0 <= result["recommended_feed_today_kg"] <= result["estimated_biomass_kg"] * .04 + .01
    assert np.isfinite(result["do_forecast_next_hour_mgl"])
    assert "Synthetic" in result["synthetic_data_warning"]


def test_features_do_not_use_future():
    sensor = core.make_sensor_data(days=10)
    changed = sensor.copy()
    changed.loc[100:, "do_mgl"] += 10
    a, b = features(sensor, 6), features(changed, 6)
    np.testing.assert_allclose(a.loc[a.time < 100, FEATURES], b.loc[b.time < 100, FEATURES])


def test_training_and_inference_features_match():
    sensor = core.make_sensor_data(days=10)
    training = features(sensor, 1).loc[100, FEATURES]
    online = features(sensor.iloc[88:101]).iloc[-1][FEATURES]
    np.testing.assert_allclose(training.to_numpy(dtype=float), online.to_numpy(dtype=float))


def test_api_validation_and_smoke():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    payload = dict(temp=30, ph=8, do=6.4, hour=15, do_history=[6.1,6.3,6.4],
                   day=60, mean_temp=29.5, fish_count=5000)
    assert client.post("/advise", json=payload).status_code == 200
    payload["do_history"] = [5]
    assert client.post("/advise", json=payload).status_code == 422
