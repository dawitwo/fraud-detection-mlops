# tests/test_api.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    # Context manager → triggers startup event → loads MLflow model
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_input(client):
    payload = {"features": [0.0] * 30}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert "is_fraud" in body
    assert "threshold" in body


def test_predict_invalid_input(client):
    payload = {"features": [0.0] * 29}  # wrong length → 422
    response = client.post("/predict", json=payload)
    assert response.status_code == 422