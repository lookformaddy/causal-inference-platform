"""FastAPI service tests — auto-skip when FastAPI is not installed."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from causal_inference.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_recovers_effect():
    response = client.post(
        "/analyze", json={"n": 2000, "true_effect": 0.8, "seed": 42}
    )
    assert response.status_code == 200
    body = response.json()
    assert abs(body["matched_ate"] - 0.8) < 0.25
    assert abs(body["ipw_ate"] - 0.8) < 0.25
    assert body["balanced"] is True
    assert body["n_matched_pairs"] > 500
