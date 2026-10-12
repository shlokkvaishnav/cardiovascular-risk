"""Tests for the per-client rate limiter."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.rate_limit import Rule, RateLimitMiddleware  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_app(clock: FakeClock, **kwargs) -> TestClient:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/auth/login")
    def login():
        return {"ok": True}

    @app.post("/predict")
    def predict():
        return {"ok": True}

    @app.get("/other")
    def other():
        return {"ok": True}

    rules = [
        Rule("auth", ("/auth/login",), 2),
        Rule("predict", ("/predict",), 3),
    ]
    app.add_middleware(
        RateLimitMiddleware,
        rules=rules,
        default_limit=5,
        clock=clock,
        enabled=True,
        **kwargs,
    )
    # Same order as app.py: CORS is added after, so it is the outer layer.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    return TestClient(app)


def test_blocks_after_the_limit_and_sets_retry_after():
    clock = FakeClock()
    client = make_app(clock)
    assert client.post("/auth/login").status_code == 200
    assert client.post("/auth/login").status_code == 200
    blocked = client.post("/auth/login")
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert "Too many requests" in blocked.json()["detail"]


def test_window_slides_so_requests_work_again_after_60_seconds():
    clock = FakeClock()
    client = make_app(clock)
    for _ in range(2):
        client.post("/auth/login")
    assert client.post("/auth/login").status_code == 429
    clock.now += 61
    assert client.post("/auth/login").status_code == 200


def test_groups_count_separately():
    clock = FakeClock()
    client = make_app(clock)
    for _ in range(2):
        client.post("/auth/login")
    assert client.post("/auth/login").status_code == 429
    assert client.post("/predict").status_code == 200


def test_clients_count_separately_by_proxy_header():
    clock = FakeClock()
    client = make_app(clock)
    for _ in range(2):
        client.post("/auth/login", headers={"X-Real-IP": "1.1.1.1"})
    assert (
        client.post("/auth/login", headers={"X-Real-IP": "1.1.1.1"}).status_code == 429
    )
    assert (
        client.post("/auth/login", headers={"X-Real-IP": "2.2.2.2"}).status_code == 200
    )


def test_forwarded_for_uses_the_last_address_not_one_the_client_set():
    clock = FakeClock()
    client = make_app(clock)
    # A client cannot pick a fresh identity by changing the first entry.
    for i in range(2):
        client.post("/auth/login", headers={"X-Forwarded-For": f"9.9.9.{i}, 3.3.3.3"})
    blocked = client.post(
        "/auth/login", headers={"X-Forwarded-For": "9.9.9.99, 3.3.3.3"}
    )
    assert blocked.status_code == 429


def test_proxy_headers_are_ignored_when_not_trusted():
    clock = FakeClock()
    client = make_app(clock, trust_proxy_headers=False)
    for i in range(2):
        client.post("/auth/login", headers={"X-Real-IP": f"5.5.5.{i}"})
    assert (
        client.post("/auth/login", headers={"X-Real-IP": "5.5.5.99"}).status_code == 429
    )


def test_health_and_preflight_are_never_limited():
    clock = FakeClock()
    client = make_app(clock)
    for _ in range(20):
        assert client.get("/health").status_code == 200
    for _ in range(20):
        response = client.options(
            "/auth/login",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200


def test_limited_response_still_has_cors_headers():
    clock = FakeClock()
    client = make_app(clock)
    for _ in range(2):
        client.post("/auth/login", headers={"Origin": "https://example.com"})
    blocked = client.post("/auth/login", headers={"Origin": "https://example.com"})
    assert blocked.status_code == 429
    assert blocked.headers["access-control-allow-origin"] == "*"


def test_default_limit_applies_to_other_routes():
    clock = FakeClock()
    client = make_app(clock)
    assert all(client.get("/other").status_code == 200 for _ in range(5))
    assert client.get("/other").status_code == 429


def test_real_app_puts_the_limiter_inside_cors():
    from src.api.app import app

    names = [m.cls.__name__ for m in app.user_middleware]
    # user_middleware lists the outermost first.
    assert names.index("CORSMiddleware") < names.index("RateLimitMiddleware")


def test_env_switch_turns_the_limits_off(monkeypatch):
    app = FastAPI()

    @app.post("/auth/login")
    def login():
        return {"ok": True}

    app.add_middleware(
        RateLimitMiddleware, rules=[Rule("auth", ("/auth/login",), 1)], default_limit=1
    )
    client = TestClient(app)

    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    assert all(client.post("/auth/login").status_code == 200 for _ in range(5))

    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    statuses = [client.post("/auth/login").status_code for _ in range(2)]
    assert statuses == [200, 429]
