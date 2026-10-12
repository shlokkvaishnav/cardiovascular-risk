"""Shared test settings."""

import pytest


@pytest.fixture(autouse=True)
def _disable_rate_limits(monkeypatch):
    # Tests create many accounts and requests from one client address.
    # test_rate_limit.py turns its own limiter on explicitly.
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
