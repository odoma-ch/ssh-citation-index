import pytest

from citation_index.config import settings


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Keep tests independent of a local .env: auth is off unless a test sets it."""
    monkeypatch.setattr(settings, "api_key", None)
