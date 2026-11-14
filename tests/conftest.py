# Shared pytest fixtures: the seed fixture directory and stores for both backends.
# MongoStore fixtures are skipped unless RUN_INTEGRATION=1 and always use a database whose name ends in _test.
from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from pantry.store.base import Store
from pantry.store.json_store import JsonStore

START: date = date(2026, 11, 14)
FIXTURE_SEED_DIR: Path = Path(__file__).parent / "fixtures" / "seed_data"


def integration_enabled() -> bool:
    return os.getenv("RUN_INTEGRATION", "").strip() == "1"


def _mongo_test_store() -> Iterator[Store]:
    if not integration_enabled():
        pytest.skip("MongoDB integration tests need RUN_INTEGRATION=1")
    from pantry.config import load_settings
    from pantry.store.mongo_store import MongoStore

    settings = load_settings()
    if not settings.mongodb_uri:
        pytest.skip("MONGODB_URI is not set")
    base = settings.mongodb_db or "pantry"
    db_name = base if base.endswith("_test") else f"{base}_test"
    assert db_name.endswith("_test")
    # No Atlas search index in tests: it is slow to build and uses a limited per-cluster slot,
    # so search_traces exercises the local fallback (atlas_unavailable=True).
    store = MongoStore(settings=settings, db_name=db_name, start_date=START, create_vector_index=False)
    store.drop_database()
    store.ensure_indexes()
    try:
        yield store
    finally:
        store.drop_database()
        store.close()


@pytest.fixture
def seed_dir() -> Path:
    return FIXTURE_SEED_DIR


@pytest.fixture
def mongo_store() -> Iterator[Store]:
    yield from _mongo_test_store()


@pytest.fixture(params=["json", "mongo"])
def any_store(request: pytest.FixtureRequest) -> Iterator[Store]:
    """Run a test against JsonStore.memory() and, with RUN_INTEGRATION=1, against MongoStore."""
    if request.param == "json":
        yield JsonStore.memory(start_date=START)
    else:
        yield from _mongo_test_store()
