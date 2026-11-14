# Builds the Store selected by STORE_BACKEND (mongo or json) so scripts never hardcode a backend.
# MongoStore is imported lazily so the json backend works without a reachable database.
from __future__ import annotations

from pantry.config import Settings, get_settings
from pantry.store.base import Store


def make_store(backend: str | None = None, settings: Settings | None = None) -> Store:
    """Return a MongoStore or a file-backed JsonStore; backend overrides STORE_BACKEND when given."""
    settings = settings or get_settings()
    chosen = (backend or settings.store_backend).lower()
    if chosen == "mongo":
        from pantry.store.mongo_store import MongoStore

        return MongoStore(settings=settings, start_date=settings.demo_start_date)
    if chosen == "json":
        from pantry.store.json_store import JsonStore

        return JsonStore(start_date=settings.demo_start_date)
    raise ValueError(f"Unknown store backend {chosen!r}; expected mongo or json")
