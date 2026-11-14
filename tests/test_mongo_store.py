# MongoStore tests: offline unit tests for conversion and search merging, plus integration tests against a
# real database that run only with RUN_INTEGRATION=1 and always use a database name ending in _test.
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from pantry.config import Settings
from pantry.llm import FakeLLM
from pantry.models import (
    BusEvent,
    Command,
    Ingredient,
    InventoryEvent,
    RetrievedTrace,
    Run,
    ShoppingListItem,
    Trace,
)
from pantry.store.base import InvariantViolation, Store
from pantry.store.mongo_store import (
    MongoStore,
    as_stored,
    atlas_score_to_cosine,
    from_doc,
    merge_search_results,
    rank_by_cosine,
    to_doc,
    vector_index_definition,
)
from tests.conftest import START


def offline_settings() -> Settings:
    return Settings(
        llm_base_url="", llm_api_key="", llm_model="", mongodb_uri="", mongodb_db="pantry_unit_test",
        linkup_api_key="", tavily_api_key="", embed_provider="openai", embed_model="text-embedding-3-small",
        mlflow_tracking_uri="", databricks_host="", databricks_token="", cache_first=False,
        store_backend="mongo", tick_seconds=2.0, demo_start_date=START,
    )


def hit(trace_id: str, similarity: float) -> RetrievedTrace:
    return RetrievedTrace(trace_id=trace_id, similarity=similarity, day_index=0, dish_name="", is_seed=False)


# ---------------------------------------------------------------------------
# Offline unit tests
# ---------------------------------------------------------------------------


def test_to_doc_stores_dates_as_iso_and_round_trips() -> None:
    item = Ingredient(item_name="milk", quantity=1000, unit="ml", expiry_date=date(2026, 11, 17))
    doc = to_doc(item)
    assert doc["expiry_date"] == "2026-11-17"
    assert isinstance(doc["last_updated"], datetime)
    restored = from_doc(Ingredient, {**doc, "_id": "ignored"})
    assert restored == as_stored(item)
    assert restored.expiry_date == item.expiry_date
    assert restored.last_updated.microsecond % 1000 == 0  # BSON keeps milliseconds only
    assert abs((restored.last_updated - item.last_updated).total_seconds()) < 0.001


def test_vector_index_definition_matches_build_prompt() -> None:
    definition = vector_index_definition(1536)
    assert definition["fields"][0] == {
        "type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"
    }
    assert {f["path"] for f in definition["fields"][1:]} == {"final_outcome", "trigger"}


def test_atlas_score_maps_back_to_cosine() -> None:
    assert atlas_score_to_cosine(1.0) == 1.0
    assert atlas_score_to_cosine(0.5) == 0.0
    assert atlas_score_to_cosine(0.0) == -1.0


def test_rank_by_cosine_orders_and_skips_wrong_dimensions() -> None:
    docs = [
        {"trace_id": "a", "embedding": [1.0, 0.0], "plan": {"dish_name": "pancakes"}, "day_index": 1},
        {"trace_id": "b", "embedding": [0.6, 0.8]},
        {"trace_id": "c", "embedding": [1.0, 0.0, 0.0]},
    ]
    ranked = rank_by_cosine([1.0, 0.0], docs, k=5)
    assert [r.trace_id for r in ranked] == ["a", "b"]
    assert ranked[0].similarity == 1.0 and ranked[0].dish_name == "pancakes" and ranked[0].day_index == 1
    assert rank_by_cosine([1.0, 0.0], docs, k=0) == []


def test_merge_deduplicates_by_trace_id_and_keeps_top_k() -> None:
    merged = merge_search_results([hit("a", 0.9), hit("b", 0.7)], [hit("a", 0.91), hit("c", 0.8)], k=2)
    assert [(h.trace_id, h.similarity) for h in merged] == [("a", 0.91), ("c", 0.8)]


def test_search_traces_merges_atlas_and_local_new_trace() -> None:
    store = MongoStore(settings=offline_settings(), client=MagicMock())
    store._atlas_search = lambda q, t, k: []  # type: ignore[method-assign]  # index not synced yet
    store._local_search = lambda q, t, k, limit: [hit("fresh", 0.99)]  # type: ignore[method-assign]
    results, atlas_unavailable = store.search_traces([0.0] * 1536, "EXPIRY_IMMINENT", k=3)
    assert [r.trace_id for r in results] == ["fresh"] and atlas_unavailable is False


def test_search_traces_falls_back_to_local_when_atlas_fails() -> None:
    store = MongoStore(settings=offline_settings(), client=MagicMock())
    calls: list[int | None] = []

    def failing(*_: object) -> list[RetrievedTrace]:
        raise RuntimeError("index not queryable")

    def local(q: object, t: object, k: int, limit: int | None) -> list[RetrievedTrace]:
        calls.append(limit)
        return [hit("local", 0.5)]

    store._atlas_search = failing  # type: ignore[method-assign]
    store._local_search = local  # type: ignore[method-assign]
    results, atlas_unavailable = store.search_traces([0.0] * 1536, "EXPIRY_IMMINENT", k=3)
    assert atlas_unavailable is True and [r.trace_id for r in results] == ["local"]
    assert calls == [None]  # the fallback scans every accepted trace, not only the recent ones
    assert "index not queryable" in (store.last_search_error or "")


def test_drop_database_refuses_non_test_names() -> None:
    store = MongoStore(settings=offline_settings(), client=MagicMock(), db_name="pantry_a")
    with pytest.raises(RuntimeError):
        store.drop_database()


def test_ensure_indexes_tolerates_vector_index_failure() -> None:
    store = MongoStore(settings=offline_settings(), client=MagicMock())
    store.vector_index_status = MagicMock(side_effect=RuntimeError("no Atlas search here"))  # type: ignore[method-assign]
    store.ensure_indexes()
    assert "no Atlas search here" in (store.vector_index_error or "")


# ---------------------------------------------------------------------------
# Integration tests (RUN_INTEGRATION=1, database name ends in _test)
# ---------------------------------------------------------------------------


def seed(store: Store) -> None:
    store.set_today(START)
    store.seed_pantry(
        [
            Ingredient(item_name="milk", quantity=1000, unit="ml", expiry_date=date(2026, 11, 17), safety_stock=200),
            Ingredient(item_name="eggs", aliases=["egg"], quantity=10, unit="pcs", safety_stock=2),
        ]
    )


def ev(item: str, kind: str, delta: float, unit: str, run_id: str = "run-1") -> InventoryEvent:
    return InventoryEvent(event_id="", item_name=item, type=kind, delta=delta, unit=unit, run_id=run_id)


def test_integration_database_name_ends_with_test(mongo_store: MongoStore) -> None:
    assert mongo_store.db_name.endswith("_test")
    assert mongo_store.ping()


def test_integration_ledger_rollback_idempotency_and_undo(mongo_store: MongoStore) -> None:
    seed(mongo_store)
    with pytest.raises(InvariantViolation):
        mongo_store.apply_events([ev("milk", "RESERVED", 100, "ml"), ev("eggs", "RESERVED", 11, "pcs")], "bad")
    with pytest.raises(InvariantViolation):
        mongo_store.apply_events([ev("milk", "RESERVED", 100, "g")], "bad-unit")
    assert mongo_store.get_item("milk").reserved == 0 and mongo_store.list_events() == []

    first = mongo_store.apply_events([ev("milk", "RESERVED", 300, "ml"), ev("egg", "RESERVED", 2, "pcs")], "k1")
    again = mongo_store.apply_events([ev("milk", "RESERVED", 300, "ml")], "k1")
    assert [e.event_id for e in again] == [e.event_id for e in first]
    assert mongo_store.get_item("milk").reserved == 300 and mongo_store.get_item("eggs").reserved == 2

    mongo_store.create_run(Run(run_id="run-1", trigger="EXPIRY_IMMINENT", item_name="milk", idempotency_key="r1"))
    assert len(mongo_store.reverse_run_events("run-1")) == 2
    assert mongo_store.reverse_run_events("run-1") == []
    assert mongo_store.get_item("milk").reserved == 0
    assert mongo_store.check_invariants() == []


def test_integration_unique_keys_feedback_and_commands(mongo_store: MongoStore) -> None:
    seed(mongo_store)
    bus = BusEvent(trigger_type="EXPIRY_IMMINENT", item_name="milk", severity="low", dedupe_key="EXPIRY_IMMINENT:milk:2026-11-15")
    assert mongo_store.emit_bus_event(bus) is True and mongo_store.emit_bus_event(bus) is False
    run = Run(run_id="run-1", trigger="EXPIRY_IMMINENT", item_name="milk", idempotency_key=bus.dedupe_key)
    assert mongo_store.create_run(run)[1] is True
    assert mongo_store.create_run(run.model_copy(update={"run_id": "run-2"}))[1] is False

    llm = FakeLLM()
    vec = llm.embed(["trigger=EXPIRY_IMMINENT; expiring=milk(2d)"])[0]
    mongo_store.save_trace(Trace(trace_id="t1", run_id="run-1", day_index=1, trigger="EXPIRY_IMMINENT",
                                 snapshot_text="x", embedding=vec, final_outcome="accepted"))
    results, atlas_unavailable = mongo_store.search_traces(vec, "EXPIRY_IMMINENT", k=3)
    assert [r.trace_id for r in results] == ["t1"]  # immediately searchable
    assert atlas_unavailable is True  # the test database has no vector index

    fb = mongo_store.save_feedback("t1", "rejected", "op-1", "user")
    assert mongo_store.save_feedback("t1", "accepted", "op-1", "user") == fb
    assert mongo_store.get_trace("t1").final_outcome == "rejected"
    assert mongo_store.search_traces(vec, "EXPIRY_IMMINENT", k=3)[0] == []

    mongo_store.add_to_shopping_list(ShoppingListItem(item_name="heavy cream", quantity=100, unit="ml", run_ids=["run-1"]))
    mongo_store.add_to_shopping_list(ShoppingListItem(item_name="double cream", quantity=100, unit="ml", run_ids=["run-1"]))
    assert [(s.item_name, s.quantity) for s in mongo_store.list_shopping_list()] == [("heavy cream", 100)]

    cmd = mongo_store.enqueue_command(Command(command_id="c1", type="ADVANCE_CLOCK"))
    assert mongo_store.enqueue_command(cmd).command_id == "c1"
    assert [c.command_id for c in mongo_store.poll_commands()] == ["c1"]
    assert mongo_store.mark_command("c1", "done").status == "done"
    assert mongo_store.poll_commands() == []
    assert mongo_store.get_run("run-1").created_at.tzinfo is not None
    assert datetime.now(timezone.utc) >= mongo_store.get_run("run-1").created_at
