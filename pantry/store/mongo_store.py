# MongoStore: the MongoDB Atlas Store used by the worker and UI, with the same semantics as JsonStore.
# Every connection passes tlsCAFile=certifi.where(); trace search merges Atlas $vectorSearch with a local numpy find.
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, TypeVar

import certifi
import numpy as np
from pydantic import BaseModel
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError
from pymongo.operations import SearchIndexModel, UpdateOne

from pantry import rules
from pantry.config import Settings, get_settings
from pantry.llm import embedding_dimension
from pantry.models import (
    TERMINAL_RUN_STATES,
    BusEvent,
    BusEventStatus,
    Command,
    CommandStatus,
    ConfirmationRequest,
    Constraints,
    Feedback,
    FeedbackOutcome,
    FeedbackSource,
    Ingredient,
    InventoryEvent,
    PlanCacheEntry,
    RetrievedTrace,
    Run,
    RunState,
    RunStep,
    SearchCacheEntry,
    ShoppingListItem,
    Trace,
    TriggerType,
    UserProfile,
    utc_now,
)
from pantry.store.base import InvariantViolation, Store
from pantry.store.json_store import COMMIT_NOTE, EPSILON, JsonStore, _json_default, _round, new_event_id

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)
R = TypeVar("R")

VECTOR_INDEX_NAME: str = "trace_vector_index"
PROFILE_ID: str = "profile"
LOCAL_RECENT_LIMIT: int = 500  # local immediate search scans at most this many newest accepted traces
INDEX_STATUS_TTL_SECONDS: float = 30.0
DEFAULT_SERVER_SELECTION_TIMEOUT_MS: int = 10_000

# Collection names (build prompt section 6), plus two internal ledger collections.
PANTRY_ITEMS = "pantry_items"
PANTRY_BASELINE = "pantry_baseline"  # internal: seed baseline used by ledger replay and reset_demo
INVENTORY_EVENTS = "inventory_events"
LEDGER_KEYS = "ledger_keys"  # internal: idempotency keys of applied event batches
USER_PROFILE = "user_profile"
BUS_EVENTS = "bus_events"
RUNS = "runs"
DECISION_TRACES = "decision_traces"
FEEDBACK = "feedback"
SHOPPING_LIST = "shopping_list"
SEARCH_CACHE = "search_cache"
PLAN_CACHE = "plan_cache"
COMMANDS = "commands"

# The four unique indexes required by the build prompt, then the ones needed for id lookups.
REQUIRED_UNIQUE_INDEXES: tuple[tuple[str, str], ...] = (
    (RUNS, "idempotency_key"),
    (BUS_EVENTS, "dedupe_key"),
    (SHOPPING_LIST, "item_name"),
    (FEEDBACK, "op_id"),
)
SUPPORT_UNIQUE_INDEXES: tuple[tuple[str, str], ...] = (
    (PANTRY_ITEMS, "item_name"),
    (PANTRY_BASELINE, "item_name"),
    (INVENTORY_EVENTS, "event_id"),
    (LEDGER_KEYS, "key"),
    (RUNS, "run_id"),
    (DECISION_TRACES, "trace_id"),
    (SEARCH_CACHE, "key"),
    (PLAN_CACHE, "key"),
    (COMMANDS, "command_id"),
)


def vector_index_definition(num_dimensions: int) -> dict[str, Any]:
    """Atlas vector search definition for decision_traces (build prompt section 10)."""
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": num_dimensions, "similarity": "cosine"},
            {"type": "filter", "path": "final_outcome"},
            {"type": "filter", "path": "trigger"},
        ]
    }


# ---------------------------------------------------------------------------
# Document conversion and pure search helpers (offline-testable)
# ---------------------------------------------------------------------------


def _bson_safe(value: Any) -> Any:
    """Recursively convert plain dates to ISO strings (BSON cannot store them) and truncate datetimes to
    the millisecond precision BSON keeps, so a returned model equals what a later read returns."""
    if isinstance(value, datetime):
        return value.replace(microsecond=value.microsecond // 1000 * 1000)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _bson_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_bson_safe(v) for v in value]
    return value


def to_doc(model: BaseModel) -> dict[str, Any]:
    """Serialise a model into a BSON-compatible dict (datetimes kept native, dates as ISO strings)."""
    return _bson_safe(model.model_dump(mode="python"))


def from_doc(cls: type[M], doc: Mapping[str, Any]) -> M:
    """Validate a stored document back into its model, dropping Mongo's _id."""
    return cls.model_validate({k: v for k, v in doc.items() if k != "_id"})


def as_stored(model: M) -> M:
    """Return the model exactly as it reads back from MongoDB (millisecond datetimes)."""
    return from_doc(type(model), to_doc(model))


def atlas_score_to_cosine(score: float) -> float:
    """Atlas cosine vectorSearchScore is (1 + cosine) / 2; map it back to plain cosine similarity."""
    return 2.0 * float(score) - 1.0


def rank_by_cosine(query_vec: Sequence[float], docs: Sequence[Mapping[str, Any]], k: int) -> list[RetrievedTrace]:
    """Rank trace documents (with embedding) by numpy cosine similarity to query_vec; skips wrong dimensions."""
    if k <= 0:
        return []
    query = np.asarray(query_vec, dtype=float)
    query_norm = float(np.linalg.norm(query))
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for doc in docs:
        embedding = doc.get("embedding") or []
        if len(embedding) != query.shape[0]:
            continue
        vec = np.asarray(embedding, dtype=float)
        denom = query_norm * float(np.linalg.norm(vec))
        scored.append((float(query @ vec / denom) if denom > 0 else 0.0, doc))
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("trace_id", ""))))
    return [_retrieved(doc, sim) for sim, doc in scored[:k]]


def merge_search_results(
    atlas: Sequence[RetrievedTrace], local: Sequence[RetrievedTrace], k: int
) -> list[RetrievedTrace]:
    """Deduplicate Atlas and local hits by trace_id (keeping the higher similarity) and return the top-k."""
    best: dict[str, RetrievedTrace] = {}
    for hit in [*atlas, *local]:
        current = best.get(hit.trace_id)
        if current is None or hit.similarity > current.similarity:
            best[hit.trace_id] = hit
    ranked = sorted(best.values(), key=lambda h: (-h.similarity, h.trace_id))
    return ranked[: max(k, 0)]


def _retrieved(doc: Mapping[str, Any], similarity: float) -> RetrievedTrace:
    """Build a RetrievedTrace from a trace document and a cosine similarity."""
    plan = doc.get("plan") or {}
    return RetrievedTrace(
        trace_id=str(doc["trace_id"]),
        similarity=round(similarity, 6),
        day_index=int(doc.get("day_index", 0)),
        dish_name=str(plan.get("dish_name", "")) if isinstance(plan, Mapping) else "",
        is_seed=bool(doc.get("is_seed", False)),
    )


# ---------------------------------------------------------------------------
# MongoStore
# ---------------------------------------------------------------------------


class MongoStore(Store):
    """Store backed by MongoDB Atlas. Multi-document writes run in a transaction when the server
    supports it; every inventory batch is fully validated in Python before anything is written."""

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        *,
        settings: Settings | None = None,
        client: MongoClient[dict[str, Any]] | None = None,
        start_date: date | None = None,
        vector_dim: int | None = None,
        create_vector_index: bool = True,
        server_selection_timeout_ms: int = DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    ) -> None:
        self._settings = settings or get_settings()
        uri = uri or self._settings.mongodb_uri
        if client is None:
            if not uri:
                raise ValueError("MongoStore needs MONGODB_URI; set STORE_BACKEND=json to run offline")
            client = MongoClient(
                uri,
                tlsCAFile=certifi.where(),
                tz_aware=True,
                serverSelectionTimeoutMS=server_selection_timeout_ms,
                appname="pantry-agent",
            )
        self._client: MongoClient[dict[str, Any]] = client
        self.db_name: str = db_name or self._settings.mongodb_db
        self._db: Database[dict[str, Any]] = client[self.db_name]
        self._start_date = start_date or self._settings.demo_start_date
        self.vector_dim = vector_dim or embedding_dimension(self._settings.embed_model)
        self.vector_index_name = VECTOR_INDEX_NAME
        self.create_vector_index = create_vector_index
        self.vector_index_error: str | None = None
        self.last_search_error: str | None = None
        self._supports_txn: bool | None = None
        self._index_status: tuple[float, bool] | None = None

    # ------------------------------------------------------------------
    # Extras beyond Store
    # ------------------------------------------------------------------

    @property
    def db(self) -> Database[dict[str, Any]]:
        """The underlying pymongo database (scripts and integration tests only)."""
        return self._db

    def ping(self) -> bool:
        """Return True if the server answers the ping command; raises on connection errors."""
        return bool(self._client.admin.command("ping").get("ok"))

    def close(self) -> None:
        """Close the client connection pool."""
        self._client.close()

    def drop_database(self) -> None:
        """Drop the whole database; refused unless its name ends with _test."""
        if not self.db_name.endswith("_test"):
            raise RuntimeError(f"Refusing to drop database {self.db_name!r}: name does not end with _test")
        self._client.drop_database(self.db_name)

    def vector_index_status(self) -> dict[str, Any] | None:
        """Return the Atlas search index description for the trace vector index, or None if absent."""
        for index in self._col(DECISION_TRACES).list_search_indexes(self.vector_index_name):
            return dict(index)
        return None

    def vector_index_ready(self, refresh: bool = False) -> bool:
        """Return True if the vector index exists and is queryable (cached for a few seconds)."""
        now = time.monotonic()
        if not refresh and self._index_status and now - self._index_status[0] < INDEX_STATUS_TTL_SECONDS:
            return self._index_status[1]
        try:
            status = self.vector_index_status()
            ready = bool(status and status.get("queryable"))
        except PyMongoError as exc:
            self.vector_index_error = f"{type(exc).__name__}: {exc}"
            ready = False
        self._index_status = (now, ready)
        return ready

    def wait_for_vector_index(self, timeout_seconds: float = 120.0, poll_seconds: float = 3.0) -> bool:
        """Poll until the vector index is queryable or the timeout passes; return the final readiness."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self.vector_index_ready(refresh=True):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_seconds)

    def prune_seed_traces(self, keep_trace_ids: Sequence[str]) -> int:
        """Delete seed traces (is_seed=true) whose trace_id is not in keep_trace_ids; return the count."""
        result = self._col(DECISION_TRACES).delete_many({"is_seed": True, "trace_id": {"$nin": list(keep_trace_ids)}})
        return result.deleted_count

    def list_feedback(self, trace_id: str | None = None) -> list[Feedback]:
        """Return feedback in insertion order, optionally for one trace (mirrors the JsonStore extra)."""
        query: dict[str, Any] = {} if trace_id is None else {"trace_id": trace_id}
        return [from_doc(Feedback, d) for d in self._col(FEEDBACK).find(query).sort("_id", ASCENDING)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _col(self, name: str) -> Collection[dict[str, Any]]:
        return self._db[name]

    def _transactions_supported(self) -> bool:
        """Transactions need a replica set or mongos (always true on Atlas)."""
        if self._supports_txn is None:
            try:
                hello = self._client.admin.command("hello")
                self._supports_txn = bool(hello.get("setName")) or hello.get("msg") == "isdbgrid"
            except PyMongoError:
                self._supports_txn = False
        return self._supports_txn

    def _txn(self, fn: Callable[[ClientSession | None], R]) -> R:
        """Run fn inside a transaction (retried on transient errors) or directly if unsupported."""
        if not self._transactions_supported():
            return fn(None)
        with self._client.start_session() as session:
            return session.with_transaction(lambda s: fn(s))

    def _today(self, session: ClientSession | None = None) -> date:
        doc = self._col(USER_PROFILE).find_one({"_id": PROFILE_ID}, session=session)
        return date.fromisoformat(doc["today"]) if doc and doc.get("today") else self._start_date

    def _profile_doc(self, session: ClientSession | None = None) -> UserProfile:
        doc = self._col(USER_PROFILE).find_one({"_id": PROFILE_ID}, session=session)
        return from_doc(UserProfile, doc) if doc else UserProfile(today=self._start_date.isoformat())

    def _write_profile(self, profile: UserProfile, session: ClientSession | None = None) -> None:
        self._col(USER_PROFILE).replace_one({"_id": PROFILE_ID}, to_doc(profile), upsert=True, session=session)

    def _items(self, session: ClientSession | None = None) -> dict[str, Ingredient]:
        return {d["item_name"]: from_doc(Ingredient, d) for d in self._col(PANTRY_ITEMS).find({}, session=session)}

    def _rescore_all(self, today: date, session: ClientSession | None = None) -> None:
        ops = [
            UpdateOne({"item_name": name}, {"$set": {"priority_score": rules.priority_score(item, today)}})
            for name, item in self._items(session).items()
        ]
        if ops:
            self._col(PANTRY_ITEMS).bulk_write(ops, session=session)

    @staticmethod
    def _resolve_name(item_name: str, items: Mapping[str, Ingredient]) -> str | None:
        """Return the stored canonical name for item_name using item aliases and ALIASES (as JsonStore)."""
        name = rules.normalize_name(item_name, rules.item_alias_map(items.values()))
        if name in items:
            return name
        cleaned = rules.clean_name(item_name)
        return cleaned if cleaned in items else None

    def _events(self, query: Mapping[str, Any] | None = None, session: ClientSession | None = None) -> list[InventoryEvent]:
        cursor = self._col(INVENTORY_EVENTS).find(dict(query or {}), session=session).sort("_id", ASCENDING)
        return [from_doc(InventoryEvent, d) for d in cursor]

    def _require_run(self, run_id: str, session: ClientSession | None = None) -> Run:
        doc = self._col(RUNS).find_one({"run_id": run_id}, session=session)
        if doc is None:
            raise KeyError(f"Unknown run_id {run_id!r}")
        return from_doc(Run, doc)

    def _update_run(self, run_id: str, update: dict[str, Any]) -> Run:
        update.setdefault("$set", {})["updated_at"] = utc_now()
        doc = self._col(RUNS).find_one_and_update({"run_id": run_id}, update, return_document=ReturnDocument.AFTER)
        if doc is None:
            raise KeyError(f"Unknown run_id {run_id!r}")
        return from_doc(Run, doc)

    # ------------------------------------------------------------------
    # Setup and reset
    # ------------------------------------------------------------------

    def ensure_indexes(self, vector_index: bool | None = None) -> None:
        """Create the unique indexes, then try to create the Atlas vector index (unless vector_index, or
        the create_vector_index constructor flag when None, is False). Vector index failures are logged
        and stored in vector_index_error instead of raised."""
        for collection, field in (*REQUIRED_UNIQUE_INDEXES, *SUPPORT_UNIQUE_INDEXES):
            self._col(collection).create_index([(field, ASCENDING)], unique=True, name=f"{field}_unique")
        self._col(INVENTORY_EVENTS).create_index([("run_id", ASCENDING)], name="run_id")
        self._col(DECISION_TRACES).create_index(
            [("final_outcome", ASCENDING), ("trigger", ASCENDING), ("_id", DESCENDING)], name="outcome_trigger"
        )
        self._col(COMMANDS).create_index([("status", ASCENDING), ("_id", ASCENDING)], name="status")
        if self.create_vector_index if vector_index is None else vector_index:
            self._ensure_vector_index()

    def _ensure_vector_index(self) -> None:
        self.vector_index_error = None
        self._index_status = None
        try:
            existing = self.vector_index_status()
            if existing is not None:
                fields = (existing.get("latestDefinition") or existing.get("definition") or {}).get("fields", [])
                dims = [f.get("numDimensions") for f in fields if f.get("type") == "vector"]
                if dims and dims[0] != self.vector_dim:
                    self.vector_index_error = (
                        f"vector index has {dims[0]} dimensions but embeddings have {self.vector_dim}; "
                        "drop the index and reseed"
                    )
                    logger.warning(self.vector_index_error)
                return
            model = SearchIndexModel(
                definition=vector_index_definition(self.vector_dim), name=self.vector_index_name, type="vectorSearch"
            )
            self._col(DECISION_TRACES).create_search_index(model)
        except Exception as exc:  # noqa: BLE001 - any failure must degrade to local search, never crash
            self.vector_index_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Atlas vector index unavailable, local search only: %s", self.vector_index_error)

    def seed_pantry(self, items: Sequence[Ingredient]) -> None:
        names = [rules.clean_name(item.item_name) for item in items]
        if len(set(names)) != len(names):
            raise ValueError("seed_pantry received duplicate item names")
        for item in items:
            if item.reserved > item.quantity + EPSILON:
                raise InvariantViolation(f"{item.item_name}: reserved exceeds quantity in seed data")
        cleaned = [item.model_copy(update={"item_name": name}) for item, name in zip(items, names)]

        def write(s: ClientSession | None) -> None:
            today = self._today(s)
            docs = [to_doc(rules.with_priority(item, today)) for item in cleaned]
            for collection in (PANTRY_ITEMS, PANTRY_BASELINE, INVENTORY_EVENTS, LEDGER_KEYS):
                self._col(collection).delete_many({}, session=s)
            if docs:
                self._col(PANTRY_BASELINE).insert_many([dict(d) for d in docs], session=s)
                self._col(PANTRY_ITEMS).insert_many([dict(d) for d in docs], session=s)

        self._txn(write)

    def reset_demo(self, start_date: date) -> dict[str, int]:
        def write(s: ClientSession | None) -> dict[str, int]:
            baseline = [from_doc(Ingredient, d) for d in self._col(PANTRY_BASELINE).find({}, session=s)]
            counts = {"pantry_items": len(baseline)}
            for collection in (INVENTORY_EVENTS, RUNS, BUS_EVENTS, COMMANDS, SHOPPING_LIST, FEEDBACK):
                counts[collection] = self._col(collection).delete_many({}, session=s).deleted_count
            counts[DECISION_TRACES] = self._col(DECISION_TRACES).delete_many({"is_seed": {"$ne": True}}, session=s).deleted_count
            self._col(LEDGER_KEYS).delete_many({}, session=s)
            self._col(PANTRY_ITEMS).delete_many({}, session=s)
            if baseline:
                self._col(PANTRY_ITEMS).insert_many(
                    [to_doc(rules.with_priority(item, start_date)) for item in baseline], session=s
                )
            profile = self._profile_doc(s).model_copy(update={"today": start_date.isoformat()})
            self._write_profile(profile, s)
            return counts

        return self._txn(write)

    # ------------------------------------------------------------------
    # Simulated clock and user profile
    # ------------------------------------------------------------------

    def get_today(self) -> date:
        return self._today()

    def set_today(self, d: date) -> None:
        def write(s: ClientSession | None) -> None:
            self._write_profile(self._profile_doc(s).model_copy(update={"today": d.isoformat()}), s)
            self._rescore_all(d, s)

        self._txn(write)

    def advance_day(self, n: int = 1) -> date:
        if n < 0:
            raise ValueError("advance_day requires n >= 0")

        def write(s: ClientSession | None) -> date:
            profile = self._profile_doc(s)
            new_day = date.fromisoformat(profile.today) + timedelta(days=n)
            self._write_profile(profile.model_copy(update={"today": new_day.isoformat()}), s)
            self._rescore_all(new_day, s)
            return new_day

        return self._txn(write)

    def get_profile(self) -> UserProfile:
        return self._profile_doc()

    def set_constraints(self, constraints: Constraints) -> UserProfile:
        profile = self._profile_doc().model_copy(update={"constraints": constraints.model_copy()})
        self._write_profile(profile)
        return as_stored(profile)

    def set_allergens(self, allergens: Sequence[str]) -> UserProfile:
        cleaned = sorted({rules.normalize_allergen_group(a) for a in allergens if a.strip()})
        profile = self._profile_doc().model_copy(update={"allergens": cleaned})
        self._write_profile(profile)
        return as_stored(profile)

    # ------------------------------------------------------------------
    # Pantry items
    # ------------------------------------------------------------------

    def list_items(self) -> list[Ingredient]:
        return sorted(self._items().values(), key=lambda i: (-i.priority_score, i.item_name))

    def get_item(self, item_name: str) -> Ingredient | None:
        items = self._items()
        name = self._resolve_name(item_name, items)
        return None if name is None else items[name]

    def recompute_priorities(self) -> list[Ingredient]:
        self._txn(lambda s: self._rescore_all(self._today(s), s))
        return self.list_items()

    # ------------------------------------------------------------------
    # Inventory ledger
    # ------------------------------------------------------------------

    def _apply_in(self, s: ClientSession | None, events: Sequence[InventoryEvent], idempotency_key: str) -> list[InventoryEvent]:
        """Validate the whole batch against current state, then write keys, items and events."""
        applied = self._col(LEDGER_KEYS).find_one({"key": idempotency_key}, session=s)
        if applied is not None:
            wanted = list(applied.get("event_ids", []))
            found = {e.event_id: e for e in self._events({"event_id": {"$in": wanted}}, s)}
            return [found[eid] for eid in wanted if eid in found]
        items = self._items(s)
        known_ids = {raw.event_id for raw in events if raw.event_id}
        if known_ids and self._col(INVENTORY_EVENTS).find_one({"event_id": {"$in": list(known_ids)}}, session=s):
            raise InvariantViolation("Duplicate event_id in batch")
        seen: set[str] = set()
        start: dict[str, tuple[float, float]] = {}
        pending: dict[str, tuple[float, float]] = {}
        prepared: list[InventoryEvent] = []
        for raw in events:
            name = self._resolve_name(raw.item_name, items)
            if name is None:
                raise InvariantViolation(f"Unknown pantry item {raw.item_name!r}")
            item = items[name]
            event = raw.model_copy(update={"item_name": name, "event_id": raw.event_id or new_event_id()})
            if event.event_id in seen:
                raise InvariantViolation(f"Duplicate event_id {event.event_id}")
            seen.add(event.event_id)
            if event.unit != item.unit:
                raise InvariantViolation(f"Unit mismatch for {name}: event {event.unit} vs item {item.unit}")
            JsonStore._validate_sign(event)
            dq, dr = JsonStore._effect(event)
            start.setdefault(name, (item.quantity, item.reserved))
            quantity, reserved = pending.get(name, (item.quantity, item.reserved))
            pending[name] = (_round(quantity + dq), _round(reserved + dr))
            prepared.append(event)
        for name, (quantity, reserved) in pending.items():
            if quantity < -EPSILON:
                raise InvariantViolation(f"{name}: quantity would become {quantity} (< 0)")
            if reserved < -EPSILON:
                raise InvariantViolation(f"{name}: reserved would become {reserved} (< 0)")
            if reserved > quantity + EPSILON:
                raise InvariantViolation(f"{name}: reserved {reserved} would exceed quantity {quantity}")
        try:
            self._col(LEDGER_KEYS).insert_one(
                {"key": idempotency_key, "event_ids": [e.event_id for e in prepared], "created_at": utc_now()}, session=s
            )
        except DuplicateKeyError:
            if s is not None:
                raise  # inside a transaction: apply_events retries and takes the idempotent path
            return self._apply_in(None, events, idempotency_key)
        today = self._today(s)
        now = utc_now()
        for name, (quantity, reserved) in pending.items():
            old_q, old_r = start[name]
            updated = items[name].model_copy(
                update={"quantity": max(quantity, 0.0), "reserved": max(reserved, 0.0), "last_updated": now}
            )
            result = self._col(PANTRY_ITEMS).update_one(
                {"item_name": name, "quantity": old_q, "reserved": old_r},
                {"$set": {
                    "quantity": updated.quantity,
                    "reserved": updated.reserved,
                    "last_updated": now,
                    "priority_score": rules.priority_score(updated, today),
                }},
                session=s,
            )
            if result.matched_count != 1:
                raise InvariantViolation(f"{name}: concurrent inventory modification, batch rolled back")
        if prepared:
            self._col(INVENTORY_EVENTS).insert_many([to_doc(e) for e in prepared], session=s)
        return [as_stored(e) for e in prepared]

    def apply_events(self, events: Sequence[InventoryEvent], idempotency_key: str) -> list[InventoryEvent]:
        try:
            return self._txn(lambda s: self._apply_in(s, events, idempotency_key))
        except DuplicateKeyError:
            # A concurrent writer committed the same idempotency key first; return its events.
            return self._txn(lambda s: self._apply_in(s, events, idempotency_key))

    def list_events(self, run_id: str | None = None, item_name: str | None = None) -> list[InventoryEvent]:
        query: dict[str, Any] = {}
        if run_id is not None:
            query["run_id"] = run_id
        if item_name is not None:
            query["item_name"] = self._resolve_name(item_name, self._items()) or rules.clean_name(item_name)
        return self._events(query)

    def reverse_run_events(self, run_id: str) -> list[InventoryEvent]:
        def write(s: ClientSession | None) -> list[InventoryEvent]:
            run_events = self._events({"run_id": run_id}, s)
            reversed_ids = {e.reverses_event_id for e in run_events if e.reverses_event_id}
            targets = [e for e in run_events if e.reverses_event_id is None and e.event_id not in reversed_ids]
            if not targets:
                return []
            reversals = [JsonStore._reversal_of(e) for e in reversed(targets)]
            return self._apply_in(s, reversals, f"reverse:{run_id}:{targets[-1].event_id}")

        return self._txn(write)

    @staticmethod
    def _outstanding_reserved(run_events: Sequence[InventoryEvent]) -> dict[str, tuple[float, str]]:
        """Return item_name -> (reserved amount still held by the run, unit)."""
        held: dict[str, tuple[float, str]] = {}
        for e in run_events:
            if e.type not in ("RESERVED", "RELEASED"):
                continue
            amount, _ = held.get(e.item_name, (0.0, e.unit))
            amount += e.delta if e.type == "RESERVED" else -e.delta
            held[e.item_name] = (_round(amount), e.unit)
        return {name: v for name, v in held.items() if v[0] > EPSILON}

    def commit_run(self, run_id: str) -> list[InventoryEvent]:
        def write(s: ClientSession | None) -> list[InventoryEvent]:
            run = self._require_run(run_id, s)
            if run.committed:
                return []
            events: list[InventoryEvent] = []
            for name, (amount, unit) in sorted(self._outstanding_reserved(self._events({"run_id": run_id}, s)).items()):
                common = {"item_name": name, "unit": unit, "estimated": True, "run_id": run_id, "note": COMMIT_NOTE}
                events.append(InventoryEvent(event_id=new_event_id(), type="CONSUMED", delta=-amount, **common))
                events.append(InventoryEvent(event_id=new_event_id(), type="RELEASED", delta=amount, **common))
            written = self._apply_in(s, events, f"commit:{run_id}") if events else []
            self._col(RUNS).update_one(
                {"run_id": run_id}, {"$set": {"committed": True, "updated_at": utc_now()}}, session=s
            )
            return written

        return self._txn(write)

    def list_uncommitted_applied_runs(self) -> list[Run]:
        query = {
            "committed": {"$ne": True},
            "$or": [{"state": "APPLIED"}, {"state": "RECORDED", "checkpoints.APPLIED": {"$exists": True}}],
        }
        return [from_doc(Run, d) for d in self._col(RUNS).find(query).sort("_id", ASCENDING)]

    def replay_ledger(self) -> dict[str, tuple[float, float]]:
        totals = {
            d["item_name"]: (float(d["quantity"]), float(d.get("reserved", 0.0)))
            for d in self._col(PANTRY_BASELINE).find({})
        }
        for e in self._events():
            quantity, reserved = totals.get(e.item_name, (0.0, 0.0))
            dq, dr = JsonStore._effect(e)
            totals[e.item_name] = (_round(quantity + dq), _round(reserved + dr))
        return totals

    def check_invariants(self) -> list[str]:
        problems: list[str] = []
        items = self._items()
        for name, item in items.items():
            if item.quantity < -EPSILON:
                problems.append(f"{name}: quantity {item.quantity} < 0")
            if item.reserved < -EPSILON:
                problems.append(f"{name}: reserved {item.reserved} < 0")
            if item.reserved > item.quantity + EPSILON:
                problems.append(f"{name}: reserved {item.reserved} > quantity {item.quantity}")
        for e in self._events():
            item = items.get(e.item_name)
            if item is None:
                problems.append(f"event {e.event_id}: unknown item {e.item_name}")
            elif e.unit != item.unit:
                problems.append(f"event {e.event_id}: unit {e.unit} != item unit {item.unit}")
        replay = self.replay_ledger()
        for name, item in items.items():
            quantity, reserved = replay.get(name, (0.0, 0.0))
            if abs(quantity - item.quantity) > 1e-6 or abs(reserved - item.reserved) > 1e-6:
                problems.append(f"{name}: stored ({item.quantity}, {item.reserved}) != replay ({quantity}, {reserved})")
        return problems

    # ------------------------------------------------------------------
    # Bus events
    # ------------------------------------------------------------------

    def emit_bus_event(self, event: BusEvent) -> bool:
        try:
            self._col(BUS_EVENTS).insert_one(to_doc(event))
        except DuplicateKeyError:
            return False
        return True

    def list_bus_events(self, status: BusEventStatus | None = None, limit: int = 50) -> list[BusEvent]:
        query = {} if status is None else {"status": status}
        cursor = self._col(BUS_EVENTS).find(query).sort("_id", DESCENDING).limit(max(limit, 0))
        return [from_doc(BusEvent, d) for d in cursor] if limit > 0 else []

    def set_bus_event_status(self, dedupe_key: str, status: BusEventStatus) -> None:
        if self._col(BUS_EVENTS).update_one({"dedupe_key": dedupe_key}, {"$set": {"status": status}}).matched_count == 0:
            raise KeyError(f"Unknown dedupe_key {dedupe_key!r}")

    # ------------------------------------------------------------------
    # Runs and checkpoints
    # ------------------------------------------------------------------

    def create_run(self, run: Run) -> tuple[Run, bool]:
        existing = self._col(RUNS).find_one({"idempotency_key": run.idempotency_key})
        if existing is not None:
            return from_doc(Run, existing), False
        try:
            self._col(RUNS).insert_one(to_doc(run))
        except DuplicateKeyError:
            existing = self._col(RUNS).find_one({"idempotency_key": run.idempotency_key})
            if existing is not None:
                return from_doc(Run, existing), False
            raise ValueError(f"run_id {run.run_id!r} already exists with a different idempotency_key") from None
        return as_stored(run), True

    def get_run(self, run_id: str) -> Run | None:
        doc = self._col(RUNS).find_one({"run_id": run_id})
        return None if doc is None else from_doc(Run, doc)

    def list_runs(self, states: Sequence[RunState] | None = None, limit: int = 50) -> list[Run]:
        if limit <= 0:
            return []
        query = {} if states is None else {"state": {"$in": list(states)}}
        return [from_doc(Run, d) for d in self._col(RUNS).find(query).sort("_id", DESCENDING).limit(limit)]

    def list_active_runs(self) -> list[Run]:
        query = {"state": {"$nin": list(TERMINAL_RUN_STATES)}}
        return [from_doc(Run, d) for d in self._col(RUNS).find(query).sort("_id", ASCENDING)]

    def save_checkpoint(self, run_id: str, state: RunState, checkpoint: dict[str, object], step: RunStep) -> Run:
        run = self._require_run(run_id)
        if run.state == state and any(s.name == step.name for s in run.steps):
            return run
        payload = _bson_safe(json.loads(json.dumps(checkpoint, default=_json_default)))
        return self._update_run(
            run_id, {"$set": {"state": state, f"checkpoints.{state}": payload}, "$push": {"steps": to_doc(step)}}
        )

    def append_step(self, run_id: str, step: RunStep) -> Run:
        return self._update_run(run_id, {"$push": {"steps": to_doc(step)}})

    def set_pending_confirmation(self, run_id: str, request: ConfirmationRequest | None) -> Run:
        value = None if request is None else to_doc(request)
        return self._update_run(run_id, {"$set": {"pending_confirmation": value}})

    # ------------------------------------------------------------------
    # Decision traces and feedback
    # ------------------------------------------------------------------

    def save_trace(self, trace: Trace) -> Trace:
        self._col(DECISION_TRACES).replace_one({"trace_id": trace.trace_id}, to_doc(trace), upsert=True)
        return as_stored(trace)

    def get_trace(self, trace_id: str) -> Trace | None:
        doc = self._col(DECISION_TRACES).find_one({"trace_id": trace_id})
        return None if doc is None else from_doc(Trace, doc)

    def get_trace_by_run(self, run_id: str) -> Trace | None:
        cursor = self._col(DECISION_TRACES).find({"run_id": run_id}).sort([("created_at", DESCENDING), ("_id", DESCENDING)]).limit(1)
        docs = list(cursor)
        return from_doc(Trace, docs[0]) if docs else None

    def list_traces(self, is_seed: bool | None = None, limit: int = 100) -> list[Trace]:
        if limit <= 0:
            return []
        query = {} if is_seed is None else ({"is_seed": True} if is_seed else {"is_seed": {"$ne": True}})
        cursor = self._col(DECISION_TRACES).find(query).sort([("created_at", DESCENDING), ("_id", DESCENDING)]).limit(limit)
        return [from_doc(Trace, d) for d in cursor]

    def _atlas_search(self, query_vec: Sequence[float], trigger: TriggerType, k: int) -> list[RetrievedTrace]:
        """Run Atlas $vectorSearch with the final_outcome and trigger filters; raises on any failure.

        The trailing $match re-checks the live documents because the search index syncs asynchronously
        (a trace rejected a moment ago may still be "accepted" in the index)."""
        if len(query_vec) != self.vector_dim:
            raise ValueError(f"query vector has {len(query_vec)} dimensions, index expects {self.vector_dim}")
        if not self.vector_index_ready():
            raise RuntimeError(self.vector_index_error or "vector index missing or not queryable yet")
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "embedding",
                    "queryVector": [float(v) for v in query_vec],
                    "numCandidates": max(100, k * 20),
                    "limit": k * 2,
                    "filter": {"final_outcome": {"$eq": "accepted"}, "trigger": {"$eq": trigger}},
                }
            },
            {"$match": {"final_outcome": "accepted", "trigger": trigger}},
            {
                "$project": {
                    "_id": 0, "trace_id": 1, "day_index": 1, "is_seed": 1, "plan.dish_name": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return [
            _retrieved(doc, atlas_score_to_cosine(doc.get("score", 0.0)))
            for doc in self._col(DECISION_TRACES).aggregate(pipeline)
        ]

    def _local_search(self, query_vec: Sequence[float], trigger: TriggerType, k: int, limit: int | None) -> list[RetrievedTrace]:
        """Plain find over accepted traces with the same trigger plus numpy cosine; visible immediately."""
        cursor = self._col(DECISION_TRACES).find(
            {"final_outcome": "accepted", "trigger": trigger},
            {"_id": 0, "trace_id": 1, "day_index": 1, "is_seed": 1, "plan.dish_name": 1, "embedding": 1},
        ).sort("_id", DESCENDING)
        if limit is not None:
            cursor = cursor.limit(limit)
        return rank_by_cosine(query_vec, list(cursor), k)

    def search_traces(
        self, query_vec: Sequence[float], trigger: TriggerType, k: int = 3
    ) -> tuple[list[RetrievedTrace], bool]:
        """Merge Atlas $vectorSearch with a local immediate numpy search over the newest accepted traces.
        If Atlas fails or the index is not queryable, rank every accepted trace locally and return True."""
        if k <= 0:
            return [], False
        try:
            atlas = self._atlas_search(query_vec, trigger, k)
            self.last_search_error = None
        except Exception as exc:  # noqa: BLE001 - Atlas problems always degrade to the local search
            self.last_search_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Atlas vector search unavailable, using local numpy search: %s", self.last_search_error)
            return self._local_search(query_vec, trigger, k, limit=None), True
        local = self._local_search(query_vec, trigger, k, limit=LOCAL_RECENT_LIMIT)
        return merge_search_results(atlas, local, k), False

    def save_feedback(self, trace_id: str, outcome: FeedbackOutcome, op_id: str, source: FeedbackSource) -> Feedback:
        def write(s: ClientSession | None) -> Feedback:
            existing = self._col(FEEDBACK).find_one({"op_id": op_id}, session=s)
            if existing is not None:
                return from_doc(Feedback, existing)
            if self._col(DECISION_TRACES).find_one({"trace_id": trace_id}, {"_id": 1}, session=s) is None:
                raise KeyError(f"Unknown trace_id {trace_id!r}")
            feedback = Feedback(op_id=op_id, trace_id=trace_id, outcome=outcome, source=source)
            self._col(FEEDBACK).insert_one(to_doc(feedback), session=s)
            self._col(DECISION_TRACES).update_one({"trace_id": trace_id}, {"$set": {"final_outcome": outcome}}, session=s)
            return as_stored(feedback)

        try:
            return self._txn(write)
        except DuplicateKeyError:
            doc = self._col(FEEDBACK).find_one({"op_id": op_id})
            if doc is None:
                raise
            return from_doc(Feedback, doc)

    # ------------------------------------------------------------------
    # Shopping list
    # ------------------------------------------------------------------

    def add_to_shopping_list(self, item: ShoppingListItem) -> ShoppingListItem:
        name = rules.normalize_name(item.item_name)

        def write(s: ClientSession | None) -> ShoppingListItem:
            doc = self._col(SHOPPING_LIST).find_one({"item_name": name}, session=s)
            if doc is None:
                stored = item.model_copy(update={"item_name": name, "run_ids": list(dict.fromkeys(item.run_ids))})
                self._col(SHOPPING_LIST).insert_one(to_doc(stored), session=s)
                return as_stored(stored)
            existing = from_doc(ShoppingListItem, doc)
            if existing.unit != item.unit:
                raise InvariantViolation(f"Shopping list unit mismatch for {name}: {existing.unit} vs {item.unit}")
            if item.run_ids and any(rid in existing.run_ids for rid in item.run_ids):
                return existing
            updated = existing.model_copy(
                update={
                    "quantity": _round(existing.quantity + item.quantity),
                    "run_ids": list(dict.fromkeys([*existing.run_ids, *item.run_ids])),
                    "reason": existing.reason or item.reason,
                    "updated_at": utc_now(),
                }
            )
            self._col(SHOPPING_LIST).replace_one({"item_name": name}, to_doc(updated), session=s)
            return as_stored(updated)

        return self._txn(write)

    def list_shopping_list(self) -> list[ShoppingListItem]:
        return [from_doc(ShoppingListItem, d) for d in self._col(SHOPPING_LIST).find({}).sort("item_name", ASCENDING)]

    def remove_from_shopping_list(self, item_name: str) -> bool:
        return self._col(SHOPPING_LIST).delete_one({"item_name": rules.normalize_name(item_name)}).deleted_count > 0

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------

    def get_search_cache(self, key: str) -> SearchCacheEntry | None:
        doc = self._col(SEARCH_CACHE).find_one({"key": key})
        return None if doc is None else from_doc(SearchCacheEntry, doc)

    def put_search_cache(self, entry: SearchCacheEntry) -> None:
        self._col(SEARCH_CACHE).replace_one({"key": entry.key}, to_doc(entry), upsert=True)

    def get_plan_cache(self, key: str) -> PlanCacheEntry | None:
        doc = self._col(PLAN_CACHE).find_one({"key": key})
        return None if doc is None else from_doc(PlanCacheEntry, doc)

    def put_plan_cache(self, entry: PlanCacheEntry) -> None:
        self._col(PLAN_CACHE).replace_one({"key": entry.key}, to_doc(entry), upsert=True)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def enqueue_command(self, command: Command) -> Command:
        try:
            self._col(COMMANDS).insert_one(to_doc(command))
        except DuplicateKeyError:
            doc = self._col(COMMANDS).find_one({"command_id": command.command_id})
            if doc is not None:
                return from_doc(Command, doc)
            raise
        return as_stored(command)

    def poll_commands(self, limit: int = 20) -> list[Command]:
        if limit <= 0:
            return []
        cursor = self._col(COMMANDS).find({"status": "pending"}).sort("_id", ASCENDING).limit(limit)
        return [from_doc(Command, d) for d in cursor]

    def get_command(self, command_id: str) -> Command | None:
        doc = self._col(COMMANDS).find_one({"command_id": command_id})
        return None if doc is None else from_doc(Command, doc)

    def mark_command(self, command_id: str, status: CommandStatus, error: str | None = None) -> Command:
        doc = self._col(COMMANDS).find_one_and_update(
            {"command_id": command_id},
            {"$set": {"status": status, "error": error, "processed_at": utc_now()}},
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            raise KeyError(f"Unknown command_id {command_id!r}")
        return from_doc(Command, doc)


__all__ = [
    "MongoStore",
    "VECTOR_INDEX_NAME",
    "atlas_score_to_cosine",
    "from_doc",
    "merge_search_results",
    "rank_by_cosine",
    "to_doc",
    "vector_index_definition",
]
