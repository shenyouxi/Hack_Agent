# JsonStore: a single-process Store backed by one JSON file (default data/pantry_store.json) or memory only.
# Used by pytest and offline runs; enforces the same ledger invariants and idempotency rules as MongoStore.
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pantry import rules
from pantry.config import DEFAULT_DEMO_START_DATE
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

DEFAULT_DATA_DIR: Path = Path("data")
DEFAULT_FILENAME: str = "pantry_store.json"
COMMIT_NOTE: str = "assumed cooked in the demo"
EPSILON: float = 1e-9
FORMAT_VERSION: int = 1


def _round(value: float) -> float:
    """Round ledger arithmetic so float noise never breaks invariant comparisons."""
    return round(value, 9)


def new_event_id() -> str:
    """Return a fresh unique inventory event id."""
    return uuid.uuid4().hex


class JsonStore(Store):
    """Store implementation keeping every collection in memory and optionally persisting to a JSON file.

    JsonStore() persists to data/pantry_store.json; JsonStore(data_dir=...) picks another directory;
    JsonStore(in_memory=True) never touches the filesystem (tests).
    """

    def __init__(
        self,
        data_dir: str | Path | None = DEFAULT_DATA_DIR,
        *,
        filename: str = DEFAULT_FILENAME,
        in_memory: bool = False,
        start_date: date | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._path: Path | None = None if in_memory or data_dir is None else Path(data_dir) / filename
        self._start_date = start_date or date.fromisoformat(DEFAULT_DEMO_START_DATE)
        self._clear_all()
        if self._path is not None and self._path.exists():
            self._load()

    @classmethod
    def memory(cls, start_date: date | None = None) -> JsonStore:
        """Return a store that lives only in memory."""
        return cls(in_memory=True, start_date=start_date)

    @property
    def path(self) -> Path | None:
        """The backing JSON file, or None for an in-memory store."""
        return self._path

    # ------------------------------------------------------------------
    # Internal state, persistence and transactions
    # ------------------------------------------------------------------

    def _clear_all(self) -> None:
        """Initialise every collection to empty."""
        self._profile = UserProfile(today=self._start_date.isoformat())
        self._items: dict[str, Ingredient] = {}
        self._baseline: dict[str, Ingredient] = {}
        self._events: list[InventoryEvent] = []
        self._applied_keys: dict[str, list[str]] = {}
        self._bus: dict[str, BusEvent] = {}
        self._runs: dict[str, Run] = {}
        self._run_keys: dict[str, str] = {}
        self._traces: dict[str, Trace] = {}
        self._feedback: dict[str, Feedback] = {}
        self._shopping: dict[str, ShoppingListItem] = {}
        self._search_cache: dict[str, SearchCacheEntry] = {}
        self._plan_cache: dict[str, PlanCacheEntry] = {}
        self._commands: dict[str, Command] = {}

    def _snapshot(self) -> dict[str, Any]:
        """Serialise the full state into JSON-compatible data."""

        def dump(models: Any) -> list[dict[str, Any]]:
            return [m.model_dump(mode="json") for m in models]

        return {
            "version": FORMAT_VERSION,
            "profile": self._profile.model_dump(mode="json"),
            "items": dump(self._items.values()),
            "baseline": dump(self._baseline.values()),
            "events": dump(self._events),
            "applied_keys": self._applied_keys,
            "bus_events": dump(self._bus.values()),
            "runs": dump(self._runs.values()),
            "traces": dump(self._traces.values()),
            "feedback": dump(self._feedback.values()),
            "shopping_list": dump(self._shopping.values()),
            "search_cache": dump(self._search_cache.values()),
            "plan_cache": dump(self._plan_cache.values()),
            "commands": dump(self._commands.values()),
        }

    def _restore(self, data: dict[str, Any]) -> None:
        """Rebuild the in-memory state from serialised data."""
        self._clear_all()
        self._profile = UserProfile.model_validate(data["profile"])
        self._items = {d["item_name"]: Ingredient.model_validate(d) for d in data.get("items", [])}
        self._baseline = {d["item_name"]: Ingredient.model_validate(d) for d in data.get("baseline", [])}
        self._events = [InventoryEvent.model_validate(d) for d in data.get("events", [])]
        self._applied_keys = {k: list(v) for k, v in data.get("applied_keys", {}).items()}
        self._bus = {d["dedupe_key"]: BusEvent.model_validate(d) for d in data.get("bus_events", [])}
        self._runs = {d["run_id"]: Run.model_validate(d) for d in data.get("runs", [])}
        self._run_keys = {run.idempotency_key: run.run_id for run in self._runs.values()}
        self._traces = {d["trace_id"]: Trace.model_validate(d) for d in data.get("traces", [])}
        self._feedback = {d["op_id"]: Feedback.model_validate(d) for d in data.get("feedback", [])}
        self._shopping = {d["item_name"]: ShoppingListItem.model_validate(d) for d in data.get("shopping_list", [])}
        self._search_cache = {d["key"]: SearchCacheEntry.model_validate(d) for d in data.get("search_cache", [])}
        self._plan_cache = {d["key"]: PlanCacheEntry.model_validate(d) for d in data.get("plan_cache", [])}
        self._commands = {d["command_id"]: Command.model_validate(d) for d in data.get("commands", [])}

    def _load(self) -> None:
        """Load state from the backing file."""
        assert self._path is not None
        with self._path.open("r", encoding="utf-8") as fh:
            self._restore(json.load(fh))

    def _save(self) -> None:
        """Atomically write state to the backing file (write a temp file, then rename)."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".store-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._snapshot(), fh, ensure_ascii=False)
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @contextmanager
    def _write(self) -> Iterator[None]:
        """Run a mutation under the lock; on any error restore the previous state, otherwise persist."""
        with self._lock:
            before = self._snapshot()
            try:
                yield
            except BaseException:
                self._restore(before)
                raise
            self._save()

    def _require_run(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id {run_id!r}")
        return run

    def _today(self) -> date:
        return date.fromisoformat(self._profile.today)

    def _rescore_all(self) -> None:
        today = self._today()
        for name, item in self._items.items():
            self._items[name] = rules.with_priority(item, today)

    # ------------------------------------------------------------------
    # Setup and reset
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> None:
        """No-op: uniqueness is enforced by the dict keys of each collection."""

    def seed_pantry(self, items: Sequence[Ingredient]) -> None:
        """Replace the pantry with items, store them as the ledger baseline, and clear the event ledger."""
        with self._write():
            names = [rules.clean_name(item.item_name) for item in items]
            if len(set(names)) != len(names):
                raise ValueError("seed_pantry received duplicate item names")
            for item in items:
                if item.reserved > item.quantity + EPSILON:
                    raise InvariantViolation(f"{item.item_name}: reserved exceeds quantity in seed data")
            cleaned = [item.model_copy(update={"item_name": name}) for item, name in zip(items, names)]
            self._baseline = {item.item_name: item.model_copy(deep=True) for item in cleaned}
            self._items = {item.item_name: item.model_copy(deep=True) for item in cleaned}
            self._events = []
            self._applied_keys = {}
            self._rescore_all()

    def reset_demo(self, start_date: date) -> dict[str, int]:
        """Restore day 1 from the baseline and clear per-run collections; keeps caches and seed traces."""
        with self._write():
            counts = {
                "pantry_items": len(self._baseline),
                "inventory_events": len(self._events),
                "runs": len(self._runs),
                "bus_events": len(self._bus),
                "commands": len(self._commands),
                "shopping_list": len(self._shopping),
                "feedback": len(self._feedback),
                "decision_traces": sum(1 for t in self._traces.values() if not t.is_seed),
            }
            self._items = {name: item.model_copy(deep=True) for name, item in self._baseline.items()}
            self._events = []
            self._applied_keys = {}
            self._runs = {}
            self._run_keys = {}
            self._bus = {}
            self._commands = {}
            self._shopping = {}
            self._feedback = {}
            self._traces = {tid: t for tid, t in self._traces.items() if t.is_seed}
            self._profile = self._profile.model_copy(update={"today": start_date.isoformat()})
            self._rescore_all()
            return counts

    # ------------------------------------------------------------------
    # Simulated clock and user profile
    # ------------------------------------------------------------------

    def get_today(self) -> date:
        with self._lock:
            return self._today()

    def set_today(self, d: date) -> None:
        with self._write():
            self._profile = self._profile.model_copy(update={"today": d.isoformat()})
            self._rescore_all()

    def advance_day(self, n: int = 1) -> date:
        if n < 0:
            raise ValueError("advance_day requires n >= 0")
        with self._write():
            new_day = self._today() + timedelta(days=n)
            self._profile = self._profile.model_copy(update={"today": new_day.isoformat()})
            self._rescore_all()
            return new_day

    def get_profile(self) -> UserProfile:
        with self._lock:
            return self._profile.model_copy(deep=True)

    def set_constraints(self, constraints: Constraints) -> UserProfile:
        with self._write():
            self._profile = self._profile.model_copy(update={"constraints": constraints.model_copy()})
            return self._profile.model_copy(deep=True)

    def set_allergens(self, allergens: Sequence[str]) -> UserProfile:
        with self._write():
            cleaned = sorted({rules.normalize_allergen_group(a) for a in allergens if a.strip()})
            self._profile = self._profile.model_copy(update={"allergens": cleaned})
            return self._profile.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Pantry items
    # ------------------------------------------------------------------

    def list_items(self) -> list[Ingredient]:
        with self._lock:
            items = [item.model_copy(deep=True) for item in self._items.values()]
        return sorted(items, key=lambda i: (-i.priority_score, i.item_name))

    def _resolve_item_name(self, item_name: str) -> str | None:
        """Return the stored canonical name for item_name using item aliases and ALIASES."""
        name = rules.normalize_name(item_name, rules.item_alias_map(self._items.values()))
        if name in self._items:
            return name
        cleaned = rules.clean_name(item_name)
        return cleaned if cleaned in self._items else None

    def get_item(self, item_name: str) -> Ingredient | None:
        with self._lock:
            name = self._resolve_item_name(item_name)
            return None if name is None else self._items[name].model_copy(deep=True)

    def recompute_priorities(self) -> list[Ingredient]:
        with self._write():
            self._rescore_all()
        return self.list_items()

    # ------------------------------------------------------------------
    # Inventory ledger
    # ------------------------------------------------------------------

    @staticmethod
    def _effect(event: InventoryEvent) -> tuple[float, float]:
        """Return the (quantity change, reserved change) an event applies.

        Sign convention: RESERVED and RELEASED carry a positive delta; CONSUMED carries a negative delta;
        ADJUSTED carries a signed delta applied to quantity.
        """
        if event.type == "RESERVED":
            return 0.0, event.delta
        if event.type == "RELEASED":
            return 0.0, -event.delta
        return event.delta, 0.0

    @staticmethod
    def _validate_sign(event: InventoryEvent) -> None:
        if not math.isfinite(event.delta) or event.delta == 0:
            raise InvariantViolation(f"Event {event.event_id}: delta must be finite and non-zero")
        if event.type in ("RESERVED", "RELEASED") and event.delta < 0:
            raise InvariantViolation(f"Event {event.event_id}: {event.type} delta must be positive")
        if event.type == "CONSUMED" and event.delta > 0:
            raise InvariantViolation(f"Event {event.event_id}: CONSUMED delta must be negative")

    def _apply_locked(self, events: Sequence[InventoryEvent], idempotency_key: str) -> list[InventoryEvent]:
        """Validate and apply a batch inside an open write; raises InvariantViolation without mutating."""
        if idempotency_key in self._applied_keys:
            wanted = set(self._applied_keys[idempotency_key])
            return [e.model_copy(deep=True) for e in self._events if e.event_id in wanted]
        known_ids = {e.event_id for e in self._events}
        pending: dict[str, tuple[float, float]] = {}
        prepared: list[InventoryEvent] = []
        for raw in events:
            name = self._resolve_item_name(raw.item_name)
            if name is None:
                raise InvariantViolation(f"Unknown pantry item {raw.item_name!r}")
            item = self._items[name]
            event = raw.model_copy(update={"item_name": name, "event_id": raw.event_id or new_event_id()})
            if event.event_id in known_ids:
                raise InvariantViolation(f"Duplicate event_id {event.event_id}")
            known_ids.add(event.event_id)
            if event.unit != item.unit:
                raise InvariantViolation(f"Unit mismatch for {name}: event {event.unit} vs item {item.unit}")
            self._validate_sign(event)
            dq, dr = self._effect(event)
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
        today = self._today()
        now = utc_now()
        for name, (quantity, reserved) in pending.items():
            updated = self._items[name].model_copy(
                update={"quantity": max(quantity, 0.0), "reserved": max(reserved, 0.0), "last_updated": now}
            )
            self._items[name] = rules.with_priority(updated, today)
        self._events.extend(prepared)
        self._applied_keys[idempotency_key] = [e.event_id for e in prepared]
        return [e.model_copy(deep=True) for e in prepared]

    def apply_events(self, events: Sequence[InventoryEvent], idempotency_key: str) -> list[InventoryEvent]:
        with self._write():
            return self._apply_locked(events, idempotency_key)

    def list_events(self, run_id: str | None = None, item_name: str | None = None) -> list[InventoryEvent]:
        with self._lock:
            name = None if item_name is None else (self._resolve_item_name(item_name) or rules.clean_name(item_name))
            return [
                e.model_copy(deep=True)
                for e in self._events
                if (run_id is None or e.run_id == run_id) and (name is None or e.item_name == name)
            ]

    @staticmethod
    def _reversal_of(event: InventoryEvent) -> InventoryEvent:
        """Build the event that cancels event's effect."""
        if event.type == "RESERVED":
            kind, delta = "RELEASED", event.delta
        elif event.type == "RELEASED":
            kind, delta = "RESERVED", event.delta
        else:
            kind, delta = "ADJUSTED", -event.delta
        return InventoryEvent(
            event_id=new_event_id(),
            item_name=event.item_name,
            type=kind,
            delta=delta,
            unit=event.unit,
            estimated=event.estimated,
            run_id=event.run_id,
            reverses_event_id=event.event_id,
            note=f"undo of {event.type} {event.event_id}",
        )

    def reverse_run_events(self, run_id: str) -> list[InventoryEvent]:
        with self._write():
            reversed_ids = {e.reverses_event_id for e in self._events if e.reverses_event_id}
            targets = [
                e for e in self._events
                if e.run_id == run_id and e.reverses_event_id is None and e.event_id not in reversed_ids
            ]
            if not targets:
                return []
            reversals = [self._reversal_of(e) for e in reversed(targets)]
            key = f"reverse:{run_id}:{targets[-1].event_id}"
            return self._apply_locked(reversals, key)

    def _outstanding_reserved(self, run_id: str) -> dict[str, tuple[float, str]]:
        """Return item_name -> (reserved amount still held by the run, unit)."""
        held: dict[str, tuple[float, str]] = {}
        for e in self._events:
            if e.run_id != run_id or e.type not in ("RESERVED", "RELEASED"):
                continue
            amount, _ = held.get(e.item_name, (0.0, e.unit))
            amount += e.delta if e.type == "RESERVED" else -e.delta
            held[e.item_name] = (_round(amount), e.unit)
        return {name: v for name, v in held.items() if v[0] > EPSILON}

    def commit_run(self, run_id: str) -> list[InventoryEvent]:
        with self._write():
            run = self._require_run(run_id)
            if run.committed:
                return []
            events: list[InventoryEvent] = []
            for name, (amount, unit) in sorted(self._outstanding_reserved(run_id).items()):
                common = {"item_name": name, "unit": unit, "estimated": True, "run_id": run_id, "note": COMMIT_NOTE}
                events.append(InventoryEvent(event_id=new_event_id(), type="CONSUMED", delta=-amount, **common))
                events.append(InventoryEvent(event_id=new_event_id(), type="RELEASED", delta=amount, **common))
            written = self._apply_locked(events, f"commit:{run_id}") if events else []
            self._runs[run_id] = run.model_copy(update={"committed": True, "updated_at": utc_now()})
            return written

    def list_uncommitted_applied_runs(self) -> list[Run]:
        with self._lock:
            return [
                run.model_copy(deep=True)
                for run in self._runs.values()
                if not run.committed and (run.state == "APPLIED" or (run.state == "RECORDED" and "APPLIED" in run.checkpoints))
            ]

    def replay_ledger(self) -> dict[str, tuple[float, float]]:
        with self._lock:
            totals = {name: (item.quantity, item.reserved) for name, item in self._baseline.items()}
            for e in self._events:
                quantity, reserved = totals.get(e.item_name, (0.0, 0.0))
                dq, dr = self._effect(e)
                totals[e.item_name] = (_round(quantity + dq), _round(reserved + dr))
            return totals

    def check_invariants(self) -> list[str]:
        with self._lock:
            problems: list[str] = []
            for name, item in self._items.items():
                if item.quantity < -EPSILON:
                    problems.append(f"{name}: quantity {item.quantity} < 0")
                if item.reserved < -EPSILON:
                    problems.append(f"{name}: reserved {item.reserved} < 0")
                if item.reserved > item.quantity + EPSILON:
                    problems.append(f"{name}: reserved {item.reserved} > quantity {item.quantity}")
            for e in self._events:
                item = self._items.get(e.item_name)
                if item is None:
                    problems.append(f"event {e.event_id}: unknown item {e.item_name}")
                elif e.unit != item.unit:
                    problems.append(f"event {e.event_id}: unit {e.unit} != item unit {item.unit}")
            replay = self.replay_ledger()
            for name, item in self._items.items():
                quantity, reserved = replay.get(name, (0.0, 0.0))
                if abs(quantity - item.quantity) > 1e-6 or abs(reserved - item.reserved) > 1e-6:
                    problems.append(
                        f"{name}: stored ({item.quantity}, {item.reserved}) != replay ({quantity}, {reserved})"
                    )
            return problems

    # ------------------------------------------------------------------
    # Bus events
    # ------------------------------------------------------------------

    def emit_bus_event(self, event: BusEvent) -> bool:
        with self._write():
            if event.dedupe_key in self._bus:
                return False
            self._bus[event.dedupe_key] = event.model_copy(deep=True)
            return True

    def list_bus_events(self, status: BusEventStatus | None = None, limit: int = 50) -> list[BusEvent]:
        with self._lock:
            events = [e.model_copy(deep=True) for e in reversed(self._bus.values()) if status is None or e.status == status]
            return events[:limit]

    def set_bus_event_status(self, dedupe_key: str, status: BusEventStatus) -> None:
        with self._write():
            event = self._bus.get(dedupe_key)
            if event is None:
                raise KeyError(f"Unknown dedupe_key {dedupe_key!r}")
            self._bus[dedupe_key] = event.model_copy(update={"status": status})

    # ------------------------------------------------------------------
    # Runs and checkpoints
    # ------------------------------------------------------------------

    def create_run(self, run: Run) -> tuple[Run, bool]:
        with self._write():
            existing_id = self._run_keys.get(run.idempotency_key)
            if existing_id is not None:
                return self._runs[existing_id].model_copy(deep=True), False
            if run.run_id in self._runs:
                raise ValueError(f"run_id {run.run_id!r} already exists with a different idempotency_key")
            self._runs[run.run_id] = run.model_copy(deep=True)
            self._run_keys[run.idempotency_key] = run.run_id
            return run.model_copy(deep=True), True

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else run.model_copy(deep=True)

    def list_runs(self, states: Sequence[RunState] | None = None, limit: int = 50) -> list[Run]:
        with self._lock:
            wanted = None if states is None else set(states)
            runs = [r.model_copy(deep=True) for r in reversed(self._runs.values()) if wanted is None or r.state in wanted]
            return runs[:limit]

    def list_active_runs(self) -> list[Run]:
        with self._lock:
            return [r.model_copy(deep=True) for r in self._runs.values() if r.state not in TERMINAL_RUN_STATES]

    def save_checkpoint(self, run_id: str, state: RunState, checkpoint: dict[str, object], step: RunStep) -> Run:
        with self._write():
            run = self._require_run(run_id)
            if run.state == state and any(s.name == step.name for s in run.steps):
                return run.model_copy(deep=True)
            checkpoints = {**run.checkpoints, state: json.loads(json.dumps(checkpoint, default=_json_default))}
            updated = run.model_copy(
                update={"state": state, "checkpoints": checkpoints, "steps": [*run.steps, step], "updated_at": utc_now()}
            )
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    def append_step(self, run_id: str, step: RunStep) -> Run:
        with self._write():
            run = self._require_run(run_id)
            updated = run.model_copy(update={"steps": [*run.steps, step], "updated_at": utc_now()})
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    def set_pending_confirmation(self, run_id: str, request: ConfirmationRequest | None) -> Run:
        with self._write():
            run = self._require_run(run_id)
            updated = run.model_copy(update={"pending_confirmation": request, "updated_at": utc_now()})
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Decision traces and feedback
    # ------------------------------------------------------------------

    def save_trace(self, trace: Trace) -> Trace:
        with self._write():
            self._traces[trace.trace_id] = trace.model_copy(deep=True)
            return trace.model_copy(deep=True)

    def get_trace(self, trace_id: str) -> Trace | None:
        with self._lock:
            trace = self._traces.get(trace_id)
            return None if trace is None else trace.model_copy(deep=True)

    def get_trace_by_run(self, run_id: str) -> Trace | None:
        with self._lock:
            matches = [t for t in self._traces.values() if t.run_id == run_id]
            if not matches:
                return None
            return max(matches, key=lambda t: t.created_at).model_copy(deep=True)

    def list_traces(self, is_seed: bool | None = None, limit: int = 100) -> list[Trace]:
        with self._lock:
            traces = [t for t in self._traces.values() if is_seed is None or t.is_seed == is_seed]
            traces.sort(key=lambda t: t.created_at, reverse=True)
            return [t.model_copy(deep=True) for t in traces[:limit]]

    def search_traces(
        self, query_vec: Sequence[float], trigger: TriggerType, k: int = 3
    ) -> tuple[list[RetrievedTrace], bool]:
        """Local numpy cosine search over accepted traces with the same trigger; never degraded here
        because JsonStore has no Atlas side to fall back from."""
        if k <= 0:
            return [], False
        query = np.asarray(query_vec, dtype=float)
        query_norm = float(np.linalg.norm(query))
        with self._lock:
            candidates = [
                t for t in self._traces.values()
                if t.final_outcome == "accepted" and t.trigger == trigger and len(t.embedding) == query.shape[0]
            ]
            scored: list[tuple[float, Trace]] = []
            for trace in candidates:
                vec = np.asarray(trace.embedding, dtype=float)
                denom = query_norm * float(np.linalg.norm(vec))
                similarity = float(query @ vec / denom) if denom > 0 else 0.0
                scored.append((similarity, trace))
        scored.sort(key=lambda pair: (-pair[0], pair[1].trace_id))
        results = [
            RetrievedTrace(
                trace_id=t.trace_id,
                similarity=round(sim, 6),
                day_index=t.day_index,
                dish_name=t.plan.dish_name if t.plan else "",
                is_seed=t.is_seed,
            )
            for sim, t in scored[:k]
        ]
        return results, False

    def save_feedback(self, trace_id: str, outcome: FeedbackOutcome, op_id: str, source: FeedbackSource) -> Feedback:
        with self._write():
            existing = self._feedback.get(op_id)
            if existing is not None:
                return existing.model_copy(deep=True)
            trace = self._traces.get(trace_id)
            if trace is None:
                raise KeyError(f"Unknown trace_id {trace_id!r}")
            feedback = Feedback(op_id=op_id, trace_id=trace_id, outcome=outcome, source=source)
            self._feedback[op_id] = feedback
            self._traces[trace_id] = trace.model_copy(update={"final_outcome": outcome})
            return feedback.model_copy(deep=True)

    def list_feedback(self, trace_id: str | None = None) -> list[Feedback]:
        """Return feedback in insertion order, optionally for one trace (JsonStore extra, for tests and UI)."""
        with self._lock:
            return [f.model_copy(deep=True) for f in self._feedback.values() if trace_id is None or f.trace_id == trace_id]

    # ------------------------------------------------------------------
    # Shopping list
    # ------------------------------------------------------------------

    def add_to_shopping_list(self, item: ShoppingListItem) -> ShoppingListItem:
        with self._write():
            name = rules.normalize_name(item.item_name)
            existing = self._shopping.get(name)
            if existing is None:
                stored = item.model_copy(update={"item_name": name, "run_ids": list(dict.fromkeys(item.run_ids))})
                self._shopping[name] = stored
                return stored.model_copy(deep=True)
            if existing.unit != item.unit:
                raise InvariantViolation(f"Shopping list unit mismatch for {name}: {existing.unit} vs {item.unit}")
            if item.run_ids and any(rid in existing.run_ids for rid in item.run_ids):
                return existing.model_copy(deep=True)
            updated = existing.model_copy(
                update={
                    "quantity": _round(existing.quantity + item.quantity),
                    "run_ids": list(dict.fromkeys([*existing.run_ids, *item.run_ids])),
                    "reason": existing.reason or item.reason,
                    "updated_at": utc_now(),
                }
            )
            self._shopping[name] = updated
            return updated.model_copy(deep=True)

    def list_shopping_list(self) -> list[ShoppingListItem]:
        with self._lock:
            return [self._shopping[name].model_copy(deep=True) for name in sorted(self._shopping)]

    def remove_from_shopping_list(self, item_name: str) -> bool:
        with self._write():
            return self._shopping.pop(rules.normalize_name(item_name), None) is not None

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------

    def get_search_cache(self, key: str) -> SearchCacheEntry | None:
        with self._lock:
            entry = self._search_cache.get(key)
            return None if entry is None else entry.model_copy(deep=True)

    def put_search_cache(self, entry: SearchCacheEntry) -> None:
        with self._write():
            self._search_cache[entry.key] = entry.model_copy(deep=True)

    def get_plan_cache(self, key: str) -> PlanCacheEntry | None:
        with self._lock:
            entry = self._plan_cache.get(key)
            return None if entry is None else entry.model_copy(deep=True)

    def put_plan_cache(self, entry: PlanCacheEntry) -> None:
        with self._write():
            self._plan_cache[entry.key] = entry.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def enqueue_command(self, command: Command) -> Command:
        with self._write():
            existing = self._commands.get(command.command_id)
            if existing is not None:
                return existing.model_copy(deep=True)
            self._commands[command.command_id] = command.model_copy(deep=True)
            return command.model_copy(deep=True)

    def poll_commands(self, limit: int = 20) -> list[Command]:
        with self._lock:
            pending = [c for c in self._commands.values() if c.status == "pending"]
            return [c.model_copy(deep=True) for c in pending[:limit]]

    def get_command(self, command_id: str) -> Command | None:
        with self._lock:
            command = self._commands.get(command_id)
            return None if command is None else command.model_copy(deep=True)

    def mark_command(self, command_id: str, status: CommandStatus, error: str | None = None) -> Command:
        with self._write():
            command = self._commands.get(command_id)
            if command is None:
                raise KeyError(f"Unknown command_id {command_id!r}")
            updated = command.model_copy(update={"status": status, "error": error, "processed_at": utc_now()})
            self._commands[command_id] = updated
            return updated.model_copy(deep=True)


def _json_default(value: Any) -> Any:
    """Serialise pydantic models, dates and sets inside checkpoints."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"Checkpoint value of type {type(value).__name__} is not JSON serialisable")
