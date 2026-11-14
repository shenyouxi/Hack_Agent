# Tests for JsonStore (ledger invariants, idempotency, reserve/commit/undo, clock, traces, feedback) and FakeLLM.
# Everything runs offline: in-memory or tmp_path stores, no network and no .env.
from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import BaseModel

from pantry import rules
from pantry.llm import FakeLLM, LLMOutputError, OpenNextLLM, prompt_key
from pantry.models import (
    BusEvent,
    Command,
    Ingredient,
    InventoryEvent,
    PlannerOutput,
    Run,
    RunStep,
    ShoppingListItem,
    Trace,
)
from pantry.store.base import InvariantViolation
from pantry.store.json_store import JsonStore

START = date(2026, 11, 14)


def seed_items() -> list[Ingredient]:
    def item(name: str, qty: float, unit: str, days: int, safety: float) -> Ingredient:
        return Ingredient(
            item_name=name, quantity=qty, unit=unit, expiry_date=START + timedelta(days=days), safety_stock=safety
        )

    return [
        item("milk", 1000, "ml", 3, 200),
        item("eggs", 10, "pcs", 14, 2),
        item("flour", 1000, "g", 180, 200),
        item("butter", 250, "g", 20, 50),
        item("sugar", 500, "g", 365, 100),
    ]


@pytest.fixture
def store() -> JsonStore:
    s = JsonStore.memory(start_date=START)
    s.seed_pantry(seed_items())
    return s


def ev(item: str, kind: str, delta: float, unit: str, run_id: str | None = "run-1", event_id: str = "") -> InventoryEvent:
    return InventoryEvent(event_id=event_id, item_name=item, type=kind, delta=delta, unit=unit, run_id=run_id)


def make_run(run_id: str = "run-1", key: str = "EXPIRY_IMMINENT:milk:2026-11-14") -> Run:
    return Run(run_id=run_id, trigger="EXPIRY_IMMINENT", item_name="milk", idempotency_key=key)


def qty(store: JsonStore, name: str) -> tuple[float, float]:
    item = store.get_item(name)
    assert item is not None
    return item.quantity, item.reserved


# --- invariants ---------------------------------------------------------------


def test_seeded_store_is_consistent(store):
    assert store.check_invariants() == []
    assert store.get_today() == START
    assert store.replay_ledger()["milk"] == (1000, 0)


def test_negative_quantity_is_rejected_and_rolled_back(store):
    with pytest.raises(InvariantViolation):
        store.apply_events([ev("flour", "ADJUSTED", -100, "g"), ev("milk", "CONSUMED", -1500, "ml")], "k1")
    assert qty(store, "flour") == (1000, 0)
    assert qty(store, "milk") == (1000, 0)
    assert store.list_events() == []
    assert store.check_invariants() == []


def test_reserved_above_quantity_is_rejected(store):
    with pytest.raises(InvariantViolation):
        store.apply_events([ev("butter", "RESERVED", 300, "g")], "k1")
    store.apply_events([ev("butter", "RESERVED", 200, "g")], "k2")
    with pytest.raises(InvariantViolation):
        store.apply_events([ev("butter", "CONSUMED", -100, "g")], "k3")
    assert qty(store, "butter") == (250, 200)


def test_releasing_more_than_reserved_is_rejected(store):
    with pytest.raises(InvariantViolation):
        store.apply_events([ev("milk", "RELEASED", 10, "ml")], "k1")


def test_unit_mismatch_is_rejected(store):
    with pytest.raises(InvariantViolation, match="Unit mismatch"):
        store.apply_events([ev("milk", "RESERVED", 100, "g")], "k1")


def test_unknown_item_and_wrong_sign_are_rejected(store):
    with pytest.raises(InvariantViolation):
        store.apply_events([ev("heavy cream", "RESERVED", 100, "ml")], "k1")
    with pytest.raises(InvariantViolation):
        store.apply_events([ev("milk", "CONSUMED", 100, "ml")], "k2")
    with pytest.raises(InvariantViolation):
        store.apply_events([ev("milk", "RESERVED", -100, "ml")], "k3")


def test_idempotency_key_takes_effect_once(store):
    first = store.apply_events([ev("milk", "RESERVED", 300, "ml")], "reserve:run-1")
    second = store.apply_events([ev("milk", "RESERVED", 300, "ml")], "reserve:run-1")
    assert [e.event_id for e in first] == [e.event_id for e in second]
    assert qty(store, "milk") == (1000, 300)
    assert len(store.list_events()) == 1


def test_alias_names_resolve_to_canonical_item(store):
    store.apply_events([ev("Whole Milk", "RESERVED", 100, "ml")], "k1")
    assert qty(store, "milk") == (1000, 100)
    assert store.list_events(item_name="whole milk")[0].item_name == "milk"


# --- reserve, commit, undo -----------------------------------------------------


def test_reserve_commit_and_commit_is_once(store):
    store.create_run(make_run())
    store.apply_events([ev("milk", "RESERVED", 300, "ml"), ev("flour", "RESERVED", 200, "g")], "reserve:run-1")
    written = store.commit_run("run-1")
    assert sorted((e.item_name, e.type) for e in written) == [
        ("flour", "CONSUMED"), ("flour", "RELEASED"), ("milk", "CONSUMED"), ("milk", "RELEASED")
    ]
    assert all(e.estimated and e.note == "assumed cooked in the demo" for e in written)
    assert qty(store, "milk") == (700, 0)
    assert qty(store, "flour") == (800, 0)
    assert store.get_run("run-1").committed
    assert store.commit_run("run-1") == []
    assert qty(store, "milk") == (700, 0)
    assert store.check_invariants() == []


def test_undo_restores_inventory_and_is_idempotent(store):
    store.create_run(make_run())
    store.apply_events([ev("milk", "RESERVED", 300, "ml")], "reserve:run-1")
    reversals = store.reverse_run_events("run-1")
    assert len(reversals) == 1 and reversals[0].type == "RELEASED"
    assert reversals[0].reverses_event_id == store.list_events()[0].event_id
    assert qty(store, "milk") == (1000, 0)
    assert store.reverse_run_events("run-1") == []
    assert store.commit_run("run-1") == []
    assert qty(store, "milk") == (1000, 0)
    assert store.check_invariants() == []


def test_undo_after_commit_restores_quantity(store):
    store.create_run(make_run())
    store.apply_events([ev("milk", "RESERVED", 300, "ml")], "reserve:run-1")
    store.commit_run("run-1")
    store.reverse_run_events("run-1")
    assert qty(store, "milk") == (1000, 0)
    assert store.check_invariants() == []


def test_uncommitted_applied_runs(store):
    store.create_run(make_run())
    store.save_checkpoint("run-1", "APPLIED", {}, RunStep(name="writeback", status="ok"))
    assert [r.run_id for r in store.list_uncommitted_applied_runs()] == ["run-1"]
    store.commit_run("run-1")
    assert store.list_uncommitted_applied_runs() == []


def test_ledger_replay_matches_after_mixed_events(store):
    store.create_run(make_run())
    store.apply_events([ev("eggs", "RESERVED", 4, "pcs")], "a")
    store.apply_events([ev("eggs", "ADJUSTED", 6, "pcs", run_id=None)], "b")
    store.commit_run("run-1")
    assert qty(store, "eggs") == (12, 0)
    assert store.replay_ledger()["eggs"] == (12, 0)
    assert store.check_invariants() == []


# --- clock and priorities ------------------------------------------------------


def test_advance_day_recomputes_priority_in_same_operation(store):
    assert store.get_item("milk").priority_score == pytest.approx(0.7 * 0.4)
    new_day = store.advance_day()
    assert new_day == START + timedelta(days=1)
    assert store.get_today() == new_day
    assert store.get_profile().today == (START + timedelta(days=1)).isoformat()
    assert store.get_item("milk").priority_score == pytest.approx(0.7 * 0.6)
    store.set_today(START)
    assert store.get_item("milk").priority_score == pytest.approx(0.7 * 0.4)


def test_triggers_fire_from_store_state_after_advance(store):
    assert rules.detect_triggers(store.list_items(), store.get_today()) == []
    store.advance_day()
    events = rules.detect_triggers(store.list_items(), store.get_today())
    assert [(e.trigger_type, e.item_name) for e in events] == [("EXPIRY_IMMINENT", "milk")]


def test_reservation_does_not_trigger_stock_depleted(store):
    store.apply_events([ev("sugar", "RESERVED", 450, "g")], "k1")
    assert rules.detect_triggers(store.list_items(), store.get_today()) == []


# --- bus, runs, commands --------------------------------------------------------


def test_bus_event_dedupe(store):
    event = BusEvent(trigger_type="EXPIRY_IMMINENT", item_name="milk", severity="low",
                     dedupe_key=rules.dedupe_key("EXPIRY_IMMINENT", "milk", START))
    assert store.emit_bus_event(event) is True
    assert store.emit_bus_event(event.model_copy()) is False
    assert len(store.list_bus_events()) == 1
    store.set_bus_event_status(event.dedupe_key, "dispatched")
    assert store.list_bus_events(status="dispatched")[0].dedupe_key == event.dedupe_key


def test_create_run_is_idempotent_by_key(store):
    run, created = store.create_run(make_run("run-1"))
    again, created_again = store.create_run(make_run("run-2"))
    assert created and not created_again
    assert again.run_id == "run-1"
    assert len(store.list_runs()) == 1


def test_save_checkpoint_same_state_and_step_is_noop(store):
    store.create_run(make_run())
    step = RunStep(name="sentinel", status="ok")
    store.save_checkpoint("run-1", "CONTEXT_BUILT", {"snapshot": "x"}, step)
    run = store.save_checkpoint("run-1", "CONTEXT_BUILT", {"snapshot": "y"}, step)
    assert len(run.steps) == 1
    assert run.checkpoints["CONTEXT_BUILT"] == {"snapshot": "x"}
    assert [r.run_id for r in store.list_active_runs()] == ["run-1"]


def test_commands_round_trip(store):
    store.enqueue_command(Command(command_id="c1", type="ADVANCE_CLOCK"))
    store.enqueue_command(Command(command_id="c1", type="ADVANCE_CLOCK"))
    assert [c.command_id for c in store.poll_commands()] == ["c1"]
    store.mark_command("c1", "done")
    assert store.poll_commands() == []
    assert store.get_command("c1").status == "done"


def test_shopping_list_merges_and_is_idempotent_per_run(store):
    store.add_to_shopping_list(ShoppingListItem(item_name="Double Cream", quantity=100, unit="ml", run_ids=["r1"]))
    store.add_to_shopping_list(ShoppingListItem(item_name="heavy cream", quantity=100, unit="ml", run_ids=["r1"]))
    merged = store.add_to_shopping_list(ShoppingListItem(item_name="heavy cream", quantity=50, unit="ml", run_ids=["r2"]))
    assert merged.quantity == 150 and merged.run_ids == ["r1", "r2"]
    with pytest.raises(InvariantViolation):
        store.add_to_shopping_list(ShoppingListItem(item_name="heavy cream", quantity=1, unit="g", run_ids=["r3"]))


# --- traces, search, feedback ---------------------------------------------------


def make_trace(trace_id: str, snapshot: str, llm: FakeLLM, outcome: str = "accepted", **kw) -> Trace:
    plan = PlannerOutput(dish_name=f"dish {trace_id}", servings=2, est_minutes=20,
                         ingredients=[{"name": "milk", "amount": 200, "unit": "ml"}])
    return Trace(trace_id=trace_id, day_index=1, trigger=kw.pop("trigger", "EXPIRY_IMMINENT"), snapshot_text=snapshot,
                 embedding=llm.embed([snapshot])[0], plan=plan, final_outcome=outcome, **kw)


def test_new_trace_is_immediately_searchable(store):
    llm = FakeLLM()
    snapshot = "trigger=EXPIRY_IMMINENT; expiring=milk(2d); have=butter,eggs,flour,sugar; constraint=25min; allergens=nuts"
    store.save_trace(make_trace("seed-1", "trigger=EXPIRY_IMMINENT; expiring=bananas(1d); have=oats", llm, is_seed=True))
    store.save_trace(make_trace("t-new", snapshot, llm))
    results, degraded = store.search_traces(llm.embed([snapshot])[0], "EXPIRY_IMMINENT", k=3)
    assert degraded is False
    assert results[0].trace_id == "t-new"
    assert results[0].similarity == pytest.approx(1.0)
    assert results[0].dish_name == "dish t-new"
    assert results[0].similarity >= results[-1].similarity


def test_search_filters_outcome_and_trigger(store):
    llm = FakeLLM()
    text = "expiring=milk(2d)"
    store.save_trace(make_trace("rejected", text, llm, outcome="rejected"))
    store.save_trace(make_trace("pending", text, llm, outcome="pending_confirmation"))
    store.save_trace(make_trace("other-trigger", text, llm, trigger="STOCK_DEPLETED"))
    store.save_trace(make_trace("ok", text, llm))
    results, _ = store.search_traces(llm.embed([text])[0], "EXPIRY_IMMINENT", k=5)
    assert [r.trace_id for r in results] == ["ok"]
    assert store.search_traces(llm.embed([text])[0], "EXPIRY_IMMINENT", k=0) == ([], False)


def test_save_feedback_is_idempotent_and_updates_outcome(store):
    llm = FakeLLM()
    store.save_trace(make_trace("t1", "x", llm, outcome="pending_confirmation"))
    fb = store.save_feedback("t1", "rejected", op_id="op-1", source="user")
    repeat = store.save_feedback("t1", "accepted", op_id="op-1", source="user")
    assert repeat.outcome == "rejected" and repeat.created_at == fb.created_at
    assert store.get_trace("t1").final_outcome == "rejected"
    store.save_feedback("t1", "accepted", op_id="auto:t1", source="auto")
    assert store.get_trace("t1").final_outcome == "accepted"
    assert len(store.list_feedback("t1")) == 2


# --- reset and persistence ---------------------------------------------------


def test_reset_demo_is_idempotent_and_keeps_seeds_and_caches(store):
    llm = FakeLLM()
    store.save_trace(make_trace("seed", "a", llm, is_seed=True))
    store.save_trace(make_trace("live", "b", llm))
    store.create_run(make_run())
    store.apply_events([ev("milk", "RESERVED", 300, "ml")], "k")
    store.commit_run("run-1")
    store.advance_day(2)
    store.reset_demo(START)
    first = (store.list_items(), [t.trace_id for t in store.list_traces()], store.get_today())
    store.reset_demo(START)
    assert (store.list_items(), [t.trace_id for t in store.list_traces()], store.get_today()) == first
    assert first[1] == ["seed"] and first[2] == START
    assert qty(store, "milk") == (1000, 0)
    assert store.list_events() == [] and store.list_runs() == []
    assert store.check_invariants() == []


def test_file_backed_store_persists_state(tmp_path):
    s = JsonStore(data_dir=tmp_path, start_date=START)
    s.seed_pantry(seed_items())
    s.create_run(make_run())
    s.apply_events([ev("milk", "RESERVED", 300, "ml")], "k")
    s.advance_day()
    reopened = JsonStore(data_dir=tmp_path)
    assert reopened.path == tmp_path / "pantry_store.json"
    assert qty(reopened, "milk") == (1000, 300)
    assert reopened.get_today() == START + timedelta(days=1)
    assert reopened.get_run("run-1") is not None
    assert reopened.check_invariants() == []
    reopened.apply_events([ev("milk", "RESERVED", 300, "ml")], "k")
    assert qty(reopened, "milk") == (1000, 300)


def test_three_rounds_keep_quantities_consistent(store):
    expected_milk = 1000.0
    for round_index in range(3):
        run_id = f"run-{round_index}"
        start_state = {i.item_name: (i.quantity, i.reserved) for i in store.list_items()}
        store.create_run(make_run(run_id, key=f"k-{round_index}"))
        store.apply_events([ev("milk", "RESERVED", 200, "ml", run_id=run_id)], f"reserve:{run_id}")
        store.save_checkpoint(run_id, "APPLIED", {}, RunStep(name="writeback", status="ok"))
        for run in store.list_uncommitted_applied_runs():
            store.commit_run(run.run_id)
        store.advance_day()
        expected_milk -= 200
        assert qty(store, "milk") == (expected_milk, 0)
        assert start_state["milk"] == (expected_milk + 200, 0)
        replay = store.replay_ledger()
        assert all(replay[i.item_name] == (i.quantity, i.reserved) for i in store.list_items())
        assert store.check_invariants() == []


# --- FakeLLM --------------------------------------------------------------------


class Answer(BaseModel):
    dish_name: str
    minutes: int


def test_fake_llm_embedding_is_deterministic_and_normalised():
    a, b = FakeLLM(), FakeLLM()
    text = "trigger=EXPIRY_IMMINENT; expiring=milk(2d)"
    va, vb = a.embed([text])[0], b.embed([text])[0]
    assert va == vb
    assert len(va) == 1536
    assert sum(v * v for v in va) == pytest.approx(1.0)
    assert a.embed([text, "something else"])[1] != va
    assert len(FakeLLM(embed_dim=8).embed(["x"])[0]) == 8


def test_fake_llm_canned_json_by_prompt_and_schema():
    llm = FakeLLM()
    llm.register("Answer", {"dish_name": "default", "minutes": 10})
    llm.register_prompt("sys", "user", {"dish_name": "pancakes", "minutes": 20})
    assert llm.chat_json("sys", "user", Answer) == {"dish_name": "pancakes", "minutes": 20}
    assert llm.chat_json("sys", "other", Answer) == {"dish_name": "default", "minutes": 10}
    assert llm.chat_json("sys", "user", Answer) == llm.chat_json("sys", "user", Answer)
    assert prompt_key("sys", "user") == prompt_key("sys", "user")


def test_fake_llm_retries_once_then_raises():
    llm = FakeLLM()
    llm.register("Answer", "not json", {"dish_name": "soup", "minutes": 15})
    assert llm.chat_json("s", "u", Answer) == {"dish_name": "soup", "minutes": 15}
    assert [c["attempt"] for c in llm.calls] == [1, 2]

    bad = FakeLLM()
    bad.register("Answer", {"dish_name": "soup"})
    with pytest.raises(LLMOutputError) as info:
        bad.chat_json("s", "u", Answer)
    assert info.value.attempts == 2 and len(bad.calls) == 2


# --- OpenNextLLM with a stub client (no network) ------------------------------------


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubClient:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.requests: list[dict] = []
        self.chat = _Obj(completions=_Obj(create=self._create))
        self.embeddings = _Obj(create=self._embed)

    def _create(self, **kw):
        self.requests.append(kw)
        return _Obj(choices=[_Obj(message=_Obj(content=self.outputs.pop(0)))])

    def _embed(self, **kw):
        self.requests.append(kw)
        return _Obj(data=[_Obj(index=1, embedding=[0.0, 1.0]), _Obj(index=0, embedding=[1.0, 0.0])])


def _settings():
    from pantry.config import Settings

    return Settings(
        llm_base_url="http://stub", llm_api_key="stub", llm_model="stub-model", mongodb_uri="", mongodb_db="",
        linkup_api_key="", tavily_api_key="", embed_provider="openai", embed_model="text-embedding-3-small",
        mlflow_tracking_uri="", databricks_host="", databricks_token="", cache_first=False, store_backend="json",
        tick_seconds=2.0, demo_start_date=START,
    )


def test_opennext_llm_uses_json_mode_and_retries_with_error():
    client = _StubClient(['{"dish_name": "x"}', '```json\n{"dish_name": "x", "minutes": 5}\n```'])
    llm = OpenNextLLM(_settings(), client=client)
    assert llm.chat_json("sys", "user", Answer) == {"dish_name": "x", "minutes": 5}
    first, retry = client.requests
    assert first["temperature"] == 0 and first["response_format"] == {"type": "json_object"}
    assert first["model"] == "stub-model"
    assert "invalid" in retry["messages"][-1]["content"]
    assert llm.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert client.requests[-1]["model"] == "text-embedding-3-small"
