# Three consecutive demo rounds (reserve, ADVANCE_CLOCK commit, undo) with FakeLLM and no network (patch A section 4).
# Runs on JsonStore.memory(); the same test runs on MongoStore when RUN_INTEGRATION=1 (database name ends in _test).
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from pantry import rules
from pantry.llm import FakeLLM
from pantry.models import InventoryEvent, PlannerOutput, RetrievedTrace, Run, RunStep, Trace
from pantry.seed import seed_store
from pantry.store.base import Store
from pantry.store.json_store import JsonStore
from tests.conftest import START

TRIGGER = "EXPIRY_IMMINENT"
Ledger = dict[str, tuple[float, float]]

ROUND_PLANS: list[dict] = [
    {
        "dish_name": "milk pancakes",
        "servings": 2,
        "est_minutes": 20,
        "ingredients": [
            {"name": "milk", "amount": 300, "unit": "ml"},
            {"name": "egg", "amount": 2, "unit": "pcs"},  # alias of eggs
            {"name": "flour", "amount": 200, "unit": "g"},
            {"name": "butter", "amount": 20, "unit": "g"},
            {"name": "sugar", "amount": 30, "unit": "g"},
            {"name": "heavy cream", "amount": 100, "unit": "ml"},  # MISSING gap, never reserved
        ],
    },
    {
        "dish_name": "baked custard",
        "servings": 2,
        "est_minutes": 25,
        "ingredients": [
            {"name": "milk", "amount": 400, "unit": "ml"},
            {"name": "eggs", "amount": 3, "unit": "pcs"},
            {"name": "sugar", "amount": 50, "unit": "g"},
        ],
    },
    {
        "dish_name": "milk crepes",
        "servings": 2,
        "est_minutes": 15,
        "ingredients": [
            {"name": "milk", "amount": 300, "unit": "ml"},
            {"name": "eggs", "amount": 2, "unit": "pcs"},
            {"name": "flour", "amount": 100, "unit": "g"},
        ],
    },
]

# Hand-computed (quantity, reserved) at the end of each round.
EXPECTED_END: list[Ledger] = [
    {"milk": (1000, 300), "eggs": (10, 2), "flour": (1000, 200), "butter": (250, 20), "sugar": (500, 30)},
    {"milk": (700, 400), "eggs": (8, 3), "flour": (800, 0), "butter": (230, 0), "sugar": (470, 50)},
    {"milk": (300, 0), "eggs": (5, 0), "flour": (800, 0), "butter": (230, 0), "sugar": (420, 0)},  # undone
]


@dataclass
class RoundResult:
    run_id: str
    trace_id: str
    retrieved: list[RetrievedTrace]
    atlas_unavailable: bool


def stored(store: Store) -> Ledger:
    return {item.item_name: (item.quantity, item.reserved) for item in store.list_items()}


def assert_consistent(store: Store) -> None:
    assert store.check_invariants() == []
    replay = store.replay_ledger()
    current = stored(store)
    assert set(replay) == set(current)
    for name, (quantity, reserved) in current.items():
        assert replay[name] == (quantity, reserved), name


def advance_clock(store: Store) -> None:
    """What the worker does for ADVANCE_CLOCK: commit every applied, uncommitted run, then advance one day."""
    for run in store.list_uncommitted_applied_runs():
        store.commit_run(run.run_id)
    store.advance_day(1)


def step(name: str) -> RunStep:
    return RunStep(name=name, status="ok")


def run_round(store: Store, llm: FakeLLM, number: int, *, undo: bool = False) -> RoundResult:
    """Sentinel -> retrieve -> plan -> reserve -> record for the milk EXPIRY_IMMINENT trigger of today."""
    today = store.get_today()
    items = store.list_items()
    profile = store.get_profile()
    bus = [e for e in rules.detect_triggers(items, today) if e.trigger_type == TRIGGER and e.item_name == "milk"]
    assert len(bus) == 1
    assert store.emit_bus_event(bus[0]) is True
    assert store.emit_bus_event(bus[0]) is False  # dedupe_key is unique

    run_id = f"run-{number}"
    run, created = store.create_run(
        Run(run_id=run_id, trigger=TRIGGER, item_name="milk", idempotency_key=bus[0].dedupe_key)
    )
    assert created and run.run_id == run_id
    assert store.create_run(run.model_copy(update={"run_id": f"{run_id}-dup"}))[1] is False

    snapshot = rules.build_snapshot_text(TRIGGER, items, today, profile.constraints.max_minutes, profile.allergens)
    query_vec = llm.embed([snapshot])[0]
    retrieved, atlas_unavailable = store.search_traces(query_vec, TRIGGER, k=3)
    store.save_checkpoint(run_id, "SEEDS_RETRIEVED", {"retrieved": retrieved}, step("seeds_retrieved"))

    llm.register("PlannerOutput", ROUND_PLANS[number - 1])
    plan = PlannerOutput.model_validate(llm.chat_json("planner system", snapshot, PlannerOutput))
    store.save_checkpoint(run_id, "PLANNED", {"plan": plan}, step("planned"))

    reservations = []
    for ingredient in plan.ingredients:
        item = store.get_item(ingredient.name)
        if item is None:
            continue  # MISSING gap: goes to the Resolver, never to the ledger
        reservations.append(
            InventoryEvent(
                event_id="", item_name=item.item_name, type="RESERVED", delta=ingredient.amount,
                unit=item.unit, estimated=True, run_id=run_id,
            )
        )
    key = f"reserve:{run_id}"
    first = store.apply_events(reservations, key)
    again = store.apply_events(reservations, key)  # same idempotency key takes effect once
    assert [e.event_id for e in again] == [e.event_id for e in first]
    store.save_checkpoint(run_id, "APPLIED", {"events": [e.event_id for e in first]}, step("applied"))

    trace_id = f"trace-{run_id}"
    store.save_trace(
        Trace(
            trace_id=trace_id, run_id=run_id, day_index=(today - START).days, trigger=TRIGGER,
            snapshot_text=snapshot, embedding=query_vec, plan=plan, retrieved_traces=retrieved,
            final_outcome="accepted", degraded=atlas_unavailable,
        )
    )
    store.save_feedback(trace_id, "accepted", f"auto:{trace_id}", "auto")

    if undo:
        reversals = store.reverse_run_events(run_id)
        assert len(reversals) == len(first)
        assert store.reverse_run_events(run_id) == []  # undo is idempotent
        store.save_feedback(trace_id, "rejected", f"undo:{run_id}", "user")
        store.save_checkpoint(run_id, "REJECTED", {"undone": True}, step("undone"))
    store.save_checkpoint(run_id, "RECORDED", {"trace_id": trace_id}, step("recorded"))
    return RoundResult(run_id, trace_id, retrieved, atlas_unavailable)


def test_three_rounds_keep_ledger_consistent(any_store: Store, seed_dir: Path) -> None:
    store = any_store
    llm = FakeLLM()
    seed_store(store, llm, seed_dir, START)
    assert_consistent(store)
    baseline = stored(store)

    results: list[RoundResult] = []
    end_state: Ledger | None = None
    for number in (1, 2, 3):
        advance_clock(store)  # ADVANCE_CLOCK: commits the previous round's reservations
        assert store.get_today() == START + timedelta(days=number)
        if end_state is not None:
            committed = {name: (q - r, 0.0) for name, (q, r) in end_state.items()}
            assert stored(store) == committed  # round N+1 starts from round N's end, committed
        assert_consistent(store)

        result = run_round(store, llm, number, undo=(number == 3))
        results.append(result)
        assert_consistent(store)
        end_state = stored(store)
        assert end_state == {k: (float(q), float(r)) for k, (q, r) in EXPECTED_END[number - 1].items()}

    # Round 1 recalls the matching accepted seed; rejected and other-trigger seeds never appear.
    round1_ids = [r.trace_id for r in results[0].retrieved]
    assert round1_ids[0] == "seed-001"
    assert "seed-003" not in round1_ids and "seed-002" not in round1_ids

    # Round 2 retrieves round 1's freshly written decision.
    round2 = {r.trace_id: r for r in results[1].retrieved}
    assert results[0].trace_id in round2
    assert round2[results[0].trace_id].is_seed is False
    assert round2[results[0].trace_id].similarity > 0.8
    if isinstance(store, JsonStore):
        assert all(not r.atlas_unavailable for r in results)

    # The undone round 3 trace is rejected and no longer retrievable.
    assert store.get_trace(results[2].trace_id).final_outcome == "rejected"
    probe = llm.embed([store.get_trace(results[2].trace_id).snapshot_text])[0]
    assert results[2].trace_id not in [r.trace_id for r in store.search_traces(probe, TRIGGER, k=5)[0]]

    # Closing ADVANCE_CLOCK: the undone run consumes nothing; milk is now expired and stops triggering.
    advance_clock(store)
    assert not [e for e in store.list_events(run_id=results[2].run_id) if e.type == "CONSUMED"]
    assert stored(store) == {k: (float(q), float(r)) for k, (q, r) in EXPECTED_END[2].items()}
    milk = store.get_item("milk")
    assert rules.is_expired(milk, store.get_today())
    assert not [e for e in rules.detect_triggers(store.list_items(), store.get_today()) if e.item_name == "milk"]
    assert_consistent(store)

    # reset_demo brings back day 1, keeps seed traces, and is idempotent.
    store.reset_demo(START)
    first_reset = (stored(store), store.get_today(), [t.trace_id for t in store.list_traces()])
    store.reset_demo(START)
    second_reset = (stored(store), store.get_today(), [t.trace_id for t in store.list_traces()])
    assert first_reset == second_reset
    assert first_reset[0] == baseline and first_reset[1] == START
    assert sorted(first_reset[2]) == ["seed-001", "seed-002", "seed-003"]
    assert store.list_runs() == [] and store.list_bus_events() == [] and store.list_events() == []
    assert_consistent(store)
