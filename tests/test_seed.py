# Tests for pantry.seed: expiry offset -> date conversion, date label import, seed traces and missing files.
# Offline only: JsonStore.memory(), FakeLLM and the fixture under tests/fixtures/seed_data.
from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from pantry import rules
from pantry.llm import FakeLLM
from pantry.models import Ingredient
from pantry.seed import (
    build_ingredients,
    build_seed_traces,
    expiry_from_offset,
    main,
    parse_date_labels,
    seed_snapshot_text,
    seed_store,
)
from pantry.store.json_store import JsonStore
from tests.conftest import START


def test_expiry_offset_converts_to_date_relative_to_start() -> None:
    assert expiry_from_offset(START, 3) == date(2026, 11, 17)
    assert expiry_from_offset(START, 0) == START
    assert expiry_from_offset(START, -1) == date(2026, 11, 13)
    assert expiry_from_offset(START, 365) == date(2027, 11, 14)
    assert expiry_from_offset(START, None) is None
    with pytest.raises(ValueError):
        expiry_from_offset(START, 1.5)  # type: ignore[arg-type]


def test_build_ingredients_follows_demo_start_date() -> None:
    rows = [{"item_name": " Milk ", "quantity": 1000, "unit": "ml", "expiry_offset_days": 3, "safety_stock": 200}]
    other_start = START + timedelta(days=10)
    item = build_ingredients(rows, other_start)[0]
    assert item.item_name == "milk"
    assert item.expiry_date == other_start + timedelta(days=3)
    assert item.allergen_tags == ["dairy"]  # defaulted from ALLERGEN_MAP
    assert item.date_label == "none" and item.packaging_unverified is False
    assert item.priority_score == rules.priority_score(item, other_start)


def test_parse_date_labels_accepts_both_layouts() -> None:
    by_item, unverified = parse_date_labels(
        {"label_types": {"Milk": "use_by", "egg": "best_before"}, "unverified_packaging": ["Eggs"]}
    )
    assert by_item == {"milk": "use_by", "eggs": "best_before"}
    assert unverified == {"eggs"}
    by_label, none = parse_date_labels({"label_types": {"use_by": ["milk"], "estimated_quality": ["flour"]}})
    assert by_label == {"milk": "use_by", "flour": "estimated_quality"}
    assert none == set()
    with pytest.raises(ValueError):
        parse_date_labels({"label_types": {"milk": "eat_soon"}})


def test_label_file_overrides_item_field() -> None:
    rows = [{"item_name": "milk", "quantity": 1, "unit": "ml", "date_label": "best_before"}]
    assert build_ingredients(rows, START)[0].date_label == "best_before"
    assert build_ingredients(rows, START, {"milk": "use_by"}, {"milk"})[0].date_label == "use_by"
    assert build_ingredients(rows, START, {"milk": "use_by"}, {"milk"})[0].packaging_unverified is True


def test_seed_store_imports_fixture(seed_dir: Path) -> None:
    store = JsonStore.memory(start_date=START)
    report = seed_store(store, FakeLLM(), seed_dir, START)
    assert report.missing_files == [] and report.warnings == []
    assert (report.items, report.traces, report.unverified) == (5, 3, 1)

    milk = store.get_item("milk")
    assert milk.expiry_date == date(2026, 11, 17)
    assert milk.date_label == "use_by" and milk.packaging_unverified is False
    eggs = store.get_item("eggs")
    assert eggs.date_label == "best_before" and eggs.packaging_unverified is True
    assert store.get_item("sugar").date_label == "none"

    profile = store.get_profile()
    assert profile.allergens == ["nuts"] and profile.constraints.max_minutes == 25
    assert store.get_today() == START
    assert store.check_invariants() == []


def test_seed_traces_are_marked_and_embedded_with_runtime_snapshot(seed_dir: Path) -> None:
    store = JsonStore.memory(start_date=START)
    llm = FakeLLM()
    seed_store(store, llm, seed_dir, START)
    traces = {t.trace_id: t for t in store.list_traces()}
    assert set(traces) == {"seed-001", "seed-002", "seed-003"}
    assert all(t.is_seed for t in traces.values())
    assert all(len(t.embedding) == 1536 for t in traces.values())

    # snapshot_text equals what the runtime Sentinel builds for the same pantry state.
    seeded_items = [i for i in store.list_items()]
    shifted = START + timedelta(days=1)  # milk at 2 days left
    runtime = rules.build_snapshot_text("EXPIRY_IMMINENT", seeded_items, shifted, 25, ["nuts"])
    assert traces["seed-001"].snapshot_text == runtime
    assert traces["seed-001"].embedding == llm.embed([runtime])[0]
    assert traces["seed-002"].snapshot_text.startswith("trigger=STOCK_DEPLETED; expiring=none; depleted=eggs;")
    assert "expiring=milk(1d)" in traces["seed-003"].snapshot_text
    assert traces["seed-002"].plan.servings == 2  # defaulted
    assert traces["seed-003"].final_outcome == "rejected"


def test_seed_is_idempotent(seed_dir: Path) -> None:
    store = JsonStore.memory(start_date=START)
    seed_store(store, FakeLLM(), seed_dir, START)
    first = ([i.model_dump(exclude={"last_updated"}) for i in store.list_items()], len(store.list_traces()))
    seed_store(store, FakeLLM(), seed_dir, START)
    second = ([i.model_dump(exclude={"last_updated"}) for i in store.list_items()], len(store.list_traces()))
    assert first == second
    assert store.list_events() == []


def test_missing_optional_files_are_reported(tmp_path: Path, seed_dir: Path) -> None:
    shutil.copy(seed_dir / "pantry_items.json", tmp_path / "pantry_items.json")
    store = JsonStore.memory(start_date=START)
    report = seed_store(store, FakeLLM(), tmp_path, START)
    assert sorted(Path(p).name for p in report.missing_files) == ["date_labels.json", "seed_traces.json"]
    assert report.items == 5 and report.traces == 0
    assert all(item.date_label == "none" for item in store.list_items())


def test_missing_pantry_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(FileNotFoundError):
        seed_store(JsonStore.memory(start_date=START), FakeLLM(), tmp_path, START)
    assert main(["--data-dir", str(tmp_path), "--backend", "json", "--fake-llm"]) == 1
    assert "missing required seed file" in capsys.readouterr().out


def test_unknown_label_item_is_a_warning(tmp_path: Path, seed_dir: Path) -> None:
    shutil.copy(seed_dir / "pantry_items.json", tmp_path / "pantry_items.json")
    (tmp_path / "date_labels.json").write_text(json.dumps({"label_types": {"cheese": "use_by"}}), encoding="utf-8")
    report = seed_store(JsonStore.memory(start_date=START), FakeLLM(), tmp_path, START)
    assert any("cheese" in w for w in report.warnings)


def test_seed_snapshot_accepts_text_expiring_and_int_constraints() -> None:
    text = seed_snapshot_text(
        {"trigger": "EXPIRY_IMMINENT", "expiring": ["milk(2d)"], "have": ["eggs"], "constraints": 20, "allergens": []},
        START,
    )
    assert text == "trigger=EXPIRY_IMMINENT; expiring=milk(2d); have=eggs; constraint=20min; allergens=none"
    with pytest.raises(ValueError):
        build_seed_traces([{"trigger": "EXPIRY_IMMINENT", "expiring": ["milk soon"], "plan": {}}], FakeLLM(), START)


def test_ingredient_model_round_trip_from_seed(seed_dir: Path) -> None:
    rows = json.loads((seed_dir / "pantry_items.json").read_text(encoding="utf-8"))["items"]
    items = build_ingredients(rows, START)
    assert all(isinstance(i, Ingredient) for i in items)
    assert {i.item_name: i.expiry_date for i in items}["sugar"] == date(2027, 11, 14)
