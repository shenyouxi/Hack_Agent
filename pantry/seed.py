# Seeds a Store from seed_data/: pantry items (offset days -> expiry dates), date labels, profile and seed traces.
# CLI: python -m pantry.seed [--data-dir DIR] [--backend mongo|json] [--fake-llm] [--no-wait-index]
"""Seed data formats (all English, all optional except pantry_items.json).

pantry_items.json  -> {"items": [...], "profile": {...}} or a bare list of items. Each item:
    {"item_name": "milk", "aliases": [], "category": "dairy", "quantity": 1000, "unit": "ml",
     "expiry_offset_days": 3, "safety_stock": 200, "allergen_tags": ["dairy"]}
    expiry_offset_days may be null (no expiry). allergen_tags defaults to the ALLERGEN_MAP groups of the name.
    profile: {"allergens": ["nuts"], "constraints": {"max_minutes": 25}} (defaults from the build prompt).

date_labels.json   -> {"label_types": {...}, "unverified_packaging": ["eggs"]}
    label_types is either item -> label ({"milk": "use_by"}) or label -> items ({"use_by": ["milk"]}).
    Labels: use_by, best_before, estimated_quality, none. It overrides any date_label inside pantry_items.json.

seed_traces.json   -> {"traces": [...]} or a bare list. Each trace uses structured fields:
    {"trace_id": "seed-001", "trigger": "EXPIRY_IMMINENT", "expiring": [{"name": "milk", "days": 2}],
     "depleted": [], "have": ["eggs", "flour"], "constraints": {"max_minutes": 25}, "allergens": ["nuts"],
     "plan": {PlannerOutput}, "final_outcome": "accepted", "day_index": 0}
    expiring also accepts "milk(2d)" strings. snapshot_text is built with rules.build_snapshot_text (the
    runtime function) and embedded with LLM.embed; every seed trace is stored with is_seed=true.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pantry import rules
from pantry.config import get_settings
from pantry.llm import LLM, FakeLLM, LLMConfigError, OpenNextLLM
from pantry.models import Constraints, Ingredient, PlannerOutput, Trace, TriggerType
from pantry.store.base import Store

DEFAULT_DATA_DIR: Path = Path("seed_data")
PANTRY_FILE: str = "pantry_items.json"
LABELS_FILE: str = "date_labels.json"
TRACES_FILE: str = "seed_traces.json"
VALID_LABELS: frozenset[str] = frozenset({"use_by", "best_before", "estimated_quality", "none"})
DEFAULT_ALLERGENS: tuple[str, ...] = ("nuts",)
DEFAULT_MAX_MINUTES: int = 25
DEFAULT_SERVINGS: int = 2
INDEX_WAIT_SECONDS: float = 120.0

_EXPIRING_TEXT = re.compile(r"^\s*(?P<name>.+?)\s*\(\s*(?P<days>-?\d+)\s*d\s*\)\s*$")


@dataclass
class SeedReport:
    """What a seed run wrote and which seed_data files were missing."""

    items: int = 0
    traces: int = 0
    labelled: int = 0
    unverified: int = 0
    missing_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reset_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _as_list(data: Any, key: str, path: Path) -> list[dict[str, Any]]:
    """Accept either a bare list or an object holding the list under key."""
    rows = data.get(key) if isinstance(data, Mapping) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list or an object with a {key!r} list")
    return rows


def expiry_from_offset(start_date: date, offset_days: int | None) -> date | None:
    """Convert expiry_offset_days into an absolute expiry date relative to DEMO_START_DATE."""
    if offset_days is None:
        return None
    if isinstance(offset_days, bool) or int(offset_days) != offset_days:
        raise ValueError(f"expiry_offset_days must be a whole number of days, got {offset_days!r}")
    return start_date + timedelta(days=int(offset_days))


def parse_date_labels(data: Mapping[str, Any]) -> tuple[dict[str, str], set[str]]:
    """Return (item_name -> date_label, unverified item names) from a date_labels.json payload."""
    raw = data.get("label_types", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("date_labels.json: label_types must be an object")
    labels: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, list):  # label -> [items]
            label = rules.clean_name(key)
            for name in value:
                labels[rules.normalize_name(name)] = label
        elif isinstance(value, str):  # item -> label
            labels[rules.normalize_name(key)] = rules.clean_name(value)
        else:
            raise ValueError(f"date_labels.json: label_types[{key!r}] must be a string or a list")
    bad = {name: label for name, label in labels.items() if label not in VALID_LABELS}
    if bad:
        raise ValueError(f"date_labels.json: unknown labels {bad}; allowed: {sorted(VALID_LABELS)}")
    unverified_raw = data.get("unverified_packaging", []) or []
    if not isinstance(unverified_raw, list):
        raise ValueError("date_labels.json: unverified_packaging must be a list of item names")
    return labels, {rules.normalize_name(name) for name in unverified_raw}


def build_ingredients(
    rows: Sequence[Mapping[str, Any]],
    start_date: date,
    labels: Mapping[str, str] | None = None,
    unverified: set[str] | None = None,
) -> list[Ingredient]:
    """Turn pantry_items.json rows into Ingredients with absolute expiry dates and date labels."""
    labels = labels or {}
    unverified = unverified or set()
    items: list[Ingredient] = []
    for row in rows:
        data = dict(row)
        name = rules.clean_name(str(data["item_name"]))
        offset = data.pop("expiry_offset_days", None)
        if "expiry_date" not in data:
            data["expiry_date"] = expiry_from_offset(start_date, offset)
        data["item_name"] = name
        data["aliases"] = [rules.clean_name(a) for a in data.get("aliases", [])]
        data.setdefault("allergen_tags", sorted(rules.allergen_groups(name)))
        data["date_label"] = labels.get(name, data.get("date_label", "none"))
        data["packaging_unverified"] = name in unverified or bool(data.get("packaging_unverified", False))
        data.pop("priority_score", None)
        data.pop("last_updated", None)
        items.append(rules.with_priority(Ingredient.model_validate(data), start_date))
    return items


# ---------------------------------------------------------------------------
# Seed traces
# ---------------------------------------------------------------------------


def _parse_expiring(entries: Sequence[Any]) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            parsed.append((rules.clean_name(str(entry["name"])), int(entry["days"])))
            continue
        match = _EXPIRING_TEXT.match(str(entry))
        if not match:
            raise ValueError(f"expiring entry {entry!r} must be {{'name', 'days'}} or 'name(Nd)'")
        parsed.append((rules.clean_name(match["name"]), int(match["days"])))
    return parsed


def _max_minutes(constraints: Any) -> int:
    if constraints is None:
        return DEFAULT_MAX_MINUTES
    if isinstance(constraints, int):
        return constraints
    if isinstance(constraints, Mapping):
        return int(constraints.get("max_minutes", DEFAULT_MAX_MINUTES))
    raise ValueError(f"constraints must be an object with max_minutes or an int, got {constraints!r}")


def seed_snapshot_text(row: Mapping[str, Any], start_date: date) -> str:
    """Build snapshot_text for a structured seed trace with the runtime rules.build_snapshot_text.

    Synthetic items reproduce the structured fields: expiring items expire in N days, depleted items sit
    at zero against a safety stock of one, and have items are plain stocked items without expiry.
    """
    items: list[Ingredient] = []
    for name, days in _parse_expiring(row.get("expiring", [])):
        items.append(Ingredient(item_name=name, quantity=1, unit="pcs", expiry_date=start_date + timedelta(days=days)))
    for name in row.get("depleted", []):
        items.append(Ingredient(item_name=rules.clean_name(name), quantity=0, unit="pcs", safety_stock=1))
    for name in row.get("have", []):
        items.append(Ingredient(item_name=rules.clean_name(name), quantity=1, unit="pcs"))
    return rules.build_snapshot_text(
        row["trigger"], items, start_date, _max_minutes(row.get("constraints")), row.get("allergens", [])
    )


def build_seed_traces(rows: Sequence[Mapping[str, Any]], llm: LLM, start_date: date) -> list[Trace]:
    """Turn structured seed rows into Traces with is_seed=true, snapshot_text and embeddings."""
    prepared: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        trace_id = str(row.get("trace_id") or f"seed-{index:03d}")
        if trace_id in seen:
            raise ValueError(f"seed_traces.json: duplicate trace_id {trace_id!r}")
        seen.add(trace_id)
        trigger: TriggerType = row["trigger"]
        max_minutes = _max_minutes(row.get("constraints"))
        plan = dict(row.get("plan") or {})
        plan.setdefault("servings", DEFAULT_SERVINGS)
        plan.setdefault("est_minutes", max_minutes)
        data = {
            "trace_id": trace_id,
            "run_id": None,
            "day_index": int(row.get("day_index", 0)),
            "trigger": trigger,
            "constraints": [f"max_minutes={max_minutes}"],
            "plan": PlannerOutput.model_validate(plan),
            "final_outcome": row.get("final_outcome", "accepted"),
            "risk_status": row.get("risk_status", "none"),
            "is_seed": True,
        }
        prepared.append((data, seed_snapshot_text(row, start_date)))
    vectors = llm.embed([text for _, text in prepared]) if prepared else []
    if len(vectors) != len(prepared):
        raise ValueError(f"LLM returned {len(vectors)} embeddings for {len(prepared)} seed traces")
    return [
        Trace.model_validate({**data, "snapshot_text": text, "embedding": vector})
        for (data, text), vector in zip(prepared, vectors)
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def seed_store(store: Store, llm: LLM, data_dir: Path, start_date: date, *, wait_index: bool = False) -> SeedReport:
    """Write pantry, profile and seed traces, build indexes, then reset to day 1. Safe to rerun."""
    report = SeedReport()
    pantry_path = data_dir / PANTRY_FILE
    labels_path = data_dir / LABELS_FILE
    traces_path = data_dir / TRACES_FILE
    if not pantry_path.exists():
        report.missing_files.append(str(pantry_path))
        raise FileNotFoundError(f"Missing required seed file {pantry_path}")

    labels: dict[str, str] = {}
    unverified: set[str] = set()
    if labels_path.exists():
        labels, unverified = parse_date_labels(_read_json(labels_path))
    else:
        report.missing_files.append(str(labels_path))

    pantry_data = _read_json(pantry_path)
    items = build_ingredients(_as_list(pantry_data, "items", pantry_path), start_date, labels, unverified)
    known = {item.item_name for item in items}
    for name in sorted((set(labels) | unverified) - known):
        report.warnings.append(f"date_labels.json mentions {name!r}, which is not a pantry item")

    traces: list[Trace] = []
    if traces_path.exists():
        traces = build_seed_traces(_as_list(_read_json(traces_path), "traces", traces_path), llm, start_date)
    else:
        report.missing_files.append(str(traces_path))

    profile = pantry_data.get("profile", {}) if isinstance(pantry_data, Mapping) else {}
    store.ensure_indexes()
    store.set_today(start_date)
    store.seed_pantry(items)
    store.set_allergens(profile.get("allergens", list(DEFAULT_ALLERGENS)))
    store.set_constraints(Constraints.model_validate(profile.get("constraints", {"max_minutes": DEFAULT_MAX_MINUTES})))
    for trace in traces:
        store.save_trace(trace)
    prune = getattr(store, "prune_seed_traces", None)
    if callable(prune) and traces_path.exists():
        removed = prune([t.trace_id for t in traces])
        if removed:
            report.warnings.append(f"removed {removed} stale seed traces no longer in {TRACES_FILE}")
    report.reset_counts = store.reset_demo(start_date)

    vector_error = getattr(store, "vector_index_error", None)
    if vector_error:
        report.warnings.append(f"vector index: {vector_error}")
    elif wait_index and hasattr(store, "wait_for_vector_index"):
        if not store.wait_for_vector_index(INDEX_WAIT_SECONDS):  # type: ignore[attr-defined]
            report.warnings.append("vector index not queryable yet; search_traces uses the local fallback until it is")

    report.items = len(items)
    report.traces = len(traces)
    report.labelled = sum(1 for item in items if item.date_label != "none")
    report.unverified = sum(1 for item in items if item.packaging_unverified)
    return report


def _make_llm(use_fake: bool) -> tuple[LLM, str | None]:
    """Return the real embedding LLM, or FakeLLM with a reason when forced or unconfigured."""
    if use_fake:
        return FakeLLM(), "--fake-llm given"
    settings = get_settings()
    if settings.embed_provider != "openai":
        return FakeLLM(), f"EMBED_PROVIDER={settings.embed_provider} is not implemented"
    try:
        return OpenNextLLM(settings), None
    except LLMConfigError as exc:
        return FakeLLM(), str(exc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pantry.seed", description="Seed the pantry store from seed_data/.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="directory with the seed JSON files")
    parser.add_argument("--backend", choices=["mongo", "json"], help="override STORE_BACKEND")
    parser.add_argument("--fake-llm", action="store_true", help="embed with the deterministic FakeLLM (offline)")
    parser.add_argument("--no-wait-index", action="store_true", help="do not wait for the Atlas vector index")
    args = parser.parse_args(argv)

    from pantry.store.factory import make_store

    settings = get_settings()
    start_date = settings.demo_start_date
    if not (args.data_dir / PANTRY_FILE).exists():
        print(f"FAIL missing required seed file: {args.data_dir / PANTRY_FILE}")
        print("     Hint: python -m pantry.seed --data-dir tests/fixtures/seed_data --backend json")
        return 1
    llm, fake_reason = _make_llm(args.fake_llm)
    if fake_reason:
        print(f"WARN embeddings use FakeLLM ({fake_reason}); Atlas similarity will not be meaningful")
    backend = args.backend or settings.store_backend
    store = make_store(backend, settings)
    report = seed_store(store, llm, args.data_dir, start_date, wait_index=not args.no_wait_index)

    print(f"Seeded backend={backend} start_date={start_date.isoformat()} data_dir={args.data_dir}")
    print(f"  pantry items: {report.items} (date labels: {report.labelled}, unverified packaging: {report.unverified})")
    print(f"  seed traces:  {report.traces}")
    print(f"  reset:        {json.dumps(report.reset_counts, sort_keys=True)}")
    for path in report.missing_files:
        print(f"  missing file: {path}")
    for warning in report.warnings:
        print(f"  warning:      {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
