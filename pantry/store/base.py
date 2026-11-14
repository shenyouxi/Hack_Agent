# Abstract Store interface shared by JsonStore (tests, offline) and MongoStore (demo).
# Signatures and docstrings only; every write is idempotent and inventory writes enforce the ledger invariants.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

from pantry.models import (
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
)


class InvariantViolation(Exception):
    """Raised when a write would break an inventory invariant; the write is rolled back."""


class Store(ABC):
    """Persistence boundary for the agent. Agents never call the database directly."""

    # ------------------------------------------------------------------
    # Setup and reset
    # ------------------------------------------------------------------

    @abstractmethod
    def ensure_indexes(self) -> None:
        """Create the unique indexes (runs.idempotency_key, bus_events.dedupe_key,
        shopping_list.item_name, feedback.op_id) and, where supported, the vector index."""

    @abstractmethod
    def seed_pantry(self, items: Sequence[Ingredient]) -> None:
        """Replace the pantry with items and record them as the ledger baseline used by replay and reset."""

    @abstractmethod
    def reset_demo(self, start_date: date) -> dict[str, int]:
        """Restore day 1: baseline inventory, today=start_date, and clear runs, bus_events, commands,
        inventory_events, shopping_list, feedback and non-seed traces. Keeps search_cache, plan_cache
        and seed traces. Idempotent; returns the number of documents removed or restored per collection."""

    # ------------------------------------------------------------------
    # Simulated clock and user profile
    # ------------------------------------------------------------------

    @abstractmethod
    def get_today(self) -> date:
        """Return the simulated current date stored in user_profile.today."""

    @abstractmethod
    def set_today(self, d: date) -> None:
        """Set the simulated date and recompute every pantry item's priority_score in the same operation."""

    @abstractmethod
    def advance_day(self, n: int = 1) -> date:
        """Move the simulated date forward by n days, recompute every priority_score in the same
        operation, and return the new date."""

    @abstractmethod
    def get_profile(self) -> UserProfile:
        """Return the user_profile document."""

    @abstractmethod
    def set_constraints(self, constraints: Constraints) -> UserProfile:
        """Replace the user's constraints and return the updated profile."""

    @abstractmethod
    def set_allergens(self, allergens: Sequence[str]) -> UserProfile:
        """Replace the user's allergen list (lowercase group names) and return the updated profile."""

    # ------------------------------------------------------------------
    # Pantry items (a projection of the inventory ledger)
    # ------------------------------------------------------------------

    @abstractmethod
    def list_items(self) -> list[Ingredient]:
        """Return every pantry item, including expired ones, sorted by priority_score descending."""

    @abstractmethod
    def get_item(self, item_name: str) -> Ingredient | None:
        """Return the item whose canonical name or alias matches item_name, or None."""

    @abstractmethod
    def recompute_priorities(self) -> list[Ingredient]:
        """Recompute priority_score for every item against get_today() and return the updated items."""

    # ------------------------------------------------------------------
    # Inventory ledger
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_events(self, events: Sequence[InventoryEvent], idempotency_key: str) -> list[InventoryEvent]:
        """Atomically append events and update quantity/reserved. The same idempotency_key takes effect
        once (a repeat returns the originally applied events). Mismatched units or any invariant
        violation rolls back the whole batch and raises InvariantViolation."""

    @abstractmethod
    def list_events(self, run_id: str | None = None, item_name: str | None = None) -> list[InventoryEvent]:
        """Return ledger events in creation order, optionally filtered by run or item."""

    @abstractmethod
    def reverse_run_events(self, run_id: str) -> list[InventoryEvent]:
        """Append one reversing event (reverses_event_id set) for every not-yet-reversed event of the run.
        Idempotent; returns the reversing events written by this call."""

    @abstractmethod
    def commit_run(self, run_id: str) -> list[InventoryEvent]:
        """Convert the run's outstanding RESERVED amounts into CONSUMED events (estimated=true, note
        "assumed cooked in the demo") plus matching RELEASED events, then mark the run committed.
        Idempotent; returns the events written by this call."""

    @abstractmethod
    def list_uncommitted_applied_runs(self) -> list[Run]:
        """Return runs that reached APPLIED (or RECORDED after APPLIED) and are not yet committed."""

    @abstractmethod
    def check_invariants(self) -> list[str]:
        """Return human-readable invariant violations (quantity >= 0, 0 <= reserved <= quantity,
        event unit equals item unit, stored values equal ledger replay); empty means consistent."""

    @abstractmethod
    def replay_ledger(self) -> dict[str, tuple[float, float]]:
        """Recompute (quantity, reserved) per item from the baseline plus all events, without writing."""

    # ------------------------------------------------------------------
    # Bus events
    # ------------------------------------------------------------------

    @abstractmethod
    def emit_bus_event(self, event: BusEvent) -> bool:
        """Insert a bus event; return False without writing if its dedupe_key already exists."""

    @abstractmethod
    def list_bus_events(self, status: BusEventStatus | None = None, limit: int = 50) -> list[BusEvent]:
        """Return bus events newest first, optionally filtered by status."""

    @abstractmethod
    def set_bus_event_status(self, dedupe_key: str, status: BusEventStatus) -> None:
        """Update the status of the bus event with this dedupe_key."""

    # ------------------------------------------------------------------
    # Runs and checkpoints
    # ------------------------------------------------------------------

    @abstractmethod
    def create_run(self, run: Run) -> tuple[Run, bool]:
        """Insert a run unless its idempotency_key exists. Returns (stored run, created flag)."""

    @abstractmethod
    def get_run(self, run_id: str) -> Run | None:
        """Return a run by id, or None."""

    @abstractmethod
    def list_runs(self, states: Sequence[RunState] | None = None, limit: int = 50) -> list[Run]:
        """Return runs newest first, optionally filtered by state."""

    @abstractmethod
    def list_active_runs(self) -> list[Run]:
        """Return every non-terminal run oldest first, for worker resume."""

    @abstractmethod
    def save_checkpoint(self, run_id: str, state: RunState, checkpoint: dict[str, object], step: RunStep) -> Run:
        """Atomically set the run state, store checkpoint under checkpoints[state] and append step.
        Re-saving the same state with the same step name is a no-op."""

    @abstractmethod
    def append_step(self, run_id: str, step: RunStep) -> Run:
        """Append a timeline step without changing state."""

    @abstractmethod
    def set_pending_confirmation(self, run_id: str, request: ConfirmationRequest | None) -> Run:
        """Set or clear the run's persistent pending confirmation request."""

    # ------------------------------------------------------------------
    # Decision traces and feedback
    # ------------------------------------------------------------------

    @abstractmethod
    def save_trace(self, trace: Trace) -> Trace:
        """Upsert a trace by trace_id and return the stored trace."""

    @abstractmethod
    def get_trace(self, trace_id: str) -> Trace | None:
        """Return a trace by id, or None."""

    @abstractmethod
    def get_trace_by_run(self, run_id: str) -> Trace | None:
        """Return the trace written for run_id, or None."""

    @abstractmethod
    def list_traces(self, is_seed: bool | None = None, limit: int = 100) -> list[Trace]:
        """Return traces newest first, optionally filtered by is_seed."""

    @abstractmethod
    def search_traces(self, query_vec: Sequence[float], trigger: TriggerType, k: int = 3) -> tuple[list[RetrievedTrace], bool]:
        """Return the top-k accepted traces with the same trigger by cosine similarity, merging Atlas
        $vectorSearch with a local immediate find (deduplicated by trace_id). The bool is True when
        Atlas was unavailable and only the local search was used (the caller marks the trace degraded)."""

    @abstractmethod
    def save_feedback(self, trace_id: str, outcome: FeedbackOutcome, op_id: str, source: FeedbackSource) -> Feedback:
        """Record feedback and set the trace's final_outcome. Idempotent on op_id: a repeat returns the
        original Feedback without applying it again."""

    # ------------------------------------------------------------------
    # Shopping list
    # ------------------------------------------------------------------

    @abstractmethod
    def add_to_shopping_list(self, item: ShoppingListItem) -> ShoppingListItem:
        """Upsert by item_name: add quantity (same unit only) and merge run_ids. Re-adding for a run_id
        already listed is a no-op."""

    @abstractmethod
    def list_shopping_list(self) -> list[ShoppingListItem]:
        """Return the shopping list sorted by item_name."""

    @abstractmethod
    def remove_from_shopping_list(self, item_name: str) -> bool:
        """Remove an item; return True if it existed."""

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------

    @abstractmethod
    def get_search_cache(self, key: str) -> SearchCacheEntry | None:
        """Return the cached search or extract result for key, or None."""

    @abstractmethod
    def put_search_cache(self, entry: SearchCacheEntry) -> None:
        """Upsert a search cache entry by key, keeping the original real call time."""

    @abstractmethod
    def get_plan_cache(self, key: str) -> PlanCacheEntry | None:
        """Return the cached plan for key, or None."""

    @abstractmethod
    def put_plan_cache(self, entry: PlanCacheEntry) -> None:
        """Upsert a plan cache entry by key."""

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @abstractmethod
    def enqueue_command(self, command: Command) -> Command:
        """Insert a pending command; a repeat command_id returns the existing command."""

    @abstractmethod
    def poll_commands(self, limit: int = 20) -> list[Command]:
        """Return pending commands oldest first, for the worker to process."""

    @abstractmethod
    def get_command(self, command_id: str) -> Command | None:
        """Return a command by id, or None (the UI uses this to wait for status done)."""

    @abstractmethod
    def mark_command(self, command_id: str, status: CommandStatus, error: str | None = None) -> Command:
        """Set a command's status (done or failed), error and processed_at."""
