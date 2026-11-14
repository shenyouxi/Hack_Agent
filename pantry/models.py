# Pydantic v2 schemas for every MongoDB collection, the agent's in-memory packages and the LLM outputs.
# Source and LinkupResult are TypedDicts because docs/conventions.md defines them that way for the tool wrappers.
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared literal types
# ---------------------------------------------------------------------------

Unit = Literal["pcs", "g", "ml"]
TriggerType = Literal["EXPIRY_IMMINENT", "STOCK_DEPLETED", "CONSTRAINT_SHIFT"]
Severity = Literal["high", "low"]
BusEventStatus = Literal["pending", "dispatched", "ignored"]
InventoryEventType = Literal["RESERVED", "RELEASED", "CONSUMED", "ADJUSTED"]
RunState = Literal[
    "TRIGGERED",
    "CONTEXT_BUILT",
    "SEEDS_RETRIEVED",
    "PLANNED",
    "GAP_CHECKED",
    "RESOLVING",
    "GATED",
    "PENDING_CONFIRMATION",
    "APPLIED",
    "REJECTED",
    "RECORDED",
]
StepStatus = Literal["started", "ok", "degraded", "failed", "skipped"]
RiskStatus = Literal["none", "allergen", "no_citation", "search_unavailable", "constraint_violation"]
FinalOutcome = Literal["accepted", "rejected", "pending_confirmation"]
FeedbackOutcome = Literal["accepted", "rejected"]
FeedbackSource = Literal["user", "auto"]
CommandType = Literal[
    "ADVANCE_CLOCK",
    "APPROVE",
    "REJECT",
    "FEEDBACK",
    "INJECT_DEMO_RESULT",
    "UNDO_RUN",
    "COOKED",
    "SET_CONSTRAINT",
]
CommandStatus = Literal["pending", "done", "failed"]
GapKind = Literal["MISSING", "INSUFFICIENT"]
Confidence = Literal["high", "medium", "low"]
ResolverTool = Literal["linkup", "tavily", "llm", "cache", "demo_injection"]
CacheTool = Literal["linkup", "tavily"]

TERMINAL_RUN_STATES: frozenset[str] = frozenset({"RECORDED"})


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class DocModel(BaseModel):
    """Base for stored documents: ignores Mongo's _id and other unknown keys."""

    model_config = ConfigDict(extra="ignore")


class LLMOutputModel(BaseModel):
    """Base for LLM outputs: unknown keys fail validation so drift triggers a retry."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Tool wrapper types (docs/conventions.md)
# ---------------------------------------------------------------------------


class Source(TypedDict):
    """One cited source; SDK "name" maps to title and "snippet" maps to content."""

    title: str
    url: str
    content: str


class LinkupResult(TypedDict):
    """Sourced answer returned by LinkupClient.search."""

    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# pantry_items, inventory_events, user_profile
# ---------------------------------------------------------------------------


class Ingredient(DocModel):
    """A pantry_items document; item_name is the English lowercase canonical name."""

    item_name: str
    aliases: list[str] = Field(default_factory=list)
    category: str = ""
    quantity: float = Field(ge=0)
    reserved: float = Field(default=0.0, ge=0)
    unit: Unit
    expiry_date: date | None = None
    safety_stock: float = Field(default=0.0, ge=0)
    allergen_tags: list[str] = Field(default_factory=list)
    priority_score: float = 0.0
    last_updated: datetime = Field(default_factory=utc_now)


class InventoryEvent(DocModel):
    """An append-only inventory_events ledger entry; delta is in the item's unit."""

    event_id: str
    item_name: str
    type: InventoryEventType
    delta: float
    unit: Unit
    estimated: bool = False
    run_id: str | None = None
    reverses_event_id: str | None = None
    note: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class Constraints(DocModel):
    """User cooking constraints."""

    max_minutes: int = Field(default=25, gt=0)


class UserProfile(DocModel):
    """The single user_profile document; today is the simulated clock as an ISO date string."""

    allergens: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    today: str


# ---------------------------------------------------------------------------
# bus_events and the Sentinel package
# ---------------------------------------------------------------------------


class BusEvent(DocModel):
    """A bus_events document; dedupe_key is trigger_type:item_name:today and is unique."""

    trigger_type: TriggerType
    item_name: str
    severity: Severity
    dedupe_key: str
    status: BusEventStatus = "pending"
    created_at: datetime = Field(default_factory=utc_now)


class TriggerEvent(BaseModel):
    """The context package the Sentinel hands to a run (expired items excluded)."""

    trigger: TriggerType
    item_name: str
    severity: Severity
    dedupe_key: str
    today: str
    items_ranked: list[Ingredient]
    constraints: Constraints
    allergens: list[str] = Field(default_factory=list)
    snapshot_text: str


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class PlanIngredient(LLMOutputModel):
    """One ingredient line of a plan; amounts of pantry items are always estimates."""

    name: str
    amount: float = Field(ge=0)
    unit: str
    estimated: bool = True


class PlannerOutput(LLMOutputModel):
    """Planner output: exactly one recipe proposal."""

    dish_name: str
    servings: int = Field(gt=0)
    est_minutes: int = Field(gt=0)
    ingredients: list[PlanIngredient] = Field(min_length=1)
    steps: list[str] = Field(default_factory=list)
    rationale: str = ""
    used_seed_trace_ids: list[str] = Field(default_factory=list)


class SubstituteComponent(LLMOutputModel):
    """One component of a substitute; amount and unit are null when the source does not state them."""

    name: str
    amount: float | None = None
    unit: str | None = None


class ResolverOutput(LLMOutputModel):
    """Resolver extraction output for one MISSING gap."""

    substitute_name: str
    components: list[SubstituteComponent] = Field(min_length=1)
    ratio_text: str | None = None
    source_url: str | None = None
    confidence: Confidence


# ---------------------------------------------------------------------------
# Gaps, steps, confirmations, runs
# ---------------------------------------------------------------------------


class Gap(BaseModel):
    """A gap found by gap detection: MISSING goes to the Resolver, INSUFFICIENT to the shopping list."""

    name: str
    needed: float = Field(ge=0)
    unit: str
    kind: GapKind


class RunStep(BaseModel):
    """One timeline entry in runs.steps and decision_traces.steps."""

    ts: datetime = Field(default_factory=utc_now)
    name: str
    status: StepStatus
    detail: dict[str, Any] = Field(default_factory=dict)


class ConfirmationRequest(BaseModel):
    """Structured pending-confirmation request written by the rule gate."""

    action: str
    evidence: str
    citation_urls: list[str] = Field(default_factory=list)
    rule: RiskStatus
    demo_injected: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class Run(DocModel):
    """A runs document; steps is the source of the UI timeline and checkpoints enable resume."""

    run_id: str
    trigger: TriggerType
    item_name: str
    state: RunState = "TRIGGERED"
    checkpoints: dict[str, Any] = Field(default_factory=dict)
    steps: list[RunStep] = Field(default_factory=list)
    pending_confirmation: ConfirmationRequest | None = None
    idempotency_key: str
    committed: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# decision_traces
# ---------------------------------------------------------------------------


class RetrievedTrace(BaseModel):
    """A seed trace returned by vector search, with its similarity score."""

    trace_id: str
    similarity: float
    day_index: int
    dish_name: str
    is_seed: bool


class ResolverAction(BaseModel):
    """One external or fallback call made by the Resolver, recorded verbatim in the trace."""

    tool: ResolverTool
    query: str
    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    degraded: bool = False
    degraded_reason: str | None = None
    cached_at: datetime | None = None
    demo_injected: bool = False


class Trace(DocModel):
    """A decision_traces document, used both as the audit record and for seed retrieval."""

    trace_id: str
    run_id: str | None = None
    day_index: int
    trigger: TriggerType
    snapshot_text: str
    embedding: list[float] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    plan: PlannerOutput | None = None
    retrieved_traces: list[RetrievedTrace] = Field(default_factory=list)
    resolver_actions: list[ResolverAction] = Field(default_factory=list)
    steps: list[RunStep] = Field(default_factory=list)
    risk_status: RiskStatus = "none"
    final_outcome: FinalOutcome
    is_seed: bool = False
    plan_cached: bool = False
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# feedback, shopping_list, caches, commands
# ---------------------------------------------------------------------------


class Feedback(DocModel):
    """A feedback document; op_id is unique so repeated writes take effect once."""

    op_id: str
    trace_id: str
    outcome: FeedbackOutcome
    source: FeedbackSource
    created_at: datetime = Field(default_factory=utc_now)


class ShoppingListItem(DocModel):
    """A shopping_list document; item_name is unique and quantities merge on repeat adds."""

    item_name: str
    quantity: float = Field(ge=0)
    unit: str
    reason: str = ""
    run_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SearchCacheEntry(DocModel):
    """A search_cache document keyed by the hash of the normalized query (or URL for Tavily)."""

    key: str
    tool: CacheTool
    query: str
    depth: str | None = None
    response: dict[str, Any]
    called_at: datetime


class PlanCacheEntry(DocModel):
    """A plan_cache document keyed by hash(snapshot_text + constraints + allergens)."""

    key: str
    snapshot_text: str
    constraints: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    plan: PlannerOutput
    called_at: datetime


class Command(DocModel):
    """A commands document written by the UI and processed only by the worker."""

    command_id: str
    type: CommandType
    payload: dict[str, Any] = Field(default_factory=dict)
    status: CommandStatus = "pending"
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    processed_at: datetime | None = None
