# Pure deterministic pantry rules: name normalization, allergen matching, priority scores and triggers.
# No I/O and no LLM calls; every date computation takes the simulated "today" explicitly.
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date

from pantry.models import BusEvent, Ingredient, Severity, TriggerType

# ---------------------------------------------------------------------------
# Constants (build prompt section 7)
# ---------------------------------------------------------------------------

EXPIRY_HORIZON_DAYS: int = 5
EXPIRY_TRIGGER_MAX_DAYS: int = 2
EXPIRY_WEIGHT: float = 0.7
LOW_STOCK_WEIGHT: float = 0.3

EXPIRY_IMMINENT: TriggerType = "EXPIRY_IMMINENT"
STOCK_DEPLETED: TriggerType = "STOCK_DEPLETED"

# Allergen group -> lowercase English synonyms and derivatives, matched on word boundaries.
ALLERGEN_MAP: dict[str, list[str]] = {
    "nuts": [
        "nut", "tree nut", "almond", "cashew", "walnut", "pecan", "hazelnut", "pistachio",
        "macadamia", "brazil nut", "pine nut", "praline", "marzipan", "frangipane", "nutella",
        "almond milk", "almond flour", "almond butter", "cashew milk", "hazelnut milk",
    ],
    "peanut": ["peanut", "groundnut", "monkey nut", "peanut butter", "peanut oil", "satay"],
    "dairy": [
        "milk", "butter", "cream", "cheese", "yogurt", "yoghurt", "whey", "casein", "ghee",
        "buttermilk", "heavy cream", "sour cream", "cream cheese", "parmesan", "mozzarella",
        "cheddar", "ricotta", "mascarpone", "custard", "ice cream", "condensed milk",
        "evaporated milk", "milk powder", "lactose",
    ],
    "egg": ["egg", "egg white", "egg yolk", "mayonnaise", "mayo", "meringue", "albumen", "albumin", "eggnog", "custard"],
    "gluten": [
        "wheat", "flour", "bread", "breadcrumb", "bread crumb", "panko", "pasta", "spaghetti",
        "couscous", "semolina", "barley", "rye", "spelt", "farro", "bulgur", "seitan", "malt",
        "soy sauce", "cracker",
    ],
    "soy": ["soy", "soya", "soybean", "soy sauce", "soy milk", "tofu", "tempeh", "edamame", "miso", "tamari", "soy lecithin"],
    "shellfish": [
        "shellfish", "shrimp", "prawn", "crab", "lobster", "crayfish", "crawfish", "scallop",
        "clam", "mussel", "oyster", "squid", "octopus",
    ],
    "fish": [
        "fish", "salmon", "tuna", "cod", "anchovy", "anchovies", "sardine", "trout", "mackerel",
        "tilapia", "halibut", "haddock", "fish sauce", "worcestershire sauce",
    ],
    "sesame": ["sesame", "sesame oil", "sesame seed", "tahini", "halva", "gomasio"],
    "mustard": ["mustard", "mustard seed", "dijon"],
}

# Phrases removed before matching a group, so plant milks are not dairy and gluten-free flours are not gluten.
ALLERGEN_EXCLUSIONS: dict[str, list[str]] = {
    "dairy": [
        "almond milk", "oat milk", "soy milk", "soya milk", "rice milk", "coconut milk", "cashew milk",
        "hazelnut milk", "plant milk", "coconut cream", "peanut butter", "almond butter", "cashew butter",
        "nut butter", "cocoa butter", "shea butter", "vegan butter", "vegan cheese", "cream of tartar",
    ],
    "gluten": [
        "rice flour", "almond flour", "coconut flour", "corn flour", "chickpea flour", "buckwheat flour",
        "tapioca flour", "potato flour", "gluten-free flour", "gluten free flour", "gluten-free bread",
        "gluten free bread", "gluten-free pasta", "gluten free pasta",
    ],
    "egg": ["vegan mayo", "vegan mayonnaise"],
}

# Alternative user spellings of allergen group names -> canonical group name.
ALLERGEN_GROUP_ALIASES: dict[str, str] = {
    "nut": "nuts",
    "tree nut": "nuts",
    "tree nuts": "nuts",
    "peanuts": "peanut",
    "milk": "dairy",
    "lactose": "dairy",
    "eggs": "egg",
    "wheat": "gluten",
    "soya": "soy",
    "sesame seeds": "sesame",
}

# Ingredients that are always assumed to be available and never count as gaps.
STAPLES: frozenset[str] = frozenset({"salt", "water", "pepper", "ice", "cooking oil", "vegetable oil"})

# Alternative ingredient name -> English lowercase canonical pantry name.
ALIASES: dict[str, str] = {
    "egg": "eggs",
    "whole milk": "milk",
    "full fat milk": "milk",
    "double cream": "heavy cream",
    "heavy whipping cream": "heavy cream",
    "whipping cream": "heavy cream",
    "all-purpose flour": "flour",
    "all purpose flour": "flour",
    "plain flour": "flour",
    "wheat flour": "flour",
    "granulated sugar": "sugar",
    "white sugar": "sugar",
    "caster sugar": "sugar",
    "unsalted butter": "butter",
    "salted butter": "butter",
    "sea salt": "salt",
    "kosher salt": "salt",
    "table salt": "salt",
    "black pepper": "pepper",
    "ground black pepper": "pepper",
    "icing sugar": "powdered sugar",
    "confectioners sugar": "powdered sugar",
    "bicarbonate of soda": "baking soda",
    "cornflour": "cornstarch",
    "spring onion": "green onion",
    "scallion": "green onion",
    "courgette": "zucchini",
    "aubergine": "eggplant",
}

_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def clean_name(name: str) -> str:
    """Lowercase, trim and collapse internal whitespace, without alias mapping."""
    return _WHITESPACE.sub(" ", name.strip().lower())


def normalize_name(name: str, extra_aliases: Mapping[str, str] | None = None) -> str:
    """Return the canonical pantry name: cleaned, then mapped through extra_aliases and ALIASES."""
    cleaned = clean_name(name)
    if extra_aliases:
        mapped = {clean_name(k): clean_name(v) for k, v in extra_aliases.items()}
        if cleaned in mapped:
            return mapped[cleaned]
    return ALIASES.get(cleaned, cleaned)


def item_alias_map(items: Iterable[Ingredient]) -> dict[str, str]:
    """Build alias -> canonical name from the pantry items' own aliases lists."""
    mapping: dict[str, str] = {}
    for item in items:
        canonical = clean_name(item.item_name)
        for alias in item.aliases:
            mapping[clean_name(alias)] = canonical
    return mapping


def is_staple(name: str) -> bool:
    """Return True if the ingredient is a staple that never counts as a gap."""
    return normalize_name(name) in STAPLES


# ---------------------------------------------------------------------------
# Allergens
# ---------------------------------------------------------------------------


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Word-boundary pattern for a lowercase phrase, allowing a simple plural suffix."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?:s|es)?(?![a-z0-9])")


def normalize_allergen_group(group: str) -> str:
    """Map a user allergen spelling (for example "tree nuts") to its ALLERGEN_MAP group name."""
    cleaned = clean_name(group)
    return ALLERGEN_GROUP_ALIASES.get(cleaned, cleaned)


def _matches_group(text: str, group: str) -> bool:
    """Return True if cleaned text contains a synonym of group after removing excluded phrases."""
    for excluded in ALLERGEN_EXCLUSIONS.get(group, []):
        text = _phrase_pattern(excluded).sub(" ", text)
    synonyms = ALLERGEN_MAP.get(group, [group])
    return any(_phrase_pattern(synonym).search(text) for synonym in synonyms)


def allergen_groups(name: str) -> set[str]:
    """Return every ALLERGEN_MAP group the ingredient name belongs to."""
    text = clean_name(name)
    return {group for group in ALLERGEN_MAP if _matches_group(text, group)}


def check_allergens(names: str | Iterable[str], user_allergens: Iterable[str]) -> list[str]:
    """Return the sorted user allergen groups matched by any of the ingredient names.

    Unknown user allergen groups are matched literally on word boundaries.
    """
    name_list = [names] if isinstance(names, str) else list(names)
    groups = {normalize_allergen_group(g) for g in user_allergens if g.strip()}
    hits = {group for group in groups for name in name_list if _matches_group(clean_name(name), group)}
    return sorted(hits)


def matches_allergen(name: str, user_allergens: Iterable[str]) -> bool:
    """Return True if the ingredient name matches any of the user's allergen groups."""
    return bool(check_allergens(name, user_allergens))


# ---------------------------------------------------------------------------
# Dates and priority (build prompt section 7)
# ---------------------------------------------------------------------------


def _as_date(value: date | str) -> date:
    """Accept a date or an ISO date string."""
    return value if isinstance(value, date) else date.fromisoformat(value)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into [low, high]."""
    return max(low, min(high, value))


def days_left(expiry_date: date | str | None, today: date | str) -> int | None:
    """Return expiry_date - today in whole days, or None when the item has no expiry date."""
    if expiry_date is None:
        return None
    return (_as_date(expiry_date) - _as_date(today)).days


def is_expired(item: Ingredient, today: date | str) -> bool:
    """Return True if the item is past its expiry date (days < 0), whatever its date_label."""
    days = days_left(item.expiry_date, today)
    return days is not None and days < 0


def expiry_u(days: int | None) -> float:
    """Expiry urgency clamp((5 - days) / 5, 0, 1); items without an expiry date score 0."""
    if days is None:
        return 0.0
    return clamp((EXPIRY_HORIZON_DAYS - days) / EXPIRY_HORIZON_DAYS, 0.0, 1.0)


def low_stock_u(quantity: float, safety_stock: float) -> float:
    """Low-stock urgency: 1 if physical quantity <= safety_stock, else 0 (reservations are ignored)."""
    return 1.0 if quantity <= safety_stock else 0.0


def priority_score(item: Ingredient, today: date | str) -> float:
    """Return 0.7 * expiry_u + 0.3 * low_stock_u for the item on the given day."""
    days = days_left(item.expiry_date, today)
    score = EXPIRY_WEIGHT * expiry_u(days) + LOW_STOCK_WEIGHT * low_stock_u(item.quantity, item.safety_stock)
    return round(score, 6)


def with_priority(item: Ingredient, today: date | str) -> Ingredient:
    """Return a copy of the item with priority_score recomputed for today."""
    return item.model_copy(update={"priority_score": priority_score(item, today)})


def plannable_items(items: Iterable[Ingredient], today: date | str) -> list[Ingredient]:
    """Return non-expired items with fresh priority scores, sorted by priority desc then name."""
    scored = [with_priority(item, today) for item in items if not is_expired(item, today)]
    return sorted(scored, key=lambda i: (-i.priority_score, i.item_name))


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


def severity(days: int | None, quantity: float) -> Severity:
    """High when the item expires today or is physically out of stock, otherwise low."""
    return "high" if days == 0 or quantity == 0 else "low"


def dedupe_key(trigger_type: TriggerType, item_name: str, today: date | str) -> str:
    """Return the bus dedupe key trigger_type:item_name:today (today as an ISO date)."""
    return f"{trigger_type}:{clean_name(item_name)}:{_as_date(today).isoformat()}"


def trigger_types(item: Ingredient, today: date | str) -> list[TriggerType]:
    """Return the triggers this item fires today; expired items (days < 0) never fire."""
    days = days_left(item.expiry_date, today)
    if days is not None and days < 0:
        return []
    fired: list[TriggerType] = []
    if days is not None and 0 <= days <= EXPIRY_TRIGGER_MAX_DAYS:
        fired.append(EXPIRY_IMMINENT)
    if item.quantity <= item.safety_stock:
        fired.append(STOCK_DEPLETED)
    return fired


def detect_triggers(items: Iterable[Ingredient], today: date | str) -> list[BusEvent]:
    """Return one pending BusEvent per (trigger, item) fired today, in input order."""
    events: list[BusEvent] = []
    for item in items:
        days = days_left(item.expiry_date, today)
        for trigger in trigger_types(item, today):
            events.append(
                BusEvent(
                    trigger_type=trigger,
                    item_name=clean_name(item.item_name),
                    severity=severity(days, item.quantity),
                    dedupe_key=dedupe_key(trigger, item.item_name, today),
                )
            )
    return events


# ---------------------------------------------------------------------------
# Snapshot text (patch A section 2: relative days only)
# ---------------------------------------------------------------------------


def build_snapshot_text(
    trigger: TriggerType,
    items: Sequence[Ingredient],
    today: date | str,
    max_minutes: int,
    allergens: Iterable[str],
) -> str:
    """Build the deterministic snapshot, for example
    "trigger=EXPIRY_IMMINENT; expiring=milk(2d); have=eggs,flour; constraint=25min; allergens=nuts".

    Expired items are excluded. expiring lists items with 0 <= days <= 2 (soonest first); depleted lists
    items at or below safety stock; have lists the other items with physical quantity > 0, by name.
    """
    live = [item for item in items if not is_expired(item, today)]
    expiring: list[tuple[int, str]] = []
    depleted: list[str] = []
    have: list[str] = []
    for item in live:
        name = clean_name(item.item_name)
        days = days_left(item.expiry_date, today)
        if days is not None and days <= EXPIRY_TRIGGER_MAX_DAYS:
            expiring.append((days, name))
        elif item.quantity <= item.safety_stock:
            depleted.append(name)
        elif item.quantity > 0:
            have.append(name)
    expiring_text = ",".join(f"{name}({days}d)" for days, name in sorted(expiring)) or "none"
    allergen_text = ",".join(sorted({normalize_allergen_group(a) for a in allergens if a.strip()})) or "none"
    parts = [f"trigger={trigger}", f"expiring={expiring_text}"]
    if depleted:
        parts.append(f"depleted={','.join(sorted(depleted))}")
    parts += [f"have={','.join(sorted(have)) or 'none'}", f"constraint={max_minutes}min", f"allergens={allergen_text}"]
    return "; ".join(parts)
