# Synthetic demo seed data

These files are Team C's three-round demo fixture. Pantry quantities, dates, profiles, events, and historical decisions are invented for testing. Recipe metadata and the cream-substitution candidate have linked sources. This is not a real food-safety or nutrition dataset.

## Files and ownership

| File | Purpose |
| --- | --- |
| `pantry_items.json` | 29 starter ingredients using the exact fields and units in `docs/conventions.md` |
| `pantry_items.csv` | Spreadsheet review copy of the same inventory |
| `recipes.json` | Six source-linked recipe candidates |
| `household_profiles.json` | Four synthetic people, including one guest with peanut and tree-nut allergies |
| `seed_traces.json` | Four synthetic historical decisions; `scripts/seed.py` must generate snapshot text and embeddings |
| `demo_script.json` | Three connected rounds, complete opening inventories, expected results, and eight optional edge cases |
| `aliases.json`, `allergen_map.json`, `staples.json` | Inputs for `pantry/rules.py` |
| `date_labels.json` | Supplementary use-by, best-before, and estimated-quality labels for demo safety checks |

Team A owns `scripts/seed.py` and storage integration. Team B owns each newly generated decision trace. Team C owns the UI and feedback commands. Team A ensures a new accepted trace becomes searchable before Round 2 begins.

## Trigger schedule

The virtual baseline is **2026-11-14 18:00 Europe/London**. `expiry_offset_days` is measured from its date. A trigger occurs once when an item first enters the `<= 2` day urgency window. Existing zero-stock items do not generate a new depletion transition on startup.

| UI label | `day_index` | New trigger | Label or estimated date |
| --- | ---: | --- | --- |
| Demo Day 1 · Round 1 | 0 | Milk | Use-by on baseline + 1 day |
| Demo Day 2 · Round 2 | 1 | Eggs | Best-before on baseline + 3 days |
| Demo Day 3 · Round 3 | 2 | Baby spinach | Estimated quality date on baseline + 4 days |

Other short-life items were set later so they do not introduce competing expiry triggers in these three rounds. The full opening stock for each round is in `demo_script.json`. Round 2 must retrieve the **live trace from Round 1**, rather than any preloaded trace. Plan acceptance alone does not consume ingredients; a simulated `MEAL_COMPLETED` event applies the stated inventory deltas. The third round pauses on a known tree-nut conflict and records a rejection without food consumption.

## Data and safety limits

`pantry_items.json` follows the agreed `Ingredient` fields exactly. The current convention does not include a date-label type. `date_labels.json` keeps this distinction for the demo, but **Team A must connect it to a deterministic safety check** before the application may suggest using milk on Demo Day 3. A past use-by item must be excluded; an estimated quality date is not a safety guarantee. Packaged ham, pancetta, and chicken stock have unverified allergen labels in this fixture; their real packaging must be checked if serving an allergic person.

The recipe ingredient lists are demo mappings, not complete cooking instructions. Some backup candidates deliberately omit optional or unavailable ingredients; `data_note` says so. The two cream substitutions are candidates for cooked dishes and must pass the Resolver's citation and constraint checks. Do not present a seeded historical decision as a live API call.

## Source links

- [Creamy garlic pasta](https://www.bbcgoodfood.com/recipes/creamy-garlic-pasta) and [creamy carbonara](https://www.bbcgoodfood.com/recipes/learn-to-make-spaghetti-carbonara): main recipe quantities and published times.
- [Double-cream substitutions](https://www.silverspoon.co.uk/baking-tips/double-cream-substitutes): milk and butter candidate for cooking; it cannot be whipped.
- [Spinach and walnut pesto pasta](https://www.bbcgoodfood.com/recipes/spaghetti-spinach-walnut-pesto): allergy-gate candidate.
- [UK Food Standards Agency date-label guidance](https://www.food.gov.uk/print/pdf/node/271) and [allergen guidance](https://www.food.gov.uk/business-guidance/allergen-guidance-for-food-businesses): safety categories.

All stock quantities, product dates, participant profiles, and decision histories are synthetic.
