# Three-round autonomous pantry demo

The canonical machine-readable script is `seed_data/demo_script.json`. The baseline clock is 2026-11-14 18:00 Europe/London. The UI shows Demo Day 1/2/3 and Round 1/2/3; the underlying `day_index` values are 0/1/2. All food stock, dates, people, and prior decisions are synthetic.

## Demo Day 1, Round 1: missing double cream

Milk has one day before its labeled use-by date. Opening stock includes 700 ml milk, 6 pcs eggs, 300 g baby spinach, 0 ml double cream, 180 g tagliatelle, 100 g butter, 5 pcs garlic cloves, 250 g plain flour, 100 ml chicken stock, 100 g parmesan, and 10 g parsley. The full 29-item opening snapshot is in the JSON script.

The Planner proposes two servings of [creamy garlic pasta](https://www.bbcgoodfood.com/recipes/creamy-garlic-pasta), which needs 100 ml double cream. The Resolver uses Linkup to find a cited cooking substitute. One source-backed candidate is 75 ml milk plus 25 g butter for 100 ml double cream, from [Silver Spoon](https://www.silverspoon.co.uk/baking-tips/double-cream-substitutes). It must pass the date, allergen, and citation rules. A Tavily extraction appears in the live timeline only if the tool actually ran.

With auto-run and non-risk auto-accept on, the plan is accepted without a click. A simulated `MEAL_COMPLETED` event confirms cooking and applies the pantry ledger. B writes the live trace, C writes auto feedback, and A makes the accepted trace searchable. Expected carryover: 625 ml milk, 6 pcs eggs, 300 g baby spinach, 30 g tagliatelle, and 65 g butter. Remaining milk keeps its original use-by date.

## Demo Day 2, Round 2: reuse the new decision

Eggs enter the two-day urgency window. Opening stock includes 625 ml milk, 6 pcs eggs, 300 g baby spinach, 0 ml double cream, 400 g spaghetti, 200 g pancetta, 2 pcs garlic cloves, 200 ml olive oil, 65 g butter, and 57.5 g parmesan. The full snapshot is in the JSON script.

The Planner proposes four servings of [creamy carbonara](https://www.bbcgoodfood.com/recipes/learn-to-make-spaghetti-carbonara), which needs 75 ml double cream. Memory must retrieve the **new live Round 1 trace**, rather than a preloaded seed trace. After checking current stock and the milk date, the Resolver may scale the cited substitute to 56.25 ml milk plus 18.75 g butter. The non-risk plan auto-accepts; a simulated meal-completion event applies consumption. Expected carryover: 568.75 ml milk, 3 pcs eggs, 300 g baby spinach, and 46.25 g butter.

## Demo Day 3, Round 3: allergen gate

Baby spinach enters its two-day estimated-quality window. Opening stock includes 300 g baby spinach, 80 g walnuts, 250 g penne, and 3 pcs eggs. The remaining 568.75 ml milk is past its use-by date and unavailable for cooking. A new guest has peanut and tree-nut allergies.

The [spinach and walnut pesto](https://www.bbcgoodfood.com/recipes/spaghetti-spinach-walnut-pesto) candidate is intentionally unsafe for this guest. The rule gate pauses the workflow, shows the affected guest and source, and waits for a decision. The user rejects it; C issues a `FEEDBACK` command linked to B's Round 3 trace. No pantry quantity changes. A generic Accept action cannot execute the known allergen candidate.

## Demo pass conditions

Rounds 1 and 2 need no acceptance clicks. Round 2 visibly retrieves the real Round 1 trace. Round 3 pauses on the allergen conflict, and the rejection persists after refresh. Each round has its own timeline and opening stock. The exact ledger deltas and eight optional edge cases are in `seed_data/demo_script.json`.
