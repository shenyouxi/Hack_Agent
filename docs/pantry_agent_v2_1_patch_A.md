# Patch for part A (v2.1): overrides the main prompt

Source: five requests from part B. Only touches files owned by A: store/, rules.py, scripts/, tests/.

## 1. New decision records must be searchable immediately
- Store provides `search_traces(query_vec, trigger, k)`. The result merges Atlas `$vectorSearch` with a local immediate query, deduplicated by trace_id, then top-k by similarity.
- Local immediate query: a plain `find` for traces with `final_outcome="accepted"` and the same `trigger` (including embedding), cosine similarity computed with numpy. A plain find is visible immediately and is not affected by Atlas index sync delay.
- If Atlas is unavailable or the index is not ready, use only the local immediate query (mark the trace as degraded).
- Test: write a new trace, call `search_traces` immediately, and it must be found (use a fake where the Atlas side returns nothing).

## 2. Current date storage and "advance one day"
- `user_profile.today` stores an ISO date string.
- Store methods: `get_today()`, `advance_day(n=1) -> new date`, `set_today(d)`. `advance_day` recomputes `priority_score` of every pantry item with the new date in the same operation.
- Config `DEMO_START_DATE` (default `2026-11-15`) is used by seed and reset. All expiry dates are computed relative to DEMO_START_DATE.
- `snapshot_text` uses relative days (for example `milk(2d)`), never absolute dates, so caches hit across rounds. Add a test for the snapshot builder in rules.py.

## 3. reset_demo.py: one command back to "day 1"
Restore: inventory (quantity, reserved, expiry), `today=DEMO_START_DATE`, `runs`, `bus_events`, `commands`, `inventory_events`, `shopping_list`, pending confirmations, and every decision trace with `is_seed=false`.
Do NOT clear: `search_cache`, `plan_cache`, traces with `is_seed=true`.
Seed traces must carry `is_seed=true`. The script is idempotent (running it twice gives the same result) and prints a summary.

## 4. Inventory write-back stays consistent across rounds
- Inventory is a projection of the event ledger: at any time `quantity == initial seed quantity + sum(events)` and `reserved == sum(RESERVED) - sum(RELEASED)`.
- Store provides `commit_run(run_id)`: converts the run's RESERVED events into CONSUMED events (estimated=true, labelled "assumed cooked in the demo") and releases the reservation. The worker calls it for every APPLIED, uncommitted run while handling `ADVANCE_CLOCK`, then advances the date. The end state of round N is the starting state of round N+1.
- New test `test_three_rounds`: run 3 consecutive rounds with fakes (trigger, plan, reserve, ADVANCE_CLOCK commit, next day). After every round assert: inventory invariants, quantity equals the hand-computed expectation, ledger replay equals stored values, and the start of round N+1 equals the end of round N.

## 5. prewarm.py covers every round
- Read `seed_data/demo_script.json` (provided by B): `{"rounds":[{"day":1,"expect_trigger":"EXPIRY_IMMINENT","queries":["..."]}, ...]}`.
- With `CACHE_FIRST=false`, run the whole multi-day demo live (including each round's ADVANCE_CLOCK and commit), write every round's plan and search results to the caches, and record the real call time.
- Cache keys: query lowercased with whitespace collapsed, then hashed; plan key is `hash(snapshot_text + constraints + allergens)`.
- Afterwards reset automatically, then run an offline replay check: `CACHE_FIRST=true`, make every real Linkup, Tavily and LLM call raise an error, and run the whole multi-day demo again. If any round misses the cache, list the missing query or snapshot and exit non-zero.
