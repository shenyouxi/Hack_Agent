# Autonomous Pantry Agent: Build Prompt (10-hour scoped version, v2.2)

Precedence: docs/conventions.md wins over this file, and docs/pantry_agent_v2_1_patch_A.md overrides sections 7, 9.2, 11 and 14 where they differ.

## A. How to work (read first)
1. Follow the order in section 13 strictly. After each step run `pytest` and report: done / not done / what I need to provide. Do not move on while tests are red.
2. All external calls (LLM, Linkup, Tavily, Mongo, MLflow) live only in thin wrappers under `tools/`, `llm.py` and `store/`. SDK parameter names and return shapes must be verified against the official docs and by `scripts/preflight.py`; record what you verify in `docs/api_notes.md`. SDK call snippets in this file are hints and may be inaccurate.
3. If a key is missing, do not stop: continue with the matching Fake implementation and list at the end which keys are missing and which features are therefore Fake.
4. Do not widen scope. P1 starts only after every P0 completion criterion in section 16 is met. No over-abstraction; every file starts with a one or two sentence header comment.
5. Secrets come only from environment variables. Provide `.env.example`; `.env` is git-ignored and must never be printed or committed.
6. All generated files, comments, docstrings, JSON, logs, error messages and UI text are in English.

## 0. Goal
Build an autonomous agent triggered by pantry state changes. With no human typing a query, it completes this loop:
sense (trigger) -> retrieve past experience -> plan -> detect gaps -> cited search for substitutes -> rule gate -> write back -> record a trace.
It stops for one human confirmation only on allergens or low confidence.

## 1. Hard principles
1. The LLM only understands and drafts JSON. Numbers, dates, gap detection, allergen checks, routing and database writes are done by deterministic code.
2. Every LLM output is validated with pydantic v2; on failure retry once with the error message; if it still fails, use the fallback. Use temperature=0.
3. Every external dependency has a fallback (section 10). A fallback must be recorded in the trace as `degraded=true` with a reason; never pretend it succeeded.
4. No fabrication: a substitute must have a citation URL returned by Linkup; no citation means the confirmation gate.
5. Every run has an idempotency key; every step writes a checkpoint; after a worker restart, runs resume from their checkpoint.
6. Ingredient names are English lowercase (for example `milk`, `heavy cream`). All external search queries are English.
7. Write tests and Fakes before features.

## 2. Scope
P0: triggers EXPIRY_IMMINENT and STOCK_DEPLETED; Sentinel, Planner, gap detection, Resolver (Linkup + Tavily), rule gate, write-back and undo, decision traces, Atlas Vector Search for seed recall, persistent pending confirmation, Streamlit UI, command-line demo, MLflow tracing.
P1: CONSTRAINT_SHIFT (a UI text box); a "Cooked it" button (reservation to consumption); feedback write-back.
Out of scope: EXTERNAL_SHOCK, Agno, AgentKit, image recognition, calorie calculation, multi-user, cross-dimension unit conversion.

## 3. Stack and environment variables
Python 3.12; pydantic>=2, pymongo, openai, linkup-sdk, tavily-python, mlflow, streamlit, numpy, pytest, python-dotenv, certifi. sentence-transformers is an optional embedding fallback only.

Environment variables: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (OpenNext, OpenAI compatible) | MONGODB_URI, MONGODB_DB | LINKUP_API_KEY | TAVILY_API_KEY | EMBED_PROVIDER (openai|local), EMBED_MODEL | MLFLOW_TRACKING_URI (optional; with Databricks also DATABRICKS_HOST, DATABRICKS_TOKEN) | CACHE_FIRST (true|false) | STORE_BACKEND (mongo|json) | TICK_SECONDS (default 2) | DEMO_START_DATE (default 2026-11-14).

Known facts: embeddings are OpenNext `text-embedding-3-small`, dimension 1536; `response_format=json_object` works; every Mongo connection must pass `tlsCAFile=certifi.where()`; load `.env` with `load_dotenv(".env")`.

## 4. Layout (create exactly this)
```
pantry/
  config.py            # reads environment variables
  models.py            # all pydantic schemas
  rules.py             # ALLERGEN_MAP, STAPLES, ALIASES, priority formula, trigger rules
  llm.py               # LLM (abstract), OpenNextLLM, FakeLLM
  store/base.py        # Store interface
  store/json_store.py  # JsonStore: only for pytest and single-process offline work
  store/mongo_store.py # MongoStore
  tools/linkup.py  tools/tavily.py  tools/cache.py   # each with a Fake version
  agents/sentinel.py planner.py resolver.py gate.py writeback.py
  orchestrator.py      # state machine, checkpoints, recovery
  tracing.py           # trace context manager: writes runs.steps / decision_traces / MLflow span together
  worker.py            # python -m pantry.worker: the only background loop
  run_demo.py          # in-process command-line demo (can use the json backend)
  ui/app.py            # streamlit run pantry/ui/app.py
scripts/preflight.py seed.py reset_demo.py prewarm.py
seed_data/             # handwritten JSON: pantry_items.json, seed_traces.json, allergen_map.json, staples.json, aliases.json, demo_script.json
tests/
docs/api_notes.md
README.md
```

## 5. Process model (mandatory)
- The worker process is the only place agent logic runs. Every TICK_SECONDS seconds it does: (1) process commands, (2) tick() (compute scores, emit triggers, dedupe onto the bus), (3) advance every non-terminal run.
- The Streamlit UI only reads Mongo and writes commands to the `commands` collection; it never runs agent logic. It polls every 2 seconds (st.fragment(run_every=...) or equivalent).
- commands: {command_id, type, payload, status (pending/done/failed), created_at}. Types: ADVANCE_CLOCK, APPROVE, REJECT, FEEDBACK, INJECT_DEMO_RESULT, UNDO_RUN, (P1) COOKED, SET_CONSTRAINT.
- Pending confirmation: the UI writes APPROVE or REJECT; the worker processes it and advances that run.

## 6. Data model (collections)
- pantry_items: item_name (English canonical), aliases[], category, quantity, reserved, unit (pcs/g/ml), expiry_date, safety_stock, allergen_tags[], priority_score, last_updated. There is no label_zh field.
- inventory_events (append only): event_id, item_name, type (RESERVED/RELEASED/CONSUMED/ADJUSTED), delta, estimated, run_id, reverses_event_id, created_at
- user_profile: allergens[], constraints{max_minutes}, today (simulated clock; all date math uses only this)
- bus_events: trigger_type, item_name, severity, dedupe_key, status, created_at
- runs: run_id, trigger, item_name, state, checkpoints{}, steps[{ts, name, status, detail}] (source of the UI timeline), pending_confirmation, idempotency_key, created_at
- decision_traces (model Trace): trace_id, run_id, day_index, trigger, snapshot_text, embedding, constraints[], plan, retrieved_traces[{trace_id, similarity, day_index, dish_name, is_seed}], resolver_actions[{tool, query, answer, sources[{title, url, content}], degraded, cached_at, demo_injected}], steps[], risk_status (none|allergen|no_citation|search_unavailable|constraint_violation), final_outcome (accepted|rejected|pending_confirmation), is_seed, created_at
- feedback: op_id (unique), trace_id, outcome (accepted|rejected), source (user|auto), created_at
- shopping_list (item_name unique), search_cache (key is a hash of the normalized query; stores the original call time), plan_cache (key is hash(snapshot_text + constraints + allergens)), commands
- Unique indexes: runs.idempotency_key, bus_events.dedupe_key, shopping_list.item_name, feedback.op_id.

Inventory invariants (asserted after every write; a violation rolls back): quantity >= 0; 0 <= reserved <= quantity; event unit equals item unit; the same idempotency key takes effect once. Mismatched units are always rejected; no conversion.

## 7. Rules and triggers (rules.py, pure code)
- days = expiry_date - today
- expiry_u = clamp((5 - days) / 5, 0, 1); low_stock_u = 1 if quantity <= safety_stock else 0 (use physical quantity, not available quantity, so the agent's own reservations do not retrigger it)
- priority_score = 0.7 * expiry_u + 0.3 * low_stock_u, recomputed on every inventory change and every clock advance
- EXPIRY_IMMINENT: 0 <= days <= 2
- STOCK_DEPLETED: quantity <= safety_stock
- days < 0 (already expired): no trigger, excluded from planning, shown in red in the UI as "needs manual handling"
- severity: days == 0 or quantity == 0 is high, otherwise low
- dedupe_key = trigger_type:item_name:today
- ALLERGEN_MAP: about 10 handwritten allergen groups (nuts, peanut, dairy, egg, gluten, soy, shellfish, fish, sesame, ...) with English synonyms and derivatives (for example nuts maps to almond, cashew, walnut, almond milk, almond flour), lowercase word-boundary matching.
- STAPLES: salt, water, pepper and similar never count as gaps.

## 8. State machine and routing (routing is plain if statements in code)
```
TRIGGERED -> CONTEXT_BUILT -> SEEDS_RETRIEVED -> PLANNED -> GAP_CHECKED
  |- no gap needing search -> GATED
  `- MISSING gap (or STOCK_DEPLETED) -> RESOLVING -> GATED
GATED -> APPLIED | PENDING_CONFIRMATION
PENDING_CONFIRMATION -> (APPROVE) APPLIED | (REJECT) REJECTED
APPLIED/REJECTED -> RECORDED
```
- Every state transition writes runs.checkpoints and one runs.steps entry and opens an MLflow span; on start the worker scans non-terminal runs and resumes them.
- Gap kinds: MISSING (not in pantry) goes to the Resolver; INSUFFICIENT (in pantry but not enough) goes only to the shopping list, no search.
- STOCK_DEPLETED: the depleted item itself is a MISSING gap handed to the Resolver to find substitutes; the Planner still plans normally without that item.
- At most 2 MISSING gaps per run, and at most 1 Resolver round per gap.

## 9. Modules
### 9.1 Sentinel (pure code)
Read inventory, compute scores, emit triggers, package {trigger, items_ranked, constraints, allergens, snapshot_text} (expired items excluded). snapshot_text is built by a deterministic template using relative days only, for example:
`trigger=EXPIRY_IMMINENT; expiring=milk(2d); have=eggs,flour,butter,sugar; constraint=25min; allergens=nuts`

### 9.2 Planner
1. Embed snapshot_text and run `$vectorSearch` on decision_traces filtered by final_outcome="accepted" and the same trigger; take top 3 as seeds and store their similarity scores in the trace. (The patch file defines the hybrid Atlas plus local immediate search.)
2. If CACHE_FIRST=true and plan_cache hits, use the cached plan (mark the trace cached). Otherwise call the LLM with the pantry (with units), constraints, allergens and seeds. Output: {dish_name, servings, est_minutes, ingredients[{name, amount, unit}], steps[], rationale, used_seed_trace_ids[]}.
3. Code validation: est_minutes <= max_minutes; no ingredient name matches ALLERGEN_MAP; units of pantry ingredients equal pantry units. On failure retry once with the specific violation.
4. Amounts of pantry ingredients are marked estimated=true.

### 9.3 Gap detection (pure code)
Match by canonical name and aliases; STAPLES never count. Output gaps[{name, needed, unit, kind: MISSING|INSUFFICIENT}].

### 9.4 Resolver (for each MISSING gap)
1. Cache: if CACHE_FIRST=true and search_cache hits, use it (trace says "cached from real call at <time>").
2. Otherwise call Linkup through LinkupClient.search(query, depth) with sourcedAnswer output; depth "standard" by default, "deep" for prewarming and multi-step reasoning. The query is English with the order written out, for example: "First list common substitutes for {gap} in baking, then give the quantity conversion for each." Timeout 25 seconds, then fall back to cache or degrade.
3. Take the top source URL and call TavilyClient.extract(url) to get the page text.
4. Ask the LLM to turn "Linkup answer + page text" into {substitute_name, components[{name, amount, unit}], ratio_text, source_url, confidence}.
5. Code validation: source_url must appear in Linkup's sources; components non-empty; on failure retry once.
6. Prefer substitutes built from pantry ingredients; components that must be bought go to the shopping list.
7. INJECT_DEMO_RESULT: the worker creates a new run for the same gap (idempotency key suffix ":demo-inject"), skips search and uses a preset result whose substitute contains almond milk, with sources URL https://example.com/demo-injected and demo_injected=true, clearly labelled "demo injection" in the UI and trace. The rule gate runs as usual and must not be bypassed.

### 9.5 Rule gate (pure code)
A run goes to PENDING_CONFIRMATION if and only if:
- substitute_name or any components.name matches a user allergen; or
- there is no citation URL, or no amount or ratio could be extracted; or
- the Planner still violates constraints after its retry; or
- search is unavailable (degraded to the last tier).
Write a structured confirmation request (what it wants to do, the evidence, citation URLs, the rule that fired).

### 9.6 Write-back (only for APPLIED runs)
- Planned amounts are written as RESERVED events and increase `reserved` (estimated=true); `quantity` is not decremented. Reservations cannot exceed quantity - reserved; any excess becomes an INSUFFICIENT gap on the shopping list.
- undo_run(run_id) appends a reversing event for every event of that run and marks the matching trace rejected.
- commit_run(run_id) converts the run's reservations into CONSUMED events (see the patch file); the worker calls it while handling ADVANCE_CLOCK.

### 9.7 Recording
- Every finished run writes one Trace (including the embedding). final_outcome: gate passes automatically means accepted; user approves means accepted; user rejects means rejected; undo means rejected; waiting means pending_confirmation. Seed recall uses only accepted traces. Every outcome change also goes through Store.save_feedback (auto-accept uses source="auto", op_id="auto:" + trace_id).
- One trace context manager in tracing.py writes runs.steps, decision_traces and the MLflow span together; do not write two instrumentations. Span tree: Sentinel -> Planner[vector_search, llm] -> Resolver[linkup, tavily, llm] -> Gate -> Writeback -> Record.

## 10. External dependencies and fallbacks
| Module | Preferred | Fallback |
|---|---|---|
| LLM structured output | response_format=json_object | prompt asks for JSON only + pydantic validation + retry |
| Embeddings | EMBED_PROVIDER=openai | local all-MiniLM-L6-v2 (384 dimensions) |
| Vector search | Atlas $vectorSearch merged with local immediate search | local numpy cosine over accepted traces only (same interface, trace marked degraded) |
| Search | Linkup standard | cache hit; if still failing, go to the confirmation gate |
| Page extraction | Tavily extract | use only the Linkup answer, confidence drops to low, go to the confirmation gate |
| Tracing | MLflow to Databricks | local mlflow ui; then the UI's own timeline |

Atlas vector index (dimension detected at startup from the real embedding; changing the embedding model requires reseeding and rebuilding the index):
```
{"fields":[
 {"type":"vector","path":"embedding","numDimensions":<DIM>,"similarity":"cosine"},
 {"type":"filter","path":"final_outcome"},
 {"type":"filter","path":"trigger"}]}
```
Create it with pymongo create_search_index(SearchIndexModel(..., type="vectorSearch")) and poll until queryable.

## 11. Seed data and scripts
- Pantry (relative to DEMO_START_DATE): milk 1000 ml (+3 days, safety 200), eggs 10 pcs (+14, safety 2), flour 1000 g (+180, safety 200), butter 250 g (+20, safety 50), sugar 500 g (+365, safety 100); allergens: nuts; max_minutes=25.
- Handwritten seed traces (15 to 30, never LLM generated) in seed_data/seed_traces.json using structured fields; at least one accepted EXPIRY_IMMINENT plan involving milk needs heavy cream so the gap appears naturally; at least 3 STOCK_DEPLETED seeds. snapshot_text and embeddings are generated by seed.py with the same function used at runtime.
- scripts/seed.py: writes inventory, rule data and seed traces (with embeddings), builds the vector index and unique indexes.
- scripts/reset_demo.py: restores day 1 (see the patch file); keeps caches and seed traces.
- scripts/prewarm.py: runs the whole multi-day demo live with CACHE_FIRST=false, fills plan_cache and search_cache, resets, then runs an offline replay check.
- scripts/preflight.py: checks and prints OK/FAIL plus the fallback to enable for: LLM JSON output, embeddings and dimension, Mongo connection and index creation, Linkup, Tavily, MLflow.

## 12. UI and demo
Single Streamlit page with four blocks: pantry table (sorted by priority_score, expired items in red) | live agent timeline (reads runs.steps; shows vector search similarity, Linkup citation URLs, cached/degraded flags) | pending confirmation card (approve/reject) | shopping list.
Demo control buttons (labelled "Demo controls"): advance clock by one day; inject a substitute result containing almond milk; undo this run.

Demo script:
1. Click "advance clock": milk has 2 days left, EXPIRY_IMMINENT fires, the pantry score rises, the bus shows a timestamped record.
2. The Planner retrieves seeds; the UI shows the matched past traces and their similarity.
3. The plan has a gap (expected heavy cream, whatever the real run gives); the Resolver calls Linkup; the UI shows the citation list; Tavily extracts the ratio.
4. The gate passes: stock is reserved and the shopping list updates automatically.
5. Show the new decision trace and the MLflow span tree.
6. Click "inject substitute result": the gate holds the run because almond milk matches nuts; a confirmation card appears; click reject and the trace is recorded as rejected.

## 13. Build order (10-hour budget; every step ends in a runnable state)
- H0-1: skeleton, config, models, rules, Store (JSON + Mongo), FakeLLM and fake tools, seed.py, preflight.py, run preflight.
- H1-3: Sentinel, bus, priorities, vector search (with numpy fallback), Planner, gap detection plus unit tests.
- H3-5: Resolver (with cache and injection), gate, write-back, undo.
- H5-6.5: orchestrator state machine, checkpoints and recovery, tracing.py, MLflow, worker.py.
- H6.5-8: Streamlit UI and the command mechanism.
- H8-9: prewarm.py, three full demo rehearsals, more tests.
- H9-10: buffer, README (Linkup does cited judgement, Tavily extracts from known pages; explain the simulated clock, the demo injection and each fallback), submission material.

Stop-loss lines: if the Resolver is still unstable at H5, demo from the cache and say so; if MLflow is not working at H7, use the UI timeline; spend at most one hour on a Databricks workspace.

## 14. Tests (write first; everything runs offline with Fakes)
Unit: priority and trigger rules (expired items do not trigger; reservations do not trigger low stock); dedupe; gap detection (aliases, staples, MISSING vs INSUFFICIENT); allergen mapping (including almond milk maps to nuts); the four gate rules; inventory invariants (negative rolls back, idempotency, undo restores); checkpoint recovery (kill the worker mid-run, then resume).
Integration: full loop; Atlas and numpy fallback return the same top-k; Linkup timeout falls back to cache; the command mechanism (APPROVE/REJECT/UNDO/FEEDBACK); pending state survives a worker restart; three consecutive rounds keep quantities correct.
LLM validation: Planner violating duration, allergen or unit constraints triggers a retry; a Resolver source_url not in the sources is rejected.
Metrics: invariant violations 0; fabricated URLs or ingredients 0; end-to-end loop time.

## 15. Runtime prompts (for the LLM)
Planner system prompt
```
You are the Planner of an autonomous pantry agent. Propose ONE recipe that uses the items most urgent to use.
Rules:
1. Output only JSON matching the given schema.
2. Respect max_minutes and never use any ingredient or derivative of the user's allergens.
3. For ingredients that are in the pantry, use exactly the unit shown in the pantry list, and use the exact item names from the pantry list.
4. Prefer ingredients with the highest priority_score. Seed plans are past accepted plans for similar situations; adapt them, do not copy blindly, and list the seed ids you used.
5. Do not invent pantry items. If you need something not in the pantry, still list it; the system will detect the gap.
6. All amounts are approximate estimates.
```
Resolver extraction prompt
```
You receive a sourced answer and page text about substituting {missing_item}. Extract ONE substitute as JSON: substitute_name, components[{name, amount, unit}], ratio_text, source_url, confidence.
Rules: use only information present in the provided text; source_url must be one of the provided source URLs; if the amounts are not stated, leave them null and set confidence to "low". Never invent numbers or URLs. Prefer substitutes made of ingredients in the pantry list.
```

## 16. P0 completion criteria
- Demo script steps 1 to 6 succeed three times in a row with CACHE_FIRST=true (run reset_demo.py between runs).
- At least one real Linkup call is cached and its citation URLs are visible in the UI.
- Vector search really runs on Atlas (similarity scores are shown).
- The gate holds a run in both the allergen case and the no-citation case, and the pending state survives a worker restart.
- All tests pass and there are 0 invariant violations.

## 17. Forbidden
Do not write LLM-generated numbers to the ledger without estimated=true; do not let the LLM call any write function; do not hardcode secrets; do not present cached or injected results as live; do not bypass the gate for the demo; do not run agent logic inside Streamlit.