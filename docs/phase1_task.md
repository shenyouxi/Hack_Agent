# Phase 1 task for part A

Read these files completely first:
1. docs/pantry_agent_build_prompt.md (main prompt)
2. docs/pantry_agent_v2_1_patch_A.md (patch for part A)
3. docs/conventions.md (naming and interfaces; wins over the other two when they conflict)

Environment facts: Python 3.12, .venv is active; embeddings use OpenNext text-embedding-3-small, dimension 1536, EMBED_PROVIDER=openai; response_format=json_object works; every Mongo connection must pass tlsCAFile=certifi.where(); load .env with load_dotenv(".env"), never the no-argument form. The .env file already exists: never commit it and never print its contents.

Language rule: all generated files, comments, docstrings, JSON, logs, error messages and UI text must be in English. The Ingredient model has no label_zh field.

Implement only part A files. Do not touch agents/, tools/, ui/, orchestrator.py or worker.py.

Phase 1 only. Create these three files, then stop and report:
1. pantry/config.py: read environment variables (including DEMO_START_DATE, default 2026-11-15).
2. pantry/models.py: pydantic v2 models covering every collection in section 6 of the main prompt, plus Ingredient, Trace (trace_id, run_id, day_index, trigger, plan, retrieved_traces, resolver_actions, steps, risk_status, final_outcome, is_seed, created_at), Run, Feedback, Command, Gap, TriggerEvent, the Planner output schema, the Resolver output schema, Source and LinkupResult.
3. pantry/store/base.py: abstract Store interface covering reads and writes for every collection, command polling, idempotent writes and inventory invariant checks. Include the signatures (signatures and docstrings only, no implementation) of get_today, advance_day, set_today, commit_run, search_traces and save_feedback.

Requirements: full type annotations; a one or two sentence header comment in each file; verify with: python -c "import pantry.models, pantry.store.base". When done, list every public class name and method signature so they can be shared with teammates.
