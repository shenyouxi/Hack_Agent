# Team C demo UI and feedback handoff

The UI should display **Demo Day N · Round N · virtual date**, the triggering ingredient and its label date, and a separate persistent timeline for each round. A timeline row should show its time, actor, action, outcome, and any real source URL. Round 2 must show the identifier of the newly created Round 1 trace that memory retrieved. Round 3 must show the affected guest, tree-nut conflict, and pending decision.

## Controls

- **Auto-run** defaults to on and advances the event-driven workflow without a Continue click for each step.
- **Auto-accept non-risk outcomes** defaults to on, but only applies when the rule gate permits it. It must create feedback with `source="auto"` and `op_id="auto:" + trace_id`.
- Provide Start/Pause, Reset, and round navigation. A single-round launch remains available as a fallback.
- Known allergen conflicts stay pending. A generic Accept click must not execute an unsafe meal for the allergic guest; the user may reject the candidate or correct an inaccurate profile and rerun the rules.

## Feedback contract from `docs/conventions.md`

B returns the real `trace_id` after writing each decision. C issues a `FEEDBACK` command with `trace_id`, `outcome`, `op_id`, and `source`. The command handler calls `Store.save_feedback(trace_id, outcome, op_id, source) -> Feedback`. A repeated `op_id` must be idempotent. The UI changes an action card to Accepted or Rejected **only after the command status is `done`**. On failure, it keeps the pending state and retries with the same `op_id`. Rejecting an applied plan releases its reservation through the agreed backend command.

Example manual rejection command payload:

```json
{
  "type": "FEEDBACK",
  "trace_id": "real_round_3_trace_id",
  "outcome": "rejected",
  "op_id": "manual:round_3:reject:001",
  "source": "manual"
}
```

## Acceptance checks

1. Rounds 1 and 2 complete with no manual acceptance click while both automation settings are on.
2. Round 2 shows the real Round 1 trace ID and begins with Round 1's carried-over stock.
3. Round 3 pauses on the tree-nut conflict and does not deduct inventory.
4. Manual rejection remains visible after refresh and is attached to the same trace.
5. Repeated clicks or network retries do not create duplicate feedback.

The demo expectations live in `seed_data/demo_script.json`; the UI must show actual runtime steps as they occur. The current API notes say Linkup works and Tavily is not yet in use, so a Tavily extraction appears in the timeline only if it was actually executed.
