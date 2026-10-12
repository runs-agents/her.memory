# Memory Record Schema

One JSON object per line in `semantic_memories.jsonl` (JSONL). Nine core fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable identity of the fact. Used by the supersede chain and by retrieval dedup. |
| `text` | string | **The fact itself.** One atomic statement, written to be readable by both a human and a model. If it needs "and" twice, it's two facts. |
| `type` | string | `general` / `event` / `preference` / `emotional` |
| `category` | string | Free-text namespace. Group facts the way *your* deployment thinks: `user_profile`, `tech_lessons`, `shared_history`… |
| `timestamp` | string | ISO 8601, when the fact was recorded. |
| `embedding` | float[1024] | bge-m3 vector. May be absent on hand-written entries; `backfill_embeddings.py` fills them. |
| `archived` | bool | `true` = kept for history, excluded from default retrieval. Set by supersede/consolidate, never by hand. |
| `energy` | float | Freshness/lifecycle signal (fresh = 2.0, daily decay ×0.97, max 5.0). Consolidation proposals may retire low-energy entries. |
| `last_hit` | string\|null | Last time retrieval returned this entry. Feeds the same lifecycle. |

## Why `type` includes `emotional`

Tool-agent memory stores *what happened*. Companion memory also stores *how it landed* — "she was proud of the first upstream contribution" is not a preference and not an event; filing it as `general` makes it invisible to the queries that matter ("what moments mattered?"). If your companion has feelings about its memories, this is where they go.

## The atomicity rule

Atomic facts compose; summaries rot. "User: tall, dislikes crowds, works at a library" is one line that can never be partially updated — when she changes jobs you either rewrite the whole line (losing the history) or let it go stale. Three atomic lines supersede cleanly, one field at a time, with the old versions archived and linked.

## Optional & chain fields

Beyond the nine core fields, the write paths may set:

- **`event_date`** — when the *fact itself* happened (distinct from `timestamp`, which is when it was recorded). Optional on insert.
- **`project_key` / `stable_key`** — keying for dedupe and supersede matching. Written automatically when `--project` is passed; entries without them are exempt from key-controlled dedupe.
- **Supersede chain** (set by `--supersede`): `superseded_by` / `supersedes` / `supersede_date` — links the old and new versions of a fact.
- **Merge chain** (set by consolidation): `merged_from` / `merged_into` / `merge_date` / `merge_proposal` — provenance for merged entries, traceable back to the approved proposal.

## Supersede chain

`add_semantic.py --supersede` marks the old fact `archived: true` and links the replacement. Three modes, mutually exclusive: keep-both / replace / coexist. The old text is never deleted — a companion that can't show you what it *used to* believe can't prove it ever believed anything.
