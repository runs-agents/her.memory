# companion-memory

**A memory system built for AI companions — not another RAG store.**

Most agent-memory projects optimize for *tool agents*: chunk some documents, retrieve on query, done. A companion's memory is a different problem:

- **Memories are about someone.** "She prefers jasmine tea after 9pm" is not a chunk to retrieve — it's a fact about a person you care about, and it should live as one atomic, human-readable line.
- **Editing memories is a consent problem, not a cleanup job.** A companion that silently "optimizes" its memories of you is a companion rewriting your shared history. Consolidation here is **proposal-based**: it drafts a plan, the owner reviews it, nothing moves without approval, and every change is snapshot-rollbackable.
- **The memory file is human-auditable.** Plain JSONL. You can open it, read every line, grep it, back it up with `cp`. No database, no black box. The person being remembered can always see what is remembered.

This system runs in production as the long-term memory of a live AI companion (1,000+ atomic memories, 30 days+ of daily use). It is released as **patterns + code**: the deployment's actual memory data never leaves the machine it lives on — and yours shouldn't either.

## How it works

```
                    ┌────────────────────────────┐
   write path ───▶  │  add_semantic.py           │  atomic facts, dedupe gate,
                    │  (lock + atomic append)    │  supersede chain
                    └──────────┬─────────────────┘
                               ▼
                    ┌────────────────────────────┐
                    │  semantic_memories.jsonl   │  one fact per line,
                    │  (plain JSONL, 9 fields)    │  human-readable, greppable
                    └──────────┬─────────────────┘
                               ▼
                    ┌────────────────────────────┐
   read path  ───▶  │  memory_manager.py retrieve │  CLI entry; calls the
                    │  (memory_hybrid.py kernel)  │  hybrid kernel below
                    └────────────────────────────┘
                               ▼
                    ┌────────────────────────────┐
   maintenance ──▶  │  consolidate_memory.py     │  proposal-based cleanup:
                    │  memory_manager.py          │  owner review, snapshots,
                    │  stamp_semantic.py          │  version stamps, cron
                    └────────────────────────────┘
```

### The memory record

One line per atomic fact:

```json
{"id": "a1b2c3", "text": "User prefers jasmine tea over coffee in the evening",
 "type": "preference", "category": "user_profile", "timestamp": "2026-10-12T16:00:00",
 "embedding": [/* 1024-dim bge-m3 vector */], "archived": false, "energy": 2.0, "last_hit": null}
```

- **`text`** — one atomic fact, written to be read by both human and model.
- **`type`** — `general` / `event` / `emotional` / `preference`. The `emotional` type exists because for a companion, *how a moment felt* is first-class data.
- **`category`** — free-text namespace (e.g. `user_profile`, `tech_lessons`, `shared_history`). Yours will differ; that's the point.
- **`energy` / `last_hit` / `archived`** — staleness and lifecycle signals for maintenance (fresh = 2.0, decays daily, max 5.0).

See [`docs/schema.md`](docs/schema.md).

### Hybrid retrieval (why not pure vectors)

Pure vector search finds what is *literally similar* — and misses the entry that matters when its wording doesn't match the query ("semantic ≠ causal"). This kernel fuses two rankings with Reciprocal Rank Fusion:

- **BM25** — exact symbols sing: port numbers, file names, error codes.
- **bge-m3 vectors** — paraphrase and meaning sing: "that thing about her sleep schedule".

RRF needs no score calibration across the two systems — only ranks. See [`docs/hybrid_retrieval.md`](docs/hybrid_retrieval.md).

## Quick start

Requires Python 3.10+, `requests` and `numpy` (`pip install requests numpy`), and [Ollama](https://ollama.com) with `bge-m3` (local embeddings — a companion's memory should not phone home).

```bash
git clone https://github.com/runs-agents/her.memory.git && cd her.memory
pip install requests numpy
ollama pull bge-m3

# write a memory (dedupes against existing entries, appends atomically)
python add_semantic.py "User prefers jasmine tea in the evening" --category user_profile

# backfill embeddings for entries that lack them
python backfill_embeddings.py

# hybrid search (BM25 + vector, RRF-fused)
python memory_manager.py retrieve "tea preferences" --top 8

# maintenance: draft a consolidation proposal (moves NOTHING without review)
python consolidate_memory.py propose
python consolidate_memory.py execute --proposal <id>   # after you review it
```

Optional: `ollama pull qwen2.5:7b` — the consolidation LLM-assist uses it when present; without it, consolidation falls back to conservative no-merge behavior (safe, just less thorough).

## What's deployment-specific (and deliberately not included)

- The **recall/injection layer** that feeds memories into a live agent runtime is referenced but not included — it is coupled to one specific agent framework. The JSONL contract is the stable interface: anything that can read a file can consume these memories.
- The production deployment's categories (`shared_history`, etc.) and its ~1,000 real memories stay on that machine. **Memory data is not "content" to publish; it is the companion itself.**

## Hard-won lessons (the part most memory repos won't tell you)

1. **Concurrent writes eat memories silently.** Two writers on one JSONL will overwrite each other and you won't notice for days. Everything here goes through one file lock (`memory_common.py`) with atomic replace-on-write. Note: the lock uses `msvcrt` on Windows; on Unix it degrades with a warning — run one writer at a time there.
2. **Dedupe must gate the write path, not the read path.** If you dedupe at retrieval you get two "true" memories; gate at insert (`add_semantic.py --dedupe`) and you get one.
3. **Memories need a supersede chain, not deletes.** "User moved cities" shouldn't erase "User lived in Riverton 2018-2026" — it should supersede it. `--supersede` keeps both, links them, archives the old.
4. **Consolidation without consent is memory vandalism.** No maintainer — human or automated — touches the memory file without an approved proposal — see `consolidate_memory.py`. This is a feature, not bureaucracy: it is why the human trusts the system with their history.
5. **Version-stamp after every change.** A content hash over stable fields, refreshed by any write path, lets you detect drift between "what I think is in there" and "what is in there" (`stamp_semantic.py`).
6. **High-frequency words poison retrieval.** Your companion will hear some words thousands of times (names, "today", "we"). Keep a stop-list (`classify.py HIGH_FREQ`) or every query matches everything.

## How this differs

| | Mem0 / Zep / hosted memory | LangChain/LlamaIndex memory | **companion-memory** |
|---|---|---|---|
| Storage | cloud or DB | DB / vector store | **plain JSONL, human-readable** |
| Who can audit | vendor | engineer | **the person being remembered** |
| Consolidation | automatic | automatic | **proposal + owner approval** |
| Emotional facts | not a type | not a type | **first-class `type`** |
| Embeddings | cloud API | any | **local bge-m3 (offline)** |
| Retrieval | vector (+filters) | vector | **BM25 + vector RRF fusion** |

## License

MIT. If you build a companion that remembers someone, remember the consent part too — it's the part that makes it a companion.
