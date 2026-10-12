# Hybrid Retrieval: BM25 + Vectors, Fused by RRF

## The failure mode that forced this design

Pure vector retrieval finds entries that are *literally similar* to the query. In production we hit the opposite case: the entry that mattered used different wording than the query, and the semantically-nearest entries were the wrong ones. Call it **"semantic ≠ causal"**:

- Query: "why did the nightly job fail last week" — the entry that matters says `nightly_job.py exit code 15 at 23:00, state file stale`. Vector similarity ranks it below three entries about *feelings about* night shifts, because those have more overlapping concept-space with "nightly".
- Query: "that tea thing" — BM25 finds nothing useful; vectors nail it.

Neither signal is sufficient. So `memory_hybrid.py` runs both and fuses the **ranks**:

```
RRF_score(d) = Σ_i  1 / (k + rank_i(d)),   k = 60
```

RRF (Reciprocal Rank Fusion) needs no score calibration between systems — BM25 scores and cosine similarities are not comparable, but ranks always are. `k=60` is the literature default: large enough to smooth score noise, small enough to keep top ranks meaningful. An entry that ranks high in *both* lists gets two large reciprocal terms and wins; an entry that is a one-signal fluke gets one.

## Tokenization (CJK-aware)

`[a-zA-Z0-9_]+|[\u4e00-\u9fff]` — English/numbers tokenize as words, Chinese tokenizes per-character. This makes BM25 work on mixed-language memories without any external segmenter. It is crude and it is enough: for a companion's memory scale (10³–10⁴ entries), per-character CJK indexing beats a dependency on jieba/pkuseg.

## What we measured

On the production corpus (1,000+ entries, mixed Chinese/English, technical + emotional content):

- Pure vector: misses exact-symbol queries (file names, error codes, port numbers) that BM25 returns at rank 1.
- Pure BM25: misses paraphrase queries that vectors return at rank 1.
- RRF fusion: returns both in the top-3. The cost is one extra in-memory index build per process — at this scale, milliseconds.

Industry cross-check (2026-08): hybrid retrieval improves recall 26–31% over pure vector on BEIR (nDCG 43→52); Mem0's own three-signal fusion (semantic + keyword + entity) confirms combined scoring beats any single signal.

## When NOT to use this

If your memory store is >10⁵ entries or needs sub-ms latency, move to a real vector DB with a BM25 sidecar. This kernel exists because at companion scale, plain files + one Python file beat an infrastructure dependency — and a file your person can `grep` is worth more than 30ms.
