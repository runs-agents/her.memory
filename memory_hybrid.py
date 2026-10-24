#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_hybrid.py — 混合检索内核（BM25 关键词 + 向量语义，RRF 融合）

解决 2026-8-29 实锤的「语义≠因果」检索盲区：
纯向量检索只找"字面像"，如果关键条目的措辞跟 query 字面不像就漏召回。
BM25 恰好相反：精确符号（端口号、文件名、报错码）一查一个准。

业界依据（2026-08 检索）：
- Hybrid search 比纯向量 recall 提升 26-31%（BEIR 基准 nDCG 43→52）
- Mem0 三信号融合（semantic + keyword + entity）验证组合分数优于单一信号
- RRF（Reciprocal Rank Fusion）不依赖分数可比性，只用排名，最稳

融合公式：RRF_score(d) = Σ 1/(k + rank_i(d))，k=60（业界默认）
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

# RRF 常数（论文默认 60，越大越平滑）
RRF_K = 60
# 中文/英文分词：中文按字切，英文/数字按词切
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")

# BM25 参数
_BM25_K1 = 1.5
_BM25_B = 0.75


def tokenize(text: str) -> List[str]:
    """极简分词：英文单词/数字整体保留（保住 9880、SoVITS 这类实体），
    中文字符逐字切（bge-m3 语义路已有词级理解，这里 BM25 路只要字面命中）。"""
    if not text:
        return []
    return _TOKEN_RE.findall(text)


class BM25Index:
    """轻量 BM25 索引（纯 Python，无外部依赖，适用 <1000 条记忆的规模）。"""

    def __init__(self, docs: Sequence[str]):
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self._avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        self.N = len(docs)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, idx: int) -> float:
        toks = self.doc_tokens[idx]
        if not toks or not self.doc_len[idx]:
            return 0.0
        tf = Counter(toks)
        dl = self.doc_len[idx]
        s = 0.0
        for term in set(tokenize(query)):
            f = tf.get(term, 0)
            if f == 0:
                continue
            s += self._idf(term) * f * (_BM25_K1 + 1) / (
                f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / (self.avgdl or 1))
            )
        return s

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def scores(self, query: str) -> List[float]:
        return [self.score(query, i) for i in range(self.N)]


def rrf_fuse(dense_ranked: Sequence[int], sparse_ranked: Sequence[int],
             top_k: int, k: int = RRF_K) -> List[int]:
    """Reciprocal Rank Fusion：输入两路已排好的索引列表，输出融合排名。"""
    return rrf_fuse_multi([dense_ranked, sparse_ranked], top_k, k)


def rrf_fuse_multi(ranked_lists: Sequence[Sequence[int]],
                   top_k: int, k: int = RRF_K) -> List[int]:
    """多路 RRF（2026-10-24 时间路升级）：任意路数的排名列表融合。

    RRF_score(d) = Σ_i 1/(k + rank_i(d) + 1)。路数增加时各路贡献
    天然被稀释，不需要重新调参——这是选 RRF 而非分数加权的原因。
    """
    rrf: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf, key=lambda i: -rrf[i])[:top_k]


# ── 时间路（2026-10-24）：temporal intent 触发式，平时不激活 ──────────
# 设计依据：EverMe/Hindsight 的 temporal 检索路，轻量版——
# query 带时间意图词时，时间倒序路才参与 RRF；普通查询两路不动，
# 行为零变化（无关近期条目不污染日常召回）。
_TIME_INTENT_RE = re.compile(
    r"最近|上次|上回|昨天|前天|今天|今晚|昨晚|上周|上个月|这周|这几天|"
    r"最新|近况|后来|后续|进展|现在还|还在吗|还记不记得|lately|recent"
)
# recency 路参与融合的窗口：只有时间倒序前这么多条拿 RRF 分，
# 防止全库按时间排队给近期条目无条件加分
RECENCY_WINDOW = 100


def temporal_intent(query: str) -> bool:
    """query 是否带时间意图（「上次聊的那个」「最近怎么样」类）。"""
    return bool(_TIME_INTENT_RE.search(query or ""))


def _parse_time(s: str) -> float:
    """event_date('2026-08-15') / timestamp(ISO) → 可比较的 epoch 秒。失败 0。"""
    if not s:
        return 0.0
    from datetime import datetime
    t = str(s).strip()
    try:
        return datetime.fromisoformat(t).timestamp()
    except ValueError:
        return 0.0


def recency_ranked(dates: Sequence[str], window: int = RECENCY_WINDOW) -> List[int]:
    """按时间倒序（新→旧）返回索引，截取前 window 条参与 RRF。

    时间解析失败的条目排最后（拿不到 recency 分，但不报错）。
    """
    order = sorted(range(len(dates)), key=lambda i: -_parse_time(dates[i]))
    return order[:window]


def hybrid_rank(query: str, docs: Sequence[str], dense_scores: Sequence[float],
                top_k: int, min_dense: float = 0.0,
                dates: Optional[Sequence[str]] = None) -> List[int]:
    """一步到位：给 query、全部文档文本、 dense 相似度分数 → 返回融合后的 top-k 索引。

    - dense 路：调用方已算好的余弦相似度，降序排名
    - sparse 路：BM25 分数，降序排名
    - 两路 RRF 融合；dense 全低于 min_dense 时仍走 sparse（兜底）
    - 时间路（2026-10-24）：dates 给了且 query 带时间意图时，
      时间倒序前 RECENCY_WINDOW 条作为第三路参与融合。
      dates 与 docs 索引对齐（event_date 优先，fallback timestamp）。
    """
    if not docs:
        return []
    n = min(len(docs), len(dense_scores))
    dense_order = sorted(range(n), key=lambda i: -float(dense_scores[i]))
    bm25 = BM25Index(list(docs)[:n])
    sparse_scores = bm25.scores(query)
    sparse_ranked = sorted(range(n), key=lambda i: -sparse_scores[i])
    # 过滤掉 BM25 全 0 的（词面完全不沾边的路不算分）
    if max(sparse_scores, default=0) <= 0:
        sparse_ranked = []
    lists = [dense_order, sparse_ranked]
    if dates and temporal_intent(query):
        rec = recency_ranked([str(d) for d in list(dates)[:n]])
        if rec:
            lists.append(rec)
    return rrf_fuse_multi(lists, top_k)