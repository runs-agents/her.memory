#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_embeddings.py — 给 semantic_memories.jsonl 里缺 embedding 的条目补算向量并写回。

背景：对话蒸馏/探索见解任务早期手写 JSON 落库，字段是 content/category、无 embedding，
     导致下游召回端无法识别。本脚本用 bge-m3 补算，统一字段为 text/type/category/embedding。

用法：python backfill_embeddings.py
安全：只追加/改写缺 embedding 的条目，已有 embedding 的条目原样保留；写前自动备份。

【P1-3 修复（2026-10-7，kit 审查报告 4.3）】原实现裸 open 读写整库，绕过了
memory_file_lock 和 load_memories_safe——与 add/maintain/tidy 并发跑会互相覆盖丢数据，
损坏行还会直接 json.loads 崩溃。现改为：
  1. 读走 load_memories_safe（损坏行跳过留痕，不崩溃）；
  2. 整库重写走 memory_file_lock(fail_fast=True) + save_memories_atomic
     （整库重写型写方必须 fail-fast：锁拿不到直接退出，绝不无锁改写）；
  3. 备份在锁外先做（备份是只读操作，不参与互斥）。
"""
import json
import os
import sys
import time
from datetime import datetime

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from hermes_config import OLLAMA_URL, SEMANTIC_MEMORIES  # noqa: E402
from memory_common import load_memories_safe, memory_file_lock, save_memories_atomic  # noqa: E402

MEMORY_FILE = SEMANTIC_MEMORIES
EMBED_MODEL = "bge-m3"


def embed_text(text: str):
    last_err = None
    for _ in (1, 2):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                return emb
        except Exception as e:  # noqa: BLE001
            last_err = e
    print(f"  ✗ embedding 失败：{last_err}", file=sys.stderr)
    return None


def main():
    # 备份（只读源文件，锁外安全）
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(BASE_DIR, "_snapshots", f"semantic_memories_backfill_{ts}.jsonl")
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    with open(backup, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"✓ 已备份到 {backup}")

    # P1-3：安全加载（损坏行跳过留痕，不崩溃）
    entries = load_memories_safe(MEMORY_FILE)

    total = len(entries)
    backfilled = 0
    skipped = 0
    now = datetime.now().isoformat()

    for m in entries:
        # 已有 embedding 的跳过
        if isinstance(m.get("embedding"), list) and m.get("embedding"):
            skipped += 1
            continue

        # 取文本：兼容 content / text
        text = str(m.get("text") or m.get("content") or "").strip()
        if not text:
            skipped += 1
            continue

        emb = embed_text(text)
        if emb is None:
            print(f"  ✗ 跳过（embedding 失败）：{text[:30]}")
            skipped += 1
            continue

        # 统一字段：text + type + category + embedding
        category = m.get("category") or "general"
        m["text"] = text
        m["type"] = m.get("type") or "general"
        m["category"] = category
        m["embedding"] = emb
        m["timestamp"] = m.get("timestamp") or now
        m["energy"] = m.get("energy", 2.0)
        m["last_hit"] = m.get("last_hit") or now
        m["archived"] = m.get("archived", False)
        # 清理旧字段 content（避免双字段混淆）
        m.pop("content", None)
        backfilled += 1
        print(f"  ✓ 补算 [{category}] {text[:30]}...")

    # P1-3：锁内原子写回（整库重写型写方 fail-fast，绝不无锁改写）
    if backfilled:
        with memory_file_lock(MEMORY_FILE, fail_fast=True):
            save_memories_atomic(entries, MEMORY_FILE)
    else:
        print("（无条目需要补算，不写回）")

    print(f"\n完成：总 {total} 条，补算 {backfilled} 条，跳过（已有向量/失败）{skipped} 条")


if __name__ == "__main__":
    main()
