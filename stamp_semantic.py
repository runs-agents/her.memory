#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stamp_semantic.py — 语义库版本戳重算（借鉴外部项目实践）

用途：每次批量变更语义库（写入/归档/重排）后跑一次，产出「全库 hash + 条目数」，
写进 semantic_version.json。得到语义库版本号 → 跨会话可确认「我读的是哪一版记忆」，
出事能精确回滚到某一版，而不是凭时间戳猜。

用法：
  python stamp_semantic.py            # 重算并写入
  python stamp_semantic.py --check    # 只比对，不写（用于自检/心跳）

hash 算法：对每条记忆的稳定字段（text/type/category/date）按行序拼接后 sha256。
不 hash embedding（浮点数组体积大且重算会变），保证「内容不变 → 版本号不变」。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
MEM_FILE = os.path.join(BASE, "semantic_memories.jsonl")
ARCH_FILE = os.path.join(BASE, "archived_memories.jsonl")  # maintain 物理移档后归档条目住这里
OUT_FILE = os.path.join(BASE, "semantic_version.json")


def _entry_key(d: dict) -> str:
    """条目稳定指纹：只用内容字段，不用 embedding / 加载时补算字段。"""
    text = str(d.get("text") or d.get("content") or "")
    typ = str(d.get("type") or "")
    cat = str(d.get("category") or "")
    date = str(d.get("date") or d.get("event_date") or "")
    return f"{typ}\x1f{cat}\x1f{date}\x1f{text}"


def compute() -> dict:
    entries_total = 0
    entries_archived = 0
    h = hashlib.sha256()
    # 2026-9-19 修：maintain 把归档条目物理移入 archived_memories.jsonl 后
    # 本脚本只扫活跃文件导致归档计数恒 0——两个文件都扫，归档文件条目计入 archived
    scan_files = [MEM_FILE] + ([ARCH_FILE] if os.path.exists(ARCH_FILE) else [])
    for path in scan_files:
      with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            entries_total += 1
            if path == ARCH_FILE or d.get("archived") or d.get("_archived"):
                entries_archived += 1
            h.update(_entry_key(d).encode("utf-8"))
            h.update(b"\n")
    digest = h.hexdigest()
    return {
        "schema_version": 1,
        "content_sha256": digest,
        "version_short": digest[:12],
        "entries_total": entries_total,
        "entries_archived": entries_archived,
        "entries_active": entries_total - entries_archived,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "语义库版本号 —— 跨会话可确认『我读的是哪一版记忆』；批量变更后重算。",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只比对不写入（不一致时退出码 1）")
    args = ap.parse_args()

    if not os.path.exists(MEM_FILE):
        print(f"✗ 找不到语义库：{MEM_FILE}", file=sys.stderr)
        return 2

    cur = compute()

    if args.check:
        if not os.path.exists(OUT_FILE):
            print(f"✗ 版本戳文件不存在：{OUT_FILE}")
            return 1
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            old = json.load(f)
        same = old.get("content_sha256") == cur["content_sha256"]
        print(f"  戳记 : {old.get('version_short')} | {old.get('entries_total')} 条 | {old.get('built_at')}")
        print(f"  实际 : {cur['version_short']} | {cur['entries_total']} 条")
        if same:
            print("  ✓ 一致")
            return 0
        print("  ⚠ 不一致 —— 语义库已变更，需要重算")
        return 1

    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT_FILE)
    print(f"✓ 版本戳已重算: {cur['version_short']} | {cur['entries_total']} 条"
          f"（活跃 {cur['entries_active']} / 归档 {cur['entries_archived']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
