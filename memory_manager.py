#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地语义记忆管理器
用法：
  python memory_manager.py add "记忆文本" --type preference
  python memory_manager.py retrieve "用户输入" --top 8
  python memory_manager.py maintain
  python memory_manager.py export            # 导出全部活跃记忆
  python memory_manager.py stats             # 查看统计
"""
import json
import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import requests

from memory_common import (
    memory_file_lock,
    load_memories_safe,
    save_memories_atomic,
    log_hit,
    drain_hits,
    clear_hits,
)

# ============ 配置区 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from hermes_config import OLLAMA_URL, SEMANTIC_MEMORIES, HOME, OLLAMA_PORT  # noqa: E402

CONFIG = {
    "ollama_url": OLLAMA_URL,
    "embed_model": "bge-m3",          # 本地嵌入模型
    "memory_file": SEMANTIC_MEMORIES,
    "archive_file": os.path.join(BASE_DIR, "archived_memories.jsonl"),

    # 按记忆类型设定合并阈值（超过则合并）
    "type_thresholds": {
        "preference": 0.90,   # 事实性偏好
        "event": 0.80,        # 事件性记忆
        "emotional": 0.75,    # 情感/关系记忆
        "general": 0.85
    },

    # 能量机制参数
    "energy_initial": 2.0,
    "energy_decay": 0.97,           # 每日衰减系数（0.97 = 更"怀旧"）
    "energy_hit_boost": 1.0,        # 被合并/命中时增加的能量
    "energy_retrieve_boost": 0.2,   # 被检索命中时的小幅增强
    "energy_max": 5.0,
    "energy_archive_threshold": 0.5,
    "archive_days": 90,             # 连续多少天未命中才归档

    "top_k": 8
}

# ============ 基础函数 ============
def load_memories(filepath, include_archived=False):
    """加载 JSONL 记忆文件（容错：损坏行跳过并留痕，不崩溃）"""
    mems = load_memories_safe(filepath)
    if include_archived:
        return mems
    # 守卫：归档文件 + 过滤模式 = 语义错误调用，直接拒绝
    # 归档条目天然 archived=True，过滤后恒空——9-28 丢账案的第三口就咬在这
    if os.path.basename(filepath) == "archived_memories.jsonl":
        raise ValueError(
            f"对归档文件 {filepath} 用 include_archived=False 语义错误"
            f"（归档条目天然 archived=True，过滤后恒空）——请用 load_archive_file()"
        )
    return [m for m in mems if not m.get("archived", False)]

def save_memories(mems, filepath):
    """保存记忆到 JSONL 文件（原子写入，须在 memory_file_lock 内调用）"""
    save_memories_atomic(mems, filepath)

def embed_text(text):
    """调用 Ollama 生成 embedding（Ollama 不在线时明确报错退出，不崩溃堆栈）"""
    try:
        r = requests.post(
            CONFIG["ollama_url"],
            json={"model": CONFIG["embed_model"], "prompt": text},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        emb = data["embedding"]
        return np.array(emb, dtype=np.float32)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Ollama 不在线（localhost:{OLLAMA_PORT} 连不上）。请先启动 Ollama 并确认 bge-m3 模型可用。"
        )
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        raise RuntimeError(f"embedding 生成失败：{e}")

def cosine(a, b):
    """余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ============ 核心操作 ============
def add_memory(text, mem_type="general", energy=None, event_date=None, category=None):
    """添加一条新记忆。

    【人格红线 2026-08-29】只追加，不覆盖、不合并：
    相似度去重/合并只在 consolidate 的提案流程做（owner review 后才动）。
    add 路径对已有记忆零修改。

    【稳定定位器 2026-09-18 · 抄点2】text 若以 [主题_子题] 开头，
    自动提取为 stable_key 字段——召回侧可先按键精确 grep、向量兜底。
    存量条目不批量补键（增量执行纪律，同抄点6）。
    """
    vec = embed_text(text)
    now = datetime.now().isoformat()

    new_mem = {
        "id": f"mem_{int(time.time() * 1000)}_{os.getpid()}",
        "text": text,
        "type": mem_type,
        "embedding": vec.tolist(),
        "timestamp": now,
        "energy": energy if energy is not None else CONFIG["energy_initial"],
        "last_hit": now,
        "archived": False
    }
    # 稳定定位器（抄点2）：[主题_子题] 前缀 → stable_key
    if text.startswith("[") and "]" in text:
        key = text[:text.find("]") + 1]
        if 2 < len(key) <= 40:            # 键长合理才收录（防误吞长段）
            new_mem["stable_key"] = key
    if event_date:
        new_mem["event_date"] = event_date
    if category:
        new_mem["category"] = category

    # 锁内 append：不读整库、不重写，与其他写方互斥
    from memory_common import append_memories_safe
    append_memories_safe([new_mem], CONFIG["memory_file"])
    mems = load_memories(CONFIG["memory_file"])
    print(f"✓ 记忆已追加（类型：{mem_type}，当前活跃记忆数：{len(mems)}）")

def retrieve_memories(query, top_k=None, include_archived=False):
    """根据用户输入检索相关记忆。

    【只读 2026-08-29】检索不写库：能量增强/last_hit 更新挪到
    daily_maintenance 批量做（维护时统一读取检索日志）。高频写库
    既有竞态风险又浪费 IO。

    【稳定定位器 2026-09-18 · 抄点2】query 含 [xxx_yyy] 形态键时，
    先按 stable_key 精确匹配（grep 语义），向量只做兜底——
    「键够用时不要先搜索」（外部项目借鉴原则）。
    """
    mems = load_memories(CONFIG["memory_file"], include_archived=include_archived)
    if not mems:
        return []

    # ── 抄点2：稳定键优先 ──────────────────────────────
    import re as _re
    key_hits = []
    key_m = _re.search(r"(\[[^\]\[]{2,40}\])", query)
    if key_m:
        want = key_m.group(1)
        key_hits = [m for m in mems
                    if m.get("stable_key") == want or m.get("text", "").startswith(want)]
        # 键命中：精确结果置顶，其余仍走向量（键兜底不排他）
        if key_hits:
            # P2-8 修复（2026-10-3）：dict 相等比较 → id() 集合——
            # 内容完全相同的两条会被 `m not in key_hits` 一并排除（dict 相等非身份），
            # id() 集合按身份判重，只排除真正命中的那几条
            hit_ids = {id(m) for m in key_hits}
            rest = [m for m in mems if id(m) not in hit_ids]
            mems = key_hits + rest

    query_vec = embed_text(query)
    scored = []
    skipped_no_emb = 0

    for m in mems:
        # include_archived=False 时才过滤归档记忆（审查 🔴-2，2026-08-29：
        # 原来无条件 continue，导致参数形同虚设）
        if not include_archived and m.get("archived", False):
            continue
        # P1-4 修复（2026-10-7，kit v2.1 同步）：裸 m["embedding"] 对缺向量条目
        # 直接 KeyError——README 自检第一步就是 retrieve，新字段/老数据一跑就炸。
        # 改为跳过并计数，结尾统一告警（不静默：静默丢弃比报错危险）。
        emb = m.get("embedding")
        if not (isinstance(emb, list) and emb):
            skipped_no_emb += 1
            continue
        sim = cosine(query_vec, np.array(emb))
        energy = m.get("energy", CONFIG["energy_initial"])
        # 能量权重压缩到 [0.5,1.0]（Qwen A2 修复马太效应）
        w = 0.5 + 0.5 * min(energy, CONFIG["energy_max"]) / CONFIG["energy_max"]
        score = sim * w
        scored.append((score, sim, energy, m))

    # === Hybrid 升级（2026-8-29）：dense+energy 路与 BM25 关键词路 RRF 融合 ===
    try:
        from memory_hybrid import hybrid_rank  # type: ignore
        docs = [m.get("text", "") for _, _, _, m in scored]
        dense = [s for s, _, _, _ in scored]
        # 时间路（2026-10-24）：event_date 优先，fallback timestamp；
        # query 带时间意图时时间倒序路参与 RRF（详见 memory_hybrid.py）
        dates = [m.get("event_date") or m.get("timestamp") or ""
                 for _, _, _, m in scored]
        fused_order = hybrid_rank(query, docs, dense, len(scored), dates=dates)
        scored = [scored[i] for i in fused_order]
    except ImportError:
        scored.sort(key=lambda x: x[0], reverse=True)  # 老路径兜底

    results = []
    for score, sim, energy, m in scored[:top_k or CONFIG["top_k"]]:
        results.append({
            "id": m.get("id", f"noid_{m.get('timestamp','?')}"),
            "text": m.get("text", ""),
            "type": m.get("type", "general"),
            "similarity": round(sim, 3),
            "energy": round(energy, 3)
        })
        # 命中 ID 落盘（2026-08-29 审查 🔴-1：原来存函数对象属性，跨进程丢失，
        # 维护进程永远读不到 → 记忆只衰减不充能。现改为 retrieval_hits.jsonl）
        mid = m.get("id") or f"noid_{m.get('timestamp','?')}"
        log_hit(mid)

    # P1-4 配套（2026-10-7）：跳过的缺向量条目不静默——提示用 backfill 补算
    if skipped_no_emb:
        print(f"⚠ retrieve 跳过 {skipped_no_emb} 条缺 embedding 的条目"
              f"（跑 backfill_embeddings.py 可补算）", file=sys.stderr)

    return results

def daily_maintenance():
    """每日维护：能量衰减、低频归档、批量结算检索命中。

    整个「读→改→写」在文件锁事务内完成，防止与 add/consolidate 并发互相覆盖。
    """
    with memory_file_lock(CONFIG["memory_file"], fail_fast=True):  # P2-7：写方 fail-fast
        mems = load_memories(CONFIG["memory_file"], include_archived=True)
        # 2026-10-1 P0-1 修归档文件条目天然 archived=True，
        # 带过滤读永远得空列表 → 9-28 假象「写入未落盘」、10-1 真丢 17 条。
        # 归档文件必须不过滤读。
        archived = load_memories(CONFIG["archive_file"], include_archived=True)
        now = datetime.now()
        active = []

        # 批量结算上次维护以来的检索命中（retrieve 已改只读，能量在这里补；
        # 命中 ID 来自 retrieval_hits.jsonl，跨进程持久化，2026-08-29 🔴-1 修复）
        # P1-1 修①缺 id 的 legacy 条目跳过而非崩溃；
        # ②drain 后移到结算成功之后——原顺序先读走清空再结算，中途崩溃=命中数据永久丢失。
        hit_ids = drain_hits(keep=True)
        if hit_ids:
            for m in mems:
                mid = m.get("id")
                if mid and mid in hit_ids:
                    m["energy"] = min(
                        CONFIG["energy_max"],
                        m.get("energy", CONFIG["energy_initial"]) + CONFIG["energy_retrieve_boost"]
                    )
                    m["last_hit"] = now.isoformat()
            # 日志已在 drain_hits() 里读走清空，无需再重置

        for m in mems:
            # 能量衰减
            m["energy"] = max(0.1, m.get("energy", CONFIG["energy_initial"]) * CONFIG["energy_decay"])

            # 如果已归档，移到归档文件
            if m.get("archived", False):
                archived.append(m)
                continue

            # 低频归档判定（Qwen A4：改单条件——超过 archive_days 未命中即归档，energy 降为辅助信号）
            last_hit = datetime.fromisoformat(m.get("last_hit", now.isoformat()))
            if (now - last_hit).days > CONFIG["archive_days"]:
                m["archived"] = True
                archived.append(m)
                continue

            active.append(m)

        with memory_file_lock(CONFIG["archive_file"], fail_fast=True):  # P2-7：写方 fail-fast
            save_memories(active, CONFIG["memory_file"])
            save_memories(archived, CONFIG["archive_file"])
        # P1-1：写盘成功后才清空命中日志（崩溃时命中数据仍在，下轮重试）
        clear_hits()
    print(f"✓ 维护完成：活跃记忆 {len(active)} 条，归档 {len(archived)} 条")

def export_memories():
    """导出全部活跃记忆（纯文本）"""
    mems = load_memories(CONFIG["memory_file"])
    if not mems:
        print("（暂无活跃记忆）")
        return
    for i, m in enumerate(mems, 1):
        extra = ""
        if m.get("event_date"):
            extra += f", 日期:{m['event_date']}"
        if m.get("category"):
            extra += f", 分类:{m['category']}"
        print(f"{i}. [{m['type']}]{extra} {m['text']} (能量:{m.get('energy', 0):.1f}, 更新:{m.get('last_hit','')[:10]})")

def stats():
    """查看统计"""
    active = load_memories(CONFIG["memory_file"])
    archived = load_memories(CONFIG["archive_file"], include_archived=True)  # P0-1 同款修复
    by_type = {}
    for m in active:
        t = m.get("type", "general")
        by_type[t] = by_type.get(t, 0) + 1
    print(f"活跃记忆: {len(active)} 条")
    print(f"归档记忆: {len(archived)} 条")
    print(f"按类型: {by_type}")

# ============ 命令行入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermes 本地语义记忆管理器")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add_parser = sub.add_parser("add", help="添加一条记忆")
    add_parser.add_argument("text", type=str, help="记忆内容")
    add_parser.add_argument("--type", default="general",
                            choices=list(CONFIG["type_thresholds"].keys()),
                            help="记忆类型")
    add_parser.add_argument("--date", type=str, default=None,
                            help="事件发生日期（如 2026-08-14）")
    add_parser.add_argument("--category", type=str, default=None,
                            help="分类（如 user_profile / tech_lessons）")

    ret_parser = sub.add_parser("retrieve", help="检索相关记忆")
    ret_parser.add_argument("query", type=str, help="用户当前输入")
    ret_parser.add_argument("--top", type=int, default=None, help="返回条数")

    sub.add_parser("maintain", help="执行每日维护")
    sub.add_parser("export", help="导出全部活跃记忆")
    sub.add_parser("stats", help="查看统计")

    args = parser.parse_args()

    if args.cmd == "add":
        add_memory(args.text, args.type, event_date=args.date, category=args.category)
    elif args.cmd == "retrieve":
        results = retrieve_memories(args.query, top_k=args.top)
        if results:
            for i, r in enumerate(results, 1):
                extra = ""
                if r.get("event_date"):
                    extra += f", 日期:{r['event_date']}"
                if r.get("category"):
                    extra += f", 分类:{r['category']}"
                print(f"{i}. [{r['type']}] {r['text']} (相似度:{r['similarity']}, 能量:{r['energy']}{extra})")
        else:
            print("（无相关记忆）")
    elif args.cmd == "maintain":
        daily_maintenance()
    elif args.cmd == "export":
        export_memories()
    elif args.cmd == "stats":
        stats()
