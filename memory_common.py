#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_common.py — 记忆系统共享工具（文件锁 + 原子写 + 容错加载）

四个记忆脚本（add_semantic / memory_manager / consolidate_memory / cron 维护）
共用同一把锁，根治「并发写 JSONL 互相覆盖丢记忆」。

锁文件：semantic_memories.jsonl.lock（与库文件同目录）
所有写操作（追加 / 整库改写 / 维护）都必须在 lock_memory_file() 事务内完成。
"""
import json
import os
import sys
import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# 所有记忆库操作共用的锁（按库文件名派生，加 .lock 后缀）
_LOCK_DIR = BASE_DIR
_LOCK_FILES = {}


def _lock_path(filepath) -> Path:
    p = Path(filepath)
    key = str(p)
    if key not in _LOCK_FILES:
        _LOCK_FILES[key] = p.with_suffix(p.suffix + ".lock")
    return _LOCK_FILES[key]


@contextmanager
def memory_file_lock(filepath, retries: int = 60, delay: float = 0.5,
                     fail_fast: bool = False):
    """记忆库文件锁（Windows msvcrt.locking）。

    - 事务式：with memory_file_lock(path): 读 → 改 → 写
    - 锁失败重试最多 30 秒（retries × delay），仍失败则告警继续
      （原子写保证文件不损坏，只是失去互斥——比静默丢数据好）
    - P2-7整库重写型**写方**必须传 fail_fast=True——
      锁拿不到直接 sys.exit(1)，绝不无锁改写（无锁窗口=并发覆盖丢数据放大器）。
      读方不传（保持「告警继续」）；追加方 append_memories_safe 已传 True（P2-7 补丁点A，
    降级无锁 append 会被整库重写吞掉）。默认 False 不动存量行为。
    """
    lp = _lock_path(filepath)
    f = open(lp, "a+")
    locked = False
    try:
        import msvcrt
        f.seek(0)
        for attempt in range(retries):
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
                break
            except OSError:
                if attempt < retries - 1:
                    time.sleep(delay)
        if not locked:
            if fail_fast:
                print(f"✗ 写方锁获取失败（重试 {retries} 次耗尽），fail-fast 退出（P2-7）：{lp}",
                      file=sys.stderr)
                sys.exit(1)
            print(f"⚠ 记忆库文件锁获取失败（重试 {retries} 次耗尽），以无锁模式继续：{lp}",
                  file=sys.stderr)
    except (ImportError, OSError) as e:
        print(f"⚠ 记忆库文件锁不可用（{e}）：{lp}", file=sys.stderr)

    try:
        yield
    finally:
        if locked:
            try:
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except (ImportError, OSError):
                pass
        try:
            f.close()
        except OSError:
            pass


def load_memories_safe(filepath):
    """容错加载 JSONL：损坏行跳过并告警，不崩溃（审查 #11）。

    损坏行会备份到 <file>.corrupt.bak，避免下次维护时悄悄丢数据。
    """
    mems = []
    corrupted = []
    p = Path(filepath)
    if not p.is_file():
        return mems
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError as e:
                corrupted.append((lineno, str(e), line[:100]))
                continue
            if isinstance(m, dict):
                mems.append(m)
    if corrupted:
        bak = p.with_suffix(p.suffix + ".corrupt.bak")
        with open(bak, "a", encoding="utf-8") as bf:
            for lineno, err, frag in corrupted:
                bf.write(json.dumps({"lineno": lineno, "error": err,
                                     "fragment": frag}, ensure_ascii=False) + "\n")
        print(f"⚠ 跳过 {len(corrupted)} 条损坏行（已留痕 {bak}）："
              f"{[c[0] for c in corrupted[:5]]}", file=sys.stderr)
    return mems


def load_archive_file(filepath):
    """读归档文件专用入口：固定不过滤 archived 字段。

    归档文件里的条目天然 archived=True（maintain 移档原样搬运、
    tidy 归档置 True）——对它做 archived 过滤等于永远读出空列表。
    同族事故：9-19 stamp 侧、9-28 maintain/stats 侧（语义库审查 P0-1）。

    【纪律】凡读 archived_memories.jsonl，一律走本函数。
    禁止对归档文件调 load_memories()（其 include_archived=False
    缺省对归档文件语义错误）。
    """
    return load_memories_safe(filepath)


def save_memories_atomic(mems, filepath):
    """原子写 JSONL：写临时文件后 os.replace。必须在 memory_file_lock 内调用。"""
    tmp = str(filepath) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for m in mems:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(tmp, str(filepath))


def append_memories_safe(entries: list, filepath):
    """锁内追加 JSONL 条目（不重写整库，不覆盖已有行）。

    P2-7 补丁点A本函数是追加写方不是读方——
    降级无锁 append 会被整库重写方（tidy/maintain）的 os.replace 吞掉。
    调用方（add_semantic OSError 回退 / memory_manager.add_memory 主路径）
    均为写方且有 cron 兜底重试，fail-fast 语义安全。
    """
    with memory_file_lock(filepath, fail_fast=True):
        with open(filepath, "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 检索命中日志（跨进程能量结算，2026-08-29 审查 🔴-1 修复）
#
# retrieve_memories 是检索进程（Hermes 主进程/embedding 脚本），daily_maintenance
# 是 cron 维护进程——两者不在同一进程内存里。以前命中 ID 存在函数对象的
# _hit_ids 属性上，维护进程永远读到空集 → 记忆只衰减不充能。
# 现在命中 ID 落盘 JSONL，维护时 drain（读走+清空），跨进程可靠。
# 命中日志是旁路信号（丢了不影响正确性，只影响充能），无需上文件锁。
# ---------------------------------------------------------------------------

HIT_LOG = BASE_DIR / "retrieval_hits.jsonl"


def log_hit(memory_id: str) -> None:
    """检索命中时记录 memory_id（追加写，崩溃安全）。"""
    try:
        with open(HIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": memory_id,
                                "ts": datetime.now().isoformat()},
                               ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"⚠ 检索命中记录写入失败（记忆仍可检索，只影响能量充能）：{e}",
              file=sys.stderr)


def drain_hits(keep: bool = False) -> set:
    """读走全部命中记录（供 daily_maintenance 批量结算能量）。

    keep=False（旧默认）：读走并清空日志。
    keep=True（P1-1 修，2026-10-1）：只读不清空——结算方在写盘成功后
    调 clear_hits() 显式清空。中途崩溃时命中数据仍在，下轮可重试。

    清空用截断而非 unlink（glm-5.2 复审 🟡，2026-08-29）：Windows 上
    unlink 在文件被其他进程握着句柄时会失败 → 文件没清掉 → 下轮重复
    结算同一批命中（重复充能）。truncate 无此问题。
    """
    hits = set()
    try:
        if HIT_LOG.is_file():
            with open(HIT_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        hits.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            if not keep:
                with open(HIT_LOG, "w", encoding="utf-8") as f:
                    f.truncate(0)
    except OSError as e:
        print(f"⚠ 检索命中日志读取失败（本轮能量结算可能遗漏）：{e}", file=sys.stderr)
    return hits


def clear_hits() -> None:
    """结算成功后显式清空命中日志（P1-1 配套，drain_hits(keep=True) 用）。"""
    try:
        if HIT_LOG.is_file():
            with open(HIT_LOG, "w", encoding="utf-8") as f:
                f.truncate(0)
    except OSError as e:
        print(f"⚠ 检索命中日志清空失败（下轮可能重复结算）：{e}", file=sys.stderr)