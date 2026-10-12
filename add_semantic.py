#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_semantic.py — 语义记忆统一落库脚本（供 cron 蒸馏/探索任务调用）

解决：cron 任务手写 JSON 追加导致字段不一致（content vs text）+ 无 embedding，
     让下游召回端无法识别。

用法：
  python add_semantic.py "记忆正文" --category 对话蒸馏 --date 2026-08-17
  python add_semantic.py "见解正文" --category 探索见解 --date 2026-08-27
  python add_semantic.py "情报正文" --category 技术情报 --project owner/repo --dedupe

查重闸门参数（2026-09-24 数据地图v2 第一步）：
  --project <owner/repo>  结构化项目键：落 project_key 字段（规范化小写、strip 前后空白）。
                          禁止事后用正则从正文猜项目名——键只在入口落一次。
  --dedupe                写前在库内按 project_key+category 查重；命中则拒写，
                          打印 "SKIP: duplicated (mem_xxx)" 到 stdout 并 exit 0（不算错误）。
  --force                 与 --dedupe 同用时强制放行（唯一绕过去重的方式）。

行为：
  - 用 bge-m3 算 embedding
  - 字段统一为 text + type + category + embedding（与 memory_manager.py 老条目一致）
  - 不合并（每条蒸馏/见解都是独立精华，不做相似度合并）
  - 只追加，不覆盖，不动 core_memory.md
  - 写入为原子写（.tmp + os.replace）；既有行按原始字节整段搬运（不重新序列化），
    字段不会漂移，追加语义不变
  - 未填 --project 的条目照常写入（不强制）；--dedupe 只对有 project_key 的条目生效
"""
import json
import os
import sys
import time
import argparse
from datetime import datetime

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from hermes_config import OLLAMA_URL, SEMANTIC_MEMORIES  # noqa: E402

MEMORY_FILE = SEMANTIC_MEMORIES
EMBED_MODEL = "bge-m3"
PROJECT_KEY_FIELD = "project_key"


def normalize_project_key(raw):
    """规范化 project_key：strip 前后空白 + 小写。空/无值返回 None（=不落字段）。"""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return key or None


def _key_tail(pk):
    """project_key 末段：'owner/repo' → 'repo'（键写歪 owner 时仍可兜底命中）。"""
    s = str(pk).strip().lower()
    return s.rsplit("/", 1)[-1] if "/" in s else s


def find_duplicate(project_key, category, filepath=MEMORY_FILE):
    """按 project_key+category 在库内查重，命中返回既有条目 id，否则 None。

    只用结构化字段匹配（project_key/category），绝不从正文正则猜项目名。
    无 project_key 的条目一律不参与去重池（未填 --project 的条目照常写入）。
    两级匹配（上游 cron 查重失灵修复）：
      1) 全键精确匹配（owner/repo == owner/repo）
      2) 末段兜底（repo == repo）——同一 cron 三天写了三个键
         （qwen3-tts / qwenlm/qwen3-tts / qwen/qwen3-tts）全漏的教训：
         owner 抄错不影响「同一项目」判定，末段相同即命中。
    """
    if not project_key:
        return None
    cat = (category or "").strip().lower()
    try:
        from memory_common import load_memories_safe
        mems = load_memories_safe(filepath)
    except Exception as e:  # noqa: BLE001
        print(f"⚠ 查重读取失败（按未命中处理，不阻塞写入）：{e}", file=sys.stderr)
        return None
    # 第一级：全键精确
    for m in mems:
        if not isinstance(m, dict):
            continue
        pk = m.get(PROJECT_KEY_FIELD)
        if not pk:
            continue
        if str(pk).strip().lower() != project_key:
            continue
        if str(m.get("category", "")).strip().lower() != cat:
            continue
        return m.get("id") or "unknown-id"
    # 第二级：末段兜底（同 category 下末段撞键 = 同一项目）
    tail = _key_tail(project_key)
    for m in mems:
        if not isinstance(m, dict):
            continue
        pk = m.get(PROJECT_KEY_FIELD)
        if not pk:
            continue
        if _key_tail(pk) != tail:
            continue
        if str(m.get("category", "")).strip().lower() != cat:
            continue
        print(f"⚠ 末段兜底命中：新键 '{project_key}' 与库内 '{pk}' 同项目（末段 {tail}）",
              file=sys.stderr)
        return m.get("id") or "unknown-id"
    return None


def _append_line_locked(entry, filepath):
    """锁内原子追加（调用者必须已持有 memory_file_lock）。

    .tmp + os.replace：读者要么看到旧文件要么看到新文件，不会读到半行。
    既有内容按字节原样搬运（不重新 json 序列化）→ 老条目字段不漂移。
    """
    new_bytes = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    # 首写防御（sandbox 暴露）：目标文件不存在时 old 置空，不炸 FileNotFoundError
    old = b""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            old = f.read()
    tmp = str(filepath) + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(old)
            if old and not old.endswith(b"\n"):
                f.write(b"\n")
            f.write(new_bytes)
        os.replace(tmp, str(filepath))
    except OSError:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def embed_text(text: str):
    """调 bge-m3 生成 embedding，带一次重试（冷启动慢）。"""
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
    print(f"✗ embedding 失败：{last_err}", file=sys.stderr)
    return None


def _stamp_version():
    """语义库版本戳刷新（P2-3 统一双源→单源，2026-10-3）。

    原实现自造口径（全文件字节 sha256 → _snapshots/semantic_version.json），
    与 stamp_semantic.py（稳定字段内容 sha256 → 根目录 semantic_version.json）
    两个文件两种口径互不知晓——版本戳双源病。现统一改调 stamp_semantic，
    单一口径单一真源。失败不影响落库。
    """
    try:
        import subprocess as _sp
        _r = _sp.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "stamp_semantic.py")],
                     capture_output=True, text=True, timeout=60)
        if _r.returncode != 0:
            print(f"⚠ 版本戳重算失败（不影响落库）: {_r.stderr.strip()[:120]}", file=sys.stderr)
    except Exception as _e:
        print(f"⚠ 版本戳刷新失败（不影响落库）: {_e}", file=sys.stderr)


def add_semantic(text: str, category: str, event_date: str = None, mem_type: str = "general",
                 project: str = None, dedupe: bool = False,
                 force: bool = False, supersede: bool = False) -> bool:
    text = (text or "").strip()
    if not text:
        print("✗ 内容为空，跳过", file=sys.stderr)
        return False

    project_key = normalize_project_key(project)
    if project is not None and project_key is None:
        print("⚠ --project 为空，按未提供处理（不落 project_key）", file=sys.stderr)
    if dedupe and not project_key:
        print("⚠ --dedupe 已忽略：未提供 --project（只对有 project_key 的条目生效）",
              file=sys.stderr)
    if supersede and not project_key:
        print("⚠ --supersede 已忽略：未提供 --project（只对有 project_key 的条目生效，照常写入）",
              file=sys.stderr)
    if supersede and (dedupe or force):
        print("✗ --supersede 与 --dedupe/--force 互斥（保旧/换新/并存三选一）", file=sys.stderr)
        return False

    # 查重快路径：命中直接拒写，连 embedding 都不用算（ollama 挂了也能正确 SKIP）
    if dedupe and project_key and not force:
        dup_id = find_duplicate(project_key, category)
        if dup_id:
            print(f"SKIP: duplicated ({dup_id})")
            return True

    emb = embed_text(text)
    if emb is None:
        return False

    now = datetime.now().isoformat()
    entry = {
        "id": f"mem_{int(time.time() * 1000)}_{os.getpid()}",
        "text": text,
        "type": mem_type,
        "category": category,
        "embedding": emb,
        "timestamp": now,
        "energy": 2.0,
        "last_hit": now,
        "archived": False,
    }
    if event_date:
        entry["event_date"] = event_date
    if project_key:
        entry[PROJECT_KEY_FIELD] = project_key
    # P2-4 修复stable_key 提取搬进 add_semantic——
    # 与 memory_manager.add_memory L129-133 同款逻辑（[主题_子题] 前缀 → 键字段），
    # 检索侧 retrieve_memories 键优先置顶由此生效。存量条目不批量补键（增量执行纪律）。
    if text.startswith("[") and "]" in text:
        _key = text[:text.find("]") + 1]
        if 2 < len(_key) <= 40:            # 键长合理才收录（防误吞长段）
            entry["stable_key"] = _key

    # 锁内「查重/supersede + 原子追加」为同一事务：关掉 check-then-write 的竞态窗口
    from memory_common import memory_file_lock, append_memories_safe
    try:
        with memory_file_lock(MEMORY_FILE, fail_fast=True):  # P2-7：写方 fail-fast
            if dedupe and project_key and not force:
                dup_id = find_duplicate(project_key, category)
                if dup_id:
                    print(f"SKIP: duplicated ({dup_id})")
                    return True
            if supersede and project_key:
                # supersede 链：
                # 锁内三步合一——查旧 → 改旧（archived+链字段）→ 写新，单次整库重写。
                # 旧条目 id 缺失（legacy）→ 跳过该条 + 警告，不炸不静默换血。
                from memory_common import load_memories_safe, save_memories_atomic
                mems = load_memories_safe(MEMORY_FILE)
                now_iso = datetime.now().isoformat()
                old_ids = []
                for m in mems:
                    if not isinstance(m, dict) or m.get("archived", False):
                        continue
                    pk = m.get(PROJECT_KEY_FIELD)
                    if not pk or str(pk).strip().lower() != project_key:
                        continue
                    if str(m.get("category", "")).strip().lower() != (category or "").strip().lower():
                        continue
                    mid = m.get("id")
                    if not mid:
                        print(f"⚠ supersede 跳过无 id 的 legacy 条目（text[:30]={str(m.get('text',''))[:30]}）",
                              file=sys.stderr)
                        continue
                    m["archived"] = True
                    m["superseded_by"] = entry["id"]
                    m["supersede_date"] = now_iso
                    m["_archive_note"] = f"superseded by {entry['id']}"
                    old_ids.append(mid)
                if old_ids:
                    entry["supersedes"] = old_ids
                    save_memories_atomic(mems + [entry], MEMORY_FILE)
                    print(f"SUPERSEDE: {len(old_ids)} 条旧条目归档链回 {entry['id']}")
                    _stamp_version()
                    return True
                # 无同键旧条目 → 走正常追加（supersede 语义退化为普通写入）
                print("SUPERSEDE: 无同键活跃条目，按普通写入")
            _append_line_locked(entry, MEMORY_FILE)
    except OSError as e:
        # 原子写失败（如 Windows 上目标文件被占）→ 退回原追加路径，绝不静默丢记忆
        # supersede 模式禁用回退——
        # 回退 append 只落新条目不归档旧条目=半事务，宁可报错重试（cron 有兜底）
        if supersede:
            print(f"✗ SUPERSEDE 原子写失败（{e}），拒绝回退（半事务），请重试", file=sys.stderr)
            return False
        print(f"⚠ 原子写失败（{e}），回退锁内追加", file=sys.stderr)
        append_memories_safe([entry], MEMORY_FILE)

    # 原则（业界同款）:每次批量变更后刷新版本戳
    _stamp_version()

    _tag = f" [project={project_key}]" if project_key else ""
    print(f"✓ 已落库 [{category}]{_tag} {text[:30]}...")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语义记忆统一落库")
    # B9 断言9：三模式互斥（互斥判定在函数层 L180-182，CLI 只传参——非 argparse 互斥组）
    parser.add_argument("text", type=str, help="记忆/见解正文")
    parser.add_argument("--category", type=str, required=True, help="分类（对话蒸馏/探索见解等）")
    parser.add_argument("--type", dest="mem_type", type=str, default="general",
                        choices=["general", "event", "emotional", "preference"],
                        help="记忆类型（general/event/emotional/preference，默认 general）")
    parser.add_argument("--date", type=str, default=None, help="事件日期（如 2026-08-17）")
    parser.add_argument("--project", type=str, default=None,
                        help="结构化项目键（owner/repo，落 project_key，规范化小写）")
    parser.add_argument("--dedupe", action="store_true",
                        help="按 project_key+category 查重，命中拒写并打印 SKIP（exit 0）")
    parser.add_argument("--force", action="store_true",
                        help="与 --dedupe 同用时强制放行写入")
    # supersede 链：三模式互斥——保旧/换新/并存
    parser.add_argument("--supersede", action="store_true",
                        help="同键旧条目归档链回+新条目写入（换新；时效性事实演进用）")
    args = parser.parse_args()

    ok = add_semantic(args.text, args.category, args.date, mem_type=args.mem_type,
                      project=args.project, dedupe=args.dedupe, force=args.force,
                      supersede=args.supersede)
    sys.exit(0 if ok else 1)
