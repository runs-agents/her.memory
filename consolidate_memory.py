#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes 长期记忆整理提炼工具（融合版方案 v1.0）
实现 propose / execute / restore 三子命令
红线：核心身份文件永不碰；全流程 owner review + 备份 + 可回滚（知情同意才动）
"""
import json
import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from memory_common import memory_file_lock, load_memories_safe, save_memories_atomic

# ============ 配置区 ============
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
from hermes_config import OLLAMA_URL, SEMANTIC_MEMORIES, OLLAMA_PORT  # noqa: E402

CONFIG = {
    "ollama_url": OLLAMA_URL,
    "embed_model": "bge-m3",
    "memory_file": Path(SEMANTIC_MEMORIES),
    "archive_file": BASE_DIR / "archived_memories.jsonl",
    "snapshot_dir": BASE_DIR / "_snapshots",
    "proposal_dir": BASE_DIR / "_proposals",
    
    # 按记忆类型设定合并阈值（general 已修正为 0.80）
    "type_thresholds": {
        "preference": 0.90,
        "event": 0.80,
        "emotional": 0.75,
        "general": 0.80  # 修正：0.85→0.80
    },
    
    # 触发机制
    "trigger_count": 100,
    "trigger_size_mb": 2.5,
    "trigger_days": 30,
    
    # 能量与归档
    "energy_archive_threshold": 0.5,
    "archive_days": 90,
}

# 安全措辞红线
SAFE_WORDS = {"整理", "归档", "提炼", "精简"}
FORBIDDEN_WORDS = {"拆分", "删除", "清理", "移除"}

# ============ 基础函数 ============
def load_memories(filepath, include_archived=False):
    """加载 JSONL 记忆文件（容错：损坏行跳过并留痕，不崩溃）"""
    mems = load_memories_safe(filepath)
    if include_archived:
        return mems
    # 守卫：归档文件 + 过滤模式 = 语义错误调用
    import os as _os
    if _os.path.basename(filepath) == "archived_memories.jsonl":
        raise ValueError(
            f"对归档文件 {filepath} 用 include_archived=False 语义错误"
            f"（归档条目天然 archived=True，过滤后恒空）——请用 load_archive_file()"
        )
    return [m for m in mems if not m.get("archived", False)]

def save_memories(mems, filepath):
    """保存记忆到 JSONL 文件（原子写入，须在 memory_file_lock 内调用）"""
    save_memories_atomic(mems, filepath)

def embed_text(text):
    """调用 Ollama 生成 embedding"""
    try:
        r = requests.post(
            CONFIG["ollama_url"],
            json={"model": CONFIG["embed_model"], "prompt": text},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        return np.array(data["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"⚠ 嵌入失败（{e}），将使用 LLM 精判兜底")
        return None

def cosine(a, b):
    """余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def check_guardrails(text):
    """检查是否触碰安全措辞红线"""
    for word in FORBIDDEN_WORDS:
        if word in text:
            return False, f"包含禁用词「{word}」"
    return True, "通过"

def llm_judge_merge(mem1, mem2, mem_type):
    """
    使用 LLM 精判两条记忆是否真的可以合并
    返回：(can_merge: bool, reason: str)
    """
    prompt = f"""请判断以下两条记忆是否应该合并。它们是同一类型（{mem_type}）的记忆。

记忆 1：{mem1['text']}
记忆 2：{mem2['text']}

判断标准：
1. 是否表达相同或高度相似的核心事实？
2. 合并后是否会丢失任何关键信息？
3. 是否有时间/状态上的冲突？

如果应该合并，请回复"YES"；如果不应该合并，请回复"NO"并简要说明原因。
只回复 YES 或 NO+原因，不要有其他内容。"""

    try:
        # 尝试本地 Ollama（如果有）
        r = requests.post(
            f"http://localhost:{OLLAMA_PORT}/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )
        if r.status_code == 200:
            response = r.json()["response"].strip()
            if response.startswith("YES"):
                return True, "LLM 判定可合并"
            else:
                reason = response.replace("NO", "").strip()
                return False, reason if reason else "LLM 判定不可合并"
    except:
        pass
    
    # 兜底：如果无法调用 LLM，保守判定为不合并
    return False, "无法调用 LLM，保守判定"

def create_backup():
    """创建记忆快照备份"""
    CONFIG["snapshot_dir"].mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 备份活跃记忆
    if CONFIG["memory_file"].exists():
        shutil.copy2(
            CONFIG["memory_file"],
            CONFIG["snapshot_dir"] / f"semantic_memories_{timestamp}.jsonl"
        )
    
    # 备份归档记忆
    if CONFIG["archive_file"].exists():
        shutil.copy2(
            CONFIG["archive_file"],
            CONFIG["snapshot_dir"] / f"archived_memories_{timestamp}.jsonl"
        )
    
    backup_path = CONFIG["snapshot_dir"] / f"backup_{timestamp}.json"
    backup_data = {
        "timestamp": timestamp,
        "active_count": len(load_memories(CONFIG["memory_file"])),
        "archived_count": len(load_memories(CONFIG["archive_file"], include_archived=True)) - len(load_memories(CONFIG["archive_file"])),
        "files": [
            str(CONFIG["snapshot_dir"] / f"semantic_memories_{timestamp}.jsonl"),
            str(CONFIG["snapshot_dir"] / f"archived_memories_{timestamp}.jsonl")
        ]
    }
    
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 备份完成：{backup_path}")
    return str(backup_path)

def generate_proposal_id():
    """生成提案 ID"""
    return f"prop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============ 核心操作 ============
def propose():
    """
    生成整理提案（只读，不修改任何数据）
    输出：提案文件路径
    """
    print("📋 开始生成整理提案...")
    
    # 守卫：检查触发条件
    mems = load_memories(CONFIG["memory_file"])
    file_size_mb = os.path.getsize(CONFIG["memory_file"]) / 1024 / 1024 if CONFIG["memory_file"].exists() else 0
    
    trigger_reason = None
    if len(mems) >= CONFIG["trigger_count"]:
        trigger_reason = f"活跃记忆数 ({len(mems)}) ≥ 触发线 ({CONFIG['trigger_count']})"
    elif file_size_mb >= CONFIG["trigger_size_mb"]:
        trigger_reason = f"文件大小 ({file_size_mb:.2f}MB) ≥ 触发线 ({CONFIG['trigger_size_mb']}MB)"
    
    if not trigger_reason:
        last_maintain = CONFIG["memory_file"].stat().st_mtime if CONFIG["memory_file"].exists() else 0
        days_since = (datetime.now().timestamp() - last_maintain) / 86400
        if days_since >= CONFIG["trigger_days"]:
            trigger_reason = f"距离上次维护已 {days_since:.1f} 天 ≥ 触发线 ({CONFIG['trigger_days']}天)"
    
    print(f"触发原因：{trigger_reason or '手动触发'}")
    print(f"当前活跃记忆：{len(mems)} 条，{file_size_mb:.2f}MB")
    
    # 按类型分组
    by_type = {}
    for m in mems:
        t = m.get("type", "general")
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(m)
    
    print(f"类型分布：{ {k: len(v) for k, v in by_type.items()} }")
    
    # 向量粗筛 + LLM 精判，识别候选合并组
    merge_candidates = []
    
    for mem_type, type_mems in by_type.items():
        if len(type_mems) < 2:
            continue
        
        threshold = CONFIG["type_thresholds"].get(mem_type, 0.80)
        print(f"\n处理 {mem_type} 类（阈值={threshold}）...")
        
        # 构建嵌入矩阵
        embeddings = []
        valid_indices = []
        for i, m in enumerate(type_mems):
            emb = embed_text(m["text"])
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)
        
        if len(embeddings) < 2:
            print(f"  ⚠ 有效嵌入不足，跳过")
            continue
        
        emb_matrix = np.array(embeddings)
        
        # 余弦相似度矩阵
        n = len(embeddings)
        merged_indices = set()
        
        for i in range(n):
            if i in merged_indices:
                continue
            
            group = [valid_indices[i]]
            
            for j in range(i + 1, n):
                if j in merged_indices:
                    continue
                
                sim = cosine(emb_matrix[i], emb_matrix[j])
                
                if sim > threshold:
                    # 向量粗筛通过，进行 LLM 精判
                    can_merge, reason = llm_judge_merge(
                        type_mems[valid_indices[i]],
                        type_mems[valid_indices[j]],
                        mem_type
                    )
                    
                    if can_merge:
                        group.append(valid_indices[j])
                        print(f"  ✓ 发现候选：{type_mems[valid_indices[i]]['text'][:30]}... ↔ {type_mems[valid_indices[j]]['text'][:30]}... (sim={sim:.3f})")
            
            if len(group) > 1:
                merged_indices.update(group)
                merge_candidates.append({
                    "type": mem_type,
                    "indices": group,
                    "memories": [type_mems[idx] for idx in group]
                })
    
    # 识别归档候选（低频/低能量）
    archive_candidates = []
    now = datetime.now()
    
    for m in mems:
        if m.get("archived", False):
            continue
        
        energy = m.get("energy", 2.0)
        last_hit = datetime.fromisoformat(m.get("last_hit", now.isoformat()))
        days_unhit = (now - last_hit).days
        
        if days_unhit > CONFIG["archive_days"] or energy < CONFIG["energy_archive_threshold"]:
            archive_candidates.append({
                "id": m["id"],
                "text": m["text"][:50] + "..." if len(m["text"]) > 50 else m["text"],
                "type": m["type"],
                "energy": energy,
                "days_unhit": days_unhit,
                "reason": f"{days_unhit}天未命中" if days_unhit > CONFIG["archive_days"] else f"能量过低 ({energy:.2f})"
            })
    
    # 识别技术类记忆（general → 技能化候选）
    skill_candidates = []
    for m in by_type.get("general", []):
        text_lower = m["text"].lower()
        # 关键词匹配：技术教训/运维坑/配置问题（关键词统一小写后再比，修复大小写 bug）
        tech_keywords = ["bug", "error", "fix", "配置", "环境", "依赖", "venv", "python", "代理", "显存", "gpu", "api", "额度"]
        if any(kw in text_lower for kw in tech_keywords):
            skill_candidates.append({
                "id": m["id"],
                "text": m["text"],
                "reason": "技术类记忆，建议沉淀进技能"
            })
    
    # 生成提案文件
    CONFIG["proposal_dir"].mkdir(exist_ok=True)
    proposal_id = generate_proposal_id()
    proposal_file = CONFIG["proposal_dir"] / f"{proposal_id}.json"
    
    proposal = {
        "proposal_id": proposal_id,
        "generated_at": datetime.now().isoformat(),
        "trigger_reason": trigger_reason or "手动触发",
        "current_stats": {
            "active_count": len(mems),
            "file_size_mb": round(file_size_mb, 2),
            "by_type": {k: len(v) for k, v in by_type.items()}
        },
        "merge_groups": [
            {
                "group_id": f"mg_{i}",
                "type": g["type"],
                "merged_from": [m["id"] for m in g["memories"]],
                "memories": g["memories"],
                "proposed_text": None,  # 待 owner review 后填充
                "status": "pending_review"
            }
            for i, g in enumerate(merge_candidates)
        ],
        "archive_candidates": archive_candidates,
        "skill_candidates": skill_candidates,
        "summary": {
            "merge_groups_count": len(merge_candidates),
            "archive_candidates_count": len(archive_candidates),
            "skill_candidates_count": len(skill_candidates),
            "estimated_reduction": len(merge_candidates)  # 粗略估计
        }
    }
    
    with open(proposal_file, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 提案已生成：{proposal_file}")
    print(f"  - 合并候选组：{len(merge_candidates)} 组")
    print(f"  - 归档候选：{len(archive_candidates)} 条")
    print(f"  - 技能化候选：{len(skill_candidates)} 条")
    print(f"\n请 owner review 提案后，使用 `execute --proposal {proposal_id}` 执行")
    
    return str(proposal_file)

def execute(proposal_id):
    """
    执行整理提案（需要 owner 确认）
    """
    print(f"🔧 开始执行提案 {proposal_id}...")
    
    proposal_file = CONFIG["proposal_dir"] / f"{proposal_id}.json"
    if not proposal_file.exists():
        print(f"❌ 提案不存在：{proposal_id}")
        return
    
    with open(proposal_file, "r", encoding="utf-8") as f:
        proposal = json.load(f)
    
    # 守卫：检查提案状态
    if proposal.get("status") == "executed":
        print(f"⚠ 提案已执行过：{proposal_id}")
        return
    
    # 创建备份
    backup_path = create_backup()
    
    # 执行合并（整个「读→改→写」在文件锁事务内完成）
    merged_count = 0
    archived_count = 0
    with memory_file_lock(CONFIG["memory_file"], fail_fast=True):  # P2-7：写方 fail-fast
        mems = load_memories(CONFIG["memory_file"])
        archived = load_memories(CONFIG["archive_file"], include_archived=True)
        
        for group in proposal.get("merge_groups", []):
            if group.get("status") != "approved":
                print(f"⚠ 跳过未批准的合并组：{group['group_id']}")
                continue
            
            merged_ids = group["merged_from"]
            merged_mems = [m for m in mems if m["id"] in merged_ids]
            
            if len(merged_mems) < 2:
                continue
            
            # 生成合并后的文本（使用三元模板）
            # TODO: 这里可以调用 LLM 生成更智能的合并文本
            merged_text = merged_mems[0]["text"]  # 暂时使用第一条
            for m in merged_mems[1:]:
                if len(m["text"]) > len(merged_text):
                    merged_text = m["text"]
            
            # 新记忆的 embedding 用合并文本现算（与 text 严格对应），
            # 算不出才退回第一条的旧向量
            merged_emb = None
            try:
                emb = embed_text(merged_text)
                if emb is not None:
                    merged_emb = emb.tolist()
            except Exception:  # noqa: BLE001
                pass
            if merged_emb is None:
                merged_emb = merged_mems[0].get("embedding")
            
            # 创建合并后的新记忆
            new_mem = {
                "id": f"mem_{int(datetime.now().timestamp() * 1000)}_{os.getpid()}",
                "text": merged_text,
                "type": group["type"],
                "embedding": merged_emb,
                "timestamp": datetime.now().isoformat(),
                "energy": max(m.get("energy", 2.0) for m in merged_mems) + 1.0,
                "last_hit": datetime.now().isoformat(),
                "archived": False,
                "merged_from": merged_ids,
                "merge_date": datetime.now().isoformat(),
                "merge_proposal": proposal_id
            }
            
            # 【人格红线 2026-08-29】旧记忆不删除，移入归档并标记去向：
            # 记忆只搬家，永远可以从 archived_memories.jsonl 找回
            merged_ids_set = set(merged_ids)
            for m in merged_mems:
                m["archived"] = True
                m["merged_into"] = new_mem["id"]
                m["merge_date"] = new_mem["merge_date"]
                m["merge_proposal"] = proposal_id
                archived.append(m)
            mems = [m for m in mems if m["id"] not in merged_ids_set]
            mems.append(new_mem)
            merged_count += 1
            
            print(f"✓ 合并组 {group['group_id']}：{len(merged_ids)} 条 → 1 条（旧记忆已移入归档，标记 merged_into）")
        
        # 执行归档
        for candidate in proposal.get("archive_candidates", []):
            for m in mems:
                if m["id"] == candidate["id"]:
                    m["archived"] = True
                    m["archive_date"] = datetime.now().isoformat()
                    m["archive_reason"] = candidate["reason"]
                    archived.append(m)
                    archived_count += 1
                    print(f"✓ 归档：{m['id']} ({candidate['reason']})")
                    break
        
        with memory_file_lock(CONFIG["archive_file"], fail_fast=True):  # P2-7：写方 fail-fast
            save_memories(mems, CONFIG["memory_file"])
            save_memories(archived, CONFIG["archive_file"])
    
    # 更新提案状态
    proposal["status"] = "executed"
    proposal["executed_at"] = datetime.now().isoformat()
    proposal["results"] = {
        "merged_count": merged_count,
        "archived_count": archived_count,
        "backup_path": backup_path
    }
    
    with open(proposal_file, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 执行完成：合并 {merged_count} 组，归档 {archived_count} 条")
    print(f"备份位置：{backup_path}")
    print(f"当前活跃记忆：{len(mems)} 条")

def restore(proposal_id):
    """
    回滚到指定提案前的状态
    """
    print(f"🔄 开始回滚提案 {proposal_id}...")
    
    proposal_file = CONFIG["proposal_dir"] / f"{proposal_id}.json"
    if not proposal_file.exists():
        print(f"❌ 提案不存在：{proposal_id}")
        return
    
    with open(proposal_file, "r", encoding="utf-8") as f:
        proposal = json.load(f)
    
    if proposal.get("status") != "executed":
        print(f"⚠ 提案未执行，无需回滚：{proposal_id}")
        return
    
    # 找到对应的备份
    backup_path = proposal.get("results", {}).get("backup_path")
    if not backup_path or not os.path.exists(backup_path):
        print(f"❌ 备份文件不存在：{backup_path}")
        return
    
    with open(backup_path, "r", encoding="utf-8") as f:
        backup_data = json.load(f)
    
    # 恢复备份
    for file_path in backup_data.get("files", []):
        if os.path.exists(file_path):
            # 从备份文件名推断目标文件名
            backup_name = os.path.basename(file_path)
            # 移除时间戳部分
            if "semantic_memories" in backup_name:
                target = str(CONFIG["memory_file"])
            elif "archived_memories" in backup_name:
                target = str(CONFIG["archive_file"])
            else:
                continue
            
            shutil.copy2(file_path, target)
            print(f"✓ 恢复：{target}")
    
    # 更新提案状态
    proposal["status"] = "restored"
    proposal["restored_at"] = datetime.now().isoformat()
    
    with open(proposal_file, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 回滚完成：已恢复到提案 {proposal_id} 前的状态")

def main():
    parser = argparse.ArgumentParser(description="Hermes 长期记忆整理提炼工具")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # propose 子命令
    sub.add_parser("propose", help="生成整理提案（只读）")
    
    # execute 子命令
    exec_parser = sub.add_parser("execute", help="执行整理提案")
    exec_parser.add_argument("--proposal", type=str, required=True, help="提案 ID")
    exec_parser.add_argument("--confirm", action="store_true", help="跳过确认直接执行")
    
    # restore 子命令
    restore_parser = sub.add_parser("restore", help="回滚到指定提案前的状态")
    restore_parser.add_argument("--proposal", type=str, required=True, help="提案 ID")
    
    args = parser.parse_args()
    
    if args.cmd == "propose":
        propose()
    elif args.cmd == "execute":
        if not args.confirm:
            confirm = input(f"⚠ 确认执行提案 {args.proposal}？(y/N): ")
            if confirm.lower() != 'y':
                print("已取消")
                return
        execute(args.proposal)
    elif args.cmd == "restore":
        restore(args.proposal)

if __name__ == "__main__":
    main()
