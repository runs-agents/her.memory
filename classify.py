# -*- coding: utf-8 -*-
"""classify.py — 代理记忆整理总纲 v4 三件套之一：确定性分类器。
用法:
  python classify.py scan <md_file>                     # 扫描 MEMORY.md/USER.md，输出分类底册(JSON)
  python classify.py scan <md_file> --out <底册.json>    # 底册落盘
职责（runbook v4 第二节）：
  - 按 § 分段，对每段打标：anchor(锚点)/tech(技术长文)/flow(可复用流程)/temp(临时)/unknown(残差,留给模型审)
  - 特征词提取：连续中文>3字+数字组合，长度降序 Top3，禁用高频词
  - 查重：对语义库【活跃条目】做特征词匹配（命中≥2词→skip+命中ID；命中归档→唤醒标记）
  - 只读，不写任何文件（--out 落盘除外）
实现约束：纯标准库；所有 open encoding='utf-8'；G1 计数排除锚点段。
"""
import sys, re, json, os, argparse
from datetime import datetime
from hermes_config import SEMANTIC_MEMORIES  # noqa: E402

SEM_FILE = SEMANTIC_MEMORIES
# 高频停用词：换成本人部署里的高频人名/称谓（伴侣名、用户称呼等）
HIGH_FREQ = {"现在", "今天", "我们", "的时候", "一个", "这个", "那个", "可以", "因为", "所以", "但是", "然后"}

# ---------- 段落切分 ----------
def split_sections(text):
    """按 § 切段；无 § 的文件按空行段返回。返回 [(段头标识, 段文本)]"""
    if "\n§\n" in text:
        parts = text.split("\n§\n")
        return [(p.strip().splitlines()[0][:40] if p.strip() else "(空)", p) for p in parts if p.strip()]
    # 无 § 分隔：按连续空行切段
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [(p.strip().splitlines()[0][:40], p) for p in paras]

# ---------- 分类判据（全部可数） ----------
RE_ANCHOR_LINE = re.compile(r"^(?:===.*|IMMUTABLE[:：].*|锚点保护.*)$", re.M)  # 只认行首结构化标记
RE_PORT   = re.compile(r"(?<![\d.])(?:[1-9]\d{3}|[1-6]\d{4})(?![\d.])")       # 端口: 1000-65535 独立数字
RE_YEAR   = re.compile(r"(?:19|20)\d{2}")                                      # 年份排除
RE_PATH   = re.compile(r"[A-Za-z]:\\[^\s，。；、）)]+|/[a-z]+/[a-z_]+")   # Windows/Unix 路径
RE_VER    = re.compile(r"v\d+(?:\.\d+)+|版本\s*\d|r\d{2,}")
RE_FLOW   = re.compile(r"以后遇到|遇到.{0,12}就(这么|一律)|一律(走|用|先)")
RE_TEMP   = re.compile(r"待验证|待办|今天(做|跑|测)了|进度：")

def classify_section(sec_text):
    """返回 (类别, 依据)。类别: anchor/tech/flow/temp/unknown"""
    if RE_ANCHOR_LINE.search(sec_text):
        return "anchor", "行首结构化锚点标记"
    ports = [p for p in RE_PORT.findall(sec_text) if not RE_YEAR.fullmatch(p)]
    hits = len(ports) + len(RE_VER.findall(sec_text)) + len(RE_PATH.findall(sec_text))
    if hits >= 2:
        return "tech", f"端口/路径/版本号命中 {hits} 处"
    if RE_FLOW.search(sec_text):
        return "flow", "可复用流程模式"
    if RE_TEMP.search(sec_text):
        return "temp", "一次性进度/临时状态"
    return "unknown", "规则不可判，留模型审残差"

# ---------- 特征词（纯正则，无模型裁量） ----------
def extract_features(sec_text, top=3):
    """连续中文(>3字)与数字组合，按长度降序取 Top3，滤高频词。"""
    cands = re.findall(r"[\u4e00-\u9fa5]{4,}\d{2,}|[\u4e00-\u9fa5]{5,}", sec_text)
    cands = [c for c in cands if c not in HIGH_FREQ]
    cands.sort(key=len, reverse=True)
    seen, out = set(), []
    for c in cands:
        if any(c in s or s in c for s in seen):
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= top:
            break
    return out

# ---------- 查重（只扫活跃；命中归档→唤醒） ----------
def load_semantic():
    active, archived = [], []
    if os.path.exists(SEM_FILE):
        with open(SEM_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                (archived if d.get("archived") else active).append(d)
    return active, archived

def dedup_check(features, active, archived):
    """返回 (skip: bool, mode, hit_ids)。mode: active_hit/archived_hit/none"""
    if len(features) < 2:
        return False, "none", []
    act_ids, arc_ids = [], []
    for m in active:
        if sum(1 for w in features if w in m.get("text", "")) >= 2:
            act_ids.append(m["id"])
    if act_ids:
        return True, "active_hit", act_ids
    for m in archived:
        if sum(1 for w in features if w in m.get("text", "")) >= 2:
            arc_ids.append(m["id"])
    if arc_ids:
        return False, "archived_hit", arc_ids   # 唤醒/续期，不跳过
    return False, "none", []

# ---------- 主扫描 ----------
def scan(md_path, out_path=None):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    active, archived = load_semantic()
    sections = split_sections(text)
    result = {
        "file": os.path.abspath(md_path),
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "total_chars": len(text),
        "semantic_active": len(active),
        "semantic_archived": len(archived),
        "sections": [],
    }
    anchor_chars = 0
    for i, (head, sec) in enumerate(sections):
        cat, why = classify_section(sec)
        feats = extract_features(sec)
        skip, mode, hit_ids = (False, "none", [])
        if cat in ("tech", "flow", "unknown"):
            skip, mode, hit_ids = dedup_check(feats, active, archived)
        if cat == "anchor":
            anchor_chars += len(sec)
        result["sections"].append({
            "idx": i, "head": head, "chars": len(sec),
            "category": cat, "reason": why,
            "features": feats,
            "dedup": {"skip": skip, "mode": mode, "hit_ids": hit_ids},
        })
    result["anchor_chars"] = anchor_chars
    result["g1_effective_chars"] = len(text) - anchor_chars   # G1 计数排除锚点段
    # 汇总
    from collections import Counter
    result["summary"] = dict(Counter(s["category"] for s in result["sections"]))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan"])
    ap.add_argument("md_file")
    ap.add_argument("--out")
    args = ap.parse_args()
    r = scan(args.md_file, args.out)
    print(f"文件: {r['file']}")
    print(f"总字符: {r['total_chars']} | 锚点段: {r['anchor_chars']} | G1 有效计数(除锚点): {r['g1_effective_chars']}")
    print(f"语义库: 活跃 {r['semantic_active']} / 归档 {r['semantic_archived']}")
    print(f"分类汇总: {r['summary']}")
    print("-" * 78)
    for s in r["sections"]:
        flag = " [锚点-不动]" if s["category"] == "anchor" else ""
        dd = f" 查重:{s['dedup']['mode']}" + (f"→{s['dedup']['hit_ids'][:2]}" if s["dedup"]["hit_ids"] else "")
        print(f"[{s['idx']:02d}] {s['category']:<7} {s['chars']:>5}字 {s['head']}{flag}{dd}")
    if args.out:
        print("-" * 78)
        print(f"底册已落盘: {args.out}")

if __name__ == "__main__":
    main()
