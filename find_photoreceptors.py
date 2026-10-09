#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""find_photoreceptors.py — 感光细胞在 MaleCNS 注释表里到底怎么写

第 1 步发现：type 列里匹配 ^R[1-8]$ 命中 0 个。
本脚本找出真正的视觉输入细胞命名，为第 2 步「复眼 → 感光细胞」做准备。
"""
import sys
import io
import re
import pandas as pd


ANN = r"E:\终末地\_handoff\data\body-annotations-male-cns-v1.0-minconf-0.5.feather"

ann = pd.read_feather(ANN, columns=["bodyId", "type", "flywireType", "superclass", "subclass",
                                    "class", "somaSide", "rootSide", "entryNerve", "receptorType",
                                    "synonyms", "status"])
print(f"总神经元 {len(ann):,}")
print()

# ---------- 1. 各 superclass 在视觉上的分布 ----------
print("=" * 72)
print("superclass 里与视觉/感觉输入相关的：")
vc = ann["superclass"].value_counts()
for k, v in vc.items():
    if any(x in str(k).lower() for x in ("ol_", "visual", "sensory", "photo")):
        print(f"  {k:28s} {v:8,}")
print()

# ---------- 2. ol_sensory 是什么 ----------
print("=" * 72)
print("ol_sensory（视叶感觉神经元）的 type 构成：")
os_ = ann[ann["superclass"] == "ol_sensory"]
print(f"  共 {len(os_):,} 个，{os_['type'].nunique()} 种")
print(os_["type"].value_counts().head(30).to_string())
print()

# ---------- 3. 找感光细胞：各种可能的命名 ----------
print("=" * 72)
print("按正则找感光细胞（在各列里试）：")
pats = {
    r"^R[1-8]$": "标准 R1-R8 感光细胞",
    r"^R[1-8][a-z]?$": "R1-R8 带后缀",
    r"R[1-8]": "含 R1-R8",
    r"photo": "含 photo",
    r"Photo": "含 Photo",
    r"^PR": "PR 开头",
    r"retina": "含 retina",
    r"ommatid": "含 ommatid",
}
for col in ["type", "flywireType", "class", "subclass", "receptorType", "synonyms"]:
    if col not in ann.columns:
        continue
    s = ann[col].dropna().astype(str)
    hits = {}
    for p, desc in pats.items():
        m = s.str.contains(p, case=False, na=False, regex=True)
        if m.any():
            hits[desc] = (int(m.sum()), sorted(s[m].unique())[:8])
    if hits:
        print(f"  列 [{col}]：")
        for desc, (cnt, ex) in hits.items():
            print(f"    {desc:24s} {cnt:>7,} 个   例：{ex}")
print()

# ---------- 4. entryNerve：感光细胞从哪根神经进脑 ----------
print("=" * 72)
print("entryNerve 取值分布（感光细胞走视神经进门）：")
en = ann["entryNerve"].dropna().astype(str)
print(en.value_counts().head(20).to_string())
print()

# ---------- 5. receptorType 已经知道是 IR/ppk，再看它属于哪些细胞 ----------
print("=" * 72)
rt = ann[ann["receptorType"].notna()]
print(f"receptorType 非空的 {len(rt):,} 个神经元：")
print(rt.groupby(["receptorType", "superclass"]).size().to_string())
print()

# ---------- 6. 直接搜 "R7" / "R8" 这类字符串出现在哪些列 ----------
print("=" * 72)
print("全表搜 'R7'/'R8'（任意列，看它们藏在哪）：")
for col in ann.columns:
    if ann[col].dtype != object and str(ann[col].dtype) not in ("str", "string", "category"):
        continue
    s = ann[col].dropna().astype(str)
    m = s.str.contains(r"\bR[78]\b|^R[78]", na=False, regex=True)
    if m.any():
        print(f"  列 [{col}]：{int(m.sum()):,} 个命中，例：{sorted(s[m].unique())[:10]}")
print()

# ---------- 7. 用 status 列看有没有未标注的感光细胞 ----------
print("=" * 72)
print("status 分布 × superclass（找未标注的感觉细胞）：")
print(pd.crosstab(ann["superclass"].fillna("(空)"), ann["status"].fillna("(空)")).to_string())
