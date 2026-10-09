#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step0_cells.py — 交接文档 §十三 待办 #1：查表确定各类细胞的类型名。

交接原文："具体叫什么我背不出来，必须查表。猜错了后面全白搭。"

要找的：
  1. 嗅觉受体神经元 ORN    —— 气味注入点（§七）
  2. 甜味感受神经元         —— 奖赏注入点（§九）
  3. 苦味感受神经元         —— 惩罚注入点（§九）
  4. 蘑菇体 Kenyon 细胞 KC  —— 三因子规则的前突触（§九）
  5. 蘑菇体输出神经元 MBON  —— 三因子规则的后突触（§九）
  6. 多巴胺能神经元 DAN     —— 三因子规则的"现在改"信号（§九）

只报告**证据**，不猜。"""
from __future__ import annotations

import os
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, "data",
                   "body-annotations-male-cns-v1.0-minconf-0.5.feather")


def S(df, col):
    """把一列安全地变成字符串（NaN → ''）。"""
    return df[col].astype("string").fillna("").astype(str)


def cnt(df, col, mask=None, top=30, label=None):
    s = S(df, col) if mask is None else S(df.loc[mask], col)
    s = s[s != ""]
    c = Counter(s)
    tot = int(mask.sum()) if mask is not None else len(df)
    print(f"  {label or col}  (候选 {tot:,} 个, 有值 {len(s):,}):")
    for k, n in c.most_common(top):
        print(f"      {n:>7,}  {k}")
    if len(c) > top:
        print(f"      ... 另有 {len(c)-top} 种")


def main():
    df = pd.read_feather(ANN)
    print(f"注释表 {len(df):,} 行 x {len(df.columns)} 列\n")

    # ================= 1. 嗅觉受体神经元 ORN =================
    print("=" * 70)
    print("【1】嗅觉受体神经元 ORN —— 气味注入点")
    print("=" * 70)
    a = S(df, "superclass") == "ol_sensory"
    b = S(df, "entryNerve") == "AN"
    print(f"  superclass == 'ol_sensory' : {int(a.sum()):,}")
    print(f"  entryNerve == 'AN'         : {int(b.sum()):,}")
    print(f"  两者交集                   : {int((a & b).sum()):,}")
    print(f"  并集                       : {int((a | b).sum()):,}")
    orn = a | b
    cnt(df, "class", orn, top=10, label="  并集里的 class")
    cnt(df, "receptorType", orn, top=45, label="  并集里的 receptorType")
    cnt(df, "somaNeuromere", orn, top=10, label="  somaNeuromere")

    # ================= 2/3. 甜味 / 苦味 =================
    print("\n" + "=" * 70)
    print("【2/3】甜味 / 苦味感受神经元 —— 奖赏 / 惩罚注入点")
    print("=" * 70)
    gus = S(df, "class") == "gustatory"
    print(f"  class == 'gustatory': {int(gus.sum()):,}")
    rt = S(df, "receptorType")
    print(f"  receptorType 非空    : {int((rt != '').sum()):,}")
    print("  receptorType 的全部取值:")
    for k, n in Counter(rt[rt != ""]).most_common(60):
        print(f"      {n:>6,}  {k}")
    print()
    print("  gustatory 里按 receptorType:")
    cnt(df, "receptorType", gus, top=40, label="    receptorType")
    print()
    print("  gustatory 里按 entryNerve:")
    cnt(df, "entryNerve", gus, top=15, label="    entryNerve")
    # 关键词扫描（在 gustatory + 全表的 receptorType 上）
    for kw in ("Gr5a", "Gr64f", "Gr64a", "Gr61a", "Gr43a", "Gr66a", "Gr89a",
               "Gr33a", "Gr22e", "Orco", "Or4", "IR2"):
        hit = rt.str.contains(kw, case=False, regex=False)
        if hit.sum():
            print(f"  receptorType 含 {kw!r}: {int(hit.sum()):,}")

    # ================= 4/5/6. 蘑菇体 + 多巴胺 =================
    print("\n" + "=" * 70)
    print("【4/5/6】KC / MBON / DAN")
    print("=" * 70)
    for cl in ("Kenyon_Cell", "MBON", "DAN", "ALPN", "ALLN", "ol_bilateral",
               "ALIN", "ALON", "olfactory", "CX"):
        m = S(df, "class") == cl
        if m.sum():
            sides = Counter(S(df.loc[m], "somaSide"))
            print(f"  class == {cl!r:16s}: {int(m.sum()):>6,}   somaSide={dict(sides)}")
    print()
    kc = S(df, "class") == "Kenyon_Cell"
    cnt(df, "superclass", kc, top=8, label="  KC 的 superclass")
    cnt(df, "subclass", kc, top=12, label="  KC 的 subclass")
    cnt(df, "type", kc, top=8, label="  KC 的 type")
    cnt(df, "somaNeuromere", kc, top=8, label="  KC 的 somaNeuromere")
    print()
    mb = S(df, "class") == "MBON"
    cnt(df, "type", mb, top=30, label="  MBON 的 type")
    print()
    dan = S(df, "class") == "DAN"
    cnt(df, "type", dan, top=30, label="  DAN 的 type")

    # ================= 汇总 =================
    print("\n" + "=" * 70)
    print("【汇总】建议的注入点 / 学习点")
    print("=" * 70)
    print(f"  ORN   嗅觉受体神经元   候选 {int(orn.sum()):,}"
          f"   （superclass=ol_sensory ∪ entryNerve=AN）")
    print(f"  KC    Kenyon 细胞      候选 {int(kc.sum()):,}   （class=Kenyon_Cell）")
    print(f"  MBON  蘑菇体输出       候选 {int(mb.sum()):,}   （class=MBON）")
    print(f"  DAN   多巴胺能         候选 {int(dan.sum()):,}   （class=DAN）")
    print(f"  味觉  gustatory        候选 {int(gus.sum()):,}   （class=gustatory）")


if __name__ == "__main__":
    main()
