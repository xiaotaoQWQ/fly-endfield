#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step0_mb.py — 学习回路三件套的可用性核查。

交接文档 §九 的三因子规则需要：
  前突触(KC) · 后突触(MBON) · 多巴胺(DAN) · 它们之间的突触
而 §九 还要求奖赏/惩罚分别走**甜味 / 苦味感受神经元**。
问题：注释表里没有任何受体基因名（Or/Gr/IR 全表 0 命中）。

所以在问用户之前，先查清楚：
  A) DAN 有没有簇名（PAM/PPL1/PPL2…）—— 有的话，奖赏/惩罚可以直接用簇区分，
     从而**绕过味觉受体**这一环
  B) KC / MBON / DAN 之间的连接在连接组里有多少条边
  C) 嗅觉通路下游（ALPN 投射神经元）能不能作为"气味→蘑菇体"的中间站
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, "data",
                   "body-annotations-male-cns-v1.0-minconf-0.5.feather")
GRAPH = os.path.join(HERE, "graph")


def S(df, col):
    return df[col].astype("string").fillna("").astype(str)


def main():
    df = pd.read_feather(ANN)
    cl, ty = S(df, "class"), S(df, "type")

    # ---------- A. DAN 亚型 ----------
    print("=" * 74)
    print("A) DAN（多巴胺能，340 个）的类型名 —— 能不能分出奖赏/惩罚簇")
    print("=" * 74)
    dan = cl == "DAN"
    tc = Counter(ty[dan])
    print(f"  共 {int(dan.sum())} 个，{len(tc)} 种 type:")
    for k, v in tc.most_common(40):
        print(f"      {v:>5,}  {k}")
    print()
    # 找 PAM / PPL 前缀
    for pref in ("PAM", "PPL", "PPL1", "PPL2", "PPL3", "PPL4", "PAL"):
        m = ty[dan].str.startswith(pref, na=False)
        if m.sum():
            print(f"  ★ type 以 {pref!r} 开头: {int(m.sum()):,}  例 "
                  f"{list(Counter(ty[dan][m]).most_common(8))}")

    # ---------- 其他蘑菇体相关类 ----------
    print()
    for c in ("Kenyon_Cell", "MBON", "DAN", "ALPN", "ALLN", "ALIN", "ALON"):
        m = cl == c
        print(f"  class={c:14s} {int(m.sum()):>6,}")

    # ---------- B. 连接组里的 KC→MBON / DAN→KC ----------
    print("\n" + "=" * 74)
    print("B) 蘑菇体连接（来自 graph_W_raw.npz）")
    print("=" * 74)
    gm = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    gm = gm.sort_values("idx").reset_index(drop=True)
    gcl = gm["class"].astype("string").fillna("").astype(str)
    idx = gm["idx"].to_numpy()

    kc_i = idx[(gcl == "Kenyon_Cell").to_numpy()]
    mb_i = idx[(gcl == "MBON").to_numpy()]
    dan_i = idx[(gcl == "DAN").to_numpy()]
    alpn_i = idx[(gcl == "ALPN").to_numpy()]
    orn_i = idx[(gcl == "olfactory").to_numpy()]
    print(f"  idx: KC {len(kc_i):,}  MBON {len(mb_i):,}  DAN {len(dan_i):,}"
          f"  ALPN {len(alpn_i):,}  ORN {len(orn_i):,}")

    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    print(f"  W {W.shape}  {W.nnz:,} 边")

    def edges(post, pre, label):
        if not len(post) or not len(pre):
            print(f"  {label}: 无细胞"); return None
        sub = W[post, :][:, pre]
        n = int(sub.nnz)
        s = float(np.abs(sub.data).sum()) if n else 0.0
        print(f"  {label:28s} {n:>8,} 条边   权重绝对值和 {s:,.1f}")
        return sub

    kc_mb = edges(mb_i, kc_i, "KC → MBON")
    dan_mb = edges(mb_i, dan_i, "DAN → MBON")
    dan_kc = edges(kc_i, dan_i, "DAN → KC")
    alpn_kc = edges(kc_i, alpn_i, "ALPN → KC")
    orn_alpn = edges(alpn_i, orn_i, "ORN → ALPN")
    if kc_mb is not None and kc_mb.nnz:
        d = kc_mb.data
        print(f"    KC→MBON 权重: 正 {int((d>0).sum()):,} / 负 {int((d<0).sum()):,}"
              f"  范围 [{d.min():.4f}, {d.max():.4f}]")
    if dan_kc is not None and dan_kc.nnz:
        d = dan_kc.data
        print(f"    DAN→KC  权重: 正 {int((d>0).sum()):,} / 负 {int((d<0).sum()):,}"
              f"  范围 [{d.min():.4f}, {d.max():.4f}]")


if __name__ == "__main__":
    main()
