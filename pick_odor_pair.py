#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pick_odor_pair.py — 挑一对 KC 重叠最低的气味通道。

为什么需要：实验丙（好恶）第一次跑，被惩罚的气味 B 反而上升。
隔离实验证明惩罚本身没错（B 单独配惩罚 Δ=−0.387），
但**只在 A 上学就能把 B 的响应抬 +0.490** —— A 的奖赏通过共享 KC 外溢到 B，
盖过了 B 自己的惩罚。

根因是 KC 活跃集合重叠太高（Jaccard 0.317）。
果蝇真实情况里 KC 编码是稀疏的（每个气味只点亮 5~10%），不同气味几乎不重叠。
这个速率模型 + 连接组权重给出的重叠偏高，所以**必须按实测挑通道对**，
不能默认任意两个肾小球都分得开。

做法：对每个通道算 KC 响应向量，两两求活跃集合 Jaccard，输出最优对。
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from odor_map import OdorEncoder                      # noqa: E402

GRAPH = os.path.join(HERE, "graph")
OUT = os.path.join(HERE, "out")
N = 211_577
A, B, TICKS = 0.5, 1.0, 10


def S(df, col):
    return df[col].astype("string").fillna("").astype(str)


def main():
    gm = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    gm = gm.sort_values("idx").reset_index(drop=True)
    cl, ty, idx = S(gm, "class"), S(gm, "type"), gm["idx"].to_numpy()
    orn = idx[(cl == "olfactory").to_numpy()]
    kc_i = idx[(cl == "Kenyon_Cell").to_numpy()]
    glom = {}
    for t, i in zip(ty[cl == "olfactory"], idx[cl == "olfactory"]):
        glom.setdefault(t, []).append(i)
    glom = {k: np.array(v, np.int64) for k, v in glom.items()}
    enc = OdorEncoder(glom, orn, verbose=False)
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()

    def kc_resp(ch: str, pct: float = 90.0):
        d = enc.vector([ch])
        r = np.zeros(N, dtype=np.float32)
        for _ in range(TICKS):
            r = (A * r + B * np.tanh(W @ r + d)).astype(np.float32)
        v = r[kc_i]
        thr = np.percentile(v, pct)
        return v, set(np.flatnonzero(v >= thr).tolist())

    vecs, acts = {}, {}
    for ch in enc.channels:
        vecs[ch], acts[ch] = kc_resp(ch)
    print(f"通道数 {len(enc.channels)}   KC {len(kc_i):,}   活跃阈值 = 前 10%")
    print("\nKC 活跃集合两两 Jaccard（越低越分得开）:")
    print("            " + "".join(f"{c.replace('ORN_',''):>8s}" for c in enc.channels))
    pairs = []
    for a, b in itertools.combinations(enc.channels, 2):
        j = len(acts[a] & acts[b]) / max(len(acts[a] | acts[b]), 1)
        pairs.append((j, a, b))
    for a in enc.channels:
        row = ""
        for b in enc.channels:
            if a == b:
                row += f"{'—':>8s}"
            else:
                j = len(acts[a] & acts[b]) / max(len(acts[a] | acts[b]), 1)
                row += f"{j:>8.3f}"
        print(f"  {a.replace('ORN_',''):>8s}  {row}")

    pairs.sort()
    print("\n最分得开的 5 对:")
    for j, a, b in pairs[:5]:
        print(f"  {a} ↔ {b}   Jaccard {j:.3f}")
    print("\n最像的 3 对（要避开）:")
    for j, a, b in pairs[-3:]:
        print(f"  {a} ↔ {b}   Jaccard {j:.3f}")
    print("\n（实验丙第一次用的 DA1 ↔ VA1d: "
          f"Jaccard {[j for j,a,b in pairs if {a,b}=={'ORN_DA1','ORN_VA1d'}][0]:.3f}）")

    best = pairs[0]
    np.savez_compressed(os.path.join(OUT, "odor_pair.npz"),
                        chA=best[1], chB=best[2], jaccard=best[0])
    print(f"\n★ 推荐气味对: {best[1]} ↔ {best[2]}  (Jaccard {best[0]:.3f})"
          f"  → 已存 out/odor_pair.npz")


if __name__ == "__main__":
    main()
