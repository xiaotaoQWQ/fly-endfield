#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_mb_groups.py — 把 97 个 MBON 按 DAN 投射模式分成 4 组，并算出每个
MBON 的奖赏/惩罚权重。

为什么需要这个：

  三因子规则原来是
      g = eta * da * relu(x - theta);  w += g[None, :]      ← 广播到所有 MBON
  每次更新 61,210 条边拿到**完全一样**的增量 → 所有 MBON 同比例推高推低，
  相对模式永远不变 → **学不出"这个味道该用哪个技能"**。

  真实蘑菇体里多巴胺是**按区室局部投射**的：PAM/PPL 各自只支配特定 MBON。
  连接组里有这个数据（DAN→MBON 3,160 条边），所以能直接算出来。

产出（存 out/mb_groups.npz）：
  · grp[i]        —— 第 i 个 MBON 属于哪一组（0..3）
  · pam_w[i]      —— 奖赏时到达第 i 个 MBON 的多巴胺强度（PAM→MBON 权重和）
  · ppl_w[i]      —— 惩罚时到达第 i 个 MBON 的多巴胺强度（PPL1/PPL2 同理）

分组用 k-means（k=4），特征是每个 MBON 的 DAN 投射向量。
纯 numpy 实现，不引 sklearn。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph")
OUT = os.path.join(HERE, "out")
K = 4


def S(df, col):
    return df[col].astype("string").fillna("").astype(str)


def kmeans(X: np.ndarray, k: int, iters: int = 200, seed: int = 0):
    """最朴素的 k-means（numpy 版）。X 已按行归一化。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    c = X[rng.choice(n, k, replace=False)].copy()
    lab = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1).astype(np.int32)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = X[m].mean(axis=0)
    return lab, c


def main():
    gm = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    gm = gm.sort_values("idx").reset_index(drop=True)
    cl, ty = S(gm, "class"), S(gm, "type")
    idx = gm["idx"].to_numpy()
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()

    mbon = idx[(cl == "MBON").to_numpy()]
    pam = idx[((cl == "DAN") & ty.str.startswith("PAM", na=False)).to_numpy()]
    ppl = idx[((cl == "DAN") & ty.str.startswith("PPL", na=False)).to_numpy()]
    dan = np.concatenate([pam, ppl])
    print(f"MBON {len(mbon)}   PAM {len(pam)}   PPL {len(ppl)}")

    # DAN → MBON 子矩阵（97 × 340）
    A = np.asarray(W[np.ix_(mbon, dan)].todense(), dtype=np.float32)
    print(f"DAN→MBON 子矩阵 {A.shape}   非零 {int((A > 0).sum()):,} 条边")

    # 每个 MBON 的奖赏/惩罚权重 = 各 DAN 的投射强度之和
    pam_w = A[:, :len(pam)].sum(axis=1)
    ppl_w = A[:, len(pam):].sum(axis=1)
    print(f"\n奖赏权重 PAM→MBON: 范围 [{pam_w.min():.4f}, {pam_w.max():.4f}]  "
          f"中位 {np.median(pam_w):.4f}   非零 {int((pam_w > 0).sum())}/{len(mbon)}")
    print(f"惩罚权重 PPL→MBON: 范围 [{ppl_w.min():.4f}, {ppl_w.max():.4f}]  "
          f"中位 {np.median(ppl_w):.4f}   非零 {int((ppl_w > 0).sum())}/{len(mbon)}")

    # 分组：按 DAN 投射向量做 k-means
    Xn = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    lab, cent = kmeans(Xn, K, seed=0)
    print(f"\n分成 {K} 组：")
    names = ty[(cl == "MBON").to_numpy()].to_numpy()
    for j in range(K):
        m = lab == j
        # 这一组更受奖赏还是惩罚支配
        pr = pam_w[m].mean()
        pu = ppl_w[m].mean()
        lean = "奖赏" if pr > pu * 1.5 else ("惩罚" if pu > pr * 1.5 else "混合")
        ex = ", ".join(names[m][:6])
        print(f"  组{j}: {int(m.sum()):>2d} 个 MBON   PAM {pr:.4f} / PPL {pu:.4f}"
              f"  偏向 {lean}   例: {ex}")

    np.savez_compressed(os.path.join(OUT, "mb_groups.npz"),
                        grp=lab.astype(np.int32),
                        pam_w=pam_w.astype(np.float32),
                        ppl_w=ppl_w.astype(np.float32),
                        mbon_idx=mbon.astype(np.int64))
    print(f"\n已存 out/mb_groups.npz")


if __name__ == "__main__":
    main()
