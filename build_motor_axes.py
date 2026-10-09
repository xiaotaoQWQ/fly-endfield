#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_motor_axes.py — 从**连接组本身**推出三条运动轴（不点名任何神经元）。

动机：
    用户要求「不用神经元直连」——原来是把 DNg100/DNa02/DNp13/DNp01/DNp10
    这几个点名的下行神经元直接接到按键上。改成从下行神经元**群体**投影出
    连续的控制量。

为什么能做：
    MaleCNS 是**全中枢神经系统**（脑 + 腹神经索），所以下行神经元在 VNC 里的
    下游靶点也在数据里。实测有 708 个 vnc_motor，其中 **255 个**是腿/胸运动
    神经元，而且 `somaNeuromere` 分出了 T1/T2/T3（前/中/后胸腿）、
    `somaSide` 左右各 128/127 很均衡。

三条轴（DN j 对腿运动神经元的两跳影响 A[i,j]）：
    推力 thrust[j] = Σ_i A[i,j]                       全体腿输出的总驱动
    横移 strafe[j] = (T1L+T2L+T3L) − (T1R+T2R+T3R)     左右**同向**不对称
    转身 turn[j]   = (T1L−T1R) − (T3L−T3R)             前腿与后腿**反向**不对称

    ★ 第三、三条的分解依据是六足步态：
      「左右同向压」= 侧向平移；「前腿和后腿反向压」= 原地转身（枢轴）。
      昆虫没有独立的"前后摆肌"（前后行走是同一运动程序的相位反转），
      所以前进/后退只能从**推力相对基线的偏移**来取 —— 这一条是设计选择，
      不是生理事实，已在文档里如实标注。

用法：
    python -X utf8 build_motor_axes.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")

ANN = os.path.join(DATA, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
LEG_KW = ("Ti ", "Tr ", "tibia", "trochanter", "coxa", "femur", "Coxa", "Fe ",
          "Sternotrochanter", "reductor", "depressor", "levator",
          "promotor", "remotor", "abductor")
SEGS = ("T1", "T2", "T3")


def main():
    t0 = time.time()
    print("载入连接组…")
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    print(f"  {W.shape[0]:,} × {W.shape[1]:,}   {W.nnz:,} 条边")

    gm = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))[
        ["bodyId", "idx", "type", "superclass", "somaSide"]]
    an = pd.read_feather(ANN)[["bodyId", "somaNeuromere"]]
    meta = gm.merge(an, on="bodyId", how="left")
    meta = meta.sort_values("idx").reset_index(drop=True)

    # ---- 腿运动神经元 ----
    is_mn = meta["superclass"].astype(str) == "vnc_motor"
    is_leg = meta["type"].astype(str).str.contains("|".join(LEG_KW),
                                                   case=False, na=False)
    leg = meta[is_mn & is_leg].copy()
    side = leg["somaSide"].astype(str).to_numpy()
    seg = leg["somaNeuromere"].astype(str).to_numpy()
    print(f"\n腿/胸运动神经元 {len(leg)} 个")
    groups = {}
    for s in SEGS:
        for sd in ("L", "R"):
            g = np.flatnonzero((seg == s) & (side == sd))
            if len(g):
                groups[(s, sd)] = leg["idx"].to_numpy()[g]
            print(f"  {s}{sd}: {len(g):4d}")
    all_leg = leg["idx"].to_numpy()
    print(f"  合计 {len(all_leg)}")

    # ---- 下行神经元 ----
    dn_mask = meta["superclass"].astype(str).str.contains("descending",
                                                          case=False, na=False)
    dn_idx = meta["idx"].to_numpy()[dn_mask.to_numpy()]
    print(f"\n下行神经元 {len(dn_idx):,}")

    # ---- 两跳影响：A[i,j] = DN j 对腿 MN i 的影响 ----
    print("\n算两跳影响 W[MN,:] @ W[:,DN] …")
    t1 = time.time()
    L = W[all_leg, :]                 # (n_leg, N)
    R = W[:, dn_idx]                  # (N, n_dn)
    A = (L @ R)
    A = np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)
    print(f"  A {A.shape}  非零 {int((A != 0).sum()):,}  用时 {time.time()-t1:.1f}s")

    # ---- 三条轴 ----
    def gsum(keys):
        # ★ 轴系数是「每个下行神经元一个」→ 长度 = A.shape[1]（不是行数）
        v = np.zeros(A.shape[1])
        for k in keys:
            if k in groups:
                # groups 里存的是 bodyId 顺序的 idx，需要映射回 A 的行号
                rows = np.searchsorted(all_leg, groups[k])
                rows = rows[(rows >= 0) & (rows < len(all_leg))]
                if len(rows):
                    v += A[rows].sum(axis=0)
        return v

    thrust = gsum([(s, d) for s in SEGS for d in ("L", "R")])
    latL = gsum([(s, "L") for s in SEGS])
    latR = gsum([(s, "R") for s in SEGS])
    strafe = latL - latR
    turn = (gsum([("T1", "L")]) - gsum([("T1", "R")])
            - gsum([("T3", "L")]) + gsum([("T3", "R")]))

    print("\n三条轴的统计（每个下行神经元一个系数）：")
    for nm, v in (("thrust 推力", thrust), ("strafe 横移", strafe),
                  ("turn 转身", turn)):
        nz = v != 0
        print(f"  {nm:14s} 非零 {int(nz.sum()):4d}/{len(v)}  "
              f"范围 [{v.min():+.4f}, {v.max():+.4f}]  "
              f"|均值| {np.abs(v[nz]).mean() if nz.any() else 0:.5f}")

    # 三条轴之间的相关性（应尽量低，否则等于同一个信号）
    def corr(a, b):
        m = (a != 0) | (b != 0)
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    print(f"\n轴间相关性  thrust~strafe {corr(thrust, strafe):+.3f}   "
          f"thrust~turn {corr(thrust, turn):+.3f}   "
          f"strafe~turn {corr(strafe, turn):+.3f}")

    np.savez_compressed(os.path.join(OUT, "motor_axes.npz"),
                        dn_idx=dn_idx.astype(np.int32),
                        thrust=thrust.astype(np.float32),
                        strafe=strafe.astype(np.float32),
                        turn=turn.astype(np.float32),
                        all_leg=all_leg.astype(np.int32))
    print(f"\n[{time.time()-t0:.1f}s] → {os.path.join(OUT, 'motor_axes.npz')}")


if __name__ == "__main__":
    main()
