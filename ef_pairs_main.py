#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ef_pairs_main.py — 给**主画面口径**重选左右下行配对。

为什么必须重选：
    `vote_design.py` 挑出的最强 5 对，是用**俯视极坐标下的合成亮斑**
    （`rm.encode(blob(ang))`）测出来的 —— 那套几何属于小地图口径。
    换成第一人称画面后，同一个细胞对"左/右"的响应可能完全不同，
    原来那 5 对未必还有对立性。这可能比相机几何更大的失配源。

做法：
    把同样 5 个方位的亮斑**按主画面几何**渲染（linear 口径下 u=(az+135)/270），
    跑网络，量每个下行神经元对各方向的响应，再按 vote_design 的口径重排配对。

朝向约定不照抄，改为**实测验证**：选完 top-k 后，喂左刺激和右刺激各算一次
margin，若符号反了就整表翻转。这样不依赖对旧文件的假设。

用法：
    python -X utf8 ef_pairs_main.py [--top 5] [--out out/vote_pairs_main.npz]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GRAPH = os.path.join(HERE, "graph")
OUT = os.path.join(HERE, "out")

# vote_design 用过的 5 个方位（0=正前，+ 为右）
ANG = {"正前": 0.0, "左前": -32.0, "右前": 32.0, "左后": -100.0, "右后": 100.0}


def blob_img(u: float, w: int, h: int, sigma_frac: float = 0.055) -> np.ndarray:
    """在横向比例 u 处画一条竖向亮带（模拟主画面里某个方位有东西）。"""
    img = np.full((h, w, 3), 18, dtype=np.float32)
    x = u * w
    s = max(2.0, sigma_frac * w)
    g = np.exp(-((np.arange(w) - x) ** 2) / (2 * s * s))
    g = np.clip(g, 0, 1) * 232 + 18
    img[:] = g[None, :, None]
    # 给一点上下结构（天空亮、地面暗），更像真实画面
    img[: h // 2] *= 1.15
    img[h // 2:] *= 0.85
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--eye-w", type=int, default=490)
    ap.add_argument("--eye-h", type=int, default=137)
    ap.add_argument("--sigma", type=float, default=0.20,
                    help="刺激亮带宽度（占画面宽比例）。★ 极关键："
                         "0.055 时最强对立度仅 0.0085，0.20 时有 0.0187 ——"
                         "比小地图口径（0.0138）还高。这些下行神经元响应的是"
                         "**大尺度**左右差异，不是局部小物体。")
    ap.add_argument("--out", default=os.path.join(OUT, "vote_pairs_main.npz"))
    ap.add_argument("--sweep", action="store_true",
                    help="只扫描刺激宽度×输入增益，不写文件")
    ap.add_argument("--sigma-help", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    from ef_eye import MainEye

    t0 = time.time()
    print("载入连接组…")
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = W.shape[0]
    print(f"  {n:,} 神经元 / {W.nnz:,} 条边")

    eye = MainEye(os.path.join(OUT, "optic_map.feather"),
                  os.path.join(GRAPH, "graph_meta.feather"),
                  hfov=90.0, map_mode="linear", ds=1, verbose=False,
                  mask_rects=[])

    def run(drive):
        r = np.zeros(n, dtype=np.float32)
        for _ in range(args.steps):
            r = (args.a * r + args.b * np.tanh(W @ r + drive)).astype(np.float32)
        return r

    # ---- 侧别（照 vote_design 的三级回退）----
    side = meta["rootSide"].astype(str).to_numpy()
    bad = ~np.isin(side, ["L", "R"])
    side[bad] = meta["somaSide"].astype(str).to_numpy()[bad]
    inst = meta["instance"].astype(str).to_numpy()
    inst_side = np.array([s[-1] if isinstance(s, str) and s.endswith(("_L", "_R"))
                          else "?" for s in inst])
    bad = ~np.isin(side, ["L", "R"])
    side[bad] = inst_side[bad]

    dn_mask = meta["superclass"].astype(str).str.contains(
        "descending", case=False, na=False).to_numpy()
    dn_idx = np.flatnonzero(dn_mask)
    print(f"下行神经元 {len(dn_idx):,}")

    # ---- 跑 5 个方位 ----
    base = run(np.zeros(n, dtype=np.float32))

    if args.sweep:
        # 扫「刺激宽度 × 输入增益」，看主画面口径到底能不能产生强左右信号。
        # 背景：小地图口径（vote_design.py，b=2.0）最强对立度 1.083，
        # 而主画面（b=1.0）只有 0.0085 —— 差 127 倍，得先搞清是哪一环。
        side_ = meta["rootSide"].astype(str).to_numpy()
        bad_ = ~np.isin(side_, ["L", "R"])
        side_[bad_] = meta["somaSide"].astype(str).to_numpy()[bad_]
        dn_m = meta["superclass"].astype(str).str.contains(
            "descending", case=False, na=False).to_numpy()
        P = []
        for t, grp in meta[dn_m].groupby("type"):
            ix = grp.index.to_numpy()
            L = [int(i) for i in ix if side_[i] == "L"]
            R = [int(i) for i in ix if side_[i] == "R"]
            P += [(a_, b_) for a_ in L for b_ in R]
        P = np.array(P, dtype=np.int64)
        print(f"\n配对 {len(P):,} 对。开始扫描：")
        print("  同一组参数下，**两种眼睛口径**的最强对立度对比")
        print(f"  {'sigma':>7s} {'b':>5s} {'步数':>5s} "
              f"{'主画面':>11s} {'小地图':>11s} {'倍数':>8s}")

        # 小地图口径：用 vote_design 的极坐标亮斑 + rm.encode
        from retina import RetinaMapper
        rm2 = RetinaMapper(os.path.join(OUT, "optic_map.feather"),
                           os.path.join(GRAPH, "graph_meta.feather"))
        G = rm2.grid
        cc0 = (G - 1) / 2.0
        yy2, xx2 = np.mgrid[0:G, 0:G].astype(np.float32)

        def blob_polar(angle_deg, width=12.0):
            a = np.deg2rad(angle_deg)
            bx = cc0 + np.sin(a) * cc0 * 0.6
            by = cc0 - np.cos(a) * cc0 * 0.6
            d2 = (xx2 - bx) ** 2 + (yy2 - by) ** 2
            return (np.exp(-d2 / (2 * width ** 2)))[..., None].repeat(3, -1).astype(np.float32)

        def maxcontra(Dm2):
            cc = np.array([(Dm2[1, li] + Dm2[3, li]) / 2 - (Dm2[2, ri] + Dm2[4, ri]) / 2
                           for li, ri in P])
            return np.abs(cc)

        for sg in (0.02, 0.05, 0.10, 0.20):
            for bb, stp in ((1.0, 10), (2.0, 12)):
                # 主画面
                DD = {}
                for name, ang in ANG.items():
                    img = blob_img((ang + 135.0) / 270.0, args.eye_w, args.eye_h, sg)
                    eye.reset()
                    eye.sample(img)
                    dv, _, _ = eye.sample(img)
                    rr = np.zeros(n, dtype=np.float32)
                    for _ in range(stp):
                        rr = (0.5 * rr + bb * np.tanh(W @ rr + dv)).astype(np.float32)
                    DD[name] = rr - base
                cs_main = maxcontra(np.array([DD[k] for k in ANG]))
                # 小地图
                DD2 = {}
                for name, ang in ANG.items():
                    rm2.reset()
                    dv = rm2.encode(blob_polar(ang))
                    rr = np.zeros(n, dtype=np.float32)
                    for _ in range(stp):
                        rr = (0.5 * rr + bb * np.tanh(W @ rr + dv)).astype(np.float32)
                    DD2[name] = rr - base
                cs_mm = maxcontra(np.array([DD2[k] for k in ANG]))
                m1, m2 = cs_main.max(), cs_mm.max()
                print(f"  {sg:>7.2f} {bb:>5.1f} {stp:>5d} {m1:>11.6f} {m2:>11.6f} "
                      f"{m2/max(m1,1e-9):>7.1f}x")
        return

    D = {}
    for name, ang in ANG.items():
        u = (ang + 135.0) / 270.0
        img = blob_img(u, args.eye_w, args.eye_h, args.sigma)
        eye.reset()
        eye.sample(img)
        drive, _, _ = eye.sample(img)
        D[name] = run(drive) - base
        print(f"  跑完 {name:>4s} (az {ang:+6.0f}° → u={u:.3f})  驱动均值 "
              f"{float(drive[eye.pr_idx].mean()):+.5f}")

    keys = list(ANG)
    Dm = np.array([D[k] for k in keys])          # (5, n)

    # ---- 构造同 type 的 L/R 配对 ----
    pairs = []
    for t, grp in meta[dn_mask].groupby("type"):
        idxs = grp.index.to_numpy()
        L = [int(i) for i in idxs if side[i] == "L"]
        R = [int(i) for i in idxs if side[i] == "R"]
        for a_ in L:
            for b_ in R:
                pairs.append((a_, b_))
    pairs = np.array(pairs, dtype=np.int64)
    print(f"\n同 type L/R 配对：{len(pairs):,} 对")

    # 对立度：L 细胞对左刺激的响应 − R 细胞对右刺激的响应
    contra = np.array([
        (Dm[1, li] + Dm[3, li]) / 2 - (Dm[2, ri] + Dm[4, ri]) / 2
        for li, ri in pairs
    ])
    strength = np.abs(contra)
    print(f"  对立度：中位 {np.median(strength):.6f}  "
          f"90分位 {np.percentile(strength, 90):.6f}  最大 {strength.max():.6f}")

    # ---- 选 top-k，并**实测**校正朝向 ----
    sel = np.argsort(strength)[::-1][:args.top]
    sel_pairs = pairs[sel].copy()
    sel_str = strength[sel]

    def margin(v, stim):
        return float(v[stim][sel_pairs[:, 0]].mean() - v[stim][sel_pairs[:, 1]].mean())

    rL = run_from = None
    # 用「左前」和「右前」判定朝向
    mL, mR = margin(Dm, 1), margin(Dm, 2)
    flipped = False
    if not (mL < 0 < mR):
        sel_pairs = sel_pairs[:, ::-1].copy()
        flipped = True
        mL, mR = margin(Dm, 1), margin(Dm, 2)
    print(f"\n  top{args.top} 朝向：{'翻转后' if flipped else '原样'}"
          f"  左前 margin {mL:+.6f} · 右前 margin {mR:+.6f}"
          f"  ({'✓ 符号相反' if mL * mR < 0 else '✗ 仍不相反'})")

    print(f"\n  {'type':>10s} {'L idx':>8s} {'R idx':>8s} {'对立度':>13s}")
    for k, (li, ri) in enumerate(sel_pairs):
        print(f"  {str(meta.iloc[li]['type']):>10s} {li:>8d} {ri:>8d} "
              f"{sel_str[k]:>+13.6f}")

    np.savez_compressed(args.out, pairs=sel_pairs, contra=contra[sel],
                        strength=sel_str, all_pairs=pairs, all_contra=contra)
    print(f"\n[{time.time()-t0:.1f}s] → {args.out}")


if __name__ == "__main__":
    main()
