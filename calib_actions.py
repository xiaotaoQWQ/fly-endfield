#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calib_actions.py — 标定「动作神经元」的活动范围，为跳跃/冲刺定阈值

交接文档第五节 + Fly64 给出的动作读出通道：
    DNg100           → 前进
    DNa02 / DNg13    → 左右差（转向）
    DNp01 / DNp10    → 跳跃

本脚本用真实游戏画面跑网络，记录这些神经元的活动分布，
并检验它们是否真的可分（不同画面下活动是否有差异）。

用法：
    python calib_actions.py --frames <帧目录> --limit 60
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GRAPH = os.path.join(HERE, "graph")
OUT = os.path.join(HERE, "out")

ACTION_GROUPS = {
    "forward": ["DNg100"],
    "turn": ["DNa02", "DNg13"],
    "jump": ["DNp01", "DNp10"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(HERE, "artifacts", "run2", "frames"))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--map-mode", choices=["fov", "linear"], default="linear")
    ap.add_argument("--out", default=os.path.join(OUT, "action_calib.csv"))
    args = ap.parse_args()

    from mc_walk import build_retina_mapper, sample_first_person

    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = W.shape[0]
    rm = build_retina_mapper(os.path.join(OUT, "optic_map.feather"),
                             os.path.join(GRAPH, "graph_meta.feather"), args.hfov)
    print()

    # 目标神经元
    targets = {}
    for grp, types in ACTION_GROUPS.items():
        for t in types:
            sub = meta[meta["type"] == t]
            for _, r in sub.iterrows():
                key = f"{t}_{r['somaSide']}"
                targets[key] = dict(idx=int(r["idx"]), group=grp, type=t,
                                    side=str(r["somaSide"]))
    print(f"动作读出神经元 {len(targets)} 个：")
    for k, v in targets.items():
        print(f"  {k:>12s}  idx={v['idx']:>7d}  组={v['group']}")
    print()

    files = sorted(glob.glob(os.path.join(args.frames, "f*.jpg")))[:args.limit]
    if not files:
        sys.exit("没有帧")
    print(f"用 {len(files)} 帧真实画面跑网络…")

    r = np.zeros(n, dtype=np.float32)
    prev_lum = None
    rows = []
    for i, p in enumerate(files):
        img = np.asarray(Image.open(p).convert("RGB"))
        drive, lum = sample_first_person(rm, img, prev_lum, args.map_mode)
        prev_lum = lum
        for _ in range(args.steps):
            r = (args.a * r + args.b * np.tanh(W @ r + drive)).astype(np.float32)
        rec = dict(frame=i + 1)
        for k, v in targets.items():
            rec[k] = float(r[v["idx"]])
        rows.append(rec)
        if i % 20 == 0:
            print(f"  {i+1}/{len(files)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print()
    print("=" * 92)
    print("各通道活动分布（用于定阈值）")
    print()
    print(f"  {'通道':>12s} {'组':>8s} {'均值':>11s} {'标准差':>10s} {'最小':>11s} "
          f"{'最大':>11s} {'极差':>10s} {'可分性*':>9s}")
    stats = {}
    for k, v in targets.items():
        x = df[k].to_numpy()
        rng = float(x.max() - x.min())
        snr = rng / max(float(x.std()), 1e-9)
        stats[k] = dict(mean=float(x.mean()), std=float(x.std()),
                        min=float(x.min()), max=float(x.max()), rng=rng, snr=snr)
        print(f"  {k:>12s} {v['group']:>8s} {x.mean():>+11.6f} {x.std():>10.6f} "
              f"{x.min():>+11.6f} {x.max():>+11.6f} {rng:>10.6f} {snr:>9.2f}")
    print()
    print("  * 可分性 = 极差 / 标准差。越大说明这个通道的活动越随画面变化。")

    # 组内左右差（转向信号）
    print()
    print("=" * 92)
    print("转向信号：组内左右差")
    print()
    for t in ACTION_GROUPS["turn"]:
        lk, rk = f"{t}_L", f"{t}_R"
        if lk in df and rk in df:
            d = df[lk].to_numpy() - df[rk].to_numpy()
            print(f"  {t:>8s}: L−R  均值 {d.mean():+.6f}  标准差 {d.std():.6f}  "
                  f"范围 {d.min():+.6f} ~ {d.max():+.6f}  可分性 {np.ptp(d)/max(d.std(),1e-9):.2f}")

    print()
    print("=" * 92)
    print("分组聚合（同组同侧取平均）")
    print()
    for grp, types in ACTION_GROUPS.items():
        for side in ("L", "R"):
            keys = [f"{t}_{side}" for t in types if f"{t}_{side}" in df]
            if keys:
                v = df[keys].mean(axis=1).to_numpy()
                print(f"  {grp:>8s}_{side}: 均值 {v.mean():+.6f}  标准差 {v.std():.6f}  "
                      f"范围 {v.min():+.6f} ~ {v.max():+.6f}")

    print()
    print(f"明细 → {args.out}")
    print()
    print("下一步：用这些范围定阈值，写进 action_map.py")
    print("  · 跳跃：jump 通道超过某分位数（且持续若干拍）")
    print("  · 冲刺：forward 通道超过高分位数且持续")


if __name__ == "__main__":
    main()
