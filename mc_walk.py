#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mc_walk.py — 闭环：果蝇看画面 → 决定方向 → 真的驱动 MC 角色行走

不用小地图，用第一人称画面作视觉输入（按用户选择）。

每一拍：
    1. 截取 MC 窗口画面 → 按**第一人称 FOV**映射到 6,006 个感光细胞
       （画面中心 = 正前方 0°，左右边缘 = ±FOV/2）
    2. 算帧间光流（果蝇视觉的关键量）
    3. 跑 MaleCNS 连接组动力学（2603 万条边）
    4. 扇区投票 → 左 / 右
    5. 执行：一直按 W 前进；判「左」则鼠标左转一点，判「右」则右转一点
    6. 记录轨迹

安全阀（交接文档第八节的三条兜底，在 MC 里都能真验证）：
    · 卡住检测：连续 N 拍画面几乎不变 → 认为卡住 → 强制换向
    · 掉出地形 / 掉血：画面出现红色（受伤）或长时间无位移
    · 强制换向：同一方向超时

用法：
    python mc_walk.py --seconds 30                  # 跑 30 秒
    python mc_walk.py --seconds 30 --turn-gain 0.8  # 转向幅度
    python mc_walk.py --dry                         # 只决策不按键（安全预览）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GRAPH = os.path.join(HERE, "graph")
OUT = os.path.join(HERE, "out")

# MC 默认水平视野约 90°（FOV 70 时）；用户可调
DEFAULT_HFOV = 90.0


def build_retina_mapper(optic_map, meta_path, hfov, grid=64):
    """重载 RetinaMapper，但把方位角映射改成「第一人称画面」口径

    原版 retina.py 假设输入是俯视小地图，方位角铺满 ±135°。
    第一人称画面只有 ±hfov/2，所以重算每个感光细胞该采画面的哪个横向位置。
    """
    from retina import RetinaMapper
    rm = RetinaMapper(optic_map, meta_path, grid=grid)
    # 感光细胞原方位角（眼面坐标，-135~135）
    az = rm._az.copy()
    half = hfov / 2.0
    # 映射到画面横向比例 0..1：-half -> 0，0 -> 0.5，+half -> 1
    u = (az + half) / hfov
    rm._fp_u = np.clip(u, -0.2, 1.2)          # 允许略微越界，后面会裁
    rm._in_fov = (az >= -half) & (az <= half)
    print(f"  第一人称映射：FOV {hfov:.0f}° → {int(rm._in_fov.sum()):,}/{len(az):,} "
          f"个感光细胞落在画面内")
    return rm


def sample_first_person(rm, img, prev_lum=None, map_mode="fov"):
    """画面 → 感光细胞驱动（第一人称口径）

    map_mode：
      "fov"    只用方位角落在画面 FOV 内的感光细胞（几何正确，但只有约 1/3 细胞工作）
      "linear" 把 270° 的细胞方位角线性压缩到画面宽度（所有细胞都工作，但失真）
    归一化只对**有效细胞**做 —— 否则视野外那批恒为 0 的值会把均值拉低，
    经 zscore 后全部变成负偏置（实测导致 22/23 拍判「左」）。
    """
    g = img.astype(np.float32)
    if g.max() > 1.5:
        g /= 255.0
    h, w = g.shape[:2]
    lum = g @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    temporal = np.abs(lum - prev_lum) if prev_lum is not None else np.zeros_like(lum)
    color = np.maximum(g[..., 1] - 0.5 * (g[..., 0] + g[..., 2]), 0)
    drive_map = np.clip(0.45 * lum + 1.6 * temporal + 0.25 * color, 0, 1)

    if map_mode == "linear":
        # 全部细胞的方位角线性铺到画面宽度
        az = rm._az
        u = (az + 135.0) / 270.0
        valid = np.ones(len(az), dtype=bool)
    else:
        u = rm._fp_u
        valid = rm._in_fov

    vals = np.zeros(len(rm._pr_idx), dtype=np.float32)
    for i in range(len(rm._pr_idx)):
        if not valid[i]:
            continue
        uu = float(np.clip(u[i], 0.0, 1.0))
        x0 = int((uu - 0.06) * w)
        x1 = int((uu + 0.06) * w)
        x0, x1 = max(0, x0), min(w, max(x1, x0 + 1))
        vals[i] = drive_map[:, x0:x1].mean()

    # ★ 归一化只针对有效细胞
    m = valid
    if m.sum() > 1:
        v = vals[m]
        mu, sd = float(v.mean()), float(v.std())
        if sd > 1e-6:
            vals[m] = (v - mu) / sd * 0.30 + 0.25 * mu
        else:
            vals[m] = v * 0.25
    out = np.zeros(rm._n, dtype=np.float32)
    out[rm._pr_idx] = vals
    return out, lum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=3.0, help="决策频率")
    ap.add_argument("--steps", type=int, default=8, help="每拍展开多少次网络更新")
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--hfov", type=float, default=DEFAULT_HFOV)
    ap.add_argument("--turn-gain", type=float, default=0.6, help="转向幅度（像素/拍）")
    ap.add_argument("--forward", action="store_true", default=True)
    ap.add_argument("--dry", action="store_true", help="只决策不按键（安全预览）")
    ap.add_argument("--map-mode", choices=["fov", "linear"], default="linear",
                    help="视觉映射口径：fov=只用视野内细胞；linear=270度压缩到画面宽度")
    ap.add_argument("--no-forward", action="store_true", help="不前进，只转向")
    ap.add_argument("--out", default=os.path.join(OUT, "mc_walk_log.csv"))
    args = ap.parse_args()

    from mc_control import MC
    from sector_vote import DirectionDecoder

    print("载入连接组…")
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = W.shape[0]
    print(f"  {n:,} 神经元 / {W.nnz:,} 条边")

    rm = build_retina_mapper(os.path.join(OUT, "optic_map.feather"),
                             os.path.join(GRAPH, "graph_meta.feather"), args.hfov)
    dec = DirectionDecoder(top_k=5)

    mc = MC()
    print(f"MC 窗口 {mc.rect}   在世界里：{mc.in_world()}")
    if not args.dry:
        if not mc.gi.is_focused():
            mc.focus()
            time.sleep(0.5)
        if not mc.gi.is_focused():
            sys.exit("❌ 游戏窗口拿不到焦点")

    print()
    print(f"参数：{args.hz} Hz 决策，每拍 {args.steps} 步网络，a={args.a} b={args.b}，"
          f"转向增益 {args.turn_gain}，前进={not args.no_forward}{'（DRY 预览）' if args.dry else ''}")
    print()

    n_tick = int(args.seconds * args.hz)
    r = np.zeros(n, dtype=np.float32)
    prev_lum = None
    prev_img = None
    rows = []
    t0 = time.time()
    stuck = 0
    held = None

    print(f"  {'拍':>4s} {'t':>6s} {'驱动':>8s} {'margin':>10s} {'方向':>5s} {'帧差':>8s} {'动作':>14s}")
    try:
        for tick in range(n_tick):
            ts = time.time() - t0
            img = mc.shot()
            drive, lum = sample_first_person(rm, img, prev_lum, args.map_mode)

            # 卡住检测：画面几乎不变
            fd = 0.0
            if prev_img is not None:
                fd = float(np.abs(img.astype(np.float32) - prev_img.astype(np.float32)).mean())
            prev_lum, prev_img = lum, img
            if fd < 1.2:
                stuck += 1
            else:
                stuck = 0

            for _ in range(args.steps):
                r = (args.a * r + args.b * np.tanh(W @ r + drive)).astype(np.float32)
            o = dec.step(r, dt=1.0 / args.hz)
            direction = dec.direction_name()
            margin = o["margin"]

            # 执行
            action = []
            if not args.dry:
                if stuck >= 3:
                    # 兜底：卡住了 → 强制大角度换向
                    sign = 1 if dec.hyst.current >= 0 else -1
                    mc.gi.mouse_move(int(sign * args.turn_gain * 6), 0, steps=6)
                    action.append("卡住→强制转")
                    stuck = 0
                else:
                    if direction == "左":
                        mc.gi.mouse_move(-int(args.turn_gain * 12), 0, steps=4)
                        action.append("左转")
                    elif direction == "右":
                        mc.gi.mouse_move(int(args.turn_gain * 12), 0, steps=4)
                        action.append("右转")
                if not args.no_forward:
                    mc.gi.key_down("w")
                    action.append("前进")
            else:
                action.append("(dry)")

            rows.append(dict(tick=tick, t=round(ts, 2), drive=float(drive[rm._pr_idx].mean()),
                             margin=float(margin), raw_margin=float(o["margin_raw"]),
                             baseline=float(o["baseline"]), direction=direction,
                             frame_diff=fd, stuck=stuck, action="+".join(action)))
            if tick % 2 == 0:
                print(f"  {tick:>4d} {ts:>6.1f} {drive[rm._pr_idx].mean():>8.5f} "
                      f"{margin:>+10.5f} {direction:>5s} {fd:>8.2f} {'+'.join(action):>14s}")

            nxt = t0 + (tick + 1) / args.hz
            sl = nxt - time.time()
            if sl > 0:
                time.sleep(sl)
    finally:
        if not args.dry:
            mc.gi.key_up("w")
            mc.gi.release_all()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print()
    print("=" * 78)
    from collections import Counter
    print(f"  决策 {len(df)} 拍 / {args.seconds:.0f} 秒")
    print(f"  方向分布：{dict(Counter(df['direction']))}")
    print(f"  切换次数：{int((df['direction'] != df['direction'].shift()).sum()) - 1}")
    print(f"  margin：{df['margin'].min():+.5f} ~ {df['margin'].max():+.5f}")
    print(f"  画面帧差：均值 {df['frame_diff'].mean():.2f}   最小 {df['frame_diff'].min():.2f}")
    print(f"  判定「卡住」的拍数：{int((df['stuck'] >= 3).sum())}")
    print()
    print(f"  日志 → {args.out}")


if __name__ == "__main__":
    main()
