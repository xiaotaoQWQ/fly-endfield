#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""endfield_walk.py — 终末地闭环：果蝇看小地图 → 决策 → 控制角色行走

与 MC 闭环的关键差异：
    感知  终末地的小地图是**俯视图**，用 retina.py 的俯视口径采样
          （MC 用的是第一人称 FOV 口径）。小地图参数已标定：
          2560×1440 下圆心 (188,236)、半径 143
    执行  走 **Interception 驱动**（应用层 SendInput 会被 ACE 丢弃）

⚠ 必须以**管理员**运行（Interception 的 create_context 需要）。

用法：
    python endfield_walk.py --seconds 30 --dry     # 只决策不按键（安全预览）
    python endfield_walk.py --seconds 30           # 真跑
    python endfield_walk.py --seconds 60 --turn-gain 0.25
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
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

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# 终末地小地图（2560×1440 标定）
MM_CX, MM_CY, MM_R, MM_H = 188, 236, 143, 1440


class BIH(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def grab(x, y, w, h):
    hdc = user32.GetDC(0)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mem, bmp)
    gdi32.BitBlt(mem, 0, 0, w, h, hdc, x, y, 0x00CC0020)
    bi = BIH()
    bi.biSize = ctypes.sizeof(BIH)
    bi.biWidth, bi.biHeight = w, -h
    bi.biPlanes, bi.biBitCount = 1, 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), 0)
    gdi32.SelectObject(mem, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(0, hdc)
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return a[:, :, ::-1].copy()


def find_endfield():
    res = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hw, lp):
        if user32.IsWindowVisible(hw):
            n = user32.GetWindowTextLengthW(hw)
            b = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hw, b, n + 1)
            if "endfield" in b.value.lower():
                r = wintypes.RECT()
                user32.GetWindowRect(hw, ctypes.byref(r))
                res.append((hw, r.left, r.top, r.right - r.left, r.bottom - r.top))
        return True

    user32.EnumWindows(cb, 0)
    return res[0] if res else None


def crop_minimap(img):
    """按分辨率缩放到小地图区域"""
    h, w = img.shape[:2]
    k = h / MM_H
    cx, cy, r = MM_CX * k, MM_CY * k, MM_R * k
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = int(cx + r), int(cy + r)
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--turn-gain", type=float, default=0.3)
    ap.add_argument("--center", choices=["none", "mean", "zscore"], default="zscore")
    ap.add_argument("--no-jump", action="store_true")
    ap.add_argument("--no-sprint", action="store_true")
    ap.add_argument("--dry", action="store_true", help="只决策不按键")
    ap.add_argument("--out", default=os.path.join(OUT, "endfield_walk.csv"))
    args = ap.parse_args()

    if not args.dry and not ctypes.windll.shell32.IsUserAnAdmin():
        print("⚠ 不是管理员 —— Interception 会失败。请以管理员运行。")

    from retina import RetinaMapper
    from sector_vote import DirectionDecoder
    from action_map import ActionMapper
    from motor import MotorPlan

    print("载入连接组…")
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = W.shape[0]
    print(f"  {n:,} 神经元 / {W.nnz:,} 条边")

    # ★ 俯视口径（终末地的小地图就是俯视图）
    rm = RetinaMapper(os.path.join(OUT, "optic_map.feather"),
                      os.path.join(GRAPH, "graph_meta.feather"))
    rm.reset()
    dec = DirectionDecoder(top_k=5)
    amap = ActionMapper.load()
    print()

    win = find_endfield()
    if not win:
        sys.exit("终末地没在运行")
    hwnd, x, y, w, h = win
    print(f"终末地窗口 ({x},{y}) {w}×{h}")

    # 执行器
    ex = None
    ic = None
    if not args.dry:
        from interception_py import Interception
        ic = Interception()
        kb = ic.keyboard_device()
        ms = ic.mouse_device()
        print(f"Interception：键盘设备 {kb}，鼠标设备 {ms}")

        class EFExec:
            name = "endfield-interception"
            KEYS = {"forward": "w", "back": "s", "jump": "space", "sprint": "shift"}

            def __init__(self):
                self.held = set()

            def _sync(self, logical, want):
                now = logical in self.held
                if want and not now:
                    ic.key_down(self.KEYS[logical], kb)
                    self.held.add(logical)
                elif not want and now:
                    ic.key_up(self.KEYS[logical], kb)
                    self.held.discard(logical)

            def apply(self, plan):
                self._sync("forward", plan.move > 0)
                self._sync("back", plan.move < 0)
                self._sync("sprint", bool(plan.sprint and plan.move > 0))
                if plan.turn and ms:
                    ic.mouse_move(int(plan.turn * args.turn_gain * 40), 0, ms)
                if plan.jump:
                    ic.tap("space", 0.09, kb)

            def release_all(self):
                for k in list(self.held):
                    try:
                        ic.key_up(self.KEYS[k], kb)
                    except Exception:
                        pass
                self.held.clear()

        ex = EFExec()

    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.0)
    print(f"前台：{user32.GetForegroundWindow() == hwnd}")
    print()

    total = int(args.seconds * args.hz)
    print(f"闭环 {args.seconds:.0f} 秒 = {total} 拍 @ {args.hz} Hz"
          f"{'（DRY：只决策不按键）' if args.dry else ''}")
    print()

    r = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    rows = []
    print(f"  {'拍':>4s} {'t':>6s} {'驱动':>9s} {'margin':>10s} {'方向':>4s} "
          f"{'跳':>3s} {'冲':>3s}  动作")
    try:
        for tick in range(total):
            ts = time.time() - t0
            full = grab(x, y, w, h)
            mm = crop_minimap(full)
            drive = rm.encode(mm, center=args.center)

            for _ in range(args.steps):
                r = (args.a * r + args.b * np.tanh(W @ r + drive)).astype(np.float32)

            o = dec.step(r, dt=1.0 / args.hz)
            direction = dec.direction_name()
            turn = -1 if direction == "左" else (1 if direction == "右" else 0)
            acts = amap.step(r)
            plan = MotorPlan(move=1, turn=turn,
                             jump=(False if args.no_jump else acts["jump"]),
                             sprint=(False if args.no_sprint else acts["sprint"]),
                             turn_amount=acts["turn"])
            if ex is not None:
                ex.apply(plan)

            rows.append(dict(tick=tick, t=round(ts, 2), drive=float(drive[rm._pr_idx].mean()),
                             margin=float(o["margin"]), direction=direction,
                             jump=bool(plan.jump), sprint=bool(plan.sprint),
                             action=plan.describe()))
            if tick % 2 == 0 or plan.jump or plan.sprint:
                print(f"  {tick:>4d} {ts:>6.1f} {drive[rm._pr_idx].mean():>9.5f} "
                      f"{o['margin']:>+10.5f} {direction:>4s} "
                      f"{'跳' if plan.jump else '  ':>3s} "
                      f"{'冲' if plan.sprint else '  ':>3s}  {plan.describe()}")

            nxt = t0 + (tick + 1) / args.hz
            sl = nxt - time.time()
            if sl > 0:
                time.sleep(sl)
    except KeyboardInterrupt:
        print("\n  中断")
    finally:
        if ex is not None:
            ex.release_all()
        if ic is not None:
            ic.close()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")
    from collections import Counter
    print()
    print("=" * 74)
    print(f"  {len(df)} 拍 / {time.time()-t0:.0f} 秒")
    print(f"  方向分布：{dict(Counter(df['direction']))}")
    print(f"  跳跃 {int(df['jump'].sum())} 次 · 冲刺 {int(df['sprint'].sum())} 次")
    print(f"  感光驱动均值：{df['drive'].mean():.5f}")
    print(f"  日志 → {args.out}")


if __name__ == "__main__":
    main()
