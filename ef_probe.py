#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ef_probe.py — 诊断：小地图采样框对不对？按 W 到底有没有动？

存四张图：整屏/小地图 × 走之前/走之后，并报差分。
看 mm_*.png 就能确认采样的到底是不是小地图。

用法（**必须管理员**）：
    python -X utf8 ef_probe.py [--walk 3.0]
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ef_shot import grab                      # noqa: E402
from ef_vision import MapOdometry, minimap_rect, phase_shift, to_gray  # noqa: E402
from endfield_walk import find_endfield       # noqa: E402

user32 = ctypes.windll.user32
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "shots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", type=float, default=3.0)
    args = ap.parse_args()

    from interception_py import Interception

    win = find_endfield()
    if not win:
        sys.exit("终末地没在运行")
    hwnd, ox, oy, w, h = win
    ic = Interception()
    kb, ms = ic.keyboard_device(), ic.mouse_device()

    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.2)
    print(f"窗口 ({ox},{oy}) {w}×{h} · 前台 {user32.GetForegroundWindow() == hwnd}")

    rect = minimap_rect(h, inner=True, ox=ox, oy=oy)
    print(f"小地图采样矩形 {rect}\n")

    def snap(tag):
        full = grab(ox, oy, w, h)
        mm = grab(*rect)
        Image.fromarray(full).save(os.path.join(OUT, f"ef_probe_{tag}_full.png"))
        Image.fromarray(mm).resize((mm.shape[1] * 3, mm.shape[0] * 3),
                                   Image.NEAREST).save(
            os.path.join(OUT, f"ef_probe_{tag}_mm.png"))
        return full, mm

    f_pre, m_pre = snap("pre")

    print(f"按住 W {args.walk} 秒 …")
    ic.key_down("w", kb)
    time.sleep(args.walk)
    ic.key_up("w", kb)
    time.sleep(0.5)

    f_post, m_post = snap("post")

    # 主画面中央差分（避开 HUD）
    def ctr(a):
        hh, ww = a.shape[:2]
        return a[int(hh * 0.25):int(hh * 0.65), int(ww * 0.35):int(ww * 0.65)]

    d_full = float(np.abs(ctr(f_pre).astype(np.int16) - ctr(f_post).astype(np.int16)).mean())
    d_mm = float(np.abs(m_pre.astype(np.int16) - m_post.astype(np.int16)).mean())

    dx, dy, sharp = phase_shift(to_gray(m_pre), to_gray(m_post), max_shift=60)

    print()
    print("=" * 66)
    print(f"  主画面中央差分  {d_full:8.3f}   （>10 说明世界变了 = 真的走了）")
    print(f"  小地图差分      {d_mm:8.3f}")
    print(f"  相位相关（整段） 内容位移 ({dx:+d}, {dy:+d})  锐度 {sharp:.2f}")
    print(f"    → 角色位移 ({-dx:+d}, {-dy:+d}) 地图像素")
    print()
    print(f"  图 → {OUT}\\ef_probe_[pre|post]_[full|mm].png")
    print("  先看 ef_probe_pre_mm.png：那应该正好是一张小地图。")

    ic.close()


if __name__ == "__main__":
    main()
