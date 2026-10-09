#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ef_combo_collect.py — 密集采集战斗帧，用来测连携技提示的位置范围。

模板已经确认很好用（自身 1.000，其他帧 ≤0.401）。
现在缺的是"它在画面上出现过多少次、都在哪" —— 一个正样本定不了搜索窗口。

战斗中每 0.6 秒存一张（1280×720 缩略），最多 200 张。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from combat import CombatMarker                # noqa: E402
from ef_shot import grab                       # noqa: E402
from ef_vision import content_rect             # noqa: E402
from endfield_walk import find_endfield        # noqa: E402

hwnd, wx, wy, ww, wh = find_endfield()
x, y, w, ch = content_rect(wx, wy, ww, wh)
OUTD = os.path.join(HERE, "shots", "combo")
os.makedirs(OUTD, exist_ok=True)
for f in os.listdir(OUTD):
    os.remove(os.path.join(OUTD, f))
mr = [28, 47, 76, 97]
mark = CombatMarker((mr[0], mr[1], mr[2], mr[3]), min_px=80, verbose=False)

SECONDS, MAXN = 420, 200
print(f"战斗帧密集采集：每 0.6s 一张，最多 {MAXN} 张")
print("等红环亮… 按 Ctrl+C 停。\n")
n = 0
last = 0.0
t0 = time.time()
while time.time() - t0 < SECONDS and n < MAXN:
    on = mark.step(grab(x + mr[0], y + mr[1], mr[2] - mr[0], mr[3] - mr[1]))
    now = time.time()
    if on and now - last >= 0.6:
        a = np.asarray(grab(x, y, w, ch))
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        n += 1
        Image.fromarray(a.astype(np.uint8)).resize((1280, 720), Image.LANCZOS).save(
            os.path.join(OUTD, f"f_{n:03d}.png"))
        if n % 10 == 0:
            print(f"  [{now-t0:6.1f}s] 已存 {n} 张")
        last = now
    time.sleep(0.15)
print(f"\n共 {n} 张 → shots/combo/")
