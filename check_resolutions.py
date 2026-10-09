#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_resolutions.py — 多分辨率兼容性验证。

思路：拿**真实战斗录屏**（951 帧，2560×1440 原生），缩放成各种分辨率，
按各分辨率下的标定重新算搜索区和模板，看检测器还认不认得出连携技提示。

★ 为什么要这么做：
  归一化坐标系只能保证"区域位置对"，**保证不了模板重采样之后还认得出**。
  模板放大/缩小会模糊，小图标尤其明显 —— 30×28 缩到 1600×900 只剩 19×18，
  还能不能匹配上？只能实测。

用法：
    python -X utf8 check_resolutions.py [录制目录]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from screen_profile import Profile  # noqa: E402
from live_frame import LiveFrame                                # noqa: E402

# 录制时 band 抓的内容坐标（见 rec_combo.py 的 BAND）
BAND_X, BAND_Y, BAND_W, BAND_H = 1240, 400, 700, 560
REF_W, REF_H = 2560, 1440

# 要测的分辨率（含非 16:9）
RESOS = [
    (2560, 1440, "16:9 原生（基准）"),
    (1920, 1080, "16:9 1080p"),
    (3840, 2160, "16:9 4K"),
    (1600, 900, "16:9 小窗"),
    (1280, 720, "16:9 720p"),
    (2560, 1080, "21:9 带鱼屏"),
    (1920, 1200, "16:10"),
    (1680, 1050, "16:10 小"),
]


def latest_rec():
    ds = sorted(glob.glob(os.path.join(HERE, "rec", "*")),
                key=os.path.getmtime, reverse=True)
    for d in ds:
        if glob.glob(os.path.join(d, "band_*.npy")):
            return d
    return None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else latest_rec()
    if not d:
        sys.exit("找不到录制目录（先跑 rec_combo.py）")
    fs = sorted(glob.glob(os.path.join(d, "band_*.npy")))
    print("=" * 72)
    print(f"  多分辨率兼容性验证")
    print(f"  录制：{os.path.basename(d)}   {len(fs)} 帧")
    print("=" * 72)
    print()

    # 先把原生帧读进来（只读一次，后面反复缩放）
    raw = []
    for p in fs:
        a = np.load(p)
        if a.shape[0] == BAND_H and a.shape[1] == BAND_W:
            raw.append(a)
    print(f"  可用帧：{len(raw)}")
    print()

    # 原生基准：先确定"哪些帧真的有提示"
    print("  ① 先用原生分辨率定出「真有提示」的帧（相关 >= 0.50）")
    p0 = Profile(0, 0, REF_W, REF_H, measured=True, verbose=False)
    lf0 = LiveFrame(p0, lambda *a: None, verbose=False)
    tpl0 = p0.template("combo_e")
    rc0 = p0.rect("combo")
    ox0, oy0 = rc0[0] - BAND_X, rc0[1] - BAND_Y
    truth = []
    # ★ 注意：raw 里存的是 band（内容坐标 1240,400 起），而 rc0 是**内容坐标**。
    #   切片时必须减去 band 的偏移，否则切到画面外 → 全是 0。
    for i, a in enumerate(raw):
        lf0.img = a
        c, _, _ = lf0.find(tpl0, lf0.at(ox0, oy0, rc0[2], rc0[3]))
        if c >= 0.50:
            truth.append(i)
    print(f"     原生下判定有提示的帧：{len(truth)} 帧")
    print()

    print("  ② 各种分辨率下能不能认出同样的帧")
    print()
    print(f"  {'分辨率':<16s} {'区域':<14s} {'模板':<10s} {'认出':<10s} "
          f"{'召回率':<9s} {'最高分':<9s} {'噪声底'}")
    print("  " + "-" * 74)

    all_ok = True
    for W, H, note in RESOS:
        sx, sy = W / REF_W, H / REF_H
        prof = Profile(0, 0, W, H, measured=True, verbose=False)
        tpl = prof.template("combo_e")
        rc = prof.rect("combo")
        if tpl is None or tpl.size == 0:
            print(f"  {W}×{H:<10d} 模板取不到 ✗")
            all_ok = False
            continue

        # 把原生帧缩放到目标分辨率下的 band
        bw, bh = max(1, int(round(BAND_W * sx))), max(1, int(round(BAND_H * sy)))
        lf = LiveFrame(prof, lambda *a: None, verbose=False)
        # ★ 带内偏移 = 区域在目标分辨率下的起点 − band 在目标分辨率下的起点。
        #   rc[0] **已经是**目标分辨率下的值，不能再乘一次 sx —— 第一版就是
        #   多乘了一次，1920×1080 下算出 -142，切片跑到画面外，全判 0 分。
        ox = rc[0] - int(round(BAND_X * sx))
        oy = rc[1] - int(round(BAND_Y * sy))
        hit, vals = 0, []
        for i, a in enumerate(raw):
            small = np.asarray(Image.fromarray(a).resize((bw, bh), Image.LANCZOS))
            lf.img = small
            c, _, _ = lf.find(tpl, lf.at(ox, oy, rc[2], rc[3]))
            vals.append(c)
            if c >= 0.50:
                hit += 1
        v = np.array(vals)
        # 召回率：原生判定有提示的那些帧，这里还认不认得出
        rec = float((v[truth] >= 0.50).mean()) if truth else float("nan")
        noise = float(np.median(v))
        good = (rec >= 0.8) if truth else True
        all_ok = all_ok and good
        flag = "" if good else "  ✗"
        print(f"  {W}×{H:<10d} {str(rc[2])+'×'+str(rc[3]):<14s} "
              f"{tpl.shape[1]}×{tpl.shape[0]:<7d} {hit:<10d} "
              f"{rec*100:5.1f}%    {v.max():+.3f}    {noise:+.3f}{flag}")

    print()
    print("  说明：")
    print("    · 「认出」= 相关 >= 0.50 的帧数（原生下有 "
          f"{len(truth)} 帧真提示）")
    print("    · 「召回率」= 原生判定有提示的帧里，这个分辨率还能认出的比例")
    print("    · 「噪声底」= 所有帧相关值的中位数（越低越好）")
    print()
    print("=" * 72)
    print("  " + ("各分辨率都能认出提示 ✓" if all_ok
                  else "有分辨率认不出来 —— 需要给低分辨率单独标模板 ✗"))
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
