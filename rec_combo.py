#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rec_combo.py — 录屏，专抓连携技提示。

为什么这么写：
  · 全屏缩略图（1280×720 JPEG）用来**看上下文** —— 提示出现时周围什么样
  · 连携技区域**原生像素**（PNG，无损）用来**框模板** —— 缩略图会把
    小图标糊掉，框出来的模板不能用
  · 只在画面变化时存 —— 1 Hz 存 10 分钟也就几十张，不然全是重复帧
  · 每帧记一行相关值 —— 事后能直接找出"哪几帧提示真的亮着"

用法：
    python -X utf8 rec_combo.py --minutes 10

产物：
    rec/<时间戳>/full_0001.jpg      全屏缩略图
    rec/<时间戳>/band_0001.png      提示带原生像素（600×450）
    rec/<时间戳>/band_0001.npy      同上，numpy 原始数据（框模板用）
    rec/<时间戳>/log.csv            每帧的时间/相关值/是否存了
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ef_shot import grab                       # noqa: E402
from endfield_walk import find_endfield        # noqa: E402
from screen_profile import build               # noqa: E402

# 提示带：比标定的搜索区大一圈，别的角色位置有偏移也能兜住
# （内容坐标）
BAND = (1240, 400, 700, 560)

# ★★ 队伍头像条（内容坐标）—— 实测这里每格下面有个黑色徽标：
#    打叉 = 该角色连携技不可用；可用时**变成按键提示**（E）。
#    原来标定到中右 (1476,560) 那套是错的/是另一种提示，
#    所以「E 不释放」的真正原因是**搜索区根本没覆盖提示所在的地方**。
PARTY = (0, 1060, 1000, 380)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=2.0, help="采样频率")
    ap.add_argument("--diff", type=float, default=1.2,
                    help="画面平均差超过它才算「变了」")
    ap.add_argument("--out", default=os.path.join(HERE, "rec"))
    # ★ --band "x,y,w,h"（内容坐标）：想抓哪一块就抓哪一块。
    #   「任务完成」这类全屏横幅通常在中上，用 --band 0,300,2560,900 抓中间一大条。
    ap.add_argument("--band", default="",
                    help="抓取区域 x,y,w,h（内容坐标）。空=默认的连携技带")
    ap.add_argument("--tag", default="combo", help="文件名前缀")
    args = ap.parse_args()

    w5 = find_endfield()
    if not w5:
        sys.exit("没找到终末地窗口")
    hwnd, wx, wy, ww, wh = w5[:5]
    prof = build(wx, wy, ww, wh, grab_fn=grab)
    print(f"游戏窗口 hwnd={hwnd}  {ww}×{wh}")
    print(f"画面区 ({prof.x},{prof.y}) {prof.w}×{prof.h}")

    # ★ 先做遮挡检查 —— 被挡住了录下来也没用
    from screen_profile import check_occlusion
    bad = check_occlusion(hwnd, prof)
    if bad:
        print("  ⚠ 有窗口挡着。**现在录下来的会是那个窗口**，"
              "请先把它最小化。10 秒后开始，来得及。")
        time.sleep(10)
        bad = check_occlusion(hwnd, prof, verbose=False)
        if bad:
            print("  ⚠ 还是有窗口挡着，照录 —— 但你可能要重来一次。")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out, stamp)
    os.makedirs(out, exist_ok=True)
    print(f"\n录制目录：{out}")
    print(f"时长 {args.minutes:.0f} 分钟 · {args.hz:g} Hz · 变了才存")
    print("开始！（去打吧）\n")

    # 提示带（内容坐标 → 屏幕坐标）
    bx, by, bw, bh = BAND
    if args.band:
        bx, by, bw, bh = (int(v) for v in args.band.split(","))
        print(f"  自定义抓取区：内容 ({bx},{by}) {bw}×{bh}")
    bsx, bsy = prof.x + bx, prof.y + by

    # 顺便跑一下连携技检测，记相关值
    kmark = None
    try:
        from combat import ComboMarker
        kmark = ComboMarker(prof.rect("combo"), verbose=False)
    except Exception as e:
        print(f"  （连携技检测没起来：{e}）")

    prev = None
    n_saved = 0
    t0 = time.time()
    csv = open(os.path.join(out, "log.csv"), "w", encoding="utf-8")
    csv.write("t,combo_corr,combo_on,saved,file\n")

    try:
        while time.time() - t0 < args.minutes * 60:
            frame = np.asarray(grab(prof.x, prof.y, prof.w, prof.h))
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            small = np.asarray(Image.fromarray(frame.astype(np.uint8)).resize(
                (320, 180), Image.BILINEAR)).astype(np.int16)

            corr, on = 0.0, False
            if kmark is not None:
                on = kmark.step(frame[
                    prof.rect("combo")[1]:prof.rect("combo")[1] + prof.rect("combo")[3],
                    prof.rect("combo")[0]:prof.rect("combo")[0] + prof.rect("combo")[2]])
                corr = kmark.corr

            changed = (prev is None or
                       float(np.abs(small - prev).mean()) > args.diff or on)
            fname = ""
            if changed:
                n_saved += 1
                idx = f"{n_saved:04d}"
                Image.fromarray(frame.astype(np.uint8)).resize(
                    (1280, 720), Image.LANCZOS).save(
                    os.path.join(out, f"full_{idx}.jpg"), quality=85)
                # ★ 提示带：**原生像素**，无损 —— 框模板必须用这个
                band = np.asarray(grab(bsx, bsy, bw, bh))
                if band.ndim == 3 and band.shape[2] == 4:
                    band = band[:, :, :3]
                Image.fromarray(band.astype(np.uint8)).save(
                    os.path.join(out, f"{args.tag}_{idx}.png"))
                np.save(os.path.join(out, f"{args.tag}_{idx}.npy"), band.astype(np.uint8))
                part = np.asarray(grab(prof.x+PARTY[0], prof.y+PARTY[1], PARTY[2], PARTY[3]))
                if part.ndim == 3 and part.shape[2] == 4:
                    part = part[:, :, :3]
                Image.fromarray(part.astype(np.uint8)).save(
                    os.path.join(out, f"party_{idx}.png"))
                np.save(os.path.join(out, f"party_{idx}.npy"), part.astype(np.uint8))
                fname = f"{args.tag}_{idx}.png"
                prev = small
            csv.write(f"{time.time()-t0:.2f},{corr:+.3f},{int(on)},{int(changed)},{fname}\n")
            csv.flush()
            if n_saved and n_saved % 20 == 0 and changed:
                print(f"  {time.time()-t0:6.0f}s  存了 {n_saved} 帧  "
                      f"当前相关 {corr:+.3f}{'  ★★ 提示亮着' if on else ''}")
            time.sleep(max(0.0, 1.0 / args.hz - 0.05))
    except KeyboardInterrupt:
        print("\n  手动停了")
    finally:
        csv.close()

    print(f"\n录完了：{out}")
    print(f"  {n_saved} 帧 · {time.time()-t0:.0f} 秒")
    # 把相关值最高的几帧挑出来 —— 那多半就是提示亮着的帧
    try:
        rows = [l.strip().split(",") for l in
                open(os.path.join(out, "log.csv"), encoding="utf-8").read().splitlines()[1:]]
        rows = [r for r in rows if r and r[-1]]
        rows.sort(key=lambda r: -float(r[1]))
        print("  相关值最高的 5 帧（多半就是提示亮着的）：")
        for r in rows[:5]:
            print(f"    t={r[0]:>6s}s  相关 {r[1]}  -> {r[-1]}")
    except Exception as e:
        print(f"  （挑帧失败：{e}）")


if __name__ == "__main__":
    main()
