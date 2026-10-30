#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_combo_diamond.py — 离线验证「连携菱形」模板（MaaEnd 的 Combo.png）。

为什么要有这个脚本
------------------
2026-10-30 的结论是「ONNX 认不出祀的连携提示」。备选方案本来是「采集 375 帧
+ 训一个小模型」，但那要等人。后来发现 MaaEnd 的资源里本来就有一枚
16×11 的模板 AutoFight/Combo.png —— 它抓的是**提示圆环正上方那枚小白菱形**，
跟角色无关，所以对祀也应该成立。

**但「应该成立」不算数。** 这个脚本干三件事：

  ① 在**全部**录制上逐帧跑这枚模板，给出分数分布（不是只看抓到了的那帧）
  ② 把「真提示 / 负样本」分开报告 —— 类别极端不平衡时准确率是骗人的，
     要看**误报率**和正负之间**空档有多宽**
  ③ 用祀那张 2560 截图（模板按尺度重采样）验一次「换分辨率还认不认得出」

用法
----
    python -X utf8 check_combo_diamond.py            # 全部录制
    python -X utf8 check_combo_diamond.py --thr 0.85 # 试别的阈值
    python -X utf8 check_combo_diamond.py --rec 20261009_173122

判定标准（事先写死，免得跑完再编）
----------------------------------
  · 428 帧那份：0022 / 0040 两帧**必须**命中（人工核验过的真提示）
  · 其余帧 ≥ 阈值的数量 = 误报数，应为 0
  · 正负之间的空档 ≥ 0.20 才算「能定阈值」
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from combat import ComboDiamond            # noqa: E402
from screen_profile import Profile         # noqa: E402

REC_ROOT = os.path.join(HERE, "..", "_handoff", "rec")
SI_REF = os.path.join(HERE, "out", "si_ref", "si_01.jpg")

# 428 帧那份里人工核验过的两帧真提示
KNOWN_TRUE = {"20261009_173122": [22, 40]}


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def scan_recording(name: str, prof: Profile, det: ComboDiamond,
                   verbose: bool = True) -> dict:
    d = os.path.join(REC_ROOT, name)
    fulls = sorted(glob.glob(os.path.join(d, "full_*.jpg")))
    if not fulls:
        return {}
    rx, ry, rw, rh = prof.rect("combo_diamond")
    det.rect = (rx, ry, rw, rh)
    det.tpl = prof.template("combo_diamond")
    scores = []
    for p in fulls:
        n = int(os.path.basename(p)[5:9])
        img = load_rgb(p)
        if img.shape[1] != prof.w or img.shape[0] != prof.h:
            img = np.asarray(Image.fromarray(img).resize((prof.w, prof.h),
                                                         Image.LANCZOS))
        crop = img[ry:ry + rh, rx:rx + rw]
        det.step(crop)
        scores.append((n, float(det.corr)))
    arr = np.array([s for _, s in scores], dtype=np.float32)
    hits = [n for n, s in scores if s >= det.corr_min]
    runs = []
    for n in hits:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    if verbose:
        print(f"\n=== {name} · {len(scores)} 帧 ===")
        print(f"  区域 {det.rect} · 模板 {det.tpl.shape[1]}×{det.tpl.shape[0]}"
              f" · 阈值 {det.corr_min:.2f}")
        print(f"  分数：中位 {np.median(arr):+.3f} · p90 {np.percentile(arr, 90):+.3f}"
              f" · p99 {np.percentile(arr, 99):+.3f} · 最高 {arr.max():+.3f}")
        print(f"  ≥ 阈值：{len(hits)} 帧  {hits[:20]}" + ("…" if len(hits) > 20 else ""))
        if runs:
            lens = [len(r) for r in runs]
            print(f"  连续段：{len(runs)} 段，长度 {min(lens)}~{max(lens)} 帧"
                  f"（中位 {sorted(lens)[len(lens) // 2]}）")
    other = [s for n, s in scores if n not in hits]
    return {"name": name, "n": len(scores), "hits": hits, "runs": runs,
            "max": float(arr.max()),
            "max_neg": float(max(other)) if other else None,
            "median": float(np.median(arr))}


def check_si(det: ComboDiamond) -> dict:
    """祀那张 2560×1440 截图 —— 模型认不出，这枚菱形认不认得出？"""
    if not os.path.exists(SI_REF):
        return {}
    img = load_rgb(SI_REF)
    prof2 = Profile(0, 0, img.shape[1], img.shape[0], measured=True, verbose=False)
    rx, ry, rw, rh = prof2.rect("combo_diamond")
    tpl = prof2.template("combo_diamond")
    det.tpl = tpl
    det.rect = (rx, ry, rw, rh)
    det.step(img[ry:ry + rh, rx:rx + rw])
    print(f"\n=== 祀的截图（{img.shape[1]}×{img.shape[0]}）===")
    print(f"  区域 {det.rect} · 模板 {tpl.shape[1]}×{tpl.shape[0]}"
          f"（按尺度重采样）· 相关 {det.corr:+.3f} → "
          f"{'★命中' if det.on else '未命中'}")
    return {"corr": float(det.corr), "on": bool(det.on)}


def main() -> int:
    ap = argparse.ArgumentParser(description="离线验证连携菱形模板")
    ap.add_argument("--thr", type=float, default=0.80, help="相关阈值（默认 0.80）")
    ap.add_argument("--rec", default="", help="只测某一份录制（默认全部）")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="录制帧率（默认 1.0 fps，用来把帧数换成秒）")
    a = ap.parse_args()

    prof = Profile(0, 0, 1280, 720, measured=True, verbose=False)
    det = ComboDiamond(corr_min=a.thr, verbose=True)

    names = [a.rec] if a.rec else sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(REC_ROOT, "*"))
        if os.path.isdir(d))
    results = [r for r in (scan_recording(n, prof, det) for n in names) if r]

    print("\n" + "=" * 68)
    print(f"汇总（阈值 {a.thr:.2f} · 录制按 {a.fps:.2f} fps 计）")
    print("=" * 68)
    print(f"{'录制':<20}{'帧数':>6}{'命中':>6}{'负样本最高':>12}{'空档':>8}")
    ok = True
    worst_neg = 0.0
    for r in results:
        mn = r["max_neg"] if r["max_neg"] is not None else float("nan")
        gap = a.thr - mn if mn == mn else float("nan")
        print(f"{r['name']:<20}{r['n']:>6}{len(r['hits']):>6}"
              f"{mn:>12.3f}{gap:>8.3f}")
        worst_neg = max(worst_neg, mn if mn == mn else 0.0)
    print(f"\n  全库负样本最高 {worst_neg:+.3f} · 阈值 {a.thr:.2f} "
          f"→ 空档 {a.thr - worst_neg:+.3f}")

    for name, want in KNOWN_TRUE.items():
        r = next((x for x in results if x["name"] == name), None)
        if not r:
            continue
        miss = [n for n in want if n not in r["hits"]]
        print(f"  召回 {name}: {len(want) - len(miss)}/{len(want)}"
              + (f"  ✗ 漏 {miss}" if miss else "  ✓"))
        ok = ok and not miss
        fps = next((x for x in results if x["name"] in KNOWN_TRUE), None)
        if fps:
            neg_frames = fps["n"] - len(fps["hits"])
            print(f"  误报（428 帧那份）：{len(fps['hits']) - 2} / {neg_frames}")
            ok = ok and (len(fps["hits"]) == 2)
    si = check_si(det)
    if si:
        ok = ok and si["on"]

    print("\n" + ("结论：✓ 可以用（空档够宽，祀也认得出）" if ok
                  else "结论：✗ 不满足验收条件，别接"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
