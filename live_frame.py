#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live_frame.py — 一帧完整画面，所有检测器都从它切片。

为什么要有这个东西
==================

改之前，每一拍要抓 **7 次**屏幕：

    红环 · 撤离面板 · 连携技 · 血条 · 受左红晕 · 受右红晕 · 小地图

两次抓取之间隔十几毫秒，而游戏是 60fps 在跑的 —— 也就是说
**"血条还剩 30%"和"红环亮着"可能不是同一时刻的状态**。
战斗里 100ms 足够让血量掉一截、让连携技提示出现或消失。

实测耗时（2560×1440）：

    整屏抓一次        87 ms
    单个小区域        14~16 ms   ← 无论区域多小都要 15ms，固定开销占大头
    7 个区域合计      ≈105 ms

所以整屏抓一次**不但更一致，还略快**。而且顺带解决两件事：

    ① **位置可以实时搜**，不用死死依赖标定坐标 —— 见 find()
    ② 所有状态来自同一帧，不会出现"血条说 30% 但红环说没挨打"这种自相矛盾

★ 果蝇的眼睛不走这条路
    眼睛线程要 30 Hz，整屏抓一次要 87ms（最多 11 Hz）—— 不够。
    所以眼睛仍然**只抓自己那条带**（小得多），单独一条快路径。
    这是刻意的取舍：眼睛要快，HUD 要准。
"""
from __future__ import annotations

import time

import numpy as np
from scipy.signal import fftconvolve


class LiveFrame:
    """一帧完整画面 + 从它切片/搜索的工具。

    用法::

        lf = LiveFrame(prof, grab)
        lf.update()                       # 每拍抓一次
        hp  = lf.region("hp")             # 按标定名切片
        sub = lf.at(1400, 590, 200, 100)  # 按内容坐标切片
        c, x, y = lf.find(tpl, lf.region("combo"))   # 在切片里找模板
    """

    def __init__(self, prof, grab_fn, verbose: bool = True):
        self.prof = prof
        self.grab = grab_fn
        self.img = None
        self.t = 0.0
        self.n = 0
        self.grab_ms = 0.0
        self._gray = None
        if verbose:
            print(f"整帧：{prof.w}×{prof.h}（每拍抓一次，所有 HUD 从它切片）")

    # ------------------------------------------------------------------
    def update(self) -> bool:
        """抓一帧。成功返回 True。"""
        t0 = time.perf_counter()
        try:
            a = np.asarray(self.grab(self.prof.x, self.prof.y,
                                     self.prof.w, self.prof.h))
        except Exception:
            return False
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        if a.shape[0] != self.prof.h or a.shape[1] != self.prof.w:
            return False                       # 尺寸不符，这一拍作废
        self.img = a
        self._gray = None
        self.t = time.time()
        self.n += 1
        self.grab_ms = (time.perf_counter() - t0) * 1000
        return True

    # ------------------------------------------------------------------
    def at(self, x: int, y: int, w: int, h: int):
        """按**内容坐标**切片。越界会裁剪，不会抛异常。

        ★ 一定要裁剪：标定值在别的分辨率下换算出界是常事，
          抛异常会让整只果蝇停摆，而少几个像素无所谓。
        """
        if self.img is None:
            return None
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1 = min(self.img.shape[1], int(x) + int(w))
        y1 = min(self.img.shape[0], int(y) + int(h))
        if x1 <= x0 or y1 <= y0:
            return None
        return self.img[y0:y1, x0:x1]

    def region(self, name: str):
        """按标定名切片（marker / evac / combo / hp / vig_l / vig_r …）。"""
        try:
            return self.at(*self.prof.rect(name))
        except Exception:
            return None

    def gray(self):
        """整帧灰度（缓存一次，多个检测器共用）。"""
        if self._gray is None and self.img is not None:
            self._gray = self.img.astype(np.float32).mean(axis=2)
        return self._gray

    # ------------------------------------------------------------------
    def find(self, tpl, sub, corr_only: bool = False):
        """在 `sub` 里找 `tpl`，返回 (相关值, 最佳位置的局部坐标)。

        ★ 这是"**实时计算位置**"的核心：不依赖标定坐标准不准，
          而是在一小块区域里**每次都重新找**。UI 元素轻微移动、
          不同分辨率下位置有偏差，都能自己纠正。

        和 combat.ComboMarker.step() 用的是同一套 FFT 归一化相关
        （逐点搜太慢，2.5Hz 下来不及）。零方差要防住 —— 平坦区域
        （纯色/黑屏/对话框）会让分母趋零、相关值爆炸。
        """
        if sub is None or tpl is None:
            return 0.0, 0, 0
        t = np.asarray(tpl, dtype=np.float32)
        if t.ndim == 3:
            t = t.mean(axis=2)
        g = np.asarray(sub, dtype=np.float32)
        if g.ndim == 3:
            g = g.mean(axis=2)
        th, tw = t.shape
        if g.shape[0] < th or g.shape[1] < tw:
            return 0.0, 0, 0
        tc = t - t.mean()
        tn = float(np.sqrt((tc * tc).sum())) + 1e-6
        ones = np.ones_like(tc)
        num = fftconvolve(g, tc[::-1, ::-1], mode="valid")
        s1 = fftconvolve(g, ones[::-1, ::-1], mode="valid")
        s2 = fftconvolve(g * g, ones[::-1, ::-1], mode="valid")
        n = tc.size
        var = np.maximum(s2 - s1 * s1 / n, 0.0)
        den = np.sqrt(var) * tn
        sc = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)
        if corr_only:
            return float(sc.max()), 0, 0
        iy, ix = np.unravel_index(int(sc.argmax()), sc.shape)
        return float(sc.max()), int(ix), int(iy)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        if self.img is None:
            return "整帧：还没抓过"
        return (f"整帧 {self.img.shape[1]}×{self.img.shape[0]} · "
                f"第 {self.n} 帧 · 抓取 {self.grab_ms:.0f}ms")


def _selftest():
    """自检：切片、灰度缓存、find 的位置和零方差防护。"""
    import sys
    ok = True

    def chk(c, m):
        nonlocal ok
        print(f"  {'OK  ' if c else 'FAIL'} {m}")
        ok = ok and c

    class _P:
        w, h, x, y = 400, 300, 0, 0

        def rect(self, name):
            return {"hp": (10, 20, 100, 8), "combo": (200, 150, 120, 80)}[name]

    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[:, :] = 40
    tpl = np.random.default_rng(0).integers(60, 200, (16, 20, 3)).astype(np.uint8)
    frame[160:176, 240:260] = tpl                     # 埋一个模板进去

    lf = LiveFrame(_P(), lambda *a: frame, verbose=False)
    chk(lf.update(), "update() 成功")
    chk(lf.at(0, 0, 50, 50).shape == (50, 50, 3), "at() 切片尺寸对")
    chk(lf.region("hp").shape == (8, 100, 3), "region('hp') 切片对")

    # 越界要裁剪而不是抛异常
    try:
        a = lf.at(380, 280, 100, 100)
        chk(a.shape == (20, 20, 3), "越界自动裁剪（不抛异常）")
    except Exception as e:
        chk(False, f"越界裁剪：{e}")

    # find 要能找到埋进去的模板，且位置对
    c, x, y = lf.find(tpl, lf.at(200, 150, 120, 80))
    chk(c > 0.99, f"find 相关值 {c:.3f} > 0.99")
    chk((x, y) == (40, 10), f"find 位置 ({x},{y}) 应为 (40,10)")

    # 零方差防护：全平的区域不能给出高分
    flat = np.full((80, 120, 3), 128, dtype=np.uint8)
    c2, _, _ = lf.find(tpl, flat)
    chk(c2 < 0.5, f"平坦区域相关值 {c2:.3f} < 0.5（零方差防住了）")

    # 灰度缓存只算一次
    g1 = lf.gray()
    chk(g1 is lf.gray(), "灰度缓存复用同一对象")

    print()
    print("  全部通过 ✓" if ok else "  有失败 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
