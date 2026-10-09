#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mission.py — 副本完成 = **任务条的「击败所有敌人」消失 N 秒**。

三次改判据的经过
================

**第一版**：模板匹配「挑战成功」横幅。
  实测 416 帧零误报 —— 但**用不了**：横幅在内容 y 250~430，
  而叠加面板铺在 y 0~300，正好压住它 ✗（用户指出）

**第二版**：数左边任务条的**金色像素**。
  428 帧录制上零误报 —— 但**实机误判**（副本没结束就清空饥饿）✗
  原因：颜色判据对**区域**极其敏感，区域里任何金色东西都能凑数。

**第三版（本文件）**：**看文字在不在**。
  用户的原话：「看数字，数字消失 4 秒后再结算」。

  ★ 这个设计正好解决了前两版的所有问题：

    ① 用的是**文字模板**，不是颜色 —— 背景怎么变都没关系
       （归一化相关是**相对**的，背景亮暗不影响）
    ② 判据是**消失**，不是出现 —— 文字消失是个干净的二元事件
    ③ **消失持续 N 秒**才算 —— 淡入淡出、切镜头、短暂遮挡全部滤掉

  ★ 为什么"消失 N 秒"是关键：
    实测「击败所有敌人」会有**短暂的淡入淡出**（相关值从 0.75 掉到 0.24
    再回 0.63）。只看下降沿会被这些抖动误触发。
    加上"必须连续消失 N 秒"，抖动（不到 1 秒）自然被滤掉，
    而真正的结束（文字再也不回来）稳稳超过 4 秒。

  实测分离度（428 帧真实录制）：
      真显示帧  +0.627 ~ +1.000
      淡出过渡  +0.235 ~ +0.246
      真消失帧  +0.142 ~ +0.294
    → 阈值 0.50 干净分离
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


class MissionWatcher:
    """盯着任务条上的文字，它**连续消失 N 秒**就判定副本结束。

    ★ 三个必要条件（缺一个都会误判）：

    ① **必须先见过它**（armed）。
       否则果蝇在副本外启动时，任务条本来就不在 —— 一开机就"消失 4 秒"，
       立刻白送 50 次奖励。

    ② **一次一局**（done）。
       同一局重复领奖没有生物学意义。

    ③ **消失要连续**。
       文字淡出/切镜头造成的短暂掉线会重置计时，不算数。
    """

    def __init__(self, rect=None, tpl=None, corr_min: float = 0.50,
                 gone_sec: float = 4.0, verbose: bool = True):
        self.rect = tuple(int(v) for v in rect) if rect else (40, 205, 220, 40)
        self.corr_min = float(corr_min)
        self.gone_sec = float(gone_sec)
        self.tpl = None
        if tpl is not None:
            t = np.asarray(tpl, dtype=np.float32)
            self.tpl = t.mean(axis=2) if t.ndim == 3 else t
        self.corr = 0.0
        self.present = False
        self.absent_for = 0.0
        self.armed = False      # 见过它一次没有
        self.done = False       # 这一局领过奖没有
        self.n = 0
        if verbose:
            if self.tpl is None:
                print("任务完成检测：**没有模板** → 不会触发（缺 kill_text.npy）")
            else:
                print(f"任务完成检测：任务条文字 {self.tpl.shape[1]}×"
                      f"{self.tpl.shape[0]} · 相关 > {corr_min} · "
                      f"**消失 {gone_sec:.1f} 秒**后结算"
                      f"（实测显示 +0.63~+1.00、消失 +0.14~+0.29）")

    # ------------------------------------------------------------------
    def corr_of(self, img) -> float:
        if self.tpl is None or img is None:
            return 0.0
        g = np.asarray(img, dtype=np.float32)
        if g.ndim == 3:
            g = g[:, :, :3].mean(axis=2)
        th, tw = self.tpl.shape
        if g.shape[0] < th or g.shape[1] < tw:
            return 0.0
        t = self.tpl - self.tpl.mean()
        tn = float(np.sqrt((t * t).sum())) + 1e-6
        ones = np.ones_like(t)
        num = fftconvolve(g, t[::-1, ::-1], mode="valid")
        s1 = fftconvolve(g, ones[::-1, ::-1], mode="valid")
        s2 = fftconvolve(g * g, ones[::-1, ::-1], mode="valid")
        n = t.size
        var = np.maximum(s2 - s1 * s1 / n, 0.0)
        den = np.sqrt(var) * tn
        # ★ 防零方差（平坦区域会让相关值爆炸）
        sc = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)
        return float(sc.max())

    # ------------------------------------------------------------------
    def step(self, img, dt: float) -> bool:
        """喂一帧，返回**这一拍是否应该结算**（True 只会出现一次）。"""
        if self.tpl is None or self.done:
            return False
        self.corr = self.corr_of(img)
        self.present = self.corr >= self.corr_min
        self.n += 1

        if self.present:
            self.armed = True           # 见过一次才算"这局开始过"
            self.absent_for = 0.0       # ★ 回来了就重置计时
            return False

        # 没见过就谈不上"消失"
        if not self.armed:
            return False

        self.absent_for += float(dt)
        if self.absent_for >= self.gone_sec:
            self.done = True
            return True
        return False

    def fire(self):
        self.done = True


def reward_burst(mb, n_times: int = 50, magnitude: float = 1.0,
                 verbose: bool = True) -> int:
    """给蘑菇体连发 n 次奖赏。

    ★ 为什么是"连发"而不是"发一个 50 倍的大数"：
      `deliver()` 是**多巴胺脉冲**，它后面还跟着 `step()` 里的
      资格迹（eligibility trace）和权重更新。发一个 50 的大数会把权重
      一次性推爆（clip 到边界），学到的不是"这样打是对的"，
      而是"所有同时活跃的突触都被顶到上限"。
      连发 50 次是**渐进**的：每次都在当时的资格迹上留一笔，
      更接近真实的爆发式多巴胺发放（真实神经元就是 burst firing）。
    """
    n = 0
    for _ in range(int(n_times)):
        try:
            mb.deliver(+float(magnitude))
            n += 1
        except Exception:
            break
    if verbose:
        print(f"  ★★★ 任务完成 → 连发 {n} 次奖赏（每次 {magnitude}）", flush=True)
    return n


if __name__ == "__main__":
    import sys
    state = [True]

    def chk(c, m):
        print(f"  {'OK  ' if c else 'FAIL'} {m}")
        state[0] = state[0] and c

    rng = np.random.default_rng(0)
    tpl = rng.integers(120, 240, (20, 120)).astype(np.uint8)

    def frame(hi):
        # ⚠ 必须用**同一个** tpl 铺进去。第一版这里又调了一次 rng.integers，
        #   于是显示帧里是另一串随机噪声，相关值根本上不去，自检全红 ✗
        a = np.full((40, 220, 3), 30, dtype=np.uint8)
        if hi:
            a[10:30, 20:140] = tpl[:, :, None]
        return a

    # ① 出现过 -> 消失 4 秒 -> 结算
    m = MissionWatcher(rect=(0, 0, 220, 40), tpl=tpl, gone_sec=4.0, verbose=False)
    chk(not m.step(frame(True), 0.4), "文字在 → 不结算")
    chk(m.armed, "见过一次 -> armed")
    # ⚠ 要 11 拍才够 4 秒（9 拍只有 3.6 秒）—— 第一版就是这里写少了，
    #   看起来像代码不工作，其实是测试没喂够时间 ✗
    for _ in range(12):
        if m.step(frame(False), 0.4):
            break
    chk(m.done, "连续消失 4 秒 -> 结算")

    # ② 中途回来 -> 计时重置
    m2 = MissionWatcher(rect=(0, 0, 220, 40), tpl=tpl, gone_sec=4.0, verbose=False)
    m2.step(frame(True), 0.4)
    for _ in range(5):
        m2.step(frame(False), 0.4)          # 消失 2 秒
    m2.step(frame(True), 0.4)               # 回来了
    chk(m2.absent_for == 0.0 and not m2.done, "文字回来 -> 计时重置，不结算")
    for _ in range(5):
        m2.step(frame(False), 0.4)          # 再消失 2 秒
    chk(not m2.done, "只消失 2 秒 -> 还不结算")

    # ③ 从没见过 -> 永远不结算（副本外启动的情形）
    m3 = MissionWatcher(rect=(0, 0, 220, 40), tpl=tpl, gone_sec=4.0, verbose=False)
    for _ in range(60):
        m3.step(frame(False), 0.4)
    chk(not m3.done and not m3.armed, "从没见过文字 -> 永不结算（副本外安全）")

    # ④ 一次一局
    m4 = MissionWatcher(rect=(0, 0, 220, 40), tpl=tpl, gone_sec=1.0, verbose=False)
    m4.step(frame(True), 0.4)
    fired = [m4.step(frame(False), 0.4) for _ in range(20)]
    chk(sum(fired) == 1, f"只结算一次（实际 {sum(fired)} 次）")

    # ⑤ 没有模板 -> 静默关闭
    m5 = MissionWatcher(rect=(0, 0, 220, 40), tpl=None, verbose=False)
    chk(m5.tpl is None and not m5.step(frame(True), 0.4), "没模板 -> 静默关闭")

    # ⑥ 可自定义 1~10 秒
    for sec in (1.0, 4.0, 10.0):
        m6 = MissionWatcher(rect=(0, 0, 220, 40), tpl=tpl, gone_sec=sec,
                            verbose=False)
        m6.step(frame(True), 0.4)
        n = 0
        while n < 200 and not m6.step(frame(False), 0.4):
            n += 1
        chk(abs((n + 1) * 0.4 - sec) <= 0.8, f"gone_sec={sec} 秒 -> 约 {(n+1)*0.4:.1f} 秒后结算")

    print()
    print("  全部通过 ✓" if state[0] else "  有失败 ✗")
    sys.exit(0 if state[0] else 1)
