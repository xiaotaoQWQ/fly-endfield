#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""odor_map.py — 交接文档 §七/§八：把游戏状态编成"气味"。

§七 原文要点：
  · 游戏里没有嗅觉。**那一路信号不是读出来的，是你造出来再塞给果蝇的。**
    生物学只规定"信号进来之后怎么处理"，信号从哪来是工程决定。
  · 注到**嗅觉受体神经元 ORN**（或等价地注到触角叶肾小球输入）。
    —— 实测这份数据里 ORN 是按**肾小球**命名的（ORN_DA1 / ORN_VA1d …），
       不是文档猜的 Or 基因名。肾小球天然就是组合编码的通道，反而更好用。
  · 果蝇嗅觉是**组合编码**：约五十类受体，一种气味激活的是一个组合。
    所以**不需要给每种敌人配一个专属通道** —— 四个通道就是十六种。
  · 三个实现细节：**持续注入不脉冲** · **要有基线活动**（否则网络静默，
    已实测确认）· 代码就三步（筛 ID → 建映射 → 每 tick 写 drive）。

§八 分三阶段：一种敌人 → 两种 → 多种。顺序别跳。
  "分化的那一刻才是真正有看头的现象：它是'学会了一种'，还是'学出了一对'。"
"""
from __future__ import annotations

import numpy as np

# 通道 → 肾小球。挑的都是实测里 ORN 数量最多的那几个，保证够强。
# ⚠ 这张表是**工程决定**，随便定都行，但要知道自己在干什么（§八 原话）。
CHANNELS = ("ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3",
            "ORN_VL2a", "ORN_VM5d", "ORN_VA2", "ORN_DL1")


class OdorEncoder:
    """把"敌人类型"编成气味向量，写进 sensory_drive 的 ORN 位置。

    阶段一：一种敌人 → 一路布尔
    阶段二：两种敌人 → 两路
    阶段三：多种   → 组合编码（每个类型占一个 bit 模式）
    """

    def __init__(self, glom: dict, pop_orn: np.ndarray, base: float = 0.02,
                 amp: float = 0.08, verbose: bool = True):
        self.glom = glom
        self.orn = np.asarray(pop_orn, np.int64)
        self.base = float(base)          # ★ 基线活动（§七：零基线会让网络静默）
        self.amp = float(amp)
        self.channels = [c for c in CHANNELS if c in glom]
        if verbose:
            print(f"气味编码：{len(self.channels)} 个通道 "
                  f"（{', '.join(self.channels)}）")
            print(f"  基线 {base} · 气味幅度 {amp} · "
                  f"理论上可表达 {2**len(self.channels)} 种组合")
        self.n = 0
        self.last_vec = None

    # ------------------------------------------------------------------
    def vector(self, active: list[str]) -> np.ndarray:
        """把激活的通道名列表编成 ORN 驱动。"""
        d = np.zeros(211_577, dtype=np.float32)
        d[self.orn] = self.base              # 全体 ORN 都有自发活动
        for c in active:
            if c in self.glom:
                d[self.glom[c]] = self.base + self.amp
        return d

    def stage1(self, enemy_present: bool) -> np.ndarray:
        """阶段一：一种敌人，一路布尔。"""
        return self.vector([self.channels[0]] if enemy_present else [])

    def stage2(self, enemy: int) -> np.ndarray:
        """阶段二：两种敌人，各占一路。"""
        return self.vector(self.channels[:1] if enemy == 0
                           else self.channels[1:2])

    def stage3(self, enemy: int) -> np.ndarray:
        """阶段三：多种敌人 → 组合编码（二进制 bit 模式）。"""
        active = [c for i, c in enumerate(self.channels)
                  if (enemy >> i) & 1]
        return self.vector(active)

    def write(self, drive: np.ndarray, active: list[str], gain: float = 1.0):
        """§七 的第三步：每个 tick 把当前向量写进 sensory_drive。

        gain 由饥饿值调制（§五 ①：越饿对食物线索越敏感）。
        """
        drive[self.orn] = self.base
        for c in active:
            if c in self.glom:
                drive[self.glom[c]] = self.base + self.amp * gain
        self.n += 1
        self.last_vec = active
        return drive
