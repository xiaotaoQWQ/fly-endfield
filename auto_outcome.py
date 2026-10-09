#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auto_outcome.py — 全自动识别「打死」和「挨打」，替掉手动 O/P 键。

用户给的两条线索：
  · **打死** → 敌人倒地/消失
  · **挨打** → 底部/左下角的血条变短

三条信号，都是**实测标定**出来的（ef_combat_frames.py 抓战斗帧 → ef_hp_probe.py 量）：

  ① 主血条（画面底部中间）
     内容坐标 (1060, 1324) 441×12
     颜色是亮青蓝 RGB(22, ~200, 253)，饱和度极高：b>170 且 b−r>80 且 g>140
     两端有**白色端帽**，所以满格宽度固定 = 441px，填充率按
     「最右侧青像素的 x − 左端」/ 441 算
     ★ 用**相对下降**判挨打，不用绝对值 —— 换角色/换血量上限都不用重标

  ② 受击红晕（画面下部的左右边缘）
     内容坐标 左 (0,1140) 12×130 · 右 (2548,1140) 12×130
     实测：受击那一下 R−G 冲到 +46，平时是 −5 ~ −17

  ③ 红环下降沿（左上角）
     敌人倒地/消失必然让战斗结束 → 红环由亮变灭。
     前提：这一段确实出过手、且战线够长（排除一闪而过）

为什么"打死"不直接认敌人倒地：
  敌人在 3D 世界里乱跑，没有干净的模板可匹配；而它一死战斗就结束，
  红环的下降沿是**等价的、且好测得多**的信号。
"""
from __future__ import annotations

import numpy as np

# ---- 实测标定（ef_hp_probe.py 量的，改分辨率要重标）----
HP_RECT = (1060, 1324, 441, 12)        # 内容坐标 x,y,w,h
VIG_L = (0, 1140, 12, 130)
VIG_R = (2548, 1140, 12, 130)


def is_cyan(a: np.ndarray) -> np.ndarray:
    """亮青蓝 = 血条填充色。草地是暗青绿，饱和度低得多，分得开。"""
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (b > 170) & ((b - r) > 80) & (g > 140)


class HPBar:
    """读血条填充比例 0..1。对**任意实数**都安全（不除零、不翻号）。"""

    def __init__(self, verbose: bool = True):
        self.fill = float("nan")
        self.n = 0
        self.valid = False
        if verbose:
            print(f"血条读取：区域 {HP_RECT}（实测标定）")

    def step(self, rgb: np.ndarray) -> float:
        a = np.asarray(rgb)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        a = a.astype(np.int16)
        if a.shape[0] < 2 or a.shape[1] < 2:
            return float("nan")
        m = is_cyan(a)
        # 每列过半数像素是青色 → 这一列在条内
        colok = m.mean(axis=0) >= 0.5
        n = len(colok)
        if n == 0:
            self.fill = 0.0
            return self.fill
        idx = np.flatnonzero(colok)
        self.valid = len(idx) >= 3      # 太少说明血条不在（非战斗）
        self.fill = float((idx.max() + 1) / n) if self.valid else float("nan")
        self.n += 1
        return self.fill


class RedVignette:
    """受击红晕：左右边缘的 R−G。实测受击时冲到 +46，平时 −5~−17。"""

    def __init__(self, thresh: float = 30.0, verbose: bool = True):
        self.thresh = float(thresh)
        self.val = 0.0
        self.on = False
        if verbose:
            print(f"受击红晕：左右边缘 R−G > {thresh} 判为挨打")

    @staticmethod
    def _redness(a):
        return float((a[:, :, 0].astype(np.int16)
                      - a[:, :, 1].astype(np.int16)).mean())

    def step(self, left: np.ndarray, right: np.ndarray) -> bool:
        self.val = max(self._redness(np.asarray(left)),
                       self._redness(np.asarray(right)))
        self.on = self.val > self.thresh
        return self.on


class AutoOutcome:
    """全自动产出奖赏/惩罚脉冲。

    打死（奖赏）：红环由亮变灭 且 这一段出过手 且 战线 ≥ min_fight 秒
    挨打（惩罚）：血条相对下降 ≥ drop_frac，**或** 受击红晕亮起
    两者都带去抖/冷却，免得一次事件被拆成好几拍重复扣。
    """

    def __init__(self, min_fight: float = 0.8, drop_frac: float = 0.010,
                 punish_cool: float = 0.8, reward: float = 1.0,
                 punish: float = 1.0, verbose: bool = True,
                 hp_rect=None, vig_l=None, vig_r=None):
        # ★ 矩形可以由 screen_profile 传进来（换分辨率时自动换算）。
        #   不传就用模块里那套 2560×1440 的实测值 —— 保证老命令还能跑。
        global HP_RECT, VIG_L, VIG_R
        if hp_rect is not None:
            HP_RECT = tuple(int(v) for v in hp_rect)
        if vig_l is not None:
            VIG_L = tuple(int(v) for v in vig_l)
        if vig_r is not None:
            VIG_R = tuple(int(v) for v in vig_r)
        self.hp = HPBar(verbose=verbose)
        self.vig = RedVignette(verbose=verbose)
        self.min_fight = float(min_fight)
        self.drop_frac = float(drop_frac)
        self.punish_cool = float(punish_cool)
        self.reward_amt = float(reward)
        self.punish_amt = float(punish)
        self.n_kill = 0
        self.n_hurt = 0
        self.last_fight_len = 0.0
        self.last_hp = None
        self.last_punish = -1e9
        self._in = False
        self._t_in = 0.0
        self._attacked = False

    def step(self, t: float, in_combat: bool, attacked: bool,
             hp_img=None, vig_left=None, vig_right=None) -> dict:
        rew = 0.0
        pun = 0.0
        why = []

        # ---------- 挨打 ----------
        hurt = False
        if hp_img is not None:
            f = self.hp.step(hp_img)
            if not np.isnan(f):
                if self.last_hp is not None:
                    drop = self.last_hp - f
                    if drop >= self.drop_frac:
                        hurt = True
                        why.append(f"掉血 {drop:.3f}")
                self.last_hp = f
        if not in_combat:
            self.last_hp = None       # 血条会消失，别把"消失"当掉血
        if vig_left is not None and vig_right is not None:
            if self.vig.step(vig_left, vig_right):
                hurt = True
                why.append(f"红晕 {self.vig.val:.0f}")
        if hurt and t - self.last_punish >= self.punish_cool:
            pun = self.punish_amt
            self.last_punish = t
            self.n_hurt += 1
        elif hurt:
            why.append("冷却中")

        # ---------- 打死 ----------
        if in_combat and not self._in:
            self._in = True
            self._t_in = t
            self._attacked = False
        elif in_combat:
            if attacked:
                self._attacked = True
        elif self._in:
            self._in = False
            self.last_fight_len = t - self._t_in
            if self._attacked and self.last_fight_len >= self.min_fight:
                rew = self.reward_amt
                self.n_kill += 1
                why.append(f"战斗 {self.last_fight_len:.1f}s 后结束")

        return dict(reward=rew, punish=pun, why=" · ".join(why),
                    hp=(self.hp.fill if self.hp.valid else float("nan")),
                    vig=self.vig.val)
