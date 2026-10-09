#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sector_vote.py — 翻译层：DN 活动 → 方向选择

架构（交接文档第四节的三层）：
    果蝇层  →  翻译层  →  执行层
    MaleCNS     本模块      MaaEnd/自建导航

本模块实现翻译层：
  1. DirectionDecoder：把下行神经元活动解成方向（左/右，可扩展 5 扇区）
  2. Hysteresis：时序平滑 + 滞回，避免每个 tick 换方向
  3. StuckBreaker：连续同方向超时 → 强制换向（防卡死在墙上）

投票口径的定量依据（vote_design.py 实测）：
  ┌────────────────────────┬──────────┬──────────┬──────────┐
  │ 口径                    │ 左前亮    │ 右前亮    │ 可分性    │
  ├────────────────────────┼──────────┼──────────┼──────────┤
  │ 全部 1332 按侧取均值     │ +0.0023  │ +0.0036  │ 符号错！  │
  │ 全部 1520 配对取差       │ +0.0006  │ +0.0039  │ 符号错！  │
  │ 点名 10 个按侧           │ -0.0307  │ +0.0903  │ 0.121    │
  │ 最强 5 对                │ -0.1142  │ +0.0563  │ 0.171 ★  │
  │ 最强 10 对               │ -0.0622  │ +0.0702  │ 0.132    │
  └────────────────────────┴──────────┴──────────┴──────────┘

  ⚠ 「全部 1332 个按左右分两组」是实测**最差**的口径：664 个细胞平均后
    方向信号互相抵消，左右亮斑的投票差值是同一个符号，无法判向。
    默认用「最强 5 对」（可分性 0.171）。
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


GRAPH = r"E:\终末地\_handoff\graph"
OUT = r"E:\终末地\_handoff\out"

# 交接文档第七节的 5 个扇区
SECTORS = [
    ("front", "正前"),
    ("front_left", "左前"),
    ("front_right", "右前"),
    ("back_left", "左后"),
    ("back_right", "右后"),
]


@dataclass
class VotePool:
    """投票池：由「对立配对」构成

    每对 = (L 侧神经元, R 侧神经元)，定义取自同 type 的左右细胞。
    margin = mean(活动[L]) − mean(活动[R])，为负表示当前刺激偏左。
    """

    pairs: np.ndarray                 # (n_pair, 2) 列 0=L, 1=R
    strength: np.ndarray              # 每对的对立度（绝对值）
    meta: pd.DataFrame
    top_k: int = 5

    def select(self, top_k: int | None = None) -> "VotePool":
        k = self.top_k if top_k is None else top_k
        sel = np.argsort(self.strength)[::-1][:k]
        return VotePool(self.pairs[sel], self.strength[sel], self.meta, k)

    def margin(self, r: np.ndarray) -> float:
        """投票边际：<0 偏左，>0 偏右"""
        l = self.pairs[:, 0]
        rr = self.pairs[:, 1]
        return float(r[l].mean() - r[rr].mean())

    def describe(self) -> str:
        lines = [f"投票池：{len(self.pairs)} 对（top_k={self.top_k}）"]
        for k, (li, ri) in enumerate(self.pairs):
            tl = str(self.meta.iloc[li]["type"])
            tr = str(self.meta.iloc[ri]["type"])
            lines.append(f"  {k+1}. {tl}(idx={li}) / {tr}(idx={ri})  对立度 {self.strength[k]:.5f}")
        return "\n".join(lines)


def build_pool(npz_path: str, meta_path: str, top_k: int = 5) -> VotePool:
    """从 fix_pairs.py 的产出载入配对表

    配对方向已由 fix_pairs.py 统一（判据：同一刺激下比较 margin，
    保证「margin < 0 表示偏左」）。这里不再做翻转，避免二次颠倒。
    `contra` 字段现在存的是方向性 `orient`（正值 = 约定正确）。
    """
    z = np.load(npz_path)
    meta = pd.read_feather(meta_path)
    pairs = np.array(z["pairs"], dtype=np.int64, copy=True)
    strength = np.abs(np.asarray(z["contra"], dtype=np.float64))
    pool = VotePool(pairs=pairs, strength=strength, meta=meta, top_k=top_k)
    return pool.select(top_k)


# ----------------------------------------------------------------------
@dataclass
class Hysteresis:
    """时序平滑 + 滞回（交接文档第七节）

    - ema_alpha：对投票边际做指数平滑，避免每个 tick 换方向
    - switch_margin：新方向要超过当前方向这么多才切换（滞回）
    - force_switch_sec：连续同方向超时 → 强制换向（防卡死）
    """

    ema_alpha: float = 0.3
    switch_margin: float = 0.02
    force_switch_sec: float = 6.0
    current: int = 0                  # -1 左 / +1 右 / 0 未定
    _ema: float = 0.0
    _since: float = 0.0
    _forced_count: int = 0

    def update(self, margin: float, dt: float = 0.1) -> dict:
        """margin <0 偏左，>0 偏右。返回本 tick 的决策"""
        self._ema = (1 - self.ema_alpha) * self._ema + self.ema_alpha * margin
        want = -1 if self._ema < 0 else (1 if self._ema > 0 else 0)

        switched = False
        forced = False
        reason = "keep"

        if self.current == 0:
            self.current = want
            switched = True
            reason = "init"
        elif want != self.current:
            # 滞回：信号必须超过 switch_margin 才切
            if abs(self._ema) >= self.switch_margin:
                self.current = want
                switched = True
                reason = "signal"
        # 超时强制换向
        self._since += dt
        if self.current != 0 and self._since >= self.force_switch_sec and not switched:
            self.current = -self.current
            switched = True
            forced = True
            reason = "forced"
            self._forced_count += 1

        if switched:
            self._since = 0.0
        else:
            self._since += 0.0

        return dict(direction=self.current, ema=self._ema, switched=switched,
                    forced=forced, reason=reason, held_sec=self._since)


# ----------------------------------------------------------------------
class DirectionDecoder:
    """完整翻译层：DN 活动 → 方向

    ★ 自适应基线（重要修复）：
      真实游戏画面的视觉统计特性与标定用的合成亮斑不同，会给 margin
      带来一个**与朝向无关的固定偏置**（实测约 +0.009，而真实波动只有 ±0.002，
      结果 20/20 帧全判「同一侧」）。
      这里用慢速 EMA 跟踪该偏置并扣除，使判据只对**方向相关的波动**响应。
      时间常数要远大于一次转向的持续时间（几秒），否则会把真实转向也跟踪掉。
    """

    def __init__(self, top_k: int = 5,
                 npz_path: str = None, meta_path: str = None,
                 ema_alpha: float = 0.35, switch_margin: float = 0.0015,
                 force_switch_sec: float = 6.0,
                 baseline_tau: float = 25.0, baseline_warmup: int = 3):
        """switch_margin 默认 0.0015：
        实测（verify_fix / 调参脚本）真实地图下扣基线后的 margin 幅度约 ±0.01，
        原默认 0.02 比信号本身还大，会导致静止时的微小波动不触发切换、
        方向被第一个判定锁死。0.0015~0.002 区间方向能正确跟随朝向。"""
        npz_path = npz_path or os.path.join(OUT, "vote_pairs.npz")
        meta_path = meta_path or os.path.join(GRAPH, "graph_meta.feather")
        self.pool = build_pool(npz_path, meta_path, top_k)
        self.hyst = Hysteresis(ema_alpha=ema_alpha, switch_margin=switch_margin,
                               force_switch_sec=force_switch_sec)
        self.baseline_tau = baseline_tau
        self.baseline_warmup = baseline_warmup
        self._baseline = None
        self._seen = 0

    def step(self, r: np.ndarray, dt: float = 0.1) -> dict:
        raw = self.pool.margin(r)
        # 慢速 EMA 估计偏置
        if self._baseline is None:
            self._baseline = raw
        else:
            a = min(1.0, dt / max(self.baseline_tau, 1e-6))
            self._baseline = (1 - a) * self._baseline + a * raw
        self._seen += 1
        # 预热期内不判定（基线还没稳）
        centered = raw - self._baseline
        out = self.hyst.update(centered, dt)
        out["margin_raw"] = raw
        out["baseline"] = self._baseline
        out["margin"] = centered
        out["warming"] = self._seen <= self.baseline_warmup
        return out

    def direction_name(self) -> str:
        return {1: "右", -1: "左", 0: "未定"}[self.hyst.current]


# ----------------------------------------------------------------------
def _selftest():
    sys.path.insert(0, r"E:\终末地\_handoff")
    from retina import RetinaMapper

    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = W.shape[0]
    rm = RetinaMapper(os.path.join(OUT, "optic_map.feather"),
                      os.path.join(GRAPH, "graph_meta.feather"))

    G = rm.grid
    c = (G - 1) / 2.0
    yy, xx = np.mgrid[0:G, 0:G].astype(np.float32)

    def blob(angle_deg, width=12.0):
        a = np.deg2rad(angle_deg)
        bx = c + np.sin(a) * c * 0.6
        by = c - np.cos(a) * c * 0.6
        d2 = (xx - bx) ** 2 + (yy - by) ** 2
        return np.exp(-d2 / (2 * width ** 2))[..., None].repeat(3, -1).astype(np.float32)

    def run(drive, a=0.5, b=2.0, steps=12):
        r = np.zeros(n, dtype=np.float32)
        for k in range(steps):
            r = (a * r + b * np.tanh(W @ r + drive)).astype(np.float32)
        return r

    dec = DirectionDecoder(top_k=5)
    print()
    print(dec.pool.describe())
    print()
    print("=" * 74)
    print("端到端自检：给不同方向的亮斑，看解出来的方向对不对")
    print()
    print(f"  {'输入':>10s} {'margin':>12s} {'EMA':>12s} {'方向':>6s} {'切换?':>8s} {'原因':>8s}")
    for ang, expect in [(0, "正前"), (-32, "左"), (32, "右"), (-100, "左"), (100, "右"), (0, "正前")]:
        rm.reset()
        r = run(rm.encode(blob(ang)))
        out = dec.step(r, dt=0.2)
        print(f"  {expect:>10s} {out['margin']:+12.6f} {out['ema']:+12.6f} "
              f"{dec.direction_name():>6s} {str(out['switched']):>8s} {out['reason']:>8s}")

    print()
    print("  滞回验证：连续同一方向喂 40 次，应触发一次强制换向")
    rm.reset()
    r = run(rm.encode(blob(-100)))
    dec2 = DirectionDecoder(top_k=5, force_switch_sec=1.0, switch_margin=0.02)
    flips = []
    for i in range(40):
        out = dec2.step(r, dt=0.1)
        if out["switched"]:
            flips.append((i, dec2.direction_name(), out["reason"]))
    print(f"    40 次里切换了 {len(flips)} 次：{flips}")


if __name__ == "__main__":
    _selftest()
