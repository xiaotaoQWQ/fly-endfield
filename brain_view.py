#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""brain_view.py — 把 211,577 个神经元的实时活动画成一张「全脑发光图」。

数据来源：
    · 胞体三维坐标 = body-annotations 的 somaLocation（141,781 个非空）
    · 与 graph 的神经元索引用 bodyId 对齐（graph_meta 里有 bodyId + idx）
    · 活动值 = 速率模型的状态向量 r

画法：
    正交投影到二维 → 用 bincount 做「密度」和「活动加权」两张累积图
    → 密度画成暗蓝的结构底、活动叠一层发光色
    纯 numpy，不走 3D 库；每次重投影约 10ms 量级，3Hz 完全够用。

用法（自检：渲染三个视角拼一张图看看）：
    python -X utf8 brain_view.py --test
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph")
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(GRAPH, "soma_xyz.npz")

ANN = os.path.join(DATA, "body-annotations-male-cns-v1.0-minconf-0.5.feather")

# 活动 → 颜色：一条**高对比彩虹热力尺**。
# 参考图那种「有活动的地方要一眼看出来」的要求，靠的就是多色相分段：
# 不活动=深蓝（几乎融进底色），越活跃越往青→绿→黄→橙→白走，色相跨得开。
CMAP_POS = np.array([0.00, 0.16, 0.34, 0.54, 0.74, 0.90, 1.00], dtype=np.float32)
CMAP_RGB = np.array([
    [0.05, 0.08, 0.24],   # 深蓝 —— 静息
    [0.10, 0.42, 0.92],   # 蓝
    [0.10, 0.88, 0.88],   # 青
    [0.26, 0.92, 0.34],   # 绿
    [1.00, 0.86, 0.16],   # 黄
    [1.00, 0.36, 0.10],   # 橙红
    [1.00, 1.00, 1.00],   # 白 —— 最活跃
], dtype=np.float32)

# 结构底色：琥珀（跟参考图一致），刻意压暗压灰，
# 好让活动色（青绿黄白）压得住、不打架。
STRUCT_RGB = np.array([0.62, 0.46, 0.28], dtype=np.float32)

_LUT = None


def _lut() -> np.ndarray:
    """预计算 256 级颜色查找表。

    原先每帧对整幅图跑 3 次 np.interp —— 588×440 上是实打实的开销。
    查表之后这一块基本免费。
    """
    global _LUT
    if _LUT is None:
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        _LUT = np.stack([np.interp(x, CMAP_POS, CMAP_RGB[:, c])
                         for c in range(3)], axis=-1).astype(np.float32)
    return _LUT


def build_cache(force: bool = False) -> str:
    """把胞体坐标对齐到神经元索引，存成 npz。"""
    if os.path.exists(CACHE) and not force:
        return CACHE

    print("构建胞体坐标缓存…")
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    ann = pd.read_feather(ANN)[["bodyId", "somaLocation"]]
    m = meta[["bodyId", "idx"]].merge(ann, on="bodyId", how="left")
    m = m.sort_values("idx")

    n = len(m)
    xyz = np.full((n, 3), np.nan, dtype=np.float32)
    loc = m["somaLocation"].to_numpy()
    have = np.array([isinstance(v, (np.ndarray, list)) and len(v) == 3
                     for v in loc])
    if have.any():
        vals = np.array([np.asarray(v, dtype=np.float32) for v in loc[have]])
        xyz[np.nonzero(have)[0]] = vals
    ok = ~np.isnan(xyz).any(axis=1)
    print(f"  有胞体坐标 {ok.sum():,} / {n:,}")

    # 存成「已居中、已缩放到 [-1,1]」的坐标，主循环里就不用再算了
    c = np.nanmean(xyz[ok], axis=0)
    q = xyz[ok] - c
    s = np.nanmax(np.abs(q))
    q = q / max(s, 1e-6)

    np.savez_compressed(CACHE, xyz=xyz, norm=q.astype(np.float32), ok=ok,
                        center=c.astype(np.float32), scale=np.float32(s))
    print(f"  → {CACHE}")
    return CACHE


class BrainView:
    """给定神经元活动向量 r，渲染一张全脑发光图（H×W×3 uint8）。"""

    def __init__(self, w: int = 560, h: int = 520, bg=(0.02, 0.02, 0.05)):
        d = np.load(build_cache())
        self.norm = d["norm"].astype(np.float32)   # (N,3)，已居中归一
        self.ok = d["ok"]
        self.n_all = len(self.ok)
        self.idx_ok = np.nonzero(self.ok)[0]
        self.n = len(self.idx_ok)
        self.w, self.h = int(w), int(h)
        self.bg = np.asarray(bg, dtype=np.float32)
        # 活动归一化的滑动参考（避免固定量程在活动整体漂移时全黑/全白）
        self._hi = 1e-3
        self._lo = 0.0
        self._blur = None
        self.last_stats = {}

    # ---------------------------------------------------------------- 投影
    def _rot(self, yaw: float, pitch: float) -> np.ndarray:
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        return Rx @ Ry

    def render(self, r: np.ndarray, yaw: float = 0.0, pitch: float = 0.35,
               gain: float = 1.0, zoom: float = 1.0) -> np.ndarray:
        W, H = self.w, self.h
        p = self.norm @ self._rot(yaw, pitch).T          # (N,3)

        # 正交投影：取 x/y 做屏幕面；等比缩放到面板内
        # zoom 默认 >1：脑的最长轴常常正对镜头，投影出来的轮廓偏小
        R = min(W, H) * 0.5 * 0.94 * zoom
        u = (p[:, 0] * R + W * 0.5).astype(np.int32)
        v = (-p[:, 1] * R + H * 0.5).astype(np.int32)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        flat = v[ok] * W + u[ok]

        # 结构底：所有胞体的密度
        dens = np.bincount(flat, minlength=W * H).astype(np.float32)

        # 活动层：按 r 加权
        a = r[self.idx_ok][ok].astype(np.float32)
        act = np.bincount(flat, weights=a, minlength=W * H).astype(np.float32)

        dens = dens.reshape(H, W)
        act = act.reshape(H, W)

        # 轻微模糊 → 发光感。别糊太狠，糊狠了颜色会糊成一团、
        # 就分不出「哪里在活动」了。
        if self._blur is None:
            try:
                from scipy.ndimage import gaussian_filter as _gf
                self._blur = _gf
            except Exception:
                self._blur = False
        if self._blur:
            act = self._blur(act, 0.8)

        # 结构底：琥珀色，对数压缩后抬一点，保证脑轮廓始终看得见
        ds = np.log1p(dens)
        ds = (ds / max(float(ds.max()), 1e-6)) ** 0.55
        base = ds[..., None] * STRUCT_RGB * 1.15

        # 量程自适应：**按累积图自身**定标，而不是按单个神经元的 r。
        # 画面上是邻域累积值，量级跟逐神经元的分位数差很多 ——
        # 用后者定标会把几乎所有活动都顶到色尺上端（只剩青/白，看不到绿/黄）。
        pos = act[act > 1e-6]
        if pos.size >= 20:
            cur = float(np.percentile(pos, 98.0))
            self._hi = cur if self._hi <= 1e-6 else (0.88 * self._hi + 0.12 * cur)
        hi = max(self._hi, 1e-6)
        an = np.clip(np.clip(act / hi, 0.0, 1.0) * gain, 0.0, 1.0)

        # 查表上色（比每帧 np.interp 快得多）
        col = _lut()[(an * 255.0).astype(np.uint8)]
        # 只在真有活动的地方覆盖底色，静息区保留琥珀结构
        live = np.clip(an * 5.0, 0.0, 1.0)[..., None]

        img = self.bg + base * (1.0 - live) + col * live * (0.35 + 0.65 * an[..., None])
        img = np.clip(img, 0.0, 1.0)
        self.last_stats = dict(act_hi=hi, act_max=float(act.max()),
                               dens_max=float(dens.max()))
        return (img * 255).astype(np.uint8)

    def adapt_range(self, r: np.ndarray, alpha: float = 0.08):
        """（保留接口）量程现在由 render() 按累积图自适应，这里只做粗初始化。"""
        v = np.asarray(r, dtype=np.float32)[self.idx_ok]
        hi = float(np.percentile(v, 99.0))
        if self._hi <= 1e-6:
            self._hi = hi


# -------------------------------------------------------------------- 自检
def _test():
    from PIL import Image

    rng = np.random.default_rng(0)
    bv = BrainView()

    # 造一个假活动：几个团块亮起来
    r = np.zeros(bv.n_all, dtype=np.float32)
    hot = rng.choice(bv.idx_ok, 20000, replace=False)
    r[hot] = rng.random(20000).astype(np.float32) * 0.12
    bv._hi = float(np.percentile(r[bv.idx_ok], 99.0))

    views = [(0.0, 0.35), (1.2, 0.0), (2.4, 0.9), (0.6, -0.6)]
    imgs = [bv.render(r, yaw=y, pitch=p) for (y, p) in views]
    gap = 6
    H = max(i.shape[0] for i in imgs)
    W = sum(i.shape[1] for i in imgs) + gap * (len(imgs) + 1)
    canvas = np.full((H + 2 * gap, W, 3), 18, dtype=np.uint8)
    x = gap
    for im in imgs:
        canvas[gap:gap + im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    out = os.path.join(HERE, "shots", "brain_test.png")
    Image.fromarray(canvas).save(out)
    print(f"已保存 {out}  {canvas.shape[1]}x{canvas.shape[0]}")
    print(f"  统计 {bv.last_stats}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    else:
        build_cache(force="--force" in sys.argv)
