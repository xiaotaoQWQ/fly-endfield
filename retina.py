#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""retina.py — 第 2 步：小地图 → 复眼采样 → sensory_drive（按真实方位角）

与第一版的区别：
  第一版只知道左右，只能按图片左右半区驱动。
  本版用 build_optic_map.py 产出的眼面坐标，给每个感光细胞算出**方位角**，
  再按方位角去小地图的对应角度取像素。

⚠ 关键设计判断（与交接文档的一处偏差，需确认）：
  文档第六节说「从游戏画面里裁出来的小地图区域」当果蝇的眼睛。
  但小地图是**俯视图**，不是第一人称视野——画面上的"上"是角色的正前方，
  左右是左右。所以本模块不照搬 Fly64 的球面眼（那是给 3D 第一人称用的），
  而是：把俯视小地图按方位角切成扇区 → 每个感光细胞按自己的方位角取值。

  角度约定（可配置，等真实截图再定）：
    方位角 0°   = 画面正上方 = 角色正前方
    方位角 +90° = 画面右侧
    方位角 -90° = 画面左侧
    方位角 ±180°= 画面正下方 = 角色正后方

光流编码照 Fly64（fly64/model.py:112-122）：
  lum = 0.2126R + 0.7152G + 0.0722B
  temporal = |lum - prev_lum|
  color = max(G - 0.5(R+B), 0)
  drive = clip(0.45*lum + 1.6*temporal + 0.25*color, 0, 1)
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# 5 扇区定义（方位角区间，单位度）
SECTORS = [
    ("front", -10.0, 10.0, "正前"),
    ("front_left", -54.0, -10.0, "左前"),
    ("front_right", 10.0, 54.0, "右前"),
    ("back_left", -180.0, -54.0, "左后"),
    ("back_right", 54.0, 180.0, "右后"),
]


@dataclass
class RetinaMapper:
    """小地图 → 感光细胞驱动"""

    optic_map_path: str
    meta_path: str
    grid: int = 64
    amp_lum: float = 0.45
    amp_temporal: float = 1.6
    amp_color: float = 0.25
    invert: bool = False
    rotate_deg: float = 0.0        # 画面朝向校正
    _pr_idx: np.ndarray = field(default=None, repr=False)
    _az: np.ndarray = field(default=None, repr=False)
    _sector_of: np.ndarray = field(default=None, repr=False)
    _prev_lum: np.ndarray = field(default=None, repr=False)
    _n: int = 0

    def __post_init__(self):
        meta = pd.read_feather(self.meta_path)
        self._n = len(meta)
        om = pd.read_feather(self.optic_map_path)

        self._pr_idx = om["idx"].to_numpy()

        # 方位角：照 fly64/retina.py:63-67
        px_x = om["px_x"].to_numpy()
        px_y = om["px_y"].to_numpy()
        side = om["side"].astype(str).to_numpy()
        horiz = (px_y % 32) / 31
        self._az = np.where(side == "L", -135.0, -8.5) + horiz * 143.5
        self._el = 72.0 - px_x / 47 * 144

        # 扇区归属
        sec = np.full(len(om), -1, dtype=np.int8)
        for i, (_, lo, hi, _) in enumerate(SECTORS):
            sec[(self._az >= lo) & (self._az < hi)] = i
        self._sector_of = sec

        print(f"RetinaMapper: 感光细胞 {len(self._pr_idx):,}   网格 {self.grid}×{self.grid}")
        print(f"  方位角 {self._az.min():.1f}° ~ {self._az.max():.1f}°"
              f"   仰角 {self._el.min():.1f}° ~ {self._el.max():.1f}°")
        print("  扇区分布：")
        for i, (key, lo, hi, name) in enumerate(SECTORS):
            c = int((sec == i).sum())
            print(f"    {name} ({lo:+.0f}°~{hi:+.0f}°)  {c:5,} 个感光细胞")

    # ------------------------------------------------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        if self.invert:
            img = 1.0 - img
        if self.rotate_deg:
            k = int(round(self.rotate_deg / 90.0)) % 4
            if k:
                img = np.rot90(img, k)
        h, w = img.shape[:2]
        yi = np.linspace(0, h - 1, self.grid).astype(np.int32)
        xi = np.linspace(0, w - 1, self.grid).astype(np.int32)
        return np.ascontiguousarray(img[yi][:, xi])

    # ------------------------------------------------------------------
    def drive_map(self, image: np.ndarray, use_temporal: bool = True) -> np.ndarray:
        """算出每像素的 drive（照 Fly64 的编码）

        use_temporal=False 时不引入帧间变化项，也不改动光流状态（对照用）。
        """
        g = self.preprocess(image)
        lum = g @ LUMA
        if use_temporal:
            temporal = np.abs(lum - self._prev_lum) if self._prev_lum is not None \
                else np.zeros_like(lum)
            self._prev_lum = lum.copy()
        else:
            temporal = np.zeros_like(lum)
        color = np.maximum(g[..., 1] - 0.5 * (g[..., 0] + g[..., 2]), 0)
        return np.clip(self.amp_lum * lum + self.amp_temporal * temporal
                       + self.amp_color * color, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _angle_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """返回每个网格像素对应的方位角、半径（俯视图极坐标）

        ⚠ 方位角约定必须和 retina.py 的眼面方位角一致（航海约定，右为正）：
            0°   = 画面正上方 = 角色正前方
            +90° = 画面右侧
            -90° = 画面左侧
            ±180°= 画面正下方 = 角色正后方
        """
        c = (self.grid - 1) / 2.0
        yy, xx = np.mgrid[0:self.grid, 0:self.grid].astype(np.float32)
        dx = xx - c                 # 右为正
        dy = c - yy                 # 上为正
        az = np.rad2deg(np.arctan2(dx, dy))     # 上=0°，右=+90°，左=-90°，下=±180°
        radius = np.sqrt(dx * dx + dy * dy)
        return az, radius

    # ------------------------------------------------------------------
    def _sample_cells(self, dm: np.ndarray, radius_frac: float = 1.0,
                      heading_deg: float = 0.0, center: str = "none",
                      keep_baseline: float = 0.25) -> np.ndarray:
        """每个感光细胞按自己的方位角，取小地图上该角度扇区内的平均 drive

        heading_deg：角色朝向。感光细胞的生理方位角（self._az，来自眼面坐标）
        加上角色朝向，才是它此刻指向的世界方位角——眼睛是跟着头转的。

        center：空间去均值方式。真实小地图常是大片同色（建筑俯视图），
          各方向采到的值几乎相同 → 网络收到均匀输入 → 输出退化成固定偏置。
          去掉空间均值后，只有**方向间的差异**成为有效信号。
            "none"  不去均值（旧行为）
            "mean"  减去全体均值
            "zscore" 减均值再除以标准差（不同对比度的图归一化到同一强度）
          keep_baseline：去均值后保留多少比例的原始基线（避免整体活动过低）
        """
        az_grid, radius = self._angle_grid()
        rmax = (self.grid / 2.0) * radius_frac
        valid = radius <= rmax
        vals = np.empty(len(self._pr_idx), dtype=np.float32)
        half = 22.0     # 取样半宽（度）
        for i, a in enumerate(self._az):
            a_world = a + heading_deg          # 眼睛此刻看的世界方位角
            lo, hi = a_world - half, a_world + half
            if lo < -180:
                m = (az_grid >= lo + 360) | (az_grid <= hi)
            elif hi > 180:
                m = (az_grid >= lo) | (az_grid <= hi - 360)
            else:
                m = (az_grid >= lo) & (az_grid <= hi)
            m &= valid
            vals[i] = dm[m].mean() if m.any() else 0.0

        if center == "mean":
            vals = vals - float(vals.mean()) + keep_baseline * float(vals.mean())
        elif center == "zscore":
            mu = float(vals.mean())
            sd = float(vals.std())
            vals = (vals - mu) / sd * 0.30 + keep_baseline * mu if sd > 1e-6 else vals * keep_baseline
        return vals.astype(np.float32)

    def encode(self, image: np.ndarray, radius_frac: float = 1.0,
               update_prev: bool = True, heading_deg: float = 0.0,
               center: str = "none", keep_baseline: float = 0.25) -> np.ndarray:
        """图 → 感光细胞驱动向量（长度 = 神经元总数）

        heading_deg：角色朝向的方位角（0=小地图上方，+90=右）。
        感光细胞的采样方向会随它旋转。
        center/keep_baseline：见 _sample_cells 的说明。
        """
        dm = self.drive_map(image, use_temporal=True)
        vals = self._sample_cells(dm, radius_frac, heading_deg=heading_deg,
                                  center=center, keep_baseline=keep_baseline)
        out = np.zeros(self._n, dtype=np.float32)
        out[self._pr_idx] = vals
        if not update_prev:
            self._prev_lum = None
        return out

    # ------------------------------------------------------------------
    def sector_drive(self, image: np.ndarray, update_prev: bool = True,
                     heading_deg: float = 0.0, center: str = "none",
                     keep_baseline: float = 0.25) -> dict[str, float]:
        """按感光细胞的方位角，聚合每个扇区的驱动均值

        这是「扇区投票」的输入口径：每个扇区的值 = 该扇区内所有感光细胞
        所采到的 drive 的平均。

        heading_deg：角色朝向的方位角（由 minimap 箭头读出）。
          小地图是「北朝上」型 ⇒ 感光细胞的生理方位角要加上角色朝向，
          才是它此刻真正看向的世界方位角。
          heading_deg=0 时退化为「假设画面上方=角色前方」（旧行为）。

        update_prev=False 时不推进光流状态（调试/对照用）。
        """
        dm = self.drive_map(image, use_temporal=True)
        vals = self._sample_cells(dm, heading_deg=heading_deg,
                                  center=center, keep_baseline=keep_baseline)
        res = {}
        for i, (key, lo, hi, name) in enumerate(SECTORS):
            m = self._sector_of == i
            res[key] = float(vals[m].mean()) if m.any() else 0.0
        if not update_prev:
            self._prev_lum = None
        return res

    # ------------------------------------------------------------------
    def sector_inputs(self, image: np.ndarray) -> dict[str, float]:
        """把画面按 5 扇区聚合，返回每个扇区的 drive 均值（用于对照/调试）"""
        dm = self.drive_map(image)
        az_grid, radius = self._angle_grid()
        valid = radius <= (self.grid / 2.0)
        res = {}
        for key, lo, hi, name in SECTORS:
            m = (az_grid >= lo) & (az_grid < hi) & valid
            res[key] = float(dm[m].mean()) if m.any() else 0.0
        return res

    def reset(self):
        self._prev_lum = None


# ----------------------------------------------------------------------
def _selftest():
    meta_path = r"E:\终末地\_handoff\graph\graph_meta.feather"
    optic_path = r"E:\终末地\_handoff\out\optic_map.feather"
    rm = RetinaMapper(optic_path, meta_path)

    G = 64
    c = (G - 1) / 2.0
    yy, xx = np.mgrid[0:G, 0:G].astype(np.float32)
    radius = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)

    def blob(angle_deg, width=12):
        """在指定方位角画一个圆斑（模拟小地图上该方向有东西）"""
        a = np.deg2rad(angle_deg)
        bx = c + np.sin(a) * c * 0.6
        by = c - np.cos(a) * c * 0.6
        d = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
        return np.exp(-(d ** 2) / (2 * width ** 2))[..., None].repeat(3, -1).astype(np.float32)

    print()
    print("=" * 78)
    print("自检：在画面不同方位角放一个亮斑，看哪个扇区被点亮")
    print()
    print(f"  {'亮斑方位':>10s} {'正前':>8s} {'左前':>8s} {'右前':>8s} {'左后':>8s} {'右后':>8s}   判定")
    cases = [(0, "正前"), (30, "右前"), (-30, "左前"), (90, "正右"), (-90, "正左"), (180, "正后")]
    for ang, expect in cases:
        img = blob(ang)
        rm.reset()
        sec = rm.sector_drive(img)
        best = max(sec, key=sec.get)
        name_of = {k: n for k, _, _, n in SECTORS}
        print(f"  {expect:>10s} " + " ".join(f"{sec[k]:8.4f}" for k, _, _, _ in SECTORS)
              + f"   → {name_of[best]}")

    print()
    print("  全黑 / 全白 对照：")
    for nm, v in [("全黑", 0.0), ("全白", 1.0)]:
        img = np.full((G, G, 3), v, np.float32)
        rm.reset()
        sec = rm.sector_drive(img)
        print(f"    {nm}: " + " ".join(f"{sec[k]:.4f}" for k, _, _, _ in SECTORS))

    print()
    print("  光流验证（关键：第 2 帧与第 1 帧完全相同 → temporal 应为 0）")
    rm.reset()
    # 用确定性的两张图，避免随机图亮度恰好相同造成误判
    imgA = np.full((G, G, 3), 0.30, np.float32)
    imgB = np.full((G, G, 3), 0.80, np.float32)
    vA1 = rm.encode(imgA)[rm._pr_idx].mean()      # 第1帧：无前帧，无光流
    vA2 = rm.encode(imgA)[rm._pr_idx].mean()      # 第2帧：同图，光流应为 0
    vB = rm.encode(imgB)[rm._pr_idx].mean()       # 第3帧：变了，有光流
    print(f"    A 帧首次  {vA1:.5f}   （0.45×0.30 = {0.45*0.30:.3f}）")
    print(f"    A 帧重复  {vA2:.5f}   ← 应与首次相同（说明 temporal 正确归零）")
    print(f"    B 帧      {vB:.5f}   （0.45×0.80 + 1.6×0.50 = {0.45*0.80+1.6*0.50:.3f}，被 clip 到 1.0）")

    print()
    print("  半径范围的影响（只取小地图中心区域，丢掉边缘）：")
    for frac in [1.0, 0.75, 0.5]:
        img = blob(30)
        rm.reset()
        sec = rm.sector_drive(img, update_prev=False)
        # 需重算带 radius_frac 的：直接用 encode
        rm.reset()
        d = rm.encode(img, radius_frac=frac)
        m = rm._sector_of == 2      # 右前
        print(f"    radius_frac={frac:<5} 右前扇区均值 {d[rm._pr_idx][m].mean():.5f}"
              f"   全部感光细胞均值 {d[rm._pr_idx].mean():.5f}")


if __name__ == "__main__":
    _selftest()
