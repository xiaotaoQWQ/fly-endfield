#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ef_eye.py — 让果蝇看**主画面**（第一人称口径）。

和看小地图的区别：
    · 小地图 = 北向固定的俯视图 → 全向地形，但**没有"前方有东西逼近"的信号**
    · 主画面 = 角色视角 → 光流 / looming（逼近膨胀），这才是真实果蝇导航用的东西

映射沿用 mc_walk.build_retina_mapper（把感光细胞方位角映到画面横向位置），
但**重写了采样**：原版对每个细胞切一条竖带做 .mean()，全屏分辨率下 6006 个细胞
要跑 ~1.8 秒。这里先把画面按列压成一维剖面，再用累积和做区间均值 —— 全部向量化。

⚠ 必须遮掉的东西（不遮就是灾难）：
    1. **我们自己的叠加面板** —— 否则果蝇会看见自己的脑活动，形成反馈环
    2. 各类 HUD —— 小地图、技能图标、血条、任务文字，都不是"世界"
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 屏幕比例坐标 (x0, y0, x1, y1) —— 这些区域是游戏 HUD，不是"世界"。
# ★ 全部按 2560×1440 实测重标过（旧表偏小，任务文字实际到 x<600/y<496，
#   右上图标行到 y<150，右侧还有情境交互提示图标 —— 漏掉它们等于给
#   眼睛喂进单侧污染，正是会造出方向偏置的那类东西）。
HUD_RECTS_FRAC = [
    (0.000, 0.000, 1.000, 0.045),   # 顶部条：标签页 / 血条
    (0.000, 0.000, 0.245, 0.350),   # 左上整块：小地图 + 图标 + 任务追踪(x<627,y<504)
    (0.500, 0.000, 1.000, 0.105),   # 右上图标行（实测从 x≈1290 起，到 y≈150）
    (0.945, 0.150, 1.000, 0.850),   # 右侧情境交互提示（x>2419, y 216..1224）
    (0.000, 0.830, 0.150, 1.000),   # 左下：角色头像 / 等级 / UID
    (0.290, 0.885, 0.735, 1.000),   # 底部中央：血条 / 技能条
    (0.700, 0.760, 1.000, 1.000),   # 右下：技能图标 + 装备制造/蓝图/设备列表
    (0.000, 0.045, 0.070, 0.280),   # 最左边缘图标列（x<180, y 65..403）
]

# 我们自己的叠加面板 —— **只在真的显示它时才遮**。
# 拆开是必须的：有一次不显示面板却仍按老表遮了这块，
# 白白挖掉宽视野右侧 24%，左右立刻失衡、方向偏成 左152:右27。
OVERLAY_RECT_FRAC = (0.700, 0.150, 1.000, 0.840)


def rect_mask(h: int, w: int, rects=HUD_RECTS_FRAC, dilate: int = 3):
    m = np.zeros((h, w), bool)
    for x0f, y0f, x1f, y1f in rects:
        x0, x1 = int(x0f * w), int(np.ceil(x1f * w))
        y0, y1 = int(y0f * h), int(np.ceil(y1f * h))
        m[max(0, y0 - dilate):min(h, y1 + dilate),
          max(0, x0 - dilate):min(w, x1 + dilate)] = True
    return m


def block_mean(a: np.ndarray, ds: int) -> np.ndarray:
    """按 ds×ds 分块求均值降采样（比跨步取样不容易混叠）。"""
    if ds <= 1:
        return a.astype(np.float32)
    h, w = a.shape[:2]
    h2, w2 = (h // ds) * ds, (w // ds) * ds
    b = a[:h2, :w2].astype(np.float32)
    if b.ndim == 3:
        return b.reshape(h2 // ds, ds, w2 // ds, ds, b.shape[2]).mean(axis=(1, 3))
    return b.reshape(h2 // ds, ds, w2 // ds, ds).mean(axis=(1, 3))


def block_any(a: np.ndarray, size) -> np.ndarray:
    """把布尔遮罩缩到 size（块内**只要有一个被遮**就算遮）—— 宁可多遮不可漏遮。"""
    h, w = a.shape[:2]
    H, W = int(size[1]), int(size[0])
    yi = (np.arange(H) * h // H)
    xi = (np.arange(W) * w // W)
    # 用累积和做块内求和，比逐块切片快
    c = np.cumsum(np.cumsum(a.astype(np.int32), 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    y1 = np.minimum((np.arange(H) + 1) * h // H, h)
    x1 = np.minimum((np.arange(W) + 1) * w // W, w)
    s = c[np.ix_(y1, x1)] - c[np.ix_(yi, x1)] - c[np.ix_(y1, xi)] + c[np.ix_(yi, xi)]
    return s > 0


class MainEye:
    """主画面 → 感光细胞驱动。"""

    def __init__(self, optic_map: str, meta_path: str, hfov: float = 90.0,
                 map_mode: str = "linear", ds: int = 4, verbose: bool = True,
                 mask_rects=None, band_u: float = 0.06, band_v: float = 0.08):
        from mc_walk import build_retina_mapper

        # ★ mask_rects 用的是**喂进来那张图**的比例坐标，不是全屏的。
        #   直接用全屏比例去遮一张裁切过的小图 = 遮错地方：
        #   实测对称裁切那版就这么被挖掉了右 30%，方向偏到 171:8。
        #   裁切框已经把 HUD/叠加面板让到框外时，这里就该传 []。
        self.mask_rects = (HUD_RECTS_FRAC if mask_rects is None
                           else list(mask_rects))

        self.rm = build_retina_mapper(optic_map, meta_path, hfov, grid=64)
        self.n = self.rm._n
        self.pr_idx = np.asarray(self.rm._pr_idx)
        self.map_mode = map_mode
        self.ds = int(ds)
        self.band_u = float(band_u)
        self.band_v = float(band_v)

        az = np.asarray(self.rm._az, dtype=np.float32)
        el = np.asarray(self.rm._el, dtype=np.float32)
        if map_mode == "linear":
            # 把 270° 的细胞方位角线性铺满画面宽度 —— 所有细胞都用得上。
            # （"fov" 口径只有 1/3 细胞在画面内，实测归一化会把其余压成负偏置）
            self.u = np.clip((az + 135.0) / 270.0, 0.0, 1.0)
            self.valid = np.ones(len(az), bool)
        else:
            self.u = np.clip(np.asarray(self.rm._fp_u), 0.0, 1.0)
            self.valid = np.asarray(self.rm._in_fov)
        # ★ 仰角也必须用上。
        #   原版 mc_walk.sample_first_person 只取 drive_map[:, x0:x1].mean()
        #   —— 整列高度的均值，**仰角信息全丢**，等于把二维视野塌缩成一维全景。
        #   实测后果：下行神经元的左右对立度只有 0.0085，
        #   而小地图（二维楔形采样）口径是 1.083 —— 差了 127 倍。
        self.v = np.clip((72.0 - el) / 144.0, 0.0, 1.0)   # 0=上（仰角+72°）

        self.prev_lum = None
        self.mask = None
        self._mshape = None
        self.last_small = None
        self.last_drive = None
        self.last_mask = None
        self.last_contrast = 0.0
        self.last_raw_std = 0.0
        if verbose:
            fr = ((self.u[self.valid] - 0.06).min(), (self.u[self.valid] + 0.06).max())
            print(f"MainEye: hfov={hfov:.0f}° mode={map_mode} ds={self.ds} "
                  f"→ 有效细胞 {int(self.valid.sum()):,}/{len(az):,}，"
                  f"横向覆盖 {fr[0]*100:.0f}%~{fr[1]*100:.0f}% 画面宽")

    # ------------------------------------------------------------------
    def sample(self, img: np.ndarray, mask: np.ndarray = None):
        """img: RGB (H,W,3) uint8 → (驱动向量, 亮度均值, drive均值)。

        mask：可选的布尔遮罩，**必须与 img 同尺寸**。
              调用方若从整屏裁切，应先在屏幕坐标下算好遮罩再裁过来 ——
              直接把「占全屏比例」的遮罩套在裁切小图上是错的（会挖错地方，
              实测把方向偏置推到 95%）。
        """
        small = block_mean(img, self.ds)              # (h, w, 3) float32 0..255
        h, w = small.shape[:2]
        if mask is not None:
            self.mask = mask
            self._mshape = (h, w)
        elif self.mask is None or self._mshape != (h, w):
            self.mask = rect_mask(h, w, self.mask_rects)
            self._mshape = (h, w)
        if self.mask.any():
            vis = ~self.mask
            small[self.mask] = small[vis].mean(axis=0)   # 填成可见区均值 = 中性

        g = small / 255.0
        lum = g @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        # 分辨率变了就丢掉上一帧（否则形状对不上直接报错）
        if self.prev_lum is not None and self.prev_lum.shape != lum.shape:
            self.prev_lum = None
        temporal = (np.abs(lum - self.prev_lum) if self.prev_lum is not None
                    else np.zeros_like(lum))
        self.prev_lum = lum
        color = np.maximum(g[..., 1] - 0.5 * (g[..., 0] + g[..., 2]), 0.0)
        drive = np.clip(0.45 * lum + 1.6 * temporal + 0.25 * color, 0.0, 1.0)

        # 留一份给调试导出（看看果蝇到底"看到"了什么）
        self.last_small = small
        self.last_drive = drive
        self.last_mask = self.mask

        # 二维采样：每个感光细胞取 (方位角, 仰角) 处的一小块，
        # 用**积分图**做 O(1) 面片均值 —— 全向量化，比逐细胞切片快几千倍。
        ii = np.zeros((h + 1, w + 1), np.float64)
        ii[1:, 1:] = drive.cumsum(0).cumsum(1)

        bu, bv = self.band_u, self.band_v
        x0 = np.clip(((self.u - bu) * w).astype(np.int32), 0, w - 1)
        x1 = np.clip(((self.u + bu) * w).astype(np.int32), 1, w)
        y0 = np.clip(((self.v - bv) * h).astype(np.int32), 0, h - 1)
        y1 = np.clip(((self.v + bv) * h).astype(np.int32), 1, h)
        cnt = np.maximum((x1 - x0) * (y1 - y0), 1).astype(np.float64)
        vals = ((ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
                / cnt).astype(np.float32)

        # ★ 失效检测指标：**归一化前**各感光细胞值的变异系数（std/mean）。
        #   相机贴到角色后背时，裁切框里几乎全是角色本人（还被遮罩填成均匀色），
        #   画面失去空间结构 → 这个值会塌下去。用它来决定要不要切回小地图口径。
        self.last_contrast = float(vals.std() / max(abs(float(vals.mean())), 1e-6))
        self.last_raw_std = float(vals.std())

        # 归一化只对有效细胞做（视野外那批恒 0，会把均值拉低）
        m = self.valid
        if m.sum() > 1:
            v = vals[m]
            mu, sd = float(v.mean()), float(v.std())
            if sd > 1e-6:
                vals[m] = (v - mu) / sd * 0.30 + 0.25 * mu
            else:
                vals[m] = v * 0.25
        vals[~m] = 0.0

        out = np.zeros(self.n, dtype=np.float32)
        out[self.pr_idx] = vals
        return out, float(lum.mean()), float(drive.mean())

    def reset(self):
        self.prev_lum = None


# ======================================================================
class EyeThread(threading.Thread):
    """**高帧率**抓主画面，供低频决策循环取用。

    为什么要单独开线程：
        原来眼睛和决策同频（1.8~2.3 Hz）。500 ms 的帧间差意味着
        drive 里的时间通道（权重 1.6，三项里最大）**整体饱和**，
        光流信息基本浪费掉了。Fly64 自己跑 50 Hz。
        实测 564×146 单帧只要 18.3 ms（抓 15.3 + 处理 3.0），30 Hz 有充裕余量。

    线程里跑完整的「抓图 → 预处理 → 二维采样 → 归一化」，主循环读最新的
    驱动向量即可。这样时间通道看到的是 33 ms 的帧间差，才是真光流。
    """

    def __init__(self, eye: "MainEye", rect, size, mask, hz: float = 30.0,
                 verbose: bool = True):
        super().__init__(daemon=True, name="eye")
        self.eye = eye
        self.rect = tuple(int(v) for v in rect)
        self.size = tuple(int(v) for v in size)
        self.mask = mask
        self.period = 1.0 / max(hz, 1.0)
        self.verbose = verbose

        self.latest = None          # (drive, contrast, t_capture)
        self.frame_id = 0
        self.fps = 0.0
        self.stop_flag = threading.Event()
        self.error = None
        self._t_hist = []

    def run(self):
        from ef_shot import grab_scaled
        while not self.stop_flag.is_set():
            t0 = time.time()
            try:
                img = grab_scaled(self.rect[0], self.rect[1], self.rect[2],
                                  self.rect[3], self.size[0], self.size[1])
                drive, _lum, _drv = self.eye.sample(img, mask=self.mask)
                self.latest = (drive, self.eye.last_contrast, time.time())
                self.frame_id += 1
                if self.verbose:
                    self._t_hist.append(t0)
                    if len(self._t_hist) > 40:
                        self._t_hist.pop(0)
                    if len(self._t_hist) > 2:
                        self.fps = (len(self._t_hist) - 1) / max(
                            self._t_hist[-1] - self._t_hist[0], 1e-6)
            except Exception as e:      # 线程里绝不能让异常静默吞掉
                self.error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    print(f"  ⚠ 眼睛线程出错：{self.error}")
                time.sleep(0.5)
            sl = self.period - (time.time() - t0)
            if sl > 0:
                time.sleep(sl)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2.0)

    def read(self):
        """返回 (drive, contrast, 帧龄秒)；还没出帧就返回 None。"""
        if self.latest is None:
            return None
        d, c, t = self.latest
        return d, c, time.time() - t
