#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ef_vision.py — 从终末地小地图里读三样东西：

    1. 位移（视觉里程计）—— 用 FFT 相位相关，看地图整体滚动了多少
    2. 朝向 —— 角色**永远在前进**，所以位移方向 ≈ 脸朝向（不需要知道 FOV）
    3. 卡住 —— 位移≈0 说明撞墙/被卡

为什么用相位相关而不是找标记：
    终末地小地图上**没有玩家位置/朝向标记**（放大扫过整张图，确认没有）。
    但地图是**北向固定**的俯视图，角色一动地图就整体平移 ——
    相位相关正好吃这个，而且不需要知道地图缩放比例（我们只要方向）。

⚠ 标定（2560×1440）：
    老代码写的是圆心 (188,236) 半径 143 —— **是错的**。
    实测暗色圆形渐晕为 圆心 (228,208) 半径 118。
    老值 x 偏 40px（=半径的 34%），采样盘整个错位并越界，
    把左侧 HUD 面板和右下游戏世界也当成地形喂给了果蝇。
"""
from __future__ import annotations

import numpy as np

# ---- 小地图标定（2560×1440 基准，按分辨率等比缩放）----
MM_CX, MM_CY, MM_R, MM_H = 215, 225, 113, 1440
# ★ MM_CY 是**画面坐标**（不含黑边）。窗口 y = MM_CY + 黑边偏移（见 content_rect）。
#   实测（2560×1600 窗口，画面 2560×1440 @y=80）：光边圆心在窗口 (215, 305)、r=113
#   → 画面坐标 y = 305 - 80 = 225。
#   ⚠ 旧值 (228, 208, 118) 是 2560×1440 全屏时代测的；那时窗口高 = 画面高，
#     两者恰好等价。分辨率改成 16:10 之后必须走 content_rect，不能直接用窗口坐标。

# 参与相关的方形区域半径（留边，别把圆外的渐晕/HUD 卷进来）
MM_INNER_R = 100


def content_rect(x: int, y: int, w: int, h: int):
    """游戏窗口里**实际画面**的区域 (x, y, w, h)。

    ★ 实测：2560×1600 的窗口里跑的是 **2560×1440 的 16:9 画面**，
      上下各 **80px 黑边**（实测 y 0..79 平均亮度为 0）。
      所以：
        · 所有按 1440 标定的 HUD 绝对坐标必须**整体下移 80px**；
        · 缩放系数要按**画面高度**算，不能按窗口高度算 ——
          按窗口算会得出 k = 1600/1440 = 1.111 的假缩放，
          小地图圆心实测偏了 74px（公式预测 y=231，实测 y=305）。

    宁可每次重新测黑边，也不要假设「窗口高 = 画面高」。
    """
    ch = min(h, int(round(w * 9.0 / 16.0)))
    oy = (h - ch) // 2
    return x, y + oy, w, ch


def minimap_rect(screen_h: int, inner: bool = True, ox: int = 0, oy: int = 0):
    """小地图的绝对屏幕矩形 (x, y, w, h)。inner=True 取保证在圆内的方形。

    ox/oy 是**画面区**左上角（见 content_rect）；screen_h 也是**画面高度**，
    不是窗口高度。传错会得出 k=1.111 的假缩放，圆心偏 74px。
    """
    k = screen_h / MM_H
    r = (MM_INNER_R if inner else MM_R) * k
    cx, cy = MM_CX * k, MM_CY * k
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    size = int(round(2 * r))
    return ox + x0, oy + y0, size, size


def crop_minimap(img: np.ndarray, inner: bool = False) -> np.ndarray:
    """从整屏图里裁出小地图（不常用；走 minimap_rect 直接抓更快）。"""
    h, w = img.shape[:2]
    k = h / MM_H
    cx, cy, r = MM_CX * k, MM_CY * k, (MM_INNER_R if inner else MM_R) * k
    x0, y0 = max(0, int(round(cx - r))), max(0, int(round(cy - r)))
    x1, y1 = int(round(cx + r)), int(round(cy + r))
    return img[y0:y1, x0:x1]


def to_gray(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return a.astype(np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


# ---------------------------------------------------------------- 鼠标指针
def cursor_mask(img: np.ndarray) -> np.ndarray:
    """找游戏鼠标指针（极亮、近白、成块）。

    指针在屏幕上是**静止**的，相位相关会把它当成"零位移"的强特征，
    从而把整幅图的相关峰拉回 0 —— 必须抠掉。
    """
    rgb = img[..., :3].astype(np.int16)
    lum = rgb.max(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    m = (lum > 232) & (sat < 28)
    if m.sum() < 40:
        return np.zeros(m.shape, bool)

    # 腐蚀：细线（地图白色道路）会消失，成块的指针留下
    er = m.copy()
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            er &= np.roll(np.roll(m, dy, 0), dx, 1)
    if er.sum() < 25:
        return np.zeros(m.shape, bool)

    # 膨胀回来并限制在指针附近
    out = er.copy()
    for dy in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        for dx in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
            out |= np.roll(np.roll(er, dy, 0), dx, 1)
    return out


def _suppress(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """抠掉指针区域，填**独立噪声**。

    为什么不用均值填：填成一块平的，会在两帧的**同一屏幕位置**留下同一条强边缘，
    相位相关就会在零位移处冒出假峰。独立噪声只在所有位移上抬高平底、
    不偏向任何位移，等效于把这块从相关里摘掉。
    """
    g = to_gray(img)
    if mask is not None and mask.any():
        keep = g[~mask]
        mu = float(keep.mean()) if keep.size else 0.0
        sd = max(float(keep.std()) if keep.size else 1.0, 1.0)
        g = g.copy()
        g[mask] = np.random.default_rng().normal(mu, sd, int(mask.sum()))
    return g


# ---------------------------------------------------------------- 相位相关
def phase_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 45) -> tuple[int, int, float]:
    """图像内容从 a 到 b 的平移量 (dx, dy)。

    约定：返回 (dx, dy) 使 b[y, x] ≈ a[y - dy, x - dx]（即内容整体挪了 (dx, dy)）。
    用汉宁窗压边界效应，并限制搜索范围避免远处伪峰。

    ⚠ 符号：FFT 互相关 A·conj(B) 的峰给的是**反向**位移，
      自检（_selftest）就是为这个写的 —— 第一版这里符号反了，被自检抓出来。
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()

    wy = np.hanning(a.shape[0])[:, None]
    wx = np.hanning(a.shape[1])[None, :]
    win = wy * wx

    A = np.fft.rfft2(a * win)
    B = np.fft.rfft2(b * win)
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-9)
    c = np.fft.irfft2(R, s=a.shape)

    h, w = c.shape
    if max_shift:
        m = np.zeros_like(c, bool)
        m[:max_shift + 1, :max_shift + 1] = True
        m[:max_shift + 1, -max_shift:] = True
        m[-max_shift:, :max_shift + 1] = True
        m[-max_shift:, -max_shift:] = True
        c = np.where(m, c, -np.inf)

    ry, rx = (int(v) for v in np.unravel_index(int(np.argmax(c)), c.shape))
    peak = float(c[ry, rx])

    # 峰值锐度：峰 / 次峰，用来判断这次相关可不可信
    flat = c.ravel().copy()
    flat[int(np.argmax(flat))] = -np.inf
    sharp = peak / (abs(float(np.max(flat))) + 1e-9)

    # 反向 → 内容位移
    dx = -(rx - w if rx > w // 2 else rx)
    dy = -(ry - h if ry > h // 2 else ry)
    return int(dx), int(dy), float(sharp)


class MapOdometry:
    """连续读小地图，累计角色位移与朝向。

    ⚠ 为什么要「长基线」而不是逐帧比：
        实测步速约 8 地图像素/秒。3Hz 逐帧只走 ~2.7px、6Hz 只有 ~1.3px，
        跟 FFT 的量化噪声（1px 起）同量级 —— 逐帧相关会大量落在噪声里。
        改成跟 **baseline 秒之前那一帧**比，位移 ~6px，信噪比就够看了。
    """

    def __init__(self, baseline: float = 0.8, min_speed: float = 1.5,
                 ema: float = 0.35, max_hist: int = 24, stuck_sec: float = 3.0):
        self.baseline = baseline
        self.min_speed = min_speed
        self.ema = ema
        self.max_hist = max_hist
        # ★ 卡住判定改成**按时间**，不是按拍数。
        #   原来 stuck = 连续 3 拍速度过低 —— 2.5Hz 下只有 1.2 秒，太灵敏。
        self.stuck_sec = float(stuck_sec)
        self._still_t0 = None
        self._last_t = 0.0
        self.stuck_now = False
        self.hist: list = []          # [(t, gray)]
        self.x = 0.0
        self.y = 0.0                  # 地图像素（相对起点）
        self.heading = 0.0            # 方位角：0=北(地图上)，+90=东(地图右)
        self.speed = 0.0              # 地图像素/秒
        self.sharp = 0.0
        self.stuck_ticks = 0
        self.moved_ticks = 0
        self.n = 0
        self.last: dict = {}

    @property
    def stuck(self) -> bool:
        """连续静止达到 stuck_sec 秒才算卡住（按时间，不按拍数）。"""
        if self._still_t0 is None:
            return False
        return (self._last_t - self._still_t0) >= self.stuck_sec

    def step(self, mm_rgb: np.ndarray, t: float | None = None) -> dict:
        """喂**小地图裁剪图**（用 minimap_rect 抓）。"""
        import time as _t
        t = _t.time() if t is None else t

        cm = cursor_mask(mm_rgb)
        g = _suppress(mm_rgb, cm)
        self.hist.append((t, g))

        # baseline 秒之前最近的一帧当参照
        ref = None
        for tt, gg in self.hist:
            if tt <= t - self.baseline:
                ref = (tt, gg)
            else:
                break

        cutoff = t - 2.0 * self.baseline
        self.hist = [h for h in self.hist if h[0] >= cutoff][-self.max_hist:]

        if ref is None or ref[1].shape != g.shape:
            self.last = dict(dx=0.0, dy=0.0, dist=0.0, speed=0.0,
                             heading=self.heading, sharp=0.0, stuck=False,
                             cursor=bool(cm.any()), primed=False)
            return self.last

        dt = max(t - ref[0], 1e-3)
        cdx, cdy, sharp = phase_shift(ref[1], g)
        self.sharp = sharp
        self.n += 1

        # 地图内容往 +x 滚 → 角色往 -x 走
        wx, wy = -float(cdx), -float(cdy)
        dist = float(np.hypot(wx, wy))
        self.speed = dist / dt

        if self.speed < self.min_speed or sharp < 1.3:
            # 静止：第一次静止时记下时间戳，之后按经过时长判卡
            if self._still_t0 is None:
                self._still_t0 = t
            self.stuck_ticks += 1
        else:
            self._still_t0 = None
            self.stuck_ticks = 0
            self.moved_ticks += 1
            self.x += wx
            self.y += wy
            # 位移方向 = 脸朝向（角色一直在前进）
            az = float(np.degrees(np.arctan2(wx, -wy))) % 360.0
            # 环形 EMA，避免 359↔1 跳变
            d = (az - self.heading + 180.0) % 360.0 - 180.0
            self.heading = (self.heading + self.ema * d) % 360.0

        self._last_t = t
        self.stuck_now = self.stuck
        self.last = dict(dx=wx, dy=wy, dist=dist, speed=self.speed,
                         heading=self.heading, sharp=sharp, stuck=self.stuck_now,
                         still_sec=(0.0 if self._still_t0 is None
                                    else t - self._still_t0),
                         cursor=bool(cm.any()), primed=True)
        return self.last

    def reset(self):
        self.__init__(self.baseline, self.min_speed, self.ema, self.max_hist,
                      self.stuck_sec)


# ---------------------------------------------------------------- 自检
def _selftest():
    rng = np.random.default_rng(0)
    base = to_gray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))
    # 加一点结构，别是纯噪声
    base[40:80, 30:120] = 240.0
    base[120:180, 60:100] = 20.0

    print("相位相关自检（应精确还原施加的平移）：")
    ok = True
    for (sx, sy) in [(0, 0), (7, -5), (12, 9), (-15, 3), (25, -20)]:
        b = np.roll(base, (sy, sx), axis=(0, 1))
        dx, dy, sh = phase_shift(base, b)
        good = (dx == sx and dy == sy)
        ok &= good
        print(f"  施 ({sx:+3d},{sy:+3d}) → 测 ({dx:+3d},{dy:+3d})  锐度 {sh:6.2f}  "
              f"{'✓' if good else '✗'}")
    print("全部正确 ✓" if ok else "有错 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
