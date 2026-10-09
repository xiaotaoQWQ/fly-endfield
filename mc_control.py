#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mc_control.py — 验证并驱动 Minecraft 角色（无小地图模式）

用途：
  1. 验证 SendInput 能不能控制 MC 角色（截图对比 + F3 坐标）
  2. 作为执行层：把方向决策变成按键

窗口：MC 是窗口化运行，位置尺寸会变，每次都重新定位。

用法：
  python mc_control.py --find                 # 找窗口
  python mc_control.py --shot                 # 截一张
  python mc_control.py --test w               # 测按 W 是否移动
  python mc_control.py --f3                   # 打开 F3 读坐标
  python mc_control.py --walk w 2.0           # 按住 W 2 秒
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_sim import GameInput, find_hwnd  # noqa: E402

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


class BIH(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def grab(x, y, w, h):
    hdc = user32.GetDC(0)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mem, bmp)
    gdi32.BitBlt(mem, 0, 0, w, h, hdc, x, y, 0x00CC0020)
    bi = BIH()
    bi.biSize = ctypes.sizeof(BIH)
    bi.biWidth, bi.biHeight = w, -h
    bi.biPlanes, bi.biBitCount = 1, 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), 0)
    gdi32.SelectObject(mem, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(0, hdc)
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return a[:, :, ::-1].copy()


def find_mc():
    """MC 窗口标题含 Minecraft"""
    for kw in ("Minecraft", "1.21.1"):
        h = find_hwnd(kw)
        if h:
            return h
    return None


class MC:
    def __init__(self, verbose=True):
        self.hwnd = find_mc()
        if self.hwnd is None:
            raise RuntimeError("没找到 Minecraft 窗口")
        r = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(r))
        self.rect = (r.left, r.top, r.right - r.left, r.bottom - r.top)
        self.gi = GameInput(self.hwnd, verbose=verbose, method="scan")
        self.verbose = verbose

    def shot(self):
        return grab(*self.rect)

    def focus(self):
        return self.gi.focus()

    def in_world(self, img=None) -> bool:
        """粗判是否在世界里：世界里画面丰富、颜色多样；
           菜单/加载界面偏暗或纯色"""
        img = self.shot() if img is None else img
        g = img.astype(np.float32)
        std = g.std()
        # 世界里颜色多样；纯色界面 std 很低
        return std > 28

    def diff(self, a, b):
        return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())

    def is_paused(self, n_probe: int = 1, gap: float = 0.0) -> bool:
        """判断游戏是否停在「游戏菜单」

        判据演进（踩了两次坑）：
          ✗ 「画面连续静止」—— MC 里角色站着不动、视野无动态元素时画面本来就静止，
             会大量误报，导致反复去"救"一个没暂停的游戏，反而真的按出菜单
          ✗ 「中央区域亮像素占比」—— 白天的森林/雪地也会命中
          ✓ 「中央区域的**低饱和度亮像素**占比」—— 菜单按钮是灰白色，
             而游戏画面（森林/水/土地）是高饱和度彩色，二者可分
        """
        img = self.shot()
        f = img.astype(np.float32)
        mx, mn = f.max(axis=2), f.min(axis=2)
        sat = (mx - mn) / np.maximum(mx, 1e-6)
        lum = f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        h, w = lum.shape
        sl = (slice(int(h * 0.30), int(h * 0.72)), slice(int(w * 0.28), int(w * 0.72)))
        btn = (lum[sl] > 150) & (sat[sl] < 0.12)
        return float(btn.mean()) > 0.25

    def ensure_playing(self, max_try: int = 3) -> bool:
        """确保游戏在运行。

        ★ 实测：MC 的暂停菜单里 **Esc 与鼠标点击都无效**（键盘通道是通的，
        但只有键盘导航管用）。有效手段是 Tab 切换按钮焦点 + Enter 激活。
        「回到游戏」在菜单顶部，所以先按若干 Tab 循环到它，再 Enter。
        """
        if not self.is_paused():
            return True

        plans = [
            ["tab", "enter"],                      # 切一格再回车
            ["tab", "tab", "enter"],
            ["enter"],
        ]
        for k, keys in enumerate(plans[:max_try]):
            print(f"  检测到游戏菜单，尝试 {'+'.join(keys)}（{k+1}/{max_try}）…")
            for key in keys:
                self.gi.tap(key, 0.14)
                time.sleep(0.35)
            time.sleep(0.9)
            if not self.is_paused():
                print("  ✓ 已回到游戏")
                return True
        print("  ✗ 仍在菜单里 —— 请人工看一眼")
        return False

    # ------------------------------------------------ 鼠标漂移
    def cursor_offset(self):
        """鼠标相对窗口中心的偏移（比例）。MC 正常捕获鼠标时它应接近 (0,0)"""
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        x, y, w, h = self.rect
        cx, cy = x + w // 2, y + h // 2
        return ((pt.x - cx) / w, (pt.y - cy) / h), (pt.x, pt.y)

    def recenter_mouse(self, tol: float = 0.22) -> bool:
        """鼠标偏离窗口中心太多就归中。

        窗口化模式下 MC 的鼠标捕获可能失效，反复的相对移动会把指针推到边界，
        之后视角行为就失控（实测导致画面被推到仰望天空）。
        """
        (fx, fy), _ = self.cursor_offset()
        if abs(fx) < tol and abs(fy) < tol:
            return False
        x, y, w, h = self.rect
        self.gi.mouse_abs(x + w // 2, y + h // 2)
        time.sleep(0.04)
        return True

    def level_view(self, sweep: int = 900, back_ratio: float = 0.5):
        """把俯仰角拉回大致水平。

        做法：先往一个方向猛推到极限（pitch 到底后不会再变），再回推一半距离。
        不精确，但足以把"仰望天空"这种极端状态拉回可玩范围。
        """
        x, y, w, h = self.rect
        self.gi.mouse_abs(x + w // 2, y + h // 2)
        time.sleep(0.1)
        for _ in range(6):
            self.gi.mouse_move(0, sweep, steps=6)     # 向下推到底
            time.sleep(0.03)
        for _ in range(6):
            self.gi.mouse_move(0, -int(sweep * back_ratio), steps=6)  # 回一半
            time.sleep(0.03)

    def turn(self, dx: int, dy: int = 0, steps: int = 4):
        """转向（相对移动），带鼠标归中保护"""
        self.recenter_mouse()
        self.gi.mouse_move(dx, dy, steps=steps)

    # ------------------------------------------------ 焦点守护
    def set_topmost(self, on: bool = True):
        """把游戏窗口设为置顶，减少被别的窗口遮挡的机会"""
        HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
        SWP_NOMOVE, SWP_NOSIZE = 0x0002, 0x0001
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST if on else HWND_NOTOPMOST,
                            0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)

    def keep_focused(self, topmost: bool = True) -> bool:
        """确保游戏保持前台。

        MC 单人默认 pauseOnLostFocus=true —— 一旦失去焦点就暂停，
        录制会被整段打断（实测两次）。这个函数在每拍调用，发现丢了就抢回来。
        返回 True 表示这一拍焦点是好的。
        """
        if topmost:
            self.set_topmost(True)
        if self.gi.is_focused():
            return True
        user32.ShowWindow(self.hwnd, 9)
        user32.SetForegroundWindow(self.hwnd)
        time.sleep(0.25)
        return self.gi.is_focused()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--shot", action="store_true")
    ap.add_argument("--out", default=r"E:\终末地\_handoff\shots\mc")
    ap.add_argument("--test")
    ap.add_argument("--walk")
    ap.add_argument("--dur", type=float, default=2.0)
    ap.add_argument("--f3", action="store_true")
    ap.add_argument("--turn", type=float, default=0.0, help="鼠标相对移动量（转视角）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mc = MC()
    print(f"MC 窗口 hwnd={mc.hwnd}  位置 {mc.rect[:2]}  尺寸 {mc.rect[2:]}")

    if args.find:
        img = mc.shot()
        print(f"  在世界里？{mc.in_world(img)}   画面 std={img.std():.1f}")
        return

    if args.shot:
        img = mc.shot()
        p = os.path.join(args.out, "mc_now.png")
        Image.fromarray(img).save(p)
        print(f"  → {p}")
        return

    # 以下都需要焦点
    if not mc.gi.is_focused():
        print("切到前台…")
        mc.focus()
        time.sleep(0.5)
    if not mc.gi.is_focused():
        sys.exit("❌ 拿不到焦点")

    if args.f3:
        print("按 F3 打开调试屏幕…")
        mc.gi.tap("f3", 0.1)
        time.sleep(0.8)
        img = mc.shot()
        p = os.path.join(args.out, "mc_f3.png")
        Image.fromarray(img).save(p)
        print(f"  → {p}  （人工看左上角的 XYZ 坐标）")
        return

    if args.turn:
        print(f"鼠标相对移动 {args.turn:+.0f} …")
        mc.gi.mouse_move(int(args.turn), 0, steps=max(1, int(abs(args.turn) / 20)))
        return

    key = args.test or args.walk
    if not key:
        print("给 --find / --shot / --test / --walk / --f3 / --turn 之一")
        return
    dur = args.dur if args.walk else 1.2

    print(f"截图 A…")
    time.sleep(0.6)
    A = mc.shot()
    Image.fromarray(A).save(os.path.join(args.out, "A.png"))

    print(f"按住 {key} {dur} 秒…")
    mc.gi.hold(key, dur)
    time.sleep(0.5)
    B = mc.shot()
    Image.fromarray(B).save(os.path.join(args.out, "B.png"))

    print(f"对照组：不按键，等 1 秒…")
    time.sleep(1.0)
    C = mc.shot()
    Image.fromarray(C).save(os.path.join(args.out, "C.png"))

    d_ab = mc.diff(A, B)
    d_ac = mc.diff(A, C)
    ratio = d_ab / max(d_ac, 1e-6)
    print()
    print("=" * 62)
    print(f"  A(静止) vs B(按键后) 差异 {d_ab:7.2f}")
    print(f"  A(静止) vs C(仍静止) 差异 {d_ac:7.2f}")
    print(f"  比值 {ratio:.2f}×")
    print()
    if ratio > 2.5:
        print(f"  ✅ 按键「{key}」生效 → 角色被控制了")
    elif ratio > 1.5:
        print(f"  ⚠ 有变化但不明显")
    else:
        print(f"  ❌ 没生效")
    print()
    print(f"  A/B/C → {args.out}")


if __name__ == "__main__":
    main()
