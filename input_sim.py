#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""input_sim.py — 把果蝇的方向决策变成真实游戏输入

这是「执行层」：果蝇输出「左/右」，这里变成键盘/鼠标动作，让角色真的走。

技术选型说明：
  · 用 Windows 的 SendInput（系统级输入注入），**不注入、不挂钩游戏进程**
  · 相比 MaaEnd 那类需要附加进程的方案，这种方式更接近「一个很怪的键盘」
  · 但仍然属于《公平运营声明》禁止的第三方工具操作范畴，用测试号

按键映射（可配置，需按游戏实际键位调整）：
    前进 W / 后退 S / 左移 A / 右移 D
    转视角：鼠标相对移动

用法：
    python input_sim.py --list                 # 列出可用按键名
    python input_sim.py --focus                # 把游戏窗口切到前台
    python input_sim.py --test                 # 测试：按一下 W，看角色是否动
    python input_sim.py --demo                 # 演示：左转→右转→前进
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
import sys
import time

sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                           errors="replace")

user32 = ctypes.windll.user32

# ---------------- SendInput 结构 ----------------
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
PUL = ctypes.POINTER(ULONG_PTR)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", PUL)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", PUL)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

# 虚拟键码
VK = {
    "w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
    "q": 0x51, "e": 0x45, "r": 0x52, "f": 0x46,
    "space": 0x20, "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35,
    "tab": 0x09, "esc": 0x1B, "enter": 0x0D,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "j": 0x4A, "h": 0x48, "t": 0x54, "y": 0x59, "6": 0x36, "7": 0x37, "8": 0x38,
    "9": 0x39, "0": 0x30, "b": 0x42, "c": 0x43, "g": 0x47, "k": 0x4B, "l": 0x4C,
    "m": 0x4D, "n": 0x4E, "o": 0x4F, "p": 0x50, "u": 0x55, "v": 0x56, "x": 0x58, "z": 0x5A,
}

# 硬件扫描码（Set 1）。游戏多用 DirectInput / Raw Input，读的是扫描码，
# 只发虚拟键码常常收不到 —— 所以默认用扫描码发送。
SCAN = {
    "esc": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B,
    "tab": 0x0F, "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14,
    "y": 0x15, "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22, "h": 0x23,
    "j": 0x24, "k": 0x25, "l": 0x26,
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30, "n": 0x31, "m": 0x32,
    "space": 0x39, "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F, "f6": 0x40,
    "f7": 0x41, "f8": 0x42, "f9": 0x43, "f10": 0x44, "f11": 0x57, "f12": 0x58,
    "up": 0x48, "down": 0x50, "left": 0x4B, "right": 0x4D,
    "enter": 0x1C,
}
# 需要扩展前缀（E0）的键
EXTENDED = {"up", "down", "left", "right"}


def _send(inp: INPUT):
    n = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if n != 1:
        err = ctypes.get_last_error()
        raise OSError(f"SendInput 失败（返回 {n}，错误码 {err}）")


def find_hwnd(kw="Endfield"):
    res = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hw, lp):
        if user32.IsWindowVisible(hw):
            n = user32.GetWindowTextLengthW(hw)
            b = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hw, b, n + 1)
            if kw.lower() in b.value.lower():
                res.append(hw)
        return True

    user32.EnumWindows(cb, 0)
    return res[0] if res else None


class GameInput:
    """游戏输入控制器

    method：
      "scan"（默认）发硬件扫描码 —— 兼容 DirectInput / Raw Input 的游戏
      "vk"          发虚拟键码 —— 兼容普通窗口程序
    两种都可用 --method 切换；若一种无效就试另一种。
    """

    def __init__(self, hwnd=None, verbose=True, method: str = "scan"):
        self.hwnd = hwnd or find_hwnd()
        self.verbose = verbose
        self.method = method
        self._held = set()
        if self.hwnd is None:
            raise RuntimeError("没找到 Endfield 窗口")

    # ------------------------------------------------ 窗口
    def is_focused(self) -> bool:
        return user32.GetForegroundWindow() == self.hwnd

    def focus(self) -> bool:
        """把游戏窗口切到前台"""
        if self.is_focused():
            return True
        user32.ShowWindow(self.hwnd, 9)          # SW_RESTORE
        ok = user32.SetForegroundWindow(self.hwnd)
        time.sleep(0.4)
        if self.verbose:
            print(f"  置前结果 {bool(ok)}，当前前台匹配：{self.is_focused()}")
        return self.is_focused()

    # ------------------------------------------------ 键盘
    def _ki(self, key: str, up: bool) -> INPUT:
        k = key.lower()
        if self.method == "scan":
            sc = SCAN.get(k)
            if sc is None:
                raise KeyError(f"未知扫描码 {key}（可用：{sorted(SCAN)}）")
            if k in EXTENDED:
                sc |= 0xE000                     # 扩展键前缀
            return INPUT(type=INPUT_KEYBOARD,
                         u=_INPUTunion(ki=KEYBDINPUT(
                             wVk=0, wScan=sc,
                             dwFlags=KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0),
                             time=0, dwExtraInfo=None)))
        vk = VK.get(k)
        if vk is None:
            raise KeyError(f"未知按键 {key}（可用：{sorted(VK)}）")
        return INPUT(type=INPUT_KEYBOARD,
                     u=_INPUTunion(ki=KEYBDINPUT(
                         wVk=vk, wScan=0,
                         dwFlags=KEYEVENTF_KEYUP if up else 0,
                         time=0, dwExtraInfo=None)))

    def key_down(self, key: str):
        _send(self._ki(key, False))
        self._held.add(key.lower())

    def key_up(self, key: str):
        _send(self._ki(key, True))
        self._held.discard(key.lower())

    def tap(self, key: str, dur: float = 0.08):
        self.key_down(key)
        time.sleep(dur)
        self.key_up(key)

    def hold(self, keys, dur: float):
        """同时按住若干键 dur 秒"""
        if isinstance(keys, str):
            keys = [keys]
        try:
            for k in keys:
                self.key_down(k)
            time.sleep(dur)
        finally:
            for k in keys:
                self.key_up(k)

    def release_all(self):
        for k in list(self._held):
            try:
                self.key_up(k)
            except Exception:
                pass

    # ------------------------------------------------ 鼠标
    def mouse_move(self, dx: int, dy: int = 0, steps: int = 1):
        """相对移动鼠标（转视角）。分步可让游戏更平滑地接收"""
        for _ in range(max(1, steps)):
            inp = INPUT(type=INPUT_MOUSE,
                        u=_INPUTunion(mi=MOUSEINPUT(dx=int(dx), dy=int(dy),
                                                    mouseData=0,
                                                    dwFlags=MOUSEEVENTF_MOVE,
                                                    time=0, dwExtraInfo=None)))
            _send(inp)
            if steps > 1:
                time.sleep(0.01)

    def mouse_abs(self, x: int, y: int):
        """把鼠标移到屏幕绝对像素坐标"""
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)
        nx = int(x * 65535 / max(sw - 1, 1))
        ny = int(y * 65535 / max(sh - 1, 1))
        inp = INPUT(type=INPUT_MOUSE,
                    u=_INPUTunion(mi=MOUSEINPUT(dx=nx, dy=ny, mouseData=0,
                                                dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                                                time=0, dwExtraInfo=None)))
        _send(inp)

    def click(self, x: int = None, y: int = None, button: str = "left"):
        """点击。给坐标则先把鼠标移过去（屏幕绝对像素）"""
        if x is not None and y is not None:
            self.mouse_abs(x, y)
            time.sleep(0.12)
        down = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
        up = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP
        for flag in (down, up):
            inp = INPUT(type=INPUT_MOUSE,
                        u=_INPUTunion(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0,
                                                    dwFlags=flag, time=0, dwExtraInfo=None)))
            _send(inp)
            time.sleep(0.06)

    def click_rect(self, rect, fx: float = 0.5, fy: float = 0.5):
        """点击窗口 rect=(x,y,w,h) 内的相对位置，默认正中"""
        self.click(int(rect[0] + rect[2] * fx), int(rect[1] + rect[3] * fy))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release_all()


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--test", action="store_true", help="按一下 W 看角色是否动")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--wait", type=float, default=3.0, help="执行前等待秒数（留时间切窗口）")
    ap.add_argument("--key", default="w")
    ap.add_argument("--method", choices=["scan", "vk"], default="scan")
    args = ap.parse_args()

    if args.list:
        print("可用按键（扫描码方式）：")
        for k in sorted(SCAN):
            print(f"  {k:>8s}  scan=0x{SCAN[k]:02X}")
        return

    gi = GameInput(method=args.method)
    print(f"游戏窗口 hwnd={gi.hwnd}   发送方式={args.method}")

    if args.focus:
        gi.focus()
        return

    if not gi.is_focused():
        print(f"⚠ 游戏不在前台。请切换到游戏窗口，{args.wait:.0f} 秒后自动执行…")
        for i in range(int(args.wait), 0, -1):
            print(f"  {i}…", end="", flush=True)
            time.sleep(1)
        print()
        gi.focus()
        time.sleep(0.5)

    if not gi.is_focused():
        print("❌ 仍然没有焦点，无法发送输入（游戏必须接收键盘焦点）")
        return

    try:
        if args.test:
            print(f"按一下 {args.key}（0.5 秒）…")
            gi.tap(args.key, 0.5)
            print("已发送。看角色有没有动。")
        elif args.demo:
            print("演示序列：")
            print("  1) 前进 0.6s")
            gi.hold("w", 0.6)
            time.sleep(0.3)
            print("  2) 左转（鼠标左移 300）")
            gi.mouse_move(-300, 0, steps=15)
            time.sleep(0.4)
            print("  3) 右转（鼠标右移 600）")
            gi.mouse_move(600, 0, steps=30)
            time.sleep(0.4)
            print("  4) 回正（鼠标左移 300）")
            gi.mouse_move(-300, 0, steps=15)
            print("演示结束")
        else:
            print("给 --test / --demo / --focus / --list 之一")
    finally:
        gi.release_all()


if __name__ == "__main__":
    main()
