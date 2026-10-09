#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""interception_py.py — Interception 驱动的 Python 封装（ctypes）

为什么需要它：
    终末地的 ACE 反作弊会丢弃应用层 SendInput 注入的输入
    （实测 SendInput 返回成功但 GetAsyncKeyState 无变化）。
    Interception 是**键盘/鼠标过滤驱动**，事件从驱动栈注入，
    不携带应用层的"注入"标记，反作弊较难区分。

前置条件：
    1. 以**管理员**运行（安装驱动 + 发送输入都需要）
    2. 安装驱动：tools\\Interception\\command line installer\\install-interception.exe /install
    3. 安装后**重启**

用法：
    python interception_py.py --check            # 检查驱动是否就绪
    python interception_py.py --devices          # 列出键盘/鼠标设备
    python interception_py.py --test w           # 发一次 W 键
    python interception_py.py --type "hello"     # 打一串字（测试用）
    python interception_py.py --mouse 100 0      # 鼠标相对移动
"""
from __future__ import annotations

import argparse
import ctypes
import io
import os
import sys
import time
from ctypes import wintypes


HERE = os.path.dirname(os.path.abspath(__file__))
DLL_X64 = os.path.join(HERE, "tools", "Interception", "Interception", "library", "x64", "interception.dll")
DLL_X86 = os.path.join(HERE, "tools", "Interception", "Interception", "library", "x86", "interception.dll")

# ---------------- 常量 ----------------
INTERCEPTION_MAX_KEYBOARD = 10
INTERCEPTION_MAX_MOUSE = 10

INTERCEPTION_KEY_DOWN = 0x00
INTERCEPTION_KEY_UP = 0x01
INTERCEPTION_KEY_E0 = 0x02
INTERCEPTION_KEY_E1 = 0x04
INTERCEPTION_KEY_TERMSRV_SET_LED = 0x08
INTERCEPTION_KEY_TERMSRV_SHADOW = 0x10
INTERCEPTION_KEY_TERMSRV_VKPACKET = 0x20

INTERCEPTION_FILTER_KEY_NONE = 0x0000
INTERCEPTION_FILTER_KEY_ALL = 0xFFFF

INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN = 0x001
INTERCEPTION_MOUSE_LEFT_BUTTON_UP = 0x002
INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN = 0x004
INTERCEPTION_MOUSE_RIGHT_BUTTON_UP = 0x008
INTERCEPTION_MOUSE_MIDDLE_BUTTON_DOWN = 0x010
INTERCEPTION_MOUSE_MIDDLE_BUTTON_UP = 0x020
INTERCEPTION_MOUSE_MOVE_RELATIVE = 0x000
INTERCEPTION_MOUSE_MOVE_ABSOLUTE = 0x001
INTERCEPTION_MOUSE_VIRTUAL_DESKTOP = 0x002
INTERCEPTION_FILTER_MOUSE_NONE = 0x0000
INTERCEPTION_FILTER_MOUSE_ALL = 0xFFFF


# ---------------- 结构 ----------------
class KeyStroke(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort),
                ("state", ctypes.c_ushort),
                ("information", ctypes.c_uint)]


class MouseStroke(ctypes.Structure):
    _fields_ = [("unitId", ctypes.c_ushort),
                ("flags", ctypes.c_ushort),
                ("rolling", ctypes.c_short),
                ("x", ctypes.c_int),
                ("y", ctypes.c_int),
                ("information", ctypes.c_uint)]


class Stroke(ctypes.Union):
    _fields_ = [("key", KeyStroke), ("mouse", MouseStroke)]


# 扫描码（Set 1）
SCAN = {
    "esc": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B,
    "minus": 0x0C, "equal": 0x0D, "backspace": 0x0E, "tab": 0x0F,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14, "y": 0x15,
    "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22, "h": 0x23,
    "j": 0x24, "k": 0x25, "l": 0x26,
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30, "n": 0x31, "m": 0x32,
    "space": 0x39, "lshift": 0x2A, "lctrl": 0x1D, "lalt": 0x38,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F, "f6": 0x40,
    "f7": 0x41, "f8": 0x42, "f9": 0x43, "f10": 0x44, "f11": 0x57, "f12": 0x58,
    "enter": 0x1C, "slash": 0x35, "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
    "numpad0": 0x52,
}
# 扩展键（需要 E0 前缀）
EXT = {"up": 0x48, "down": 0x50, "left": 0x4B, "right": 0x4D,
       "home": 0x47, "end": 0x4F, "rctrl": 0x1D, "ralt": 0x38}


# ---------------- 封装 ----------------
class Interception:
    def __init__(self, dll_path: str = None, verbose: bool = True):
        path = dll_path or (DLL_X64 if ctypes.sizeof(ctypes.c_void_p) == 8 else DLL_X86)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 {path}")
        self.dll_path = path
        self.verbose = verbose
        try:
            self.dll = ctypes.WinDLL(path)
        except OSError as e:
            raise RuntimeError(
                f"加载 interception.dll 失败：{e}\n"
                f"（常见原因：缺 VC 运行库，或 DLL 位数与 Python 不符）") from e

        # 原型声明
        self.dll.interception_create_context.restype = ctypes.c_void_p
        self.dll.interception_destroy_context.argtypes = [ctypes.c_void_p]
        self.dll.interception_is_keyboard.argtypes = [ctypes.c_int]
        self.dll.interception_is_keyboard.restype = ctypes.c_int
        self.dll.interception_is_mouse.argtypes = [ctypes.c_int]
        self.dll.interception_is_mouse.restype = ctypes.c_int
        self.dll.interception_set_filter.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                    ctypes.c_ushort]
        self.dll.interception_get_filter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.interception_get_filter.restype = ctypes.c_ushort
        self.dll.interception_send.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                              ctypes.POINTER(Stroke), ctypes.c_uint]
        self.dll.interception_send.restype = ctypes.c_int
        self.dll.interception_receive.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.POINTER(Stroke), ctypes.c_uint]
        self.dll.interception_receive.restype = ctypes.c_int
        self.dll.interception_get_hardware_id.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                         ctypes.c_void_p, ctypes.c_uint]
        self.dll.interception_get_hardware_id.restype = ctypes.c_uint
        self.dll.interception_wait.argtypes = [ctypes.c_void_p]
        self.dll.interception_wait.restype = ctypes.c_int

        self.ctx = self.dll.interception_create_context()
        if not self.ctx:
            raise RuntimeError(
                "interception_create_context 失败 —— 驱动未安装或未启动。\n"
                "  请以管理员运行：\n"
                f"  {os.path.join(HERE, 'tools', 'Interception', 'Interception', 'command line installer', 'install-interception.exe')} /install\n"
                "  然后重启。")

    # ------------------------------------------------ 设备
    def keyboards(self):
        """键盘设备号是 1..10"""
        return [d for d in range(1, INTERCEPTION_MAX_KEYBOARD + 1)
                if self.dll.interception_is_keyboard(d)]

    def mice(self):
        """鼠标设备号是 11..20

        ★ Interception 的设备号是分段的（键盘 1-10、鼠标 11-20），
        之前按 1..10 查鼠标永远返回空，导致鼠标事件发不出去。
        """
        lo = INTERCEPTION_MAX_KEYBOARD + 1
        return [d for d in range(lo, lo + INTERCEPTION_MAX_MOUSE)
                if self.dll.interception_is_mouse(d)]

    def hardware_id(self, device: int) -> str:
        buf = ctypes.create_unicode_buffer(512)
        n = self.dll.interception_get_hardware_id(self.ctx, device, buf,
                                                  ctypes.sizeof(buf))
        return buf.value if n > 0 else ""

    # ------------------------------------------------ 发送
    def _send_keys(self, device: int, strokes) -> int:
        arr = (Stroke * len(strokes))()
        for i, s in enumerate(strokes):
            if isinstance(s, tuple):
                code, state = s
                arr[i].key = KeyStroke(code=code, state=state, information=0)
        return self.dll.interception_send(self.ctx, device, arr, len(strokes))

    def key_down(self, key: str, device: int = None):
        dev = device or self.keyboard_device()
        k = key.lower()
        if k in EXT:
            return self._send_keys(dev, [(EXT[k], INTERCEPTION_KEY_DOWN | INTERCEPTION_KEY_E0)])
        if k not in SCAN:
            raise KeyError(f"未知按键 {key}")
        return self._send_keys(dev, [(SCAN[k], INTERCEPTION_KEY_DOWN)])

    def key_up(self, key: str, device: int = None):
        dev = device or self.keyboard_device()
        k = key.lower()
        if k in EXT:
            return self._send_keys(dev, [(EXT[k], INTERCEPTION_KEY_UP | INTERCEPTION_KEY_E0)])
        if k not in SCAN:
            raise KeyError(f"未知按键 {key}")
        return self._send_keys(dev, [(SCAN[k], INTERCEPTION_KEY_UP)])

    def tap(self, key: str, dur: float = 0.08, device: int = None):
        self.key_down(key, device)
        time.sleep(dur)
        self.key_up(key, device)

    def hold(self, keys, dur: float, device: int = None):
        if isinstance(keys, str):
            keys = [keys]
        dev = device or self.keyboard_device()
        for k in keys:
            self.key_down(k, dev)
        time.sleep(dur)
        for k in keys:
            self.key_up(k, dev)

    def mouse_move(self, dx: int, dy: int = 0, device: int = None):
        dev = device or self.mouse_device()
        arr = (Stroke * 1)()
        arr[0].mouse = MouseStroke(unitId=0, flags=INTERCEPTION_MOUSE_MOVE_RELATIVE,
                                   rolling=0, x=int(dx), y=int(dy), information=0)
        return self.dll.interception_send(self.ctx, dev, arr, 1)

    def mouse_click(self, button: str = "left", device: int = None):
        dev = device or self.mouse_device()
        dn = (INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN if button == "left"
              else INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN)
        up = (INTERCEPTION_MOUSE_LEFT_BUTTON_UP if button == "left"
              else INTERCEPTION_MOUSE_RIGHT_BUTTON_UP)
        for flag in (dn, up):
            arr = (Stroke * 1)()
            arr[0].mouse = MouseStroke(unitId=0, flags=flag, rolling=0, x=0, y=0,
                                       information=0)
            self.dll.interception_send(self.ctx, dev, arr, 1)
            time.sleep(0.05)

    _kb = None
    _ms = None

    def keyboard_device(self) -> int:
        if self._kb is None:
            ks = self.keyboards()
            if not ks:
                raise RuntimeError("没有可用的键盘设备（驱动没装好？）")
            self._kb = ks[0]
        return self._kb

    def mouse_device(self) -> int:
        if self._ms is None:
            ms = self.mice()
            if not ms:
                raise RuntimeError("没有可用的鼠标设备")
            self._ms = ms[0]
        return self._ms

    def close(self):
        if self.ctx:
            self.dll.interception_destroy_context(self.ctx)
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="检查驱动是否就绪")
    ap.add_argument("--devices", action="store_true")
    ap.add_argument("--test", help="发一次某个键")
    ap.add_argument("--type", help="打一串字")
    ap.add_argument("--mouse", nargs=2, type=int, metavar=("DX", "DY"))
    ap.add_argument("--dll")
    args = ap.parse_args()

    print(f"Python 位数：{ctypes.sizeof(ctypes.c_void_p) * 8} 位")
    print(f"DLL：{args.dll or DLL_X64}")
    try:
        ic = Interception(args.dll)
    except Exception as e:
        print(f"\n❌ {e}")
        return 1

    print("✓ 驱动已就绪，上下文创建成功")
    if args.check:
        ic.close()
        return 0

    ks, ms = ic.keyboards(), ic.mice()
    print(f"\n键盘设备 {len(ks)} 个：{ks}")
    for d in ks:
        print(f"  {d}: {ic.hardware_id(d)}")
    print(f"鼠标设备 {len(ms)} 个：{ms}")
    for d in ms:
        print(f"  {d}: {ic.hardware_id(d)}")

    if args.devices:
        ic.close()
        return 0

    if args.test:
        print(f"\n发送按键 {args.test} …")
        ic.tap(args.test, 0.12)
        print("已发送")
    elif args.type:
        print(f"\n输入：{args.type}")
        for ch in args.type:
            key = {" ": "space"}.get(ch, ch.lower())
            if key in SCAN:
                ic.tap(key, 0.04)
                time.sleep(0.03)
        print("完成")
    elif args.mouse:
        dx, dy = args.mouse
        print(f"\n鼠标相对移动 ({dx}, {dy})")
        ic.mouse_move(dx, dy)
        print("已发送")

    ic.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
