#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""find_game_window.py — 定位游戏窗口的实际位置和尺寸

背景：用户有两个显示器（主屏 2048×1152，副屏 1707×1067），
      但游戏截图是 2560×1440。需要先确认游戏渲染在哪块屏、窗口多大，
      才能正确换算小地图的坐标。
"""
import ctypes
from ctypes import wintypes
import sys

user32 = ctypes.windll.user32

# 让进程 DPI 感知，否则拿到的坐标是缩放后的
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_AWARE_V2
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


def enum_windows():
    out = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        # 进程名
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if title and w > 100 and h > 100:
            out.append((hwnd, title, pid.value, rect.left, rect.top, w, h))
        return True

    user32.EnumWindows(cb, 0)
    return out


print("=" * 78)
print("所有可见窗口（宽高 > 100）")
print()
print(f"  {'hwnd':>10s} {'pid':>7s} {'位置':>16s} {'尺寸':>12s}  标题")
for hwnd, title, pid, x, y, w, h in sorted(enum_windows(), key=lambda r: -r[5] * r[6]):
    print(f"  {hwnd:>10d} {pid:>7d} ({x:>6d},{y:>6d}) {w:>5d}x{h:<5d}  {title[:44]}")

print()
print("=" * 78)
print("匹配 Endfield 的窗口")
print()
found = [r for r in enum_windows() if "endfield" in r[1].lower()]
for hwnd, title, pid, x, y, w, h in found:
    print(f"  hwnd={hwnd}  pid={pid}  位置 ({x},{y})  尺寸 {w}×{h}  标题「{title}」")
    # 客户区（去掉边框标题栏）
    rc = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    print(f"    客户区 {rc.right - rc.left}×{rc.bottom - rc.top}")
if not found:
    print("  没找到（可能标题不含 Endfield）")

print()
print("=" * 78)
print("显示器")
print()
for i in range(1, 9):
    try:
        ctypes.windll.user32.SetProcessDpiAwareness
    except Exception:
        pass
    import ctypes as C

    class DEVMODE(C.Structure):
        _fields_ = [("dmDeviceName", C.c_wchar * 32), ("dmSpecVersion", C.c_ushort),
                    ("dmDriverVersion", C.c_ushort), ("dmSize", C.c_ushort),
                    ("dmDriverExtra", C.c_ushort), ("dmFields", C.c_ulong),
                    ("dmPositionX", C.c_long), ("dmPositionY", C.c_long),
                    ("dmDisplayOrientation", C.c_ulong), ("dmDisplayFixedOutput", C.c_ulong),
                    ("dmColor", C.c_short), ("dmDuplex", C.c_short), ("dmYResolution", C.c_short),
                    ("dmTTOption", C.c_short), ("dmCollate", C.c_short),
                    ("dmFormName", C.c_wchar * 32), ("dmLogPixels", C.c_ushort),
                    ("dmBitsPerPel", C.c_ulong), ("dmPelsWidth", C.c_ulong),
                    ("dmPelsHeight", C.c_ulong), ("dmDisplayFlags", C.c_ulong),
                    ("dmDisplayFrequency", C.c_ulong)]
    dm = DEVMODE()
    dm.dmSize = C.sizeof(DEVMODE)
    if C.windll.user32.EnumDisplaySettingsW(None, i - 1, C.byref(dm)):
        print(f"  显示器 {i}: {dm.dmPelsWidth}×{dm.dmPelsHeight} @ ({dm.dmPositionX},{dm.dmPositionY})"
              f"  {dm.dmDisplayFrequency}Hz")
    else:
        break
