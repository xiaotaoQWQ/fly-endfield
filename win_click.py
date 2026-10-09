#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""win_click.py — 用 **SendInput** 发鼠标点击。

★ 为什么鼠标按键不能走 Interception，键盘却必须走：

    实测（同一台机器、同一个游戏、同一次会话）：

      · 键盘  Interception → 系统看不到？**看得到**，游戏也吃（W 能走）
               SendInput    → 游戏**不吃**（ACE 把注入按键丢了）
      · 鼠标  Interception → interception_send 返回 1（"成功"），
                             **但 GetAsyncKeyState 看不到、游戏也不响应**
               SendInput    → **游戏吃**（角色会挥刀）

    所以这个项目必须**两条路都留着**：
        键盘 → Interception（见 interception_py.py）
        鼠标按键 → SendInput（本模块）

    别把鼠标按键也改成 Interception —— 试过了，10 个鼠标设备全试过，
    11/12/13 返回成功但系统收不到，14~20 直接返回 0。

判别方法（当时怎么确认的）：
    开一个线程高频轮询 GetAsyncKeyState(VK_LBUTTON)，主线程注入。
    先用 SendInput 打一发——抓到了；再用 Interception 打一发——抓不到。
    这就排除了"检测方法失灵"，坐实是 Interception 的鼠标按键没发出去。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time

user32 = ctypes.windll.user32

INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


def _send(flag: int):
    inp = INPUT(type=INPUT_MOUSE,
                u=_INPUTUNION(mi=MOUSEINPUT(0, 0, 0, flag, 0, None)))
    return user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def down(button: str = "left"):
    """只按下，不抬起 —— 配合 up() 做**跨拍长按**。

    ★ 为什么长按必须拆成两次调用：普攻改成按住 1 秒后，
      时长（1000ms）已经**超过决策拍**（400ms）。
      如果在一次调用里 sleep 1 秒，整个主循环会被卡住 —— 走路、
      闪避、挨打判定全部停摆。所以按下和抬起必须落在不同的拍上。
    """
    return _send(MOUSEEVENTF_LEFTDOWN if button == "left"
                 else MOUSEEVENTF_RIGHTDOWN)


def up(button: str = "left"):
    """只抬起。见 down() 的说明。"""
    return _send(MOUSEEVENTF_LEFTUP if button == "left"
                 else MOUSEEVENTF_RIGHTUP)


def click(button: str = "left", hold: float = 0.03):
    """按一下鼠标键（阻塞式）。hold = 按下到抬起之间的秒数。

    ⚠ hold 超过决策拍就会卡住主循环 —— 长按请用 down()/up() 拆到两拍。
    """
    n1 = down(button)
    time.sleep(hold)
    n2 = up(button)
    return n1, n2


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    gap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    print(f"发 {n} 次左键，间隔 {gap}s")
    for i in range(n):
        r = click("left")
        print(f"  {i+1}: SendInput 返回 {r}")
        time.sleep(gap)
    print("完成")
