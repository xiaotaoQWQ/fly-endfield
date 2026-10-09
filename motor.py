#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motor.py — 动作输出与执行器抽象（方便在 MC / 终末地 之间切换）

分层：
    果蝇连接组 → action_map（神经活动 → 动作）→ **MotorPlan** → Executor（按键）

MotorPlan 是与游戏无关的动作描述：
    move    前进（-1 后退 / 0 停 / 1 前进）
    turn    转向（-1 左 / 0 直 / 1 右）
    jump    跳
    sprint  冲刺

Executor 负责把 MotorPlan 翻成具体按键。目前实现：
    MCExecutor         Minecraft（W / 鼠标 / Space / Ctrl）
    EndfieldExecutor   预留：终末地的键位与输入通道（ACE 会拦注入，见 DECISIONS 第九节）
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


@dataclass
class MotorPlan:
    move: int = 1          # -1 / 0 / 1
    turn: int = 0          # -1 / 0 / 1
    jump: bool = False
    sprint: bool = False
    turn_amount: float = 0.0    # 精细转向量（action_map 的连续值）

    def describe(self) -> str:
        p = []
        if self.move > 0:
            p.append("前进")
        elif self.move < 0:
            p.append("后退")
        if self.turn < 0:
            p.append("左转")
        elif self.turn > 0:
            p.append("右转")
        if self.jump:
            p.append("跳")
        if self.sprint:
            p.append("冲刺")
        return "+".join(p) if p else "停"


# ----------------------------------------------------------------------
class BaseExecutor:
    """执行器基类：把 MotorPlan 变成游戏输入"""

    name = "base"
    #: 按键绑定，子类可覆盖
    KEYS = {}

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._held = set()

    def apply(self, plan: MotorPlan):
        raise NotImplementedError

    def release_all(self):
        raise NotImplementedError

    # 小工具：按下/松开某个逻辑键
    def _press(self, logical: str):
        k = self.KEYS.get(logical)
        if not k:
            return False
        self.gi.key_down(k)
        self._held.add(logical)
        return True

    def _release(self, logical: str):
        k = self.KEYS.get(logical)
        if not k:
            return False
        self.gi.key_up(k)
        self._held.discard(logical)
        return True

    def _sync_hold(self, logical: str, want: bool):
        """按住/松开某个键，只在状态变化时发事件"""
        now = logical in self._held
        if want and not now:
            self._press(logical)
        elif not want and now:
            self._release(logical)


# ----------------------------------------------------------------------
class MCExecutor(BaseExecutor):
    """Minecraft 执行器

    键位（MC 默认）：
        W 前进 / S 后退 / A 左移 / D 右移
        Space 跳跃 / Ctrl 冲刺（潜行是 Shift）
    转向用鼠标相对移动（更接近"转头看"），不用 A/D 平移。
    """

    name = "minecraft"
    KEYS = {"forward": "w", "back": "s", "left": "a", "right": "d",
            "jump": "space", "sprint": "ctrl"}

    def __init__(self, mc, turn_gain: float = 0.3, verbose: bool = True):
        super().__init__(verbose)
        self.mc = mc
        self.gi = mc.gi
        self.turn_gain = turn_gain

    def apply(self, plan: MotorPlan):
        # 前进/后退
        self._sync_hold("forward", plan.move > 0)
        self._sync_hold("back", plan.move < 0)
        # 冲刺（只在前进时有效）
        self._sync_hold("sprint", bool(plan.sprint and plan.move > 0))
        # 转向：鼠标相对移动
        if plan.turn:
            self.mc.turn(int(plan.turn * self.turn_gain * 12), 0, steps=4)
        # 跳跃（点按一次）
        if plan.jump:
            self.gi.tap("space", 0.09)

    def release_all(self):
        for logical in list(self._held):
            try:
                self._release(logical)
            except Exception:
                pass
        try:
            self.gi.release_all()
        except Exception:
            pass


# ----------------------------------------------------------------------
class EndfieldExecutor(BaseExecutor):
    """终末地执行器（占位）

    ⚠ 已知障碍：终末地的 ACE 内核反作弊会丢弃程序注入的输入
      （实测 SendInput 返回成功但 GetAsyncKeyState 无变化，见 DECISIONS 第九节）。
      所以这个执行器目前只是把键位准备好，真正启用需要先解决输入通道
      （硬件模拟 / 驱动级注入）。
    """

    name = "endfield"
    KEYS = {"forward": "w", "back": "s", "left": "a", "right": "d",
            "jump": "space", "sprint": "shift"}   # 终末地的冲刺键待确认

    def __init__(self, gi, turn_gain: float = 0.3, verbose: bool = True):
        super().__init__(verbose)
        self.gi = gi
        self.turn_gain = turn_gain

    def apply(self, plan: MotorPlan):
        self._sync_hold("forward", plan.move > 0)
        self._sync_hold("back", plan.move < 0)
        self._sync_hold("sprint", bool(plan.sprint and plan.move > 0))
        if plan.turn:
            self.gi.mouse_move(int(plan.turn * self.turn_gain * 12), 0, steps=4)
        if plan.jump:
            self.gi.tap("space", 0.09)

    def release_all(self):
        for logical in list(self._held):
            try:
                self._release(logical)
            except Exception:
                pass


# ----------------------------------------------------------------------
class InterceptionExecutor(BaseExecutor):
    """走 Interception 驱动的执行器

    为什么需要：终末地的 ACE 反作弊会丢弃应用层 SendInput
    （实测 SendInput 返回成功但 GetAsyncKeyState 无变化）。
    Interception 是键盘/鼠标**过滤驱动**（键盘类 UpperFilters 里能看到
    `keyboard` 排在 `kbdclass` 前面），事件从驱动栈注入，不带应用层注入标记。

    ⚠ 要求：
      · 驱动已安装并**重启**过（`interception_py.py --check` 能通过）
      · 本进程需要**管理员权限**（create_context 会失败，否则）
    """

    name = "interception"

    def __init__(self, verbose: bool = True, dll: str = None):
        super().__init__(verbose)
        from interception_py import Interception
        self.ic = Interception(dll)
        self.kb = self.ic.keyboard_device()
        self.ms = self.ic.mouse_device()

    def apply(self, plan: MotorPlan):
        # 前进 / 后退
        self._sync_hold("forward", plan.move > 0)
        self._sync_hold("back", plan.move < 0)
        # 冲刺
        self._sync_hold("sprint", bool(plan.sprint and plan.move > 0))
        # 转向：鼠标相对移动
        if plan.turn and self.ms:
            self.ic.mouse_move(int(plan.turn * 12 * 0.3), 0)
        # 跳跃
        if plan.jump:
            self.ic.tap("space", 0.09, self.kb)

    def _sync_hold(self, logical: str, want: bool):
        now = logical in self._held
        if want and not now:
            self.ic.key_down(self.KEYS[logical], self.kb)
            self._held.add(logical)
        elif not want and now:
            self.ic.key_up(self.KEYS[logical], self.kb)
            self._held.discard(logical)

    def release_all(self):
        for logical in list(self._held):
            try:
                self.ic.key_up(self.KEYS[logical], self.kb)
            except Exception:
                pass
        self._held.clear()

    # 键位（终末地待确认；MC 也是这套）
    KEYS = {"forward": "w", "back": "s", "left": "a", "right": "d",
            "jump": "space", "sprint": "shift"}


# ----------------------------------------------------------------------
def build_executor(kind: str, **kw) -> BaseExecutor:
    if kind == "mc":
        return MCExecutor(**kw)
    if kind == "endfield":
        return EndfieldExecutor(**kw)
    if kind == "interception":
        return InterceptionExecutor(**kw)
    raise ValueError(f"未知执行器 {kind}")


EXECUTORS = {"mc": MCExecutor, "endfield": EndfieldExecutor,
             "interception": InterceptionExecutor}
