#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dodge_probe.py — 10 秒钟搞清楚「闪避到底是哪个键」。

为什么要有这个：日志里闪避触发了 16 次，但用户看不到角色闪 ——
可能是 (a) 键不是 Shift (b) 需要配合方向键 (c) 注入没进游戏。
与其再跑 5 分钟战斗，不如**逐个按键试一遍**，人眼看哪个让角色翻滚。

用法：
    dodge_probe.cmd            # 双击（会自动提权）
    python -X utf8 dodge_probe.py --dry   # 只打印，不按键
"""
from __future__ import annotations

import argparse
import sys
import time

import interception_py as icp


def countdown(n=3):
    for i in range(n, 0, -1):
        print(f"   {i}...", flush=True)
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="不真的按键")
    ap.add_argument("--gap", type=float, default=3.0, help="每个试验之间隔几秒")
    a = ap.parse_args()

    print("=" * 60)
    print("  闪避键探测：一共 6 个试验，每个间隔 3 秒")
    print("  ★ 请**盯着角色**，看它有没有翻滚 / 快速位移")
    print("  ★ 记下是第几个试验起作用的，跑完告诉我")
    print("=" * 60)

    ic = None
    if not a.dry:
        ic = icp.Interception(verbose=True)
        kbs = ic.keyboards()
        print(f"\n键盘设备：{kbs}")
        for d in kbs:
            print(f"  dev{d}: {ic.hardware_id(d)}")
    else:
        print("\n[DRY] 不按键，只走流程")

    trials = [
        ("1) 单按 Shift（0.15s）", "shift", None),
        ("2) 按住 W + Shift（0.15s）", "shift", "w"),
        ("3) 单按 Space", "space", None),
        ("4) 单按 Alt", "alt", None),
        ("5) 单按 Ctrl", "ctrl", None),
        ("6) 单按鼠标右键", "mouse_right", None),
    ]

    for name, key, hold in trials:
        print(f"\n>>> {name}")
        countdown(3)
        if ic is None:
            print("   [DRY] 跳过", flush=True)
            continue
        try:
            if hold:
                ic.key_down(hold)
                time.sleep(0.15)
            if key.startswith("mouse_"):
                import win_click as wc
                wc.click(key.split("_", 1)[1])
                print("   已发送鼠标右键", flush=True)
            else:
                r1 = ic.key_down(key)
                time.sleep(0.15)
                r2 = ic.key_up(key)
                print(f"   已发送 {key}（send 返回 {r1}/{r2}）", flush=True)
            if hold:
                time.sleep(0.15)
                ic.key_up(hold)
                print(f"   已松开 {hold}", flush=True)
        except Exception as e:
            print(f"   ⚠ {type(e).__name__}: {e}", flush=True)
        time.sleep(a.gap)

    print("\n" + "=" * 60)
    print("  跑完了。哪一个让角色翻滚/位移了？把编号告诉我。")
    print("  全都没有 → 检查：游戏在前台？是管理员？Interception 装了吗？")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
