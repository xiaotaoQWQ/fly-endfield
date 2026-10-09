#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fly_app.py — 打包成 exe 之后的入口。

它做三件事，顺序不能换：

  ① **提权**：Interception 驱动需要管理员。已经提权就跳过；
     没提权就用 `runas` 把自己重新拉起来。**用户拒绝 UAC 时不退出** ——
     提示一句"键鼠注入会失效"继续跑，窗口照常打开。宁可功能残废，
     也不要"双击了没反应、什么提示都没有"。

  ② **检查终末地**：没开游戏就打印提示并等一会儿再退出，
     免得用户对着一个灰掉的启动按钮发呆。

  ③ **跑主程序**，默认进 `--panel`（控制窗口）模式。

注意：这里**不写死任何参数**，只用默认值 + 命令行里传进来的东西。
这样 exe 和直接 `python endfield_fly.py` 的行为完全一致，出一致的问题。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(sys.executable if getattr(
    sys, "frozen", False) else __file__))


def _force_utf8_io():
    """把 stdout/stderr 强制成 UTF-8，并且**永不因为编码报错而崩**。

    ★ 打包成 exe 之后没有 `python -X utf8` 那层保护了，控制台默认是 GBK
      代码页。程序里打印一个 `✓`（U+2713）就：

          UnicodeEncodeError: 'gbk' codec can't encode character '\\u2713'

      —— **一句 print 把整只果蝇弄死**，而且因为提权后的进程有自己的
      控制台，那个 traceback 根本没人看得到。

      errors="replace" 是关键：万一还有别的字符编不出来，顶多显示成 `?`，
      绝不再抛异常。任何程序都不该因为"打不出一个字"而停摆。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass


def _install_file_log():
    """把 stdout/stderr 同时抄一份到 exe 旁边的 fly_app.log。

    ★ 为什么必须有：
      ① 提权后的进程有自己的控制台窗口，一闪就没了，错误信息根本看不到；
      ② 用户报问题只会说"打不开"，而真正的原因在 stdout 里；
      ③ 我调试的时候就是被这个卡住的 —— exe 提权后直接退出，
         进程列表里什么都没有，无从下手。

    日志写在 exe 同目录（放不下就退到 %TEMP%）。写不进去也**不能让程序挂** ——
    日志是辅助，不是功能。
    """
    paths = [os.path.join(HERE, "fly_app.log"),
             os.path.join(os.environ.get("TEMP", "."), "fly_app.log")]
    fh = None
    for p in paths:
        try:
            fh = open(p, "a", encoding="utf-8", errors="replace", buffering=1)
            break
        except Exception:
            continue
    if fh is None:
        return None

    class _Tee:
        def __init__(self, orig, f):
            self.orig, self.f = orig, f

        def write(self, s):
            try:
                self.f.write(s)
            except Exception:
                pass
            try:
                return self.orig.write(s)
            except Exception:
                return len(s or "")

        def flush(self):
            for x in (self.f, self.orig):
                try:
                    x.flush()
                except Exception:
                    pass

        def __getattr__(self, k):
            return getattr(self.orig, k)

    import datetime
    try:
        fh.write(f"\n{'=' * 60}\n"
                 f"  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
                 f"frozen={getattr(sys, 'frozen', False)}  "
                 f"argv={sys.argv[1:]}\n{'=' * 60}\n")
    except Exception:
        pass
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return fh


_force_utf8_io()
_LOG_FH = _install_file_log()

# 打包后 exe 默认带的参数（用户自己在命令行追加的会覆盖不了这些，
# 所以只放"一定要有"的：面板模式 + 那套已经调好的默认值）
DEFAULT_ARGS = [
    "--panel",
    "--hz", "2.5",
    "--motor", "pop",
    "--eye", "main", "--eye-hz", "30",
    "--overlay",
    "--stuck-sec", "3.0", "--escape-grace", "10",
    "--forage", "--evac", "--auto-outcome", "--combo", "--skill-auto", "--ult",
    "--attack-hold", "1.0",
]

GAME = "Endfield.exe"


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def try_elevate() -> bool:
    """把自己以管理员身份重新拉起。返回 True = 新进程已起来，本进程该退。

    ★ 传的是 `sys.argv[1:]`，**不是 sys.argv**。
      第一版写了 `list2cmdline(sys.argv)` —— 那会把 argv[0]（exe 自己的
      路径）当成第一个参数传给新进程，argparse 收到一个多余的位置参数，
      直接报错退出。表现是"UAC 点了、然后什么都没有"，而且因为提权后的
      进程有自己的控制台，错误信息根本看不到。
    """
    try:
        argv = subprocess.list2cmdline(sys.argv[1:])
        r = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, argv, None, 1)
        return int(r) > 32
    except Exception:
        return False


def game_running() -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {GAME}"],
                             capture_output=True, text=True, timeout=15).stdout
        return GAME.lower() in (out or "").lower()
    except Exception:
        return False


def main():
    # ---- ① 提权 ----
    # --no-elevate 供测试/调试用：不起新进程，就在当前权限下跑
    no_elev = "--no-elevate" in sys.argv
    if no_elev:
        sys.argv.remove("--no-elevate")
    if not is_admin() and not no_elev:
        print("  正在请求管理员权限（Interception 输入驱动需要）…", flush=True)
        if try_elevate():
            return 0
        print()
        print("  [!] 管理员权限被拒绝或超时。")
        print("      程序会继续运行，但**键鼠注入不会生效** ——")
        print("      你能看到界面和状态，角色不会动。")
        print()
    elif not is_admin():
        print("  [!] --no-elevate：跳过提权（键鼠注入不会生效）\n")

    # ---- ② 检查游戏 ----
    if not game_running():
        print("  ============================================================")
        print("    没有检测到终末地。")
        print()
        print("    请先打开游戏，再重新运行本程序。")
        print("    （程序本来也会在启动前再检查一次，这里只是提前告诉你）")
        print("  ============================================================")
        time.sleep(6)
        # ★ 不直接退出：万一 tasklist 抽风，让用户自己决定。
        #   控制窗口里的「启动」按钮仍然会是灰的。

    # ---- ③ 跑主程序 ----
    sys.path.insert(0, HERE)
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from endfield_fly import main as fly_main

    # 已经有 --panel 就不再重复加
    extra = [] if any(a == "--panel" for a in sys.argv[1:]) else DEFAULT_ARGS
    sys.argv = [sys.argv[0]] + extra + sys.argv[1:]
    fly_main()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        # ★ 兜底：任何异常都写进日志再退，绝不"闪一下就没了"
        import traceback
        print("\n  ✗ 启动失败：", flush=True)
        traceback.print_exc()
        try:
            print(f"\n  日志：{os.path.join(HERE, 'fly_app.log')}", flush=True)
        except Exception:
            pass
        try:
            input("\n  按回车退出…")
        except Exception:
            import time as _t
            _t.sleep(15)
        sys.exit(1)
