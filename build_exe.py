#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_exe.py — 打包成 Windows exe。

用法：
    python -X utf8 build_exe.py            # 完整打包
    python -X utf8 build_exe.py --clean    # 先清掉旧产物

产物：dist/果蝇全脑驾驶终末地/  整个目录可以直接 zip 发出去。

★ 为什么是 --onedir 而不是 --onefile：
    运行期要带 graph/（连接组本身，206 MB）。onefile 每次启动都要把
    200+ MB 解压到临时目录 —— 启动要十几秒，还容易被杀软盯上。
    onedir 就是一个目录，双击里面的 exe 秒开。

★ 为什么显式列 hidden-import：
    这个项目有一堆"函数内部才 import"的模块（为了启动快、也为了
    某个功能没开时不拖累别人）。PyInstaller 的静态分析**看不到**这些，
    必须一个个写出来，否则打包出来的 exe 一开功能就 ImportError。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "果蝇全脑驾驶终末地"
ENTRY = "fly_app.py"

# 运行期必须跟着走的目录（相对 _handoff/）
DATA_DIRS = [
    ("graph", "graph"),          # 连接组，216 MB，**必须**
    ("templates", "templates"),  # 标定模板
    # ★ Interception 的 DLL 和驱动安装器 —— 只有 1.1 MB，但少了它
    #   启动就 FileNotFoundError（而且报的是一长串路径，很难一眼看懂）。
    #   用户拿到 exe 之后还要用它装驱动：tools/Interception/command line
    #   installer/install-interception.exe
    ("tools", "tools"),
]
# data/ 是 1 GB 的原始连接组，只有 build_graph.py 需要它，**不打包**。

# out/ 里**运行期要读**的文件（都是标定/预处理产物，几十 KB）
#
# ★ 故意**不带 fly_brain_patch.npz** —— 那是果蝇学到的东西（蘑菇体突触补丁）。
#   打包出去应该是"一只什么都没学过的果蝇"，而不是作者那只。
#   它在 out/ 里不存在时会自动从零开始 ✓
OUT_FILES = [
    "optic_map.feather",     # 眼面坐标 → 每个感光细胞的方位角
    "vote_pairs.npz",        # 群体解码用的配对
    "motor_axes.npz",        # 三条运动轴
    "mb_groups.npz",         # 蘑菇体四个分区 → 四个技能
    "odor_pair.npz",         # 气味通道对
    "action_calib.json",     # 动作标定
]


def find_local_modules() -> list:
    """从入口做**传递闭包**，只挑真正用得到的本地模块。

    ★ 第一版是把 `_handoff/*.py` 全塞进 hidden-import（133 个），
      连 `mb_learn.bak` 和一堆一次性诊断脚本都带上了。
      多带的模块不只是占体积 —— PyInstaller 会去分析它们的 import，
      把一个本来就有的依赖网络拖进来，251 MB 里有一大半是这么来的。

    做法：从 fly_app / endfield_fly 开始，扫源码里的 import 语句，
    遇到本地模块就加进队列继续扫。**函数内部 import 也能扫到** ——
    这正是 hidden-import 要解决的问题，静态分析看不到它们。
    """
    local = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
    seen, queue = set(), ["fly_app", "endfield_fly"]
    pat = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w]*)", re.M)
    while queue:
        m = queue.pop()
        if m in seen or m not in local:
            continue
        seen.add(m)
        p = os.path.join(HERE, m + ".py")
        try:
            src = open(p, encoding="utf-8").read()
        except Exception:
            continue
        for dep in pat.findall(src):
            if dep in local and dep not in seen:
                queue.append(dep)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="先删掉 build/ 和 dist/")
    ap.add_argument("--console", action="store_true",
                    help="保留控制台窗口（默认保留 —— 这个程序输出很多，"
                         "出问题时没有控制台就只能靠猜）")
    ap.add_argument("--no-graph", action="store_true",
                    help="不打包 graph/（体积从 ~250MB 降到 ~40MB，"
                         "但用户得自己先跑 build_graph.py）")
    args = ap.parse_args()

    os.chdir(HERE)
    if args.clean:
        for d in ("build", "dist", f"{NAME}.spec"):
            p = os.path.join(HERE, d)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)
        print("  已清理 build/ dist/ *.spec")

    # ★ 打包前先跑模板自检。
    #   尺寸对不上的模板会**静默失效** —— 不报错、不崩溃、不打印，
    #   只是那个功能永远不触发。实测就是这么丢掉 E 键的：
    #   combo_e 被重采样成 128×116，塞不进降采样后 104×97 的搜索区，
    #   ComboMarker.step() 撞上 `if g.shape[0] < th: return False` 直接返回，
    #   表现是"果蝇不会按 E"。
    print("  打包前自检：")
    r0 = subprocess.run([sys.executable, "-X", "utf8", "check_templates.py"],
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    for ln in (r0.stdout or "").strip().splitlines()[-4:]:
        print("   ", ln)
    if r0.returncode != 0:
        print("\n  ✗ 模板自检没过，先修好再打包")
        return 2

    # ★★ argparse 自检：跑一遍 --help。
    #    argparse 会对 help 做一次 help % params —— 字符串里裸的 %
    #    会被当成格式符，直接 ValueError: unsupported format character。
    #    实测这个 bug **只在 exe 上暴露**（源码跑没走到那条路径），
    #    用户双击才看到崩。所以打包前必须跑一遍。
    print("  argparse 自检（--help）：")
    # ★ 跑的是 endfield_fly.py，**不是 fly_app.py**。
    #   fly_app 会先请求提权，--help 根本到不了 argparse ——
    #   第一版就踩了这个：自检"通过"了，但输出只有 34 字节
    #   （那是正在请求管理员权限），等于没测。
    r_h = subprocess.run([sys.executable, "-X", "utf8",
                          "endfield_fly.py", "--help"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    if r_h.returncode != 0:
        print("    ✗ --help 跑不起来，先修好再打包：")
        for ln in (r_h.stderr or "").strip().splitlines()[-6:]:
            print("      ", ln)
        return 3
    _hout = r_h.stdout or ""
    if len(_hout) < 2000:
        print(f"    ✗ --help 只输出 {len(_hout)} 字节，不像是正常的帮助信息")
        print(f"      （跑错脚本了？还是 argparse 提前退出了？）")
        return 3
    print(f"    OK（输出 {len(_hout)} 字节）")

    # ★ 多分辨率检查（可选，要录制数据）。没有录制就跳过。
    print("  多分辨率检查：")
    import glob as _g
    if _g.glob(os.path.join(HERE, "rec", "*", "band_*.npy")):
        r1 = subprocess.run([sys.executable, "-X", "utf8",
                             "check_resolutions.py"],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
        for ln in (r1.stdout or "").strip().splitlines()[-4:]:
            print("   ", ln)
        if r1.returncode != 0:
            print("\n  ⚠ 多分辨率检查没过（继续打包，但请看一眼上面的表）")
    else:
        print("    （没有 rec/ 录制数据，跳过）")

    mods = find_local_modules()
    print(f"  本地模块 {len(mods)} 个：{', '.join(mods)}")

    # ---- 组装 PyInstaller 参数 ----
    sep = ";" if os.name == "nt" else ":"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm",
           "--name", NAME, "--onedir", "--console"]

    for m in mods:
        cmd += ["--hidden-import", m]

    for src, dst in DATA_DIRS:
        if src == "graph" and args.no_graph:
            print("  ⚠ 跳过 graph/（--no-graph）")
            continue
        p = os.path.join(HERE, src)
        if not os.path.isdir(p):
            print(f"  ⚠ 没有 {src}/，跳过")
            continue
        sz = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fs in os.walk(p) for f in fs) / 1e6
        print(f"  带上 {src}/  ({sz:.0f} MB)")
        cmd += ["--add-data", f"{p}{sep}{dst}"]

    # out/ 里那几个标定文件 —— 逐个加，不整个目录端过去
    # （整个 out/ 有 64 MB，还有作者那只果蝇的记忆和一堆日志）
    n_out = 0
    for f in OUT_FILES:
        p = os.path.join(HERE, "out", f)
        if not os.path.exists(p):
            print(f"  ⚠ out/{f} 不存在 —— exe 里这一项会缺，启动时会报错")
            continue
        cmd += ["--add-data", f"{p}{sep}out"]
        n_out += 1
    print(f"  带上 out/ 里 {n_out}/{len(OUT_FILES)} 个标定文件"
          f"（不含 fly_brain_patch.npz —— 果蝇从零开始学）")

    # scipy / pandas 的隐藏依赖
    for m in ("scipy.fft", "scipy.signal", "scipy.ndimage",
              "pandas._libs.tslibs.base"):
        cmd += ["--hidden-import", m]
    # 明确不要的东西（省体积）
    # ★★ **不要排除 unittest / test**：numpy.testing 需要 unittest，
    #    而 scipy._lib._array_api → scipy._external.array_api_compat →
    #    numpy.testing 这条链在 import scipy.sparse 时就会走到。
    #    实测排除它的后果是 exe 一启动就 ModuleNotFoundError: No module
    #    named 'unittest'，而且 PyInstaller 打包时**一声不吭**。
    #    体积省不了多少，坑很大。
    for m in ("matplotlib", "tkinter.test"):
        cmd += ["--exclude-module", m]

    cmd += [ENTRY]

    print(f"\n  开始打包（第一次要几分钟）…\n")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n  ✗ 打包失败（退出码 {r.returncode}）")
        return r.returncode

    out = os.path.join(HERE, "dist", NAME)
    exe = os.path.join(out, NAME + ".exe")
    if not os.path.exists(exe):
        # PyInstaller 对非 ASCII 名字有时会改写，找一下真实的
        for f in os.listdir(os.path.join(HERE, "dist")):
            if f.lower().endswith(".exe") or os.path.isdir(
                    os.path.join(HERE, "dist", f)):
                pass
        print(f"\n  ⚠ 没找到 {exe}，看看 dist/ 里实际生成了什么：")
        for f in os.listdir(os.path.join(HERE, "dist")):
            print("   ", f)
        return 1

    # ★ 使用须知与免责声明：每次打包都刷一份到发布目录里。
    #   放在**发布目录里**而不是只放仓库 —— 用户拿到 zip 解压就看到，
    #   不用去翻 GitHub。写成 utf-8-sig（带 BOM）是为了 Windows 记事本
    #   能正确识别 UTF-8，否则满屏乱码。
    try:
        import io as _io
        _notice = _io.open(os.path.join(HERE, "NOTICE.txt"),
                           encoding="utf-8").read()
        with _io.open(os.path.join(out, "免责声明与使用须知.txt"), "w",
                      encoding="utf-8-sig") as _f:
            _f.write(_notice)
        print("  已放入：免责声明与使用须知.txt")
    except Exception as _e:
        print(f"  ⚠ 免责声明没放进去：{_e}")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(out) for f in fs) / 1e6
    print(f"\n  ✓ 打包完成")
    print(f"     {exe}")
    print(f"     整个目录 {total:.0f} MB")
    print(f"\n  发出去：把 dist/{NAME}/ 整个目录打包成 zip 即可。")
    print(f"  （exe 单独拿出来不行，它依赖同目录下的 _internal/）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
