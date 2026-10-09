#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""screen_profile.py — 分辨率/窗口自适应层。

为什么需要它：
  这个项目里所有标定值原来都是**写死的内容坐标**（2560×1440 下量的），
  比如"撤离面板在 (24,45) 213×53"。换一台电脑、换一个分辨率、换一个
  窗口大小，全部作废。

  这个模块做三件事：
    ① **实测**画面区（黑边位置），不再靠 `ch = min(h, w*9/16)` 猜
    ② 把所有标定值表示成**画面区的归一化比例**，运行时再换算成像素
    ③ 模板按当前尺度**重采样**（模板是从某个分辨率截的图）

★ 诚实的边界：归一化能解决"同宽高比、不同分辨率"。
  宽高比变了（16:9 → 21:9 → 4:3），HUD 的锚点位置会变，归一化只是近似。
  所以另有 `verify()` —— 每个模板检测都会实测相关值，匹配不上就报告，
  而不是默默用错的位置跑。**宁可告诉你"这个分辨率下没标定"，也不要乱判。**
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TPL_DIR = os.path.join(HERE, "templates")

# 标定时用的参考画面尺寸。所有归一化值都以它为分母。
REF_W, REF_H = 2560, 1440

# ---------------------------------------------------------------- 归一化标定
# (nx, ny, nw, nh) —— 全部是**画面区的比例**，不是像素
REGIONS = {
    # 左上角红环（战斗中变红）
    "marker": (28 / REF_W, 47 / REF_H, 48 / REF_W, 50 / REF_H),
    # 左上角「撤离」面板（副本内才有）
    "evac": (24 / REF_W, 45 / REF_H, 213 / REF_W, 53 / REF_H),
    # 主血条（亮青蓝）
    "hp": (1060 / REF_W, 1324 / REF_H, 441 / REF_W, 12 / REF_H),
    # 受击红晕：画面下部左右边缘
    "vig_l": (0.0, 1140 / REF_H, 12 / REF_W, 130 / REF_H),
    "vig_r": (2548 / REF_W, 1140 / REF_H, 12 / REF_W, 130 / REF_H),
    # ★★ 副本完成 = **左边那条任务条变金**（副本内才有）。
    #
    #    用户先说的是上面『任务完成』会被可视化遮住，换成看左边的击杀数。
    #    录屏确认了两件事：
    #      · 横幅实际写的是「挑战成功」（不是任务完成）
    #      · 它在内容 y 250~430 —— **正好在叠加面板（y 0~300）底下** ✗
    #
    #    而左边这条任务条在内容 y ≈ 380~490 —— 面板下面，不会被遮 ✓
    #
    #        战斗中:  ▮ 高危材料管理室  /  □ 击败所有敌人 4/5
    #        完成时:  ▮ 高危材料管理室 完成  <- ★ 整条变**金黄**
    #                 ✓ 击败所有敌人 5/5
    #
    #    判据用**颜色比例**（不用模板）：实测 428 帧里，
    #    完成帧金色占比约 5.4%，其余帧 0~0.005% —— 0 对 1200，极干净。
    # ★★ 任务条的「击败所有敌人」文字（副本结束时它会消失）。
    #    判据 = **这块文字连续消失 N 秒**，不是颜色。
    #    位置：内容 (124,424) 248×44 附近，搜索区放大到 300×70 留容差。
    "mission": (100 / REF_W, 410 / REF_H, 300 / REF_W, 70 / REF_H),

    # 连携技提示：**队友圆形头像旁边的 E 键徽标**（圆角方块 + 白色 E）
    #
    # ★★ 这块标定重做过。原值 (1476,560) 208×196 是错的 ——
    #    实测扫了 172 帧录制画面：那个区域的最高相关只有 +0.460，
    #    而且抓到的模板根本是**另一个 UI 元素**（圆形徽标，不是 E 键提示）。
    #
    #    真正的 E 徽标在内容坐标 (1468,637) 附近，是 30×28 的圆角方块。
    #    区域取 (1400,590) 200×100 时分离度最好：
    #        真提示 +0.75 ~ +1.000
    #        中位数 +0.333（没有提示时的噪声底）
    #    对比 (1240,400) 700×560 的宽区域：噪声底 +0.437、>=0.45 有 75/172 帧 ✗
    "combo": (1400 / REF_W, 590 / REF_H, 200 / REF_W, 100 / REF_H),
    # 小地图：圆心按画面**高度**归一（它是个圆，横竖必须同一个尺度）
    "minimap": (215 / REF_W, 305 / REF_H, 113 / REF_H),
    # 果蝇眼睛取景带
    "eye": (152 / REF_W, 579 / REF_H, 2256 / REF_W, 586 / REF_H),
}

# 模板：文件名 → **这个模板是在多大的画面上截的**
#
# ★★ 这个数字必须对，否则重采样会把模板缩错、相关值直接归零。
#    实测踩坑：evac_panel 写成 1280×720，于是 213×53 的模板被放大成
#    426×106 —— 跟画面里 213×53 的「撤离」面板完全对不上，
#    相关值从 +1.000 掉到 +0.009。表现是「在副本里被判成副本外」，
#    而且**白填充仍然是 0.734（面板明明在）**，两个判据自相矛盾。
#    判断方法：模板尺寸 ≈ REGIONS[对应区域] 在参考分辨率下的尺寸
#    → 它就是参考分辨率(2560×1440)截的。
TEMPLATES = {
    # evac_panel **铺满**整个 evac 区域 → 尺寸必须正好等于区域尺寸，
    # 这条可以机械校验（见 check_templates）。
    "evac_panel": (2560, 1440),     # 213×53，正好等于 evac 区域的标定尺寸
    # ★★ combo_e 的尺度是 **2560×1440**，不是 1280×720 —— 虽然模板文件
    #    本身是 64×58，看着像 1280 尺度的东西。
    #
    #    原因：ComboMarker.step() 内部会把抓到的画面**降采样 2 倍**
    #    （`g = a[::2, ::2]`），所以它要匹配的模板必须是"内容尺度的一半"。
    #    模板在 1280×720 下是 64×58，而内容区是 2560×1440 → 除以 2 之后
    #    正好还是 64×58 ✓。所以对 Template 来说，它的参考画面就是 2560×1440。
    #
    #    第一版这里写了 (1280, 720)，于是重采样把模板放大成 128×116，
    #    比降采样后的搜索区（约 104×98）**还大** → 撞上 step() 里的
    #    `if g.shape[0] < th` 防线 → **永远返回 False → 永远不按 E**，
    #    而且一声不吭。表现就是"果蝇不会用 E 键"。
    "combo_e": (2560, 1440),        # 30×28，**内容尺度**（step 已不再降采样）
}


def check_occlusion(hwnd: int, prof: "Profile", verbose: bool = True) -> dict:
    """★ 检查关键区域有没有被**别的窗口**压住。

    为什么必须有这个检查：`grab()` 抓的是**屏幕**，不是游戏窗口的 DC
    （实测 PrintWindow 对 DirectX 游戏返回全黑，拿不到）。所以任何压在
    游戏上面的窗口，果蝇都会把它当成"游戏画面"读进去 ——
    而且**不报错、不崩溃、只是那个功能永远不触发**。

    实测踩的坑（2026-10）：聊天窗口盖住了连携技搜索区和血条区域 →
      · E 键永远检测不到提示 → 永远不按连携技
      · 血量读不到 → 自动挨打失效、大招永远不放（血条不可读就不放大招）
    这两个症状看起来毫不相关，根因是同一个：**有个窗口压在上面**。

    做法：取每个区域中心点，看那一点的窗口是不是游戏窗口自己。
    WindowFromPoint 返回的就是**可见的最上层**窗口，正好是这个判据。
    """
    import ctypes
    import ctypes.wintypes as wt
    u = ctypes.windll.user32
    bad = {}
    for name in ("combo", "hp", "evac", "marker", "vig_l", "vig_r",
                 "mission"):
        try:
            r = prof.screen_rect(name)
        except Exception:
            continue
        cx, cy = r[0] + r[2] // 2, r[1] + r[3] // 2
        h = u.WindowFromPoint(wt.POINT(cx, cy))
        if h == hwnd:
            continue
        n = u.GetWindowTextLengthW(h)
        b = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, b, n + 1)
        bad[name] = (b.value or "（无标题窗口）", (cx, cy))
    if verbose:
        if bad:
            print("\n  ⚠⚠⚠ 有窗口压在游戏上面，果蝇看到的是它，不是游戏：")
            for name, (title, pt) in bad.items():
                print(f"      · {name:<7s} 区域被 {title[:38]!r} 挡住")
            print("      → 这些功能会**静默失效**（不报错，只是永远不触发）：")
            if "mission" in bad:
                print("        副本完成检测会失效 → 打完副本不会给那 50 次奖励")
            if "combo" in bad:
                print("        连携技 E 永远检测不到")
            if "hp" in bad:
                print("        血量读不到 → 自动挨打失效、大招不放")
            print("      → 请把这些窗口最小化，或者挪到游戏画面之外。\n")
        else:
            print("  遮挡检查：关键区域都没被挡住 ✓")
    return bad

    """自检：铺满区域的模板，尺寸必须和区域尺寸对得上。

    ★ 这条检查是有来历的：evac_panel 曾被错标成 1280×720 尺度，
      重采样把它从 213×53 放大成 426×106 —— 跟画面里的面板完全对不上，
      相关值从 +1.000 掉到 +0.009。表现是「在副本里被判成副本外」，
      而且**白填充仍然是 0.734**，两个判据自相矛盾，非常难查。
    """
    ok = True
    for name, (rw, rh) in TEMPLATES.items():
        p = os.path.join(TPL_DIR, name + ".npy")
        if not os.path.exists(p):
            print(f"  {name}: 模板文件不存在 ✗")
            ok = False
            continue
        t = np.load(p)
        th, tw = t.shape[:2]
        if name != "evac_panel":        # 只有它铺满区域
            print(f"  {name}: {tw}×{th} @ {rw}×{rh} 尺度（子区域模板，不做尺寸校验）")
            continue
        nx, ny, nw, nh = REGIONS["evac"]
        ew, eh = int(round(nw * rw)), int(round(nh * rh))
        good = (ew, eh) == (tw, th)
        ok = ok and good
        print(f"  {name}: 文件 {tw}×{th} · 声明尺度 {rw}×{rh} · "
              f"按区域推算应为 {ew}×{eh} → {'一致 ✓' if good else '不一致 ✗'}")
    return ok


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _run_around_center(v, th):
    """【已废弃】取包含中心的那一段连续亮区。

    保留是因为它记录了前两版为什么失败 —— 见 measure_content_rect 的注释。
    现在改用"整行/整列纯黑占比"，这个函数不再被调用。
    """
    n = len(v)
    bright = v > th
    c = n // 2
    if not bright[c]:
        off = np.flatnonzero(bright)
        if len(off) == 0:
            return 0, n - 1
        c = int(off[np.argmin(np.abs(off - c))])
    lo = c
    while lo > 0 and bright[lo - 1]:
        lo -= 1
    hi = c
    while hi < n - 1 and bright[hi + 1]:
        hi += 1
    return lo, hi


def measure_content_rect(x: int, y: int, w: int, h: int, grab_fn,
                          thresh: float = 8.0, ds: int = 4,
                          dark_frac: float = 0.5):
    """**实测**画面区（黑边在哪），而不是靠宽高比公式猜。

    ★★★ 判据是「**整行/整列有多黑**」，不是"某一条线上第一个亮点在哪"。

    这个函数改过三版，前两版都被同一个东西打败：**屏幕上压着别的窗口**
    （实测 DSH 窗口同时盖住画面底部和顶部黑边）。

      第一版：正中取一条竖线一条横线找黑边。
              → 副本里那一行左边有 115px 是暗的，画面区算成 (115,80) 2445×1440，
                所有标定区域右移 111px，「撤离」面板抓到暗背景 → 副本内判成副本外。
      第二版：5 条线取中位数。
              → 被盖住的线超过一半时中位数就跟着错，画面区算成 (0,38) 2560×1482。

      第三版（现在）：黑边是**整条整条的黑** —— 它是全局特征，不是局部特征。
              所以按行/列统计"纯黑像素占多少"，占比高的才算黑边。
              别的窗口盖住一部分？剩下的部分照样是纯黑，占比仍然过半。
              游戏内容里几乎不会有整行纯黑（纯黑 RGB<8 很罕见）。

    一次抓整窗（降采样 ds 倍），比来回抓十几条线还快。
    """
    try:
        img = np.asarray(grab_fn(x, y, w, h)).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, :3]
        img = img[::ds, ::ds]
        lum = img.mean(axis=2)
    except Exception:
        return None
    if lum.size == 0:
        return None
    dark = lum < thresh
    rowdark = dark.mean(axis=1)          # 每行纯黑像素占比
    coldark = dark.mean(axis=0)          # 每列纯黑像素占比
    H, W = len(rowdark), len(coldark)

    def _band(v, n):
        """从两端往中间吃掉占比高的部分，返回 (起点, 终点)，闭区间。"""
        lo = 0
        while lo < n - 1 and v[lo] > dark_frac:
            lo += 1
        hi = n - 1
        while hi > lo and v[hi] > dark_frac:
            hi -= 1
        return lo, hi

    t, b = _band(rowdark, H)
    l, r = _band(coldark, W)
    # 剩下的区域太小 → 判据失效，返回 None 让上层退回公式推算
    if (b - t + 1) < H * 0.4 or (r - l + 1) < W * 0.4:
        return None
    # ★★ 对称性校验：画面是**居中**的 —— 上下黑边必须一样宽、左右黑边必须一样宽。
    #    这是几何事实，正好拿来判定"这次测准了没有"。
    #    实测踩坑：DSH 窗口盖住底部黑边时，那一带纯黑占比掉到 50% 以下，
    #    底部黑边没被认出来，画面区算成 (0,80) 2560×1520（下黑边 0px）——
    #    上下黑边一个 80 一个 0，一眼就不对。这种情况**宁可退回公式推算**，
    #    也不要用一个畸形的画面区去换算所有标定区域（会连锁判错）。
    if abs(t - (H - 1 - b)) > max(2, H // 200):
        return None
    if abs(l - (W - 1 - r)) > max(2, W // 200):
        return None
    return (x + l * ds, y + t * ds,
            max(1, (r - l + 1) * ds), max(1, (b - t + 1) * ds))


class Profile:
    """一台机器上的画面几何。所有标定坐标都从它换算。"""

    def __init__(self, x: int, y: int, w: int, h: int,
                 measured: bool = False, verbose: bool = True):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.measured = measured
        # 缩放系数：以参考画面为 1.0。取宽高各自的比值里**较小**的那个，
        # 保证算出来的矩形不会超出画面边界。
        self.sx = self.w / REF_W
        self.sy = self.h / REF_H
        self.s = min(self.sx, self.sy)
        self.aspect = self.w / max(self.h, 1)
        if verbose:
            src = "实测黑边" if measured else "按宽高比推算"
            print(f"画面区 ({self.x},{self.y}) {self.w}×{self.h}  "
                  f"[{src}]  宽高比 {self.aspect:.3f}  "
                  f"缩放 {self.sx:.3f}×{self.sy:.3f}（取 {self.s:.3f}）")
            if abs(self.aspect - REF_W / REF_H) > 0.02:
                print(f"  ⚠ 宽高比和标定时（{REF_W/REF_H:.3f}）差得多 —— "
                      f"HUD 锚点会偏，建议跑一次标定")

    # ------------------------------------------------------------------
    def rect(self, name: str) -> tuple:
        """取一个标定区域的**内容坐标** (x, y, w, h)。

        ⚠ 内容坐标 = 画面区左上角为 (0,0)。抓屏时还要加上画面区的屏幕原点。
        """
        nx, ny, nw, nh = REGIONS[name]
        return (int(round(nx * self.w)), int(round(ny * self.h)),
                max(1, int(round(nw * self.w))), max(1, int(round(nh * self.h))))

    def screen_rect(self, name: str) -> tuple:
        """取一个标定区域的**屏幕绝对坐标**，可以直接喂给 grab()。"""
        cx, cy, cw, chh = self.rect(name)
        return (self.x + cx, self.y + cy, cw, chh)

    def minimap(self) -> tuple:
        """小地图 (圆心x, 圆心y, 半径)，内容坐标。

        半径按**高度**归一 —— 它是个圆，横竖必须同一个尺度，
        按宽度归一会把圆压扁。
        """
        nx, ny, nr = REGIONS["minimap"]
        return (int(round(nx * self.w)), int(round(ny * self.h)),
                max(4, int(round(nr * self.h))))

    # ------------------------------------------------------------------
    def template_path(self, name: str) -> str:
        """模板文件的真实路径。

        ★ 为什么需要：`template()` 会**重采样**到当前分辨率，
          但有些检测器（MissionMarker）要自己决定怎么用 —— 给它路径，
          让它自己读、自己判断缺不缺。
        """
        import os
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "templates", name + ".npy")

    def template(self, name: str, target_rect: tuple = None) -> np.ndarray:
        """读模板，并**重采样到当前分辨率**。

        模板是从某个分辨率（记在 TEMPLATES 里）截的图。换分辨率后要等比放大，
        否则归一化相关会整体偏低、认不出来。

        做法：先算"模板标定时覆盖画面的多大比例"，再按当前画面换回像素。
        比直接用屏幕缩放系数稳 —— 它不受黑边变化影响。
        """
        p = os.path.join(TPL_DIR, name + ".npy")
        if not os.path.exists(p):
            return None
        t = np.load(p)
        ref_w, ref_h = TEMPLATES.get(name, (REF_W, REF_H))
        # 模板是灰度还是彩色都接受
        if t.ndim == 3:
            t = t.astype(np.float32).mean(axis=2)
        tw, th = t.shape[1], t.shape[0]
        if target_rect is not None:
            # 模板应该铺满 target_rect 的同一位置 → 直接按目标尺寸缩放
            ow, oh = max(2, int(target_rect[2])), max(2, int(target_rect[3]))
        else:
            ow = max(2, int(round(tw * self.w / ref_w)))
            oh = max(2, int(round(th * self.h / ref_h)))
        if (ow, oh) == (tw, th):
            return t.astype(np.float32)
        im = Image.fromarray(np.clip(t, 0, 255).astype(np.uint8)).resize(
            (ow, oh), Image.LANCZOS)
        return np.asarray(im).astype(np.float32)

    def describe(self) -> str:
        return (f"{self.w}×{self.h} @({self.x},{self.y}) "
                f"缩放{self.s:.3f} 比例{self.aspect:.3f}"
                + ("" if self.measured else " [未实测黑边]"))


def build(x, y, w, h, grab_fn=None, verbose: bool = True) -> Profile:
    """从窗口矩形建一个 Profile。给了 grab_fn 就实测黑边。"""
    if grab_fn is not None:
        m = measure_content_rect(x, y, w, h, grab_fn)
        if m and m[2] > 64 and m[3] > 64:
            return Profile(*m, measured=True, verbose=verbose)
    # 退回宽高比推算（16:9 内容居中）
    ch = min(h, int(round(w * 9.0 / 16.0)))
    oy = (h - ch) // 2
    return Profile(x, y + oy, w, ch, measured=False, verbose=verbose)


def verify(grab_fn, prof: Profile, verbose: bool = True) -> dict:
    """跑一遍自检：每个标定区域现在长什么样，模板还认不认得出。

    这是这个模块最重要的一个函数 —— 换机器之后**先跑它**，
    它会告诉你哪些功能在这个分辨率下还能用、哪些已经失效。
    比闷头跑然后行为诡异强得多。
    """
    out = {}
    if verbose:
        print("=" * 62)
        print(f"标定自检   {prof.describe()}")
        print("=" * 62)
    # ① 各区域能不能抓到、亮度是否合理
    for name in REGIONS:
        if name == "minimap":
            continue                    # 小地图是(圆心,半径)，不是矩形
        try:
            sr = prof.screen_rect(name)
            a = np.asarray(grab_fn(*sr)).astype(np.float32)
            if a.ndim == 3:
                a = a[:, :, :3]
            lum = a.mean()
            std = a.std()
            out[name] = dict(rect=sr, lum=float(lum), std=float(std))
            if verbose:
                print(f"  {name:<8s} {str(sr):<26s} 亮度 {lum:6.1f}  "
                      f"起伏 {std:6.1f}")
        except Exception as e:
            out[name] = dict(error=str(e))
            if verbose:
                print(f"  {name:<8s} ✗ {e}")
    # ② 模板匹配：拿当前画面跟模板算，看还认得出多少
    if verbose:
        print("\n  模板匹配（拿实时画面比模板，越接近 1 越好）：")
    for tname in TEMPLATES:
        t = prof.template(tname)
        if t is None:
            if verbose:
                print(f"    {tname:<12s} ✗ 模板文件不存在")
            out[f"tpl_{tname}"] = None
            continue
        key = "evac" if "evac" in tname else "combo"
        try:
            sr = prof.screen_rect(key)
            a = np.asarray(grab_fn(*sr)).astype(np.float32)
            if a.ndim == 3:
                a = a[:, :, :3]
            g = a.mean(axis=2)
            g = np.asarray(Image.fromarray(
                np.clip(g, 0, 255).astype(np.uint8)).resize(
                (t.shape[1], t.shape[0]), Image.LANCZOS)).astype(np.float32)
            tt = t - t.mean()
            gg = g - g.mean()
            corr = float((tt * gg).sum()
                         / (np.sqrt((tt * tt).sum()) * np.sqrt((gg * gg).sum())
                            + 1e-6))
            out[f"tpl_{tname}"] = corr
            if verbose:
                hint = ("✅ 绿" if abs(corr) > 0.45 else
                        "（当前画面本来就没有它，相关低是正常的）")
                print(f"    {tname:<12s} {t.shape[1]}×{t.shape[0]}  "
                      f"相关 {corr:+.3f}  {hint}")
        except Exception as e:
            out[f"tpl_{tname}"] = None
            if verbose:
                print(f"    {tname:<12s} ✗ {e}")
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    from ef_shot import grab                       # noqa: E402
    from endfield_walk import find_endfield        # noqa: E402

    w5 = find_endfield()
    if not w5:
        print("没找到终末地窗口")
        sys.exit(1)
    hwnd, wx, wy, ww, wh = w5[:5]
    print(f"窗口 hwnd={hwnd} ({wx},{wy}) {ww}×{wh}")
    p = build(wx, wy, ww, wh, grab_fn=grab)
    print()
    for k in REGIONS:
        if k == "minimap":
            continue                      # 小地图是 (圆心,半径)，不是矩形
        print(f"  {k:<8s} 内容 {p.rect(k)}   屏幕 {p.screen_rect(k)}")
    print(f"  minimap  内容 圆心+半径 {p.minimap()}")
    print()
    print('模板自检：')
    check_templates()
    print()
    verify(grab, p)
