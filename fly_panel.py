#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fly_panel.py — 叠加面板（**顶部横幅版**，1800×340，四列横排）。

为什么从"右侧竖条 680×900"改成"顶部横幅 1800×340"：
    单屏下，画在屏幕上的面板和"主画面口径的眼睛"抢同一块像素。
    右侧竖条会落进眼睛的裁切框里，逼得我们只能遮罩它 ——
    而遮罩会注入方向偏置（实测空场景基线被推偏 0.35、方向偏成 85% 往左）。
    顶部横幅放在 y 160..500，眼睛框在 y 530..1080，**完全不重叠**，
    于是零遮罩、零偏置，视野宽度也不用妥协。
    ★ 两者面积完全相同（612,000 px²），内容一条没砍，只是横过来。

    附带好处：右侧竖条会挡住游戏自己的技能图标和 UI，录像里很难看；
    顶部横幅盖的是天空和远景，而且**角色本体（y 576..1080）完全不被遮挡**。

版面：
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 果蝇全脑·实时神经活动  │ 输入信号·复眼 │ 输出信号·下行 │ 本拍指令    │ 实时指令流 │
    │ 211,577 神经元·26.03M │ 正前左前右前  │ 前进 转向 跳跃 │ 左 ◀ margin │ 150.2s < … │
    │ ┌───────────────┐    │ 左后 右后     │               │ 按键 W S S X │ 151.5s < … │
    │ │   2D 全脑      │    │ 状态行        │               │ 累计计数     │ …          │
    │ └───────────────┘    │ 放电 / 步数   │               │              │            │
    └──────────────────────────────────────────────────────────────────────┘

用法（自检出一张静态 PNG 看排版）：
    python -X utf8 fly_panel.py --test
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

FONT_CN = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
           r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]
FONT_MONO = [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"]

BG = (8, 10, 16)
BG_A = 205
# 战斗条高度（只在战斗模式打开时加到面板底部）——
# 让走路布局一像素不动。
COMBAT_H = 88
# 觅食闭环条（《攻击奖励方案·第二版》）。同样**恒定预留**、关闭时透明 ——
# Overlay.update() 尺寸不符会抛异常，动态高度会在切模式时崩掉。
FORAGE_H = 84
FG = (236, 242, 250)
DIM = (136, 148, 168)
FAINT = (74, 84, 102)
ACCENT = (86, 208, 255)
WARM = (255, 186, 74)
HOT = (255, 92, 96)
OK = (110, 235, 150)
IDLE = (86, 104, 132)
CELL = (14, 17, 24)


def _pick(paths, size):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


class FlyPanel:
    """顶部横幅。默认 1800×340，四列。"""

    def __init__(self, w: int = 1800, h: int = 300,
                 strips=("combat", "forage")):
        """strips = 底部要预留哪几条。

        ★ 按需预留，不是无条件都留：--forage 模式下战斗条永远空着，白占 88px。
          Overlay 尺寸在启动时固定，所以这里一次性定死。
          （运行时用 0 键临时开战斗模式时，没预留的那条会被裁掉。）
        """
        from brain_view import BrainView

        self.w, self.h = int(w), int(h)
        self.strips = tuple(strips)
        self.c_h = COMBAT_H if "combat" in self.strips else 0
        self.f_h = FORAGE_H if "forage" in self.strips else 0
        self.extra_h = self.c_h + self.f_h
        self.pad = 12
        # 四列几何
        self.c1 = (12, 452)       # 标题 + 全脑
        self.c2 = (464, 948)      # 输入 / 输出 / 状态 / 放电
        self.c3 = (960, 1330)     # 指令 / 按键 / 计数
        self.c4 = (1342, 1788)    # 实时指令流
        self.brain_w = self.c1[1] - self.c1[0]
        self.brain_h = 232

        self.bv = BrainView(w=self.brain_w, h=self.brain_h)

        self.f_title = _pick(FONT_CN, 19)
        self.f_sub = _pick(FONT_CN, 10)
        self.f_lab = _pick(FONT_CN, 12)
        self.f_val = _pick(FONT_CN, 16)
        self.f_big = _pick(FONT_CN, 34)
        self.f_key = _pick(FONT_CN, 12)
        self.f_mono = _pick(FONT_MONO, 12)
        self.f_log = _pick(FONT_CN, 12)
        self.f_tiny = _pick(FONT_CN, 10)

    # ---------------------------------------------------------------- 组件
    def _bar_center(self, d, x, y, w, h, signed, col):
        d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=(28, 33, 45))
        cx = x + w / 2
        d.line([cx, y - 2, cx, y + h + 2], fill=(66, 78, 98), width=1)
        v = max(-1.0, min(1.0, float(signed)))
        x0, x1 = (cx, cx + v * w / 2) if v >= 0 else (cx + v * w / 2, cx)
        if x1 - x0 > 1:
            d.rounded_rectangle([x0, y, x1, y + h], radius=h // 2, fill=col)

    def _tri(self, d, cx, cy, size, direction, col):
        s = size
        if direction > 0:
            pts = [(cx - s * 0.5, cy - s), (cx - s * 0.5, cy + s), (cx + s * 0.8, cy)]
        else:
            pts = [(cx + s * 0.5, cy - s), (cx + s * 0.5, cy + s), (cx - s * 0.8, cy)]
        d.polygon(pts, fill=col)

    def _key(self, d, x, y, w, h, key, label, on, col):
        d.rounded_rectangle([x, y, x + w, y + h], radius=5,
                            fill=col if on else (26, 31, 42),
                            outline=col if on else (52, 62, 80), width=2)
        d.text((x + w / 2, y + h * 0.34), key, font=self.f_key,
               fill=(10, 12, 18) if on else FG, anchor="mm")
        d.text((x + w / 2, y + h * 0.74), label, font=self.f_tiny,
               fill=(10, 12, 18) if on else DIM, anchor="mm")

    def _stat(self, d, x, y, label, value, unit="", col=FG):
        d.text((x, y), label, font=self.f_tiny, fill=DIM)
        d.text((x, y + 12), value, font=self.f_val, fill=col)
        if unit:
            d.text((x + d.textlength(value, font=self.f_val) + 3, y + 17),
                   unit, font=self.f_tiny, fill=DIM)

    # -------------------------------------------------- 输入信号（复眼扇区）
    def _row_input(self, d, x, y, w, h, sectors, wins):
        d.rectangle([x, y, x + w, y + h], fill=CELL)
        # ★ 标题和扇区名会撞：标题必须短，且扇区起点要留够。
        #   第一版用「输入信号 · 复眼」+ 起点 x+74，渲染出来是「复眼正前」叠字。
        d.text((x + 8, y + 4), "输入·复眼", font=self.f_tiny, fill=DIM)
        d.text((x + 8, y + h - 16), f"{int(sum(wins))} 拍", font=self.f_tiny, fill=FAINT)
        if not sectors:
            return
        mx = max([v for _, v in sectors] + [1e-6])
        n = len(sectors)
        bx0 = x + 62
        bw = (w - 62 - 8) / n
        for i, (nm, v) in enumerate(sectors):
            bx = bx0 + i * bw
            top = v >= mx - 1e-9
            col = WARM if top else IDLE
            d.text((bx, y + 4), nm, font=self.f_tiny, fill=col)
            d.text((bx + bw - 12, y + 4), f"{wins[i]}", font=self.f_tiny,
                   fill=(WARM if wins[i] else FAINT), anchor="ra")
            byy = y + 22
            d.rounded_rectangle([bx, byy, bx + bw - 12, byy + 7], radius=3,
                                fill=(28, 33, 45))
            fw = (v / mx) * (bw - 12)
            if fw > 1:
                d.rounded_rectangle([bx, byy, bx + fw, byy + 7], radius=3, fill=col)
            d.text((bx, byy + 10), f"{v:.4f}", font=self.f_tiny,
                   fill=col if top else DIM)

    # ------------------------------------------- 输出信号（下行神经元通道）
    def _row_output(self, d, x, y, w, h, chans):
        d.rectangle([x, y, x + w, y + h], fill=CELL)
        d.text((x + 8, y + 4), "输出·下行神经元", font=self.f_tiny, fill=DIM)
        d.text((x + 8, y + h - 16), "刻度=阈值", font=self.f_tiny, fill=FAINT)
        if not chans:
            return
        n = len(chans)
        bx0 = x + 112          # 让开标题（第一版 x+108 会和「前进」叠字）
        bw = (w - 112 - 8) / n
        for i, c in enumerate(chans):
            bx = bx0 + i * bw
            col = (OK if c["over"] else IDLE) if c["key"] != "turn" else \
                  (ACCENT if c["over"] else IDLE)
            d.text((bx, y + 4), c["name"], font=self.f_tiny, fill=col)
            span = c["thr"] - c["q50"]
            norm = (c["v"] - c["q50"]) / span if abs(span) > 1e-9 else 0.0
            byy = y + 22
            bwid = bw - 62
            d.rounded_rectangle([bx, byy, bx + bwid, byy + 7], radius=3,
                                fill=(28, 33, 45))
            fw = max(0.0, min(1.0, norm / 2.0)) * bwid
            if fw > 1:
                d.rounded_rectangle([bx, byy, bx + fw, byy + 7], radius=3, fill=col)
            d.line([bx + bwid / 2, byy - 2, bx + bwid / 2, byy + 9],
                   fill=(220, 228, 240), width=1)
            d.text((bx + bwid + 3, byy + 8), f"{c['v']:+.4f}", font=self.f_tiny,
                   fill=col)

    # ---------------------------------------------------------------- 主渲染
    def render(self, st: dict) -> np.ndarray:
        # ★ 高度**恒定**为 h + COMBAT_H，战斗关闭时底部那 88 行保持透明。
        #   为什么不做成动态高度：Overlay.update() 在尺寸不符时会抛异常，
        #   而运行时按 0 切换战斗模式会立刻改变面板高度 —— 用恒定尺寸 + 透明区
        #   就不需要重建窗口。
        combat_on = bool(st.get("combat_on"))
        forage_on = bool(st.get("forage_on"))
        W, H = self.w, self.h + self.extra_h
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # ⚠ rectangle 包含终点 —— 写 self.h 会多填一行，把透明区戳破一个像素
        d.rectangle([0, 0, W, self.h - 1], fill=(*BG, BG_A))

        # ================= 第 1 列：标题 + 全脑 =================
        x0, x1 = self.c1
        d.text((x0, 12), "果蝇全脑 · 实时神经活动", font=self.f_title, fill=FG)
        running = bool(st.get("running", True))
        d.ellipse([x1 - 78, 18, x1 - 68, 28], fill=OK if running else HOT)
        d.text((x1 - 62, 14), "运行中" if running else "已停止",
               font=self.f_lab, fill=OK if running else HOT)
        d.text((x0, 36), f"MaleCNS v1.0 · {self.bv.n_all:,} 神经元 · "
                         f"26,028,386 突触 · 胞体 {self.bv.n:,} · "
                         f"{st.get('exec', 'Interception')}",
               font=self.f_sub, fill=DIM)

        by = 54
        d.rectangle([x0, by, x1, by + self.brain_h], fill=(4, 5, 10))
        brain = self.bv.render(np.asarray(st["r"], dtype=np.float32),
                               yaw=0.0, pitch=float(st.get("pitch", 0.30)),
                               zoom=float(st.get("zoom", 2.60)))
        img.paste(Image.fromarray(brain), (x0, by))
        d.rectangle([x0, by, x1, by + self.brain_h], outline=(48, 58, 76), width=1)
        from brain_view import CMAP_POS, CMAP_RGB
        lw2, lx, lyy = 132, x1 - 140, by + 6
        ramp = np.zeros((8, lw2, 4), dtype=np.uint8)
        ramp[..., 3] = 235
        xs = np.linspace(0, 1, lw2)
        for c in range(3):
            ramp[..., c] = (np.interp(xs, CMAP_POS, CMAP_RGB[:, c]) * 255).astype(np.uint8)
        img.paste(Image.fromarray(ramp), (lx, lyy))
        d.text((lx, lyy + 10), "静息", font=self.f_tiny, fill=DIM)
        d.text((lx + lw2, lyy + 10), "最活跃", font=self.f_tiny, fill=DIM, anchor="ra")

        # ================= 第 2 列：信号 + 状态 =================
        cx0, cx1 = self.c2
        cw = cx1 - cx0
        y = 14
        self._row_input(d, cx0, y, cw, 74, st.get("in_sectors") or [],
                        st.get("sector_wins") or [0] * 5)
        y += 78
        self._row_output(d, cx0, y, cw, 74, st.get("out_chans") or [])
        y += 78
        d.rectangle([cx0, y, cx1, y + 74], fill=CELL)
        q = cw / 4
        self._stat(d, cx0 + 8, y + 8, "朝向", f"{float(st.get('heading', 0)):.0f}°")
        self._stat(d, cx0 + q + 8, y + 8, "速度",
                   f"{float(st.get('speed', 0)):.2f}", "px/s")
        self._stat(d, cx0 + 2 * q + 8, y + 8, "感光驱动",
                   f"{float(st.get('drive', 0)):.4f}")
        self._stat(d, cx0 + 3 * q + 8, y + 8, "累计位移",
                   f"{float(st.get('dist', 0)):.0f}", "px")
        y += 78
        d.rectangle([cx0, y, cx1, y + 74], fill=CELL)
        t = cw / 3
        self._stat(d, cx0 + 8, y + 8, "累计放电",
                   f"{int(st.get('spikes_total', 0)):,}", "活跃神经元·拍", col=WARM)
        self._stat(d, cx0 + t + 8, y + 8, "模拟步数",
                   f"{int(st.get('steps_total', 0)):,}",
                   f"={st.get('tick', 0)}拍×{st.get('steps', 10)}", col=ACCENT)
        self._stat(d, cx0 + 2 * t + 8, y + 8, "本拍放电",
                   f"{int(st.get('spikes', 0)):,}", "μ+2σ 以上",
                   col=OK if st.get("spikes", 0) else DIM)

        # ================= 第 3 列：本拍指令 + 按键 + 计数 =================
        gx0, gx1 = self.c3
        gw = gx1 - gx0
        dirn = st.get("direction", "未定")
        margin = float(st.get("margin", 0.0))
        dxs = -1 if dirn == "左" else (1 if dirn == "右" else 0)
        col = DIM if dxs == 0 else (ACCENT if dxs > 0 else WARM)
        md = int(st.get("turn_dx", 0))
        zt = float(st.get("z_thrust") or 0)
        zs = float(st.get("z_strafe") or 0)
        zr = float(st.get("z_turn") or 0)
        pop = ("z_thrust" in st)

        y = 14
        d.rectangle([gx0, y, gx1, y + 134], fill=CELL)
        d.text((gx0 + 8, y + 4),
               "本拍动作 · 群体解码三条轴" if pop else "本拍指令",
               font=self.f_tiny, fill=DIM)
        if pop:
            d.text((gx0 + 8, y + 18), st.get("action", ""), font=self.f_lab, fill=FG)
            for i, (lab, zv, cc) in enumerate((
                    ("推力 W/S", zt, OK), ("横移 A/D", zs, ACCENT),
                    ("转身 鼠标", zr, WARM))):
                yy = y + 40 + i * 28
                d.text((gx0 + 8, yy), lab, font=self.f_tiny, fill=DIM)
                d.text((gx0 + 72, yy - 2), f"{zv:+5.2f}", font=self.f_mono, fill=cc)
                self._bar_center(d, gx0 + 128, yy + 1, gw - 138, 9, zv / 2.5, cc)
            d.text((gx0 + 8, y + 118), f"鼠标 X {md:+d} px", font=self.f_mono, fill=cc)
        else:
            d.text((gx0 + 8, y + 18), dirn, font=self.f_big, fill=col)
            if dxs:
                self._tri(d, gx0 + 104, y + 38, 14, dxs, col)
            d.text((gx0 + 8, y + 62), f"margin {margin:+.5f}",
                   font=self.f_tiny, fill=DIM)
            self._bar_center(d, gx0 + 8, y + 78, gw - 16, 10, margin / 0.03, col)
            d.text((gx0 + 8, y + 94), f"鼠标 X {md:+d} px", font=self.f_mono, fill=col)
            d.text((gx0 + 8, y + 112), st.get("action", ""), font=self.f_lab, fill=FG)

        y = 156
        d.rectangle([gx0, y, gx1, y + 66], fill=CELL)
        d.text((gx0 + 8, y + 4), "按键（亮=已按下）", font=self.f_tiny, fill=DIM)
        kw, gx = (gw - 20 - 3 * 6) / 4, 6
        if pop:
            # ★ 6 个键：WASD + Space + Shift（用户要求把跳跃/冲刺也显示出来）
            kw, gx = (gw - 20 - 5 * 6) / 6, 6
            hits = [("W", "前进", st.get("key_w"), OK),
                    ("A", "左移", st.get("key_a"), ACCENT),
                    ("S", "后退", st.get("key_s"), WARM),
                    ("D", "右移", st.get("key_d"), ACCENT),
                    ("Space", "跳跃", st.get("jump"), WARM),
                    ("Shift", "冲刺", st.get("sprint"), ACCENT)]
        else:
            hits = [("W", "前进", st.get("move", True), OK),
                    ("Space", "跳跃", st.get("jump"), WARM),
                    ("Shift", "冲刺", st.get("sprint"), ACCENT),
                    ("X", "转向", abs(md) > 0, col if dxs else DIM)]
        for i, (k, lab, on, c) in enumerate(hits):
            self._key(d, gx0 + 8 + i * (kw + gx), y + 18, kw, 42, k, lab,
                      bool(on), c)

        y = 230
        d.rectangle([gx0, y, gx1, y + 98], fill=CELL)
        d.text((gx0 + 8, y + 4), "累计", font=self.f_tiny, fill=DIM)
        cnt = st.get("counts", {})
        for i, (lab, v, c) in enumerate([
                ("左转", cnt.get("left", 0), WARM), ("右转", cnt.get("right", 0), ACCENT),
                ("跳跃", cnt.get("jump", 0), WARM),
                ("冲刺", cnt.get("sprint", 0), ACCENT),
                ("撞墙", cnt.get("esc", 0), HOT),
                ("卡住", cnt.get("stuck", 0), HOT)]):
            cx = gx0 + 8 + (i % 3) * (gw - 16) / 3
            cy = y + 22 + (i // 3) * 32
            d.text((cx, cy), lab, font=self.f_tiny, fill=DIM)
            d.text((cx + 30, cy - 3), f"{v}", font=self.f_lab, fill=c)

        # ================= 第 4 列：实时指令流 =================
        lx0, lx1 = self.c4
        d.text((lx0, 14), "实时指令流（果蝇决定 → 程序执行）",
               font=self.f_lab, fill=DIM)
        d.text((lx1, 16), f"{st.get('tick', 0)} 拍", font=self.f_tiny,
               fill=FAINT, anchor="ra")

        # ★★ 果蝇当前状态**单独一块**，不混进指令流（用户要求）。
        #   指令流回答的是"它这一拍做了什么"，状态回答的是"它现在怎么样"——
        #   两件事混在一起，观众分不清哪行是刚发生的、哪行是持续的。
        #   实测原来那一列里塞着 h=0.72 / 饿 / 逃 这种持续量，
        #   一屏 8 行里有一半不是"指令"。
        sy = 36
        sh = 74
        d.rectangle([lx0, sy, lx1, sy + sh], fill=(14, 16, 24),
                    outline=(52, 46, 34))
        d.text((lx0 + 8, sy + 4), "果蝇当前状态", font=self.f_tiny, fill=WARM)
        _st_pairs = [
            ("方向", str(st.get("direction", "-"))),
            ("朝向", f"{float(st.get('heading') or 0):.0f}°"),
            ("速度", f"{float(st.get('speed') or 0):.1f} px/s"),
            ("卡住", "是" if st.get("stuck") else "否"),
            ("饥饿 h", f"{float(st.get('f_h') or 0):.2f}"),
            ("出手阈值", f"{float(st.get('f_thr') or 0):.2f}"),
            ("驱动力", f"{float(st.get('f_urge') or 0):.2f}"),
            ("漂移", f"{float(st.get('f_drift') or 0):.0f}"),
        ]
        for k, (lab, val) in enumerate(_st_pairs):
            cx = lx0 + 10 + (k % 4) * 118
            cy = sy + 22 + (k // 4) * 22
            d.text((cx, cy), lab, font=self.f_tiny, fill=DIM)
            d.text((cx + 54, cy), val, font=self.f_tiny, fill=FG)

        ly = sy + sh + 8
        # ★ 走路布局的高度必须用 self.h，不能用加了战斗条的 H ——
        #   否则第 4 列日志框会跟着长下去，战斗关闭时底部就不是透明区了。
        box_h = self.h - ly - self.pad
        d.rectangle([lx0, ly, lx1, ly + box_h], fill=(11, 13, 19),
                    outline=(44, 52, 68))
        logs = st.get("log") or []
        lh = 22
        maxn = max(1, (box_h - 8) // lh)
        for i, e in enumerate(logs[-maxn:]):
            self._logline(d, lx0 + 8, ly + 5 + i * lh, e)

        # ================= 战斗条（只在战斗模式打开时）=================
        if combat_on and self.c_h:
            self._combat(d, self.h, st)
        # ================= 觅食闭环条 =================
        if forage_on and self.f_h:
            self._forage(d, self.h + self.c_h, st)

        return np.asarray(img)

    # ------------------------------------------------------------------
    def _forage(self, d, y0: int, st: dict):
        """觅食闭环（《果蝇攻击奖励方案·第二版》§三 三层 / §五 饱足感）。

        要让观众看出这是**一只在觅食的果蝇**，不是一个自动打怪脚本，所以画：
          · 饥饿值 h —— 循环的节拍器（饿→找→吃→饱→停→又饿）
          · 气味有没有（阶段一：红环布尔）
          · MBON 响应 + 驱动力 + 阈值线 —— 出手是它自己算出来的
          · 奖赏/惩罚次数 + 权重漂移 —— 学没学、忘了多少
        """
        d.rectangle([0, y0, self.w, y0 + FORAGE_H], fill=(14, 18, 12))
        yy = y0 + 6

        d.text((12, yy), "觅食闭环", font=self.f_lab, fill=FG)
        d.ellipse([86, yy + 3, 96, yy + 13], fill=OK)
        d.text((100, yy - 1), "ON", font=self.f_lab, fill=OK)
        # ★ 「吃饱」倒计时（任务完成后 10 秒不涨饥饿）
        _hold = float(st.get("f_hold") or 0)
        if _hold > 0:
            d.rectangle([272, yy - 2, 380, yy + 15], fill=(120, 96, 20))
            d.text((277, yy - 1), f"已吃饱 {_hold:4.1f}s", font=self.f_tiny,
                   fill=(20, 16, 4))

        # 气味状态
        if st.get("f_odor"):
            d.rectangle([132, yy - 2, 206, yy + 15], fill=ACCENT)
            d.text((137, yy - 1), "闻到气味", font=self.f_tiny, fill=(8, 12, 16))
        else:
            d.text((132, yy), "无气味", font=self.f_tiny, fill=FAINT)
        # 副本判定：左上角「撤离」面板在不在（在 = 战斗状态开着）
        if st.get("f_evac"):
            d.rectangle([216, yy - 2, 268, yy + 15], fill=WARM)
            d.text((221, yy - 1), "副本内", font=self.f_tiny, fill=(12, 8, 8))
        else:
            d.text((216, yy), "副本外", font=self.f_tiny, fill=FAINT)

        # --- 饥饿值：循环的节拍器 ---
        h = float(st.get("f_hunger") or 0.0)
        d.text((12, yy + 24), "饥饿值 h", font=self.f_tiny, fill=DIM)
        d.text((72, yy + 22), f"{h:.2f}", font=self.f_mono,
               fill=HOT if h > 0.5 else DIM)
        d.rectangle([116, yy + 24, 296, yy + 36], fill=(32, 34, 28))
        d.rectangle([116, yy + 24, 116 + 180 * min(max(h, 0), 1), yy + 36],
                    fill=HOT if h > 0.35 else (90, 96, 80))
        d.text((12, yy + 46), "饿就上，饱就停", font=self.f_tiny, fill=FAINT)

        # --- 驱动力 vs 阈值 ---
        urge = float(st.get("f_urge") or 0.0)
        thr = float(st.get("f_thr") or 0.35)
        atk = bool(st.get("f_attack"))
        x = 330
        d.text((x, yy), "攻击驱动力（MBON × 饥饿）", font=self.f_tiny, fill=DIM)
        d.text((x, yy + 18), f"{urge:.3f}", font=self.f_mono,
               fill=HOT if atk else DIM)
        d.rectangle([x + 70, yy + 20, x + 310, yy + 34], fill=(32, 34, 28))
        d.rectangle([x + 70, yy + 20,
                     x + 70 + 240 * min(max(urge / 1.0, 0), 1), yy + 34],
                    fill=HOT if atk else (80, 90, 76))
        tx = x + 70 + 240 * min(max(thr, 0), 1)
        d.line([tx, yy + 17, tx, yy + 37], fill=FG, width=1)   # 阈值刻线
        if atk:
            d.rectangle([x + 330, yy + 18, x + 382, yy + 35], fill=HOT)
            d.text((x + 336, yy + 19), "进食", font=self.f_tiny, fill=(12, 8, 8))

        # --- 学习状态 ---
        z = 760
        d.text((z, yy), "学习（蘑菇体 KC→MBON 三因子）", font=self.f_tiny, fill=DIM)
        d.text((z, yy + 18),
               f"奖赏 {st.get('f_reward', 0)}   惩罚 {st.get('f_punish', 0)}",
               font=self.f_tiny, fill=OK)
        d.text((z, yy + 36),
               f"权重漂移 {float(st.get('f_drift') or 0):.0f}   "
               f"MBON {float(st.get('f_mbon') or 0):+.3f}",
               font=self.f_tiny, fill=WARM)
        d.text((z, yy + 54), f"通道 {st.get('f_ch', '')}", font=self.f_tiny,
               fill=FAINT)

        # --- 提示 ---
        # ★ O/P 手动键已去掉，所以这里**不再显示按键说明**。
        #   全自动那套还没开时也只说"自动奖惩未开"，不提手按 ——
        #   否则画面上会出现已经不存在的东西。
        if st.get("f_auto"):
            hp = st.get("f_hp")
            vg = st.get("f_vig")
            hps = f"{hp*100:.0f}%" if isinstance(hp, float) and hp == hp else "—"
            vgs = f"{vg:+.0f}" if isinstance(vg, float) and vg == vg else "—"
            d.text((1120, yy + 18), f"★全自动 血条 {hps}  受击红晕 {vgs}",
                   font=self.f_tiny, fill=OK)
            d.text((1120, yy + 36), "打死/挨打由画面自己认 —— 不用手按",
                   font=self.f_tiny, fill=DIM)
        else:
            d.text((1120, yy + 18), "自动奖惩未开", font=self.f_tiny, fill=FAINT)
            d.text((1120, yy + 36), "加 --auto-outcome 让画面自己认打死/挨打",
                   font=self.f_tiny, fill=FAINT)

    # ------------------------------------------------------------------
    def _combat(self, d, y0: int, st: dict):
        """战斗归因（交接文档 §八）。

        要能让观众看出**这一下是果蝇决定的，还是惯性/脚本给的**，所以：
          · 两条通道各画**变化率**（不是活动值 —— 变化率才是真正在按键的量）
          · 每次按键标**来源**：果蝇触发 / 被冷却拦下 / 上层脚本
          · 画**犹豫度**：两条通道的活动差。差值小 → 抖动灰（它在犹豫），
            差值大 → 实心。★ 饱和的时候全是实心，瞟一眼就知道出事了。
        """
        W = self.w
        d.rectangle([0, y0, W, y0 + COMBAT_H], fill=(16, 12, 12))
        yy = y0 + 6

        # --- 左：模式 + 计数 ---
        d.text((12, yy), "战斗模式", font=self.f_lab, fill=FG)
        d.ellipse([86, yy + 3, 96, yy + 13], fill=OK)
        d.text((100, yy - 1), "ON", font=self.f_lab, fill=OK)
        # 左上角红叉（游戏在战斗中）—— 它会把攻击阈值乘上 urge，必须能看见
        if st.get("cm_mark"):
            d.rectangle([132, yy - 2, 206, yy + 15], fill=HOT)
            d.text((137, yy - 1), "★ 战斗中", font=self.f_tiny, fill=(12, 8, 8))
            d.text((212, yy), f"阈值 ×{float(st.get('cm_urge') or 1):.2f}",
                   font=self.f_tiny, fill=HOT)
        elif st.get("cm_red_px") is not None:
            d.text((132, yy), f"无战斗标记（红像素 {st.get('cm_red_px', 0)}）",
                   font=self.f_tiny, fill=FAINT)
        cnt = st.get("cm_cnt") or {}
        d.text((12, yy + 22),
               f"果蝇触发 {cnt.get('fly', 0)}", font=self.f_lab, fill=OK)
        d.text((12, yy + 44),
               f"冷却拦下 {cnt.get('cooldown', 0)}", font=self.f_lab, fill=WARM)
        d.text((12, yy + 66),
               f"上层脚本 {cnt.get('script', 0)}", font=self.f_lab, fill=FAINT)

        # --- 中：两条通道 ---
        for i, (lab, zk, dk, src, hit, col) in enumerate((
                ("攻击 → 鼠标左键", "cm_z_a", "cm_der_a", "cm_src_a",
                 st.get("cm_atk"), HOT),
                ("技能 → 逼近回路", "cm_z_s", "cm_der_s", "cm_src_s",
                 st.get("cm_skl"), ACCENT))):
            x = 330 + i * 460      # 让开左侧的「战斗模式 + 战斗标记」块（占到 ~300）
            z = float(st.get(zk) or 0.0)
            der = float(st.get(dk) or 0.0)
            d.text((x, yy), lab, font=self.f_lab, fill=col if hit else DIM)
            if hit:
                d.rectangle([x + 148, yy - 1, x + 178, yy + 15],
                            fill=col, outline=FG)
                d.text((x + 154, yy), "按下", font=self.f_tiny,
                       fill=(10, 10, 10))
            d.text((x, yy + 22), "水平 z", font=self.f_tiny, fill=DIM)
            d.text((x + 52, yy + 20), f"{z:+5.2f}", font=self.f_mono,
                   fill=col if z > 1.0 else FAINT)
            pp = (z + 3.0) / 6.0
            d.rectangle([x + 108, yy + 22, x + 288, yy + 34],
                        fill=(34, 34, 40))
            d.rectangle([x + 108, yy + 22, x + 108 + 180 * min(max(pp, 0), 1),
                         yy + 34], fill=col if z > 1.0 else (70, 74, 86))
            # 触发阈值的刻线
            tx = x + 108 + 180 * ((1.0 + 3.0) / 6.0)
            d.line([tx, yy + 20, tx, yy + 36], fill=FG, width=1)
            # ★ 这里必须用中文字体：f_mono 是 Consolas，**没有中文字形**，
            #   用它画「变化率」只会得到三个豆腐块。f_mono 仅用于纯 ASCII。
            d.text((x, yy + 42), f"变化率 {der:+.5f}/s", font=self.f_tiny,
                   fill=DIM)
            src = st.get(src) or "-"
            d.text((x + 200, yy + 42),
                   {"fly": "← 果蝇触发", "cooldown": "← 冷却拦下",
                    "sustain": "← 战斗中持续",
                    "hold": "← 持续中"}.get(src, ""),
                   font=self.f_tiny,
                   fill=OK if src == "fly" else (
                       WARM if src in ("cooldown", "sustain") else FAINT))

        # --- 右：犹豫度 ---
        hx = 1170
        hes = float(st.get("cm_hesit") or 0.0)
        d.text((hx, yy), "犹豫度（两通道活动差）", font=self.f_tiny, fill=DIM)
        certain = abs(hes) > 0.25
        for k in range(14):
            bx = hx + k * 12
            # 不确定时画成抖动的断续条纹，确定时画成实心粗条
            if certain or (k % 2 == 0):
                d.rectangle([bx, yy + 20, bx + 9, yy + 40],
                            fill=HOT if certain else (86, 90, 100))
        d.text((hx + 176, yy + 18), f"{hes:+.2f}", font=self.f_mono,
               fill=HOT if certain else DIM)
        d.text((hx + 176, yy + 36),
               "很确定" if certain else "在犹豫", font=self.f_tiny,
               fill=HOT if certain else DIM)

    def _logline(self, d, x, y, e):
        """一行指令。**按键列用英文**（用户要求）：W A S D SPACE SHIFT E SK1..4 LMB ULT1..4"""
        sym = e.get("dir", "-")
        col = ACCENT if sym == ">" else (WARM if sym == "<" else DIM)
        # 群体解码口径那一列是「转身轴 z」，旧口径是「margin」—— 前缀要分开
        pre = "z" if e.get("pop") else "m"
        d.text((x, y), f"{e.get('t', 0):6.1f}s {sym} {pre}{e.get('margin', 0):+.4f}",
               font=self.f_mono, fill=col)

        # ★★ 按键列：这一拍真正发出去的键，英文大写。
        #   放在动作文字**前面**单独一列，这样扫一眼就能看出
        #   "它决定了什么"和"实际按了什么"是不是一回事。
        keys = e.get("keys") or []
        kx = x + 152
        kw = 8
        for k in keys:
            w = max(16, 7 * len(k))
            # 战斗键用暖色，移动键用冷色 —— 一眼分得开
            is_combat = k in ("LMB", "RMB", "E") or k.startswith(("SK", "ULT"))
            bc = (58, 30, 30) if is_combat else (26, 34, 48)
            fc = HOT if is_combat else (150, 200, 255)
            d.rectangle([kx, y - 1, kx + w, y + 15], fill=bc,
                        outline=(70, 80, 100))
            d.text((kx + w // 2, y), k, font=self.f_tiny, fill=fc, anchor="ma")
            kx += w + 3
            if kx > x + 300:
                break

        if e.get("esc"):
            acol = HOT
        elif e.get("jump"):
            acol = WARM
        elif e.get("sprint"):
            acol = ACCENT
        else:
            acol = (198, 208, 224)
        d.text((x + 320, y), e.get("action", ""), font=self.f_log, fill=acol)


# -------------------------------------------------------------------- 自检
def _test():
    st = dict(r=np.zeros(211577, dtype=np.float32), heading=237.0,
              direction="左", margin=-0.0173, turn_dx=-350, speed=7.09,
              drive=0.0705, jump=False, sprint=True, esc=False, stuck=False,
              move=True, dist=124.0, yaw=0.0, pitch=0.30, tick=251,
              spikes=15230, spikes_total=3_845_120, steps_total=2510, steps=10,
              counts=dict(left=97, right=154, jump=21, sprint=34, esc=64, stuck=89),
              action="前进 + 左转 + 冲刺",
              in_sectors=[("正前", 0.0681), ("左前", 0.0544), ("右前", 0.0921),
                          ("左后", 0.0477), ("右后", 0.0603)],
              sector_wins=[41, 22, 118, 15, 55],
              out_chans=[
                  dict(key="forward", name="前进", v=0.0165, q50=0.0158,
                       thr=0.01649, over=True),
                  dict(key="turn", name="转向", v=-0.0091, q50=0.0,
                       thr=0.0021, over=False),
                  dict(key="jump", name="跳跃", v=-0.0080, q50=-0.0107,
                       thr=-0.00805, over=True)])
    st["log"] = [
        dict(t=150.2 + i * 0.65, dir="<", margin=-0.028 + i * 0.001, dx=-350,
             action=["前进 + 左转", "前进 + 左转 + 冲刺", "撞墙反射：原地转",
                     "前进 + 左转"][i % 4],
             jump=(i % 4 == 1), sprint=(i % 4 == 2), esc=(i % 4 == 2))
        for i in range(14)]

    p = FlyPanel()
    rng = np.random.default_rng(5)
    st["r"][rng.choice(p.bv.idx_ok, 8000, replace=False)] = \
        rng.random(8000).astype(np.float32) * 0.10
    a = None
    for _ in range(12):
        a = p.render(st)
    out = os.path.join(HERE, "shots", "fly_panel_wide.png")
    Image.fromarray(a).save(out)
    print(f"已保存 {out}  {a.shape[1]}x{a.shape[0]}")


if __name__ == "__main__":
    _test()
