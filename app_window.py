#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""app_window.py — 桌面应用程序窗口（tkinter，零依赖）。

为什么是 tkinter：
  · Python 标准库自带，**不用装任何东西** —— 开源软件最怕"先 pip install 一堆"
  · 真原生窗口：任务栏有条目、能最小化、关闭就是退出
  · 之前这个项目试 tkinter 崩过一次（0xC0000005），但那是别的原因；
    实测 Tcl/Tk 9.0.4 在这台机器上正常

为什么不重写成原生控件：
  控制台那套 UI（8 组滑块、每个参数下面有一行说明、实时状态读数）
  已经在 HTML 里写好了，而且写得不难看。这里用 tkinter 把它们**按同一份
  GROUPS 数据重新搭一遍** —— 数据源是同一个，不会两边不同步。

★ 线程模型：tkinter 必须独占一个线程（这里让它占主线程）。
  果蝇的主循环跑在**另一个线程**里，两边只通过 `ctl` 字典和只读的状态回调
  通信 —— 绝不让 tkinter 去碰果蝇的数据结构。
"""
from __future__ import annotations

import threading
import time

import tkinter as tk
from tkinter import ttk

# 配色跟网页版保持一致，谁先看都不别扭
BG = "#0b0d12"
PANEL = "#111722"
CELL = "#161d2a"
FG = "#e8eef7"
DIM = "#7c8aa0"
ACC = "#4ea1ff"
OK = "#7ee787"
WARN = "#ffd479"
BAD = "#ff8f8f"
FAINT_DIM = "#5a6678"


class AppWindow:
    """一个 tkinter 窗口，是控制台的门面。

    它只做三件事：读 `ctl` 字典、写 `ctl` 字典、调 `status_fn` 显示读数。
    **不碰果蝇的任何数据结构。**
    """

    def __init__(self, args, ctl: dict, status_fn=None, game_fn=None,
                 tune=None, on_close=None, port: int = 8791, log_fn=None,
                 mem=None):
        """★ 构造函数**不碰 tkinter**。

        Tcl/Tk 9 要求所有 Tcl 调用都在**创建 Tk 对象的那个线程**里。
        第一版在这里直接 `Tk()`，而 `mainloop()` 在另一个线程跑 ——
        启动就报 `RuntimeError: Calling Tcl from different apartment`。
        所以：参数先存着，真正的窗口在 `run()` 里建，全在同一个线程。
        """
        self.args = args
        self.ctl = ctl
        self.status_fn = status_fn
        self.game_fn = game_fn
        self.tune = tune if tune is not None else {}
        self.on_close = on_close
        self.port = port
        self.log_fn = log_fn
        # ★ 记忆存档回调（存/读），由主程序传进来。
        #   界面**不直接动蘑菇体权重** —— 只发请求，主循环在自己的节拍上执行。
        self.mem = mem or {}
        self.mem_btns = {}
        self._slot_taken = {}
        self.rows = {}
        self.toggle_rows = {}
        self.select_rows = {}
        self._closed = False
        self._last_log = None
        self._was_running = False
        # ★ 必须是 False。request_show() 里那个 `= True` 是"有人请求显示"，
        #   这里是初始状态。第一版漏了这行，_tick 第一拍就 AttributeError，
        #   窗口直接建不出来 —— 而当时的检查脚本还误判成"已存在"
        #   （它匹配到了 request_show 里的那一行）。
        self._show_req = False
        self._close_req = False        # 别的线程只能改这个标志，不能碰 tk
        self.root = None

    # ------------------------------------------------------------------
    def run(self):
        """建窗口 + 跑事件循环。**必须在同一个线程里全部做完。**"""
        _dbg = bool(__import__("os").environ.get("EF_DEBUG_UI"))

        def _p(msg):
            if _dbg:
                print(f"  [ui] {msg}", flush=True)

        _p("run() 开始")
        # ★ 高分屏适配：不开 DPI 感知的话，Windows 会把窗口整体拉伸，
        #   文字发虚（实测 150% 缩放下 1080 逻辑像素 → 735 物理像素，糊）。
        #   开了之后 tkinter 自己按物理像素算，字是清的。
        #   必须在 Tk() **之前**设，设晚了没用。
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)   # Win8.1+
                _p("DPI 感知已设（shcore）")
            except Exception as _e:
                ctypes.windll.user32.SetProcessDPIAware()        # 老系统
                _p(f"DPI 感知（user32 回退）：{_e}")
        except Exception as _e:
            _p(f"DPI 设置失败（不影响）：{_e}")
        _p("准备 Tk()")
        self.root = tk.Tk()
        _p("Tk() 成功")
        self.root.title("果蝇全脑驾驶终末地")
        self.root.configure(bg=BG)
        self.root.geometry("1180x880")
        self.root.minsize(880, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        _p("开始 _build()")
        self._build()
        _p("_build() 完成")
        self._apply_defaults()
        _p("开始 _tick() / mainloop")
        self._tick()
        self.root.mainloop()
        _p("mainloop 退出")

    def _apply_defaults(self):
        """把 args 里当前的参数值灌进滑块（让界面反映真实状态）。"""
        for k, (var, val, s, step) in self.rows.items():
            v = getattr(self.args, k, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    var.set(float(v))
                    val.config(text=f"{float(v):.{self._dec(step)}f}")
                except Exception:
                    pass

    def request_close(self):
        """**线程安全**的关闭请求：只置标志，让 _tick 在 tk 线程里处理。"""
        self._close_req = True

    def request_show(self):
        """**线程安全**的"请把窗口显示出来"请求。

        ★ 为什么需要：主线程在窗口建好之后会 `ShowWindow(game_hwnd, SW_RESTORE)`
          把全屏游戏恢复/提到前台，tkinter 的窗口就被压在下面、而且**从未被映射**
          （实测 IsWindowVisible=False、标题是空的，ShowWindow 从外面也救不回来）。
          加上 --overlay 时更明显：叠加面板是 WS_EX_TOPMOST，一建出来就顶在最前。
          所以要在那之后再"推"它一把。
        """
        self._show_req = True

    def _build(self):
        r = self.root
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("TScale", background=BG, troughcolor="#1e2634")
        st.configure("TCombobox", fieldbackground=CELL, background=CELL,
                     foreground=FG)

        # ---------- 顶部：游戏检测横幅 ----------
        self.banner = tk.Label(r, text="正在检测终末地…", bg="#2a2417",
                               fg=WARN, anchor="w", padx=14, pady=9,
                               font=("Microsoft YaHei UI", 10))
        self.banner.pack(fill="x", padx=12, pady=(12, 8))

        # ---------- 控制按钮 ----------
        bar = tk.Frame(r, bg=BG)
        bar.pack(fill="x", padx=12, pady=(0, 8))

        def btn(text, color, bgc, cmd):
            b = tk.Button(bar, text=text, command=cmd, bg=bgc, fg=color,
                          activebackground="#243044", activeforeground=color,
                          relief="flat", padx=18, pady=8,
                          font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2")
            b.pack(side="left", padx=(0, 8))
            return b

        self.b_start = btn("▶  启动", OK, "#17351f", lambda: self._ctl("start"))
        self.b_pause = btn("⏸  暂停", WARN, "#3a2e14", lambda: self._ctl("pause"))
        self.b_stop = btn("■  停止", BAD, "#3a1a1a", lambda: self._ctl("stop"))

        tk.Label(bar, text="运行时长", bg=BG, fg=DIM,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(14, 6))
        self.secs = ttk.Combobox(bar, width=9, state="readonly",
                                 values=["3 分钟", "6 分钟", "10 分钟",
                                         "30 分钟", "不限时"])
        self.secs.current(1)
        self.secs.pack(side="left")

        tk.Button(bar, text="在浏览器打开", command=self._open_browser,
                  bg=CELL, fg=ACC, relief="flat", padx=10, pady=8,
                  cursor="hand2").pack(side="right")

        # ---------- 记忆存档（最多 10 个）----------
        # ★ 为什么要多个槽位：**做对照实验要能回到同一个起点**。
        #   跑一轮 → 存下来 → 关掉某个脑区再跑一轮 —— 两次从同一份记忆出发，
        #   差异才归因得到那个脑区上。只有一个存档的话，第二轮是从第一轮
        #   学到的记忆开始的，「关掉的到底是脑区还是记忆」就分不清了。
        membar = tk.Frame(r, bg=BG)
        membar.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(membar, text="记忆存档", bg=BG, fg=DIM,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(0, 8))
        self.mem_lab = tk.Label(membar, text="", bg=BG, fg=FAINT_DIM,
                                font=("Microsoft YaHei UI", 8), anchor="w")
        for i in range(1, 11):
            b = tk.Button(membar, text=str(i), width=2,
                          bg=CELL, fg=FG, relief="flat", cursor="hand2",
                          font=("Consolas", 9),
                          command=lambda n=i: self._mem_menu(n))
            b.pack(side="left", padx=1)
            self.mem_btns[i] = b
        tk.Button(membar, text="清空记忆", bg="#3a1a1a", fg=BAD, relief="flat",
                  padx=8, pady=1, cursor="hand2",
                  font=("Microsoft YaHei UI", 8),
                  command=self._mem_clear).pack(side="left", padx=(12, 0))
        self.mem_lab.pack(side="left", padx=(10, 0))

        # ★ 当前键位一行 —— 改错了必须能**一眼看见**。
        #   键位是从面板改的，改错之后的症状是行为莫名其妙（按错键、
        #   提示不消失），从画面上完全看不出问题在哪。这一行是唯一的线索。
        self.keylab = tk.Label(r, text="", bg=BG, fg="#8fa3bd",
                               font=("Microsoft YaHei UI", 8), anchor="w")
        self.keylab.pack(fill="x", padx=14, pady=(0, 6))

        # ---------- 主体：左参数 / 右状态 ----------
        body = tk.Frame(r, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # 左：可滚动参数区
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_params(left)

        # 右：状态读数 + 日志
        right = tk.Frame(body, bg=BG, width=370)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        tk.Label(right, text="实时状态", bg=BG, fg=DIM,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        self.status = tk.Frame(right, bg=PANEL)
        self.status.pack(fill="x", pady=(4, 10))

        # ★ 日志面板：出问题时不用再去翻控制台窗口。
        #   之前只能听用户说"卡住了/不动了"，而真正的原因全在 stdout 里。
        tk.Label(right, text="日志（最近十几行）", bg=BG, fg=DIM,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        self.logbox = tk.Text(right, bg="#07090d", fg="#9fb0c8", bd=0,
                              wrap="none", height=14, font=("Consolas", 8))
        self.logbox.pack(fill="both", expand=True, pady=(4, 0))
        self.logbox.configure(state="disabled")

    def _build_params(self, parent):
        """按 GROUPS 生成滑块。数据源和网页版是同一份，不会不同步。"""
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        # ★ 对照组（消融实验）放最前面 —— 它比调参重要：
        #   调参是调多少合适，对照组是这个部件到底在不在起作用。
        # ---- 键位 / 技能含义（下拉框）----
        sels = self.tune.get("selects", [])
        if sels:
            gs = tk.LabelFrame(inner, text=" 键位与技能含义 ", bg=BG, fg=ACC,
                               font=("Microsoft YaHei UI", 9, "bold"),
                               bd=1, relief="groove", labelanchor="nw")
            gs.pack(fill="x", padx=2, pady=(2, 10))
            import tkinter.ttk as _ttk
            for spec in sels:
                k, lab, opts, hint = spec[0], spec[1], spec[2], spec[3]
                if k.startswith("__hdr_"):
                    tk.Label(gs, text=lab, bg=BG, fg="#8fa3bd", anchor="w",
                             font=("Microsoft YaHei UI", 8, "bold")).pack(
                        fill="x", padx=8, pady=(8, 0))
                    continue
                row = tk.Frame(gs, bg=BG)
                row.pack(fill="x", padx=8, pady=(4, 0))
                tk.Label(row, text=lab, bg=BG, fg=FG, width=15, anchor="w",
                         font=("Microsoft YaHei UI", 9)).pack(side="left")
                labels = [(o[1] if isinstance(o, (list, tuple)) else o) for o in opts]
                cur = str(getattr(self.args, k, ""))
                # 找到当前值对应的显示文字
                shown = cur
                for o in opts:
                    if isinstance(o, (list, tuple)) and o[0] == cur:
                        shown = o[1]
                cb = _ttk.Combobox(row, values=labels, state="readonly", width=26)
                cb.set(shown)
                # ★★ 必须屏蔽滚轮改值。
                #    ttk.Combobox 在鼠标**悬停**时会跟着滚轮改值 ——
                #    用户一边玩游戏一边滚滚轮，鼠标扫过面板就把键位**静默改掉**了。
                #    实测被改成 combo_key=q，于是连携技永远按错键、提示不消失、
                #    每拍都按一次，连刷 11 次 —— 看起来像"果蝇在乱按键"，
                #    根因其实是滚轮。
                #    只读下拉框本来就不该响应滚轮，直接吃掉事件。
                cb.bind("<MouseWheel>", lambda e: "break")
                cb.bind("<Button-4>", lambda e: "break")     # Linux 上滚
                cb.bind("<Button-5>", lambda e: "break")     # Linux 下滚
                cb.pack(side="left", fill="x", expand=True, padx=8)
                cb.bind("<<ComboboxSelected>>",
                        lambda e, kk=k, oo=opts, w=cb: self._select(kk, w, oo))
                self.select_rows[k] = (cb, opts, labels)
                if hint:
                    tk.Label(gs, text=hint, bg=BG, fg="#6b7a90", anchor="w",
                             justify="left", wraplength=560,
                             font=("Microsoft YaHei UI", 8)).pack(
                        fill="x", padx=(26, 8))

        tgs = self.tune.get("toggles", [])
        if tgs:
            g0 = tk.LabelFrame(inner, text=" 对照组（消融实验） ", bg=BG,
                               fg=WARN, font=("Microsoft YaHei UI", 9, "bold"),
                               bd=1, relief="groove", labelanchor="nw")
            g0.pack(fill="x", padx=2, pady=(2, 10))
            for spec in tgs:
                k, lab, defv, hint = spec[0], spec[1], spec[2], spec[3]
                var = tk.BooleanVar(value=bool(getattr(self.args, k, defv)))
                cb = tk.Checkbutton(
                    g0, text=lab, variable=var, bg=BG, fg=FG,
                    selectcolor=CELL, activebackground=BG, activeforeground=FG,
                    font=("Microsoft YaHei UI", 9), anchor="w",
                    command=lambda kk=k, vv=var: self._toggle(kk, vv.get()))
                cb.pack(fill="x", padx=8, pady=(4, 0))
                self.toggle_rows[k] = var
                if hint:
                    tk.Label(g0, text=hint, bg=BG, fg="#6b7a90", anchor="w",
                             justify="left", wraplength=560,
                             font=("Microsoft YaHei UI", 8)).pack(
                        fill="x", padx=(26, 8))

        for title, ps in self.tune.get("groups", []):
            grp = tk.LabelFrame(inner, text=" " + title + " ", bg=BG, fg="#8fa3bd",
                                font=("Microsoft YaHei UI", 9), bd=1,
                                relief="groove", labelanchor="nw")
            grp.pack(fill="x", padx=2, pady=(2, 10))
            for spec in ps:
                k, lab, lo, hi, step = spec[0], spec[1], spec[2], spec[3], spec[4]
                hint = spec[5] if len(spec) > 5 else ""
                row = tk.Frame(grp, bg=BG)
                row.pack(fill="x", padx=8, pady=(4, 0))
                tk.Label(row, text=lab, bg=BG, fg=FG, width=17, anchor="w",
                         font=("Microsoft YaHei UI", 9)).pack(side="left")
                var = tk.DoubleVar(value=float(self.tune["args"].__dict__.get(k, lo)))
                val = tk.Label(row, text=f"{var.get():g}", bg=BG, fg=WARN,
                               width=8, anchor="e",
                               font=("Consolas", 9))
                val.pack(side="right")
                s = ttk.Scale(row, from_=lo, to=hi, variable=var,
                              orient="horizontal", style="TScale",
                              command=lambda v, kk=k, vv=val: self._slide(kk, v, vv))
                s.pack(side="left", fill="x", expand=True, padx=8)
                self.rows[k] = (var, val, s, step)
                if hint:
                    tk.Label(grp, text=hint, bg=BG, fg="#6b7a90", anchor="w",
                             justify="left", wraplength=560,
                             font=("Microsoft YaHei UI", 8)).pack(
                        fill="x", padx=(26, 8))

    # ------------------------------------------------------------------
    def _toggle(self, key, on):
        """对照组开关。★ 直接写 args 上的 bool —— 循环下一拍就读到了。"""
        setattr(self.args, key, bool(on))
        print(f"  [对照组] {key} = {'开' if on else '★关掉'}", flush=True)
        for h in self.tune.get("hooks", []):
            try:
                h()
            except Exception:
                pass

    def _mem_menu(self, n):
        """点槽位号码弹菜单：存 / 读 / 删。

        ★ 为什么用弹出菜单而不是三个按钮：10 个槽位 × 3 个动作 = 30 个按钮，
          窗口直接被塞满。一个号码 + 右键菜单，10 个按钮搞定。
        """
        m = tk.Menu(self.root, tearoff=0, bg=CELL, fg=FG,
                    activebackground="#243044", activeforeground=FG,
                    font=("Microsoft YaHei UI", 9))
        taken = bool(self._slot_taken.get(n, False))
        m.add_command(label=f"存入槽位 {n}", command=lambda: self._mem("save", n))
        if taken:
            m.add_command(label=f"读取槽位 {n}", command=lambda: self._mem("load", n))
            m.add_separator()
            m.add_command(label=f"删除槽位 {n}", command=lambda: self._mem("delete", n))
        try:
            m.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            m.grab_release()

    def _mem(self, act, n):
        """存档操作。★ 不直接动蘑菇体权重 —— 读档只挂待办，主循环下一拍执行。"""
        if act == "load":
            if self.ctl is not None:
                self.ctl["load_slot"] = int(n)
            self._flash(f"槽位 {n} 读取请求已发出（下一拍生效）")
            print(f"  [存档] 请求读取槽位 {n}", flush=True)
            return
        fn = self.mem.get(act)
        if fn is None:
            self._flash("这一轮没有可存的记忆（觅食闭环没开）", BAD)
            return
        try:
            ok = bool(fn(n))
            self._flash(f"槽位 {n} " + ("已保存 ✓" if ok else "保存失败（看日志）"),
                        FG if ok else BAD)
            print(f"  [存档] {act} 槽位 {n} -> {'成功' if ok else '失败'}", flush=True)
        except Exception as e:
            self._flash(f"槽位 {n} 出错：{e}", BAD)

    def _select(self, key, widget, opts):
        """下拉框改了 —— 把**显示文字**换回**内部值**再写进 args。

        ★ 选项是 (值, 显示文字) 的二元组：值是给代码用的（attack / mouse_left），
          显示文字是给人看的（"攻击 · 主攻，永远在线"）。
          直接写显示文字的话，argparse 那边收到的是中文长句 —— 表面上不报错，
          但语义全丢了（roles=("攻击 · 主攻，永远在线", ...) 一个都匹配不上）。
        """
        shown = widget.get()
        val = shown
        for o in opts:
            if isinstance(o, (list, tuple)):
                if o[1] == shown:
                    val = o[0]
                    break
            elif o == shown:
                val = o
                break
        setattr(self.args, key, val)
        print(f"  [键位] {key} = {val}（显示为「{shown}」）", flush=True)
        for h in self.tune.get("hooks", []):
            try:
                h()
            except Exception:
                pass

    def _mem_clear(self):
        """清空记忆 —— 权重回到**连接组基线**（那只什么都没学过的果蝇）。

        ★ 和「读一个空槽位」不是一回事：槽位里存的可能是某个中间状态，
          而 reset 是明确的忘掉一切、回到先天。
        """
        from tkinter import messagebox
        if not messagebox.askyesno(
                "清空记忆",
                "把蘑菇体权重恢复到连接组基线？\n\n"
                "这会**忘掉所有学到的东西**（奖赏/惩罚留下的痕迹全部清零），"
                "但不会影响这一轮的计数和饥饿值。"):
            return
        fn = self.mem.get("reset")
        try:
            ok = bool(fn()) if fn else False
            self._flash("记忆已清空（回到连接组基线）" if ok else "清空失败", FG if ok else BAD)
            print(f"  [存档] 清空记忆 -> {'成功' if ok else '失败'}", flush=True)
        except Exception as e:
            self._flash(f"清空出错：{e}", BAD)

    def _slide(self, key, value, label):
        v = float(value)
        _, _, _, step = self.rows[key]
        label.config(text=f"{v:.{self._dec(step)}f}")
        # 只在松手后才提交太反直觉 —— ttk.Scale 每动一下就回调，
        # 直接写进去，反正后端会夹范围
        setattr(self.args, key, v)
        for h in self.tune.get("hooks", []):
            try:
                h()
            except Exception:
                pass

    @staticmethod
    def _dec(step):
        t = str(step)
        return 0 if "." not in t else len(t) - t.index(".") - 1

    def _ctl(self, a):
        # ★ 界面上的每一次点击都打日志 —— 用户说点了没反应的时候，
        #   第一件要确认的就是点击到底有没有进来。
        print(f"  [界面] 点击「{a}」", flush=True)
        if a == "start":
            if self.game_fn:
                ok, info = self.game_fn()
                if not ok:
                    self._flash(f"没有检测到终末地（{info}）", BAD)
                    return
            self.ctl["running"] = True
            self.ctl["stop"] = False
            n = self.secs.current()
            self.args.seconds = [180, 360, 600, 1800, 0][n]
            self._flash("已启动")
        elif a == "pause":
            self.ctl["running"] = False
            self._flash("已暂停")
        elif a == "stop":
            self.ctl["running"] = False
            self.ctl["stop"] = True
            self._flash("已停止")

    def _open_browser(self):
        try:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{self.port}/")
        except Exception:
            pass

    def _flash(self, msg, color=FG):
        self.banner.config(text=msg, bg="#16281a" if color == FG else "#2a1717",
                           fg=color)

    # ------------------------------------------------------------------
    def _tick(self):
        """每 700ms 刷新一次界面。不阻塞主循环 —— 只读共享状态。"""
        if self._close_req:
            self._close()
            return
        if self._closed:
            return
        if self._show_req:
            self._show_req = False
            try:
                self.root.deiconify()
                self.root.lift()
                self.root.attributes("-topmost", True)
                self.root.after(
                    2500, lambda: self.root.attributes("-topmost", False))
            except Exception:
                pass
        try:
            if self.game_fn:
                ok, info = self.game_fn()
            else:
                ok, info = False, ""
            running = bool(self.ctl.get("running"))
            if ok:
                self.banner.config(
                    text=f"已检测到终末地 ({info})    "
                         + ("运行中" if running else "已停止"),
                    bg="#16281a", fg=OK)
            else:
                self.banner.config(
                    text="没有检测到终末地 —— 启动已禁用。请先打开游戏。",
                    bg="#2a1717", fg=BAD)
            # ★ 变空闲的那一刻把窗口提到最前 —— 不然它藏在全屏游戏后面，
            #   用户看到屏幕上一块冻住的面板，会以为"界面卡死了"。
            if self._was_running and not running:
                try:
                    self.root.deiconify()
                    self.root.lift()
                    self.root.attributes("-topmost", True)
                    self.root.after(
                        1500, lambda: self.root.attributes("-topmost", False))
                except Exception:
                    pass
            self._was_running = running
            self.b_start.config(state="normal" if (ok and not running) else "disabled")
            self.b_pause.config(state="normal" if running else "disabled")
            self.b_stop.config(state="normal" if running else "disabled")

            # 状态读数
            if self.status_fn:
                for w in self.status.winfo_children():
                    w.destroy()
                html = self.status_fn()
                import re
                txt = re.sub(r"<[^>]+>", "|", html).strip("|")
                parts = [p.strip() for p in txt.split("|") if p.strip()]
                for i in range(0, len(parts) - 1, 2):
                    row = tk.Frame(self.status, bg=PANEL)
                    row.pack(fill="x", padx=10, pady=1)
                    tk.Label(row, text=parts[i], bg=PANEL, fg=DIM, anchor="w",
                             font=("Microsoft YaHei UI", 9)).pack(side="left")
                    tk.Label(row, text=parts[i + 1], bg=PANEL, fg=OK, anchor="e",
                             font=("Consolas", 9)).pack(side="right")

            # ★ 这里**不要**刷新缓存：find_endfield() 要枚举窗口，
            #   从非主线程调用会卡住（实测把 HTTP 请求和窗口一起拖死）。
            #   刷新交给主循环和 main 线程的空闲循环去做。

            # ★ 当前键位一行 —— 改错了必须能一眼看见。
            #   键位是从面板改的，改错之后症状是"行为莫名其妙"（按错键、
            #   提示不消失），从界面上完全看不出来。这一行是**唯一的线索**。
            try:
                _keys = ("键位  "
                         f"普攻 {getattr(self.args, 'attack_key', '?')}"
                         f" · 技能 {getattr(self.args, 'key_skill1', '?')}"
                         f"{getattr(self.args, 'key_skill2', '?')}"
                         f"{getattr(self.args, 'key_skill3', '?')}"
                         f"{getattr(self.args, 'key_skill4', '?')}"
                         f" · 连携 {getattr(self.args, 'combo_key', '?')}"
                         f" · 大招 {getattr(self.args, 'key_ult', '?')}")
                self.keylab.config(text=_keys)
            except Exception:
                pass

            # 槽位按钮状态：有内容的画成绿底
            try:
                import live_tune as _lt
                for sl in _lt.list_slots():
                    n = sl["slot"]; taken = not sl.get("empty")
                    self._slot_taken[n] = taken
                    btn = self.mem_btns.get(n)
                    if btn is not None:
                        want = ("#17351f", "#2f6b3d") if taken else (CELL, "#2b3a52")
                        if btn.cget("bg") != want[0]:
                            btn.config(bg=want[0], activebackground=want[0],
                                       highlightbackground=want[1])
                _n = sum(1 for v in self._slot_taken.values() if v)
                self.mem_lab.config(text=f"{_n} / 10 已存" +
                                         ("　（右键槽位 = 读取）" if _n else ""))
            except Exception:
                pass

            # 日志尾巴
            if self.log_fn:
                lines = self.log_fn()
                txt = "\n".join(lines)
                if txt != self._last_log:
                    self._last_log = txt
                    self.logbox.configure(state="normal")
                    self.logbox.delete("1.0", "end")
                    self.logbox.insert("1.0", txt)
                    self.logbox.see("end")
                    self.logbox.configure(state="disabled")
        except Exception:
            pass
        self.root.after(700, self._tick)

    # ------------------------------------------------------------------
    def _close(self):
        self._closed = True
        if self.on_close:
            try:
                self.on_close()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass


def start_app(args, ctl: dict, status_fn=None, game_fn=None, tune=None,
              on_close=None, port: int = 8791, verbose: bool = True,
              log_fn=None, mem=None):
    """在**后台线程**里起这个窗口，主线程继续跑果蝇。

    ★ 为什么让窗口占子线程而不是反过来：果蝇的主循环是一个上千行的整体，
      搬到子线程风险更大。而 Tcl/Tk 9 只要求"创建和使用在同一个线程" ——
      所以只要**窗口对象本身在这个线程里创建**就没问题（见 __init__ 的注释）。
    """
    app = AppWindow(args, ctl, status_fn=status_fn, game_fn=game_fn,
                    tune=tune, on_close=on_close, port=port, log_fn=log_fn,
                    mem=mem)

    def _run_guarded():
        # ★ 线程里抛的异常默认只打到 stderr，而这个程序常常跑在
        #   "输出重定向到文件" 的启动器里 —— 结果就是窗口没出来、
        #   日志里一个字都没有，完全查不出原因。必须自己接住并打印。
        try:
            app.run()
        except Exception as e:
            import traceback
            print(f"\n  ✗ 应用程序窗口启动失败：{type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()

    th = threading.Thread(target=_run_guarded, name="app-window", daemon=True)
    th.start()
    if verbose:
        print("应用程序窗口已打开（右上角关闭 = 退出）")
    return app, th


if __name__ == "__main__":
    # 单独跑一下看窗口长什么样（不接果蝇）
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from live_tune import GROUPS, make_status_html

    a = argparse.Namespace(**{p[0]: p[2] for _, ps in GROUPS for p in ps})
    a.seconds = 360
    c = {"running": False, "pause": False, "stop": False}
    tune = {"groups": GROUPS, "args": a, "hooks": []}
    st = lambda: make_status_html([("饥饿值 h", "0.720"), ("血条", "83%"),
                                   ("攻击驱动力", "0.580"),
                                   ("蘑菇体输出", "+1.187"), ("权重漂移", "103.1")])
    gm = lambda: (True, "hwnd=123456 2560×1600")
    app, _ = start_app(a, c, status_fn=st, game_fn=gm, tune=tune)
    app.run()
