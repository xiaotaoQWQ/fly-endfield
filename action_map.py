#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""action_map.py — 从下行神经元活动映射出「移动 / 跳跃 / 冲刺」

与 sector_vote（只做左右选择）的区别：这里读出**动作通道**。

依据（交接文档第五节 + Fly64 技术笔记）：
    DNg100        → 前进
    DNa02 / DNg13 → 左右差（转向）
    DNp01 / DNp10 → 跳跃

★ 关键处理：各通道的**基线偏置差异极大**（标定实测 DNg100 均值 +0.0002、
DNa02 −0.019、DNp10 +0.019）。所以不能用一个统一阈值，必须
  1. 减去各自的标定基线
  2. 用标定数据的**分位数**定触发阈值
  3. 加持续时长与冷却，避免抖动/连跳

用法：
    # 标定（从 mc_record 录下的帧算统计）
    python action_map.py calibrate --frames artifacts/run2/frames
    # 之后 mc_walk/mc_longrun 会自动加载 out/action_calib.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GRAPH = os.path.join(HERE, "graph")
OUT = os.path.join(HERE, "out")
CALIB_JSON = os.path.join(OUT, "action_calib.json")

CHANNELS = {
    "forward": ["DNg100"],
    "turn": ["DNa02", "DNg13"],
    "jump": ["DNp01", "DNp10"],
}


# ----------------------------------------------------------------------
@dataclass
class ActionMapper:
    """读出行神经元活动 → 动作开关

    每个动作通道：
      value  = 该组神经元活动的平均 − 标定基线
      trigger= value > 阈值（阈值 = 标定时该通道值的 q_hi 分位数）
    再加两个时间约束：
      hold_steps  连续多少拍超过阈值才真正触发（防抖）
      cooldown    触发后多少拍内不再触发（防连发）
    """

    calib: dict
    hold_steps: dict = field(default_factory=lambda: {"jump": 1, "sprint": 3})
    cooldown: dict = field(default_factory=lambda: {"jump": 5, "sprint": 12})
    _streak: dict = field(default_factory=dict)
    _last: dict = field(default_factory=dict)
    _tick: int = 0
    _sprint_on: bool = False
    _hist: dict = field(default_factory=dict)     # 在线自适应阈值的滑动窗口
    last_thr: dict = field(default_factory=dict)  # 当拍实际生效的阈值（面板要画刻度）

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str = CALIB_JSON, **kw) -> "ActionMapper":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"缺少标定文件 {path}；先跑：python action_map.py calibrate --frames <帧目录>")
        with open(path, encoding="utf-8") as f:
            calib = json.load(f)
        return cls(calib=calib, **kw)

    # ------------------------------------------------------------------
    def hist_q(self, key: str, q: float):
        """滑动窗口里该通道的分位数（没有足够样本时返回 None）。

        面板画刻度要用它，**不能**用标定文件里的 q50 ——
        实测两次运行之间 raw 的分布会整体漂移好几个数量级
        （jump 一次在 [−0.011, −0.0002]，下一次在 [−0.003, +0.0095]），
        标定文件里的分位数很快就过期了。
        """
        d = self._hist.get(key)
        if d is None or len(d) < 20:
            return None
        return float(np.percentile(np.fromiter(d, float), q))

    # ------------------------------------------------------------------
    def read(self, r: np.ndarray) -> dict:
        """从网络状态 r 读出各动作通道的「相对基线值」"""
        out = {}
        for name, ch in self.calib["channels"].items():
            v = float(np.mean([r[i] for i in ch["idx"]]))
            out[name] = v - ch["baseline"]
        return out

    # ------------------------------------------------------------------
    def step(self, r: np.ndarray) -> dict:
        """返回本拍的动作：{jump, sprint, turn, raw}

        · jump 是**事件**（点按一次）：超阈值 + 持续 hold 拍 + 冷却
        · sprint 是**状态**（按住不放）：带滞回 —— 超过进入阈值就按住，
          回落到退出阈值才松手。
          ★ 这点很关键：原先把冲刺也做成"事件+冷却"，实测触发率只有 0.6%，
          因为 forward 通道波动大，很难连续多拍超阈值；而游戏里的冲刺本来
          就是「按住持续跑」，用状态机才符合语义。
        """
        raw = self.read(r)
        self._tick += 1
        acts = {}

        # ---- 在线自适应阈值 ----
        # 为什么要自适应：实测两次运行的 raw 分布会**明显漂移**
        # （jump 的 q60 一次是 −0.0123、下一次是 −0.0080，而整个跨度才 0.011）。
        # 固定绝对阈值很容易整段落在外，表现就是「跳跃突然一次都不触发」。
        # 用滑动窗口的分位数，触发率天然锁在 (100−q)% 附近。
        adapt = (self.calib.get("meta", {}) or {}).get("adaptive")
        if adapt:
            from collections import deque
            w = int(adapt.get("window", 400))
            for k in ("jump", "forward"):
                d = self._hist.get(k)
                if d is None or d.maxlen != w:
                    d = deque(d or (), maxlen=w)
                    self._hist[k] = d
                d.append(float(raw.get(k, 0.0)))

        def _thr(key: str, q: float, fixed: float) -> float:
            if adapt:
                d = self._hist.get(key)
                if d is not None and len(d) >= int(adapt.get("warmup", 25)):
                    return float(np.percentile(np.fromiter(d, float), q))
            return fixed

        # ---- 跳跃（事件）----
        if "jump" in self.calib["channels"]:
            jc = self.calib["channels"]["jump"]
            thr = _thr("jump", float((adapt or {}).get("jump_q", 60.0)),
                       jc["q_hi"])
            over = raw["jump"] > thr
            s = self._streak.get("jump", 0)
            s = s + 1 if over else 0
            self._streak["jump"] = s
            last = self._last.get("jump", -10 ** 9)
            ready = (self._tick - last) >= self.cooldown.get("jump", 5)
            fire = (s >= self.hold_steps.get("jump", 1)) and ready
            if fire:
                self._last["jump"] = self._tick
                self._streak["jump"] = 0
            acts["jump"] = bool(fire)
            self.last_thr["jump"] = thr
        else:
            acts["jump"] = False

        # ---- 冲刺（带滞回的状态）----
        if "sprint" in self.calib["channels"]:
            sc = self.calib["channels"]["sprint"]
            hi = _thr("forward", float((adapt or {}).get("sprint_q", 85.0)),
                      sc["q_hi"])
            # ★ 退出阈值优先用标定给的 q_lo（另一个分位数）。
            #   原先固定写 hi*0.5 —— 实测 forward 通道只有 ~4% 的拍会落到
            #   一半以下，于是**一旦进入冲刺就再也退不出来**：模拟显示不管
            #   进入阈值取 q85 还是 q98，占空比都是 96%。用分位数才可控。
            lo = _thr("forward", float((adapt or {}).get("sprint_exit_q", 70.0)),
                      sc.get("q_lo", hi * 0.5))
            if lo > hi:
                lo = hi
            v = float(raw.get("forward", 0.0))
            self._sprint_on = (v > lo) if self._sprint_on else (v > hi)
            acts["sprint"] = bool(self._sprint_on)
            self.last_thr["sprint_hi"] = hi
            self.last_thr["sprint_lo"] = lo
        else:
            acts["sprint"] = False

        # ---- 转向（连续量，同组左右差，已各自扣基线）----
        if "turn_L" in raw and "turn_R" in raw:
            acts["turn"] = raw["turn_L"] - raw["turn_R"]
        else:
            acts["turn"] = 0.0
        acts["raw"] = raw
        return acts


# ----------------------------------------------------------------------
def do_calibrate(args):
    from PIL import Image
    from mc_walk import build_retina_mapper, sample_first_person

    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = W.shape[0]
    rm = build_retina_mapper(os.path.join(OUT, "optic_map.feather"),
                             os.path.join(GRAPH, "graph_meta.feather"), args.hfov)
    print()

    # 收集每个通道的神经元索引
    ch_idx = {}
    for grp, types in CHANNELS.items():
        for t in types:
            sub = meta[meta["type"] == t]
            for _, row in sub.iterrows():
                key = f"{t}_{row['somaSide']}"
                ch_idx[key] = int(row["idx"])
    # 组装成「按组按侧」的通道
    channels = {}
    for grp, types in CHANNELS.items():
        for side in ("L", "R"):
            keys = [f"{t}_{side}" for t in types if f"{t}_{side}" in ch_idx]
            if keys:
                channels[f"{grp}_{side}"] = [ch_idx[k] for k in keys]
    # 前进与跳跃还要一个「双侧合并」通道（触发用）
    for grp in ("forward", "jump"):
        keys = [f"{grp}_{s}" for s in ("L", "R") if f"{grp}_{s}" in channels]
        if keys:
            channels[grp] = [i for k in keys for i in channels[k]]
    print(f"通道：{ {k: len(v) for k, v in channels.items()} }")

    files = sorted(glob.glob(os.path.join(args.frames, "f*.jpg")))[:args.limit]
    if not files:
        sys.exit("没有帧")
    print(f"用 {len(files)} 帧跑网络…")

    r = np.zeros(n, dtype=np.float32)
    prev_lum = None
    rows = []
    for i, p in enumerate(files):
        img = np.asarray(Image.open(p).convert("RGB"))
        drive, lum = sample_first_person(rm, img, prev_lum, args.map_mode)
        prev_lum = lum
        for _ in range(args.steps):
            r = (args.a * r + args.b * np.tanh(W @ r + drive)).astype(np.float32)
        rec = {k: float(np.mean([r[i2] for i2 in v])) for k, v in channels.items()}
        rows.append(rec)
        if i % 20 == 0:
            print(f"  {i+1}/{len(files)}")

    df = pd.DataFrame(rows)
    out = {"channels": {}, "meta": dict(frames=len(files), steps=args.steps,
                                        a=args.a, b=args.b, hfov=args.hfov,
                                        map_mode=args.map_mode,
                                        source=os.path.abspath(args.frames))}
    print()
    print(f"  {'通道':>12s} {'基线(均值)':>13s} {'q70':>12s} {'q80':>12s} {'q90':>12s} {'极差':>11s}")
    for k, idxs in channels.items():
        x = df[k].to_numpy()
        base = float(x.mean())
        out["channels"][k] = dict(
            idx=idxs, baseline=base,
            q50=float(np.percentile(x, 50)), q70=float(np.percentile(x, 70)),
            q80=float(np.percentile(x, 80)), q90=float(np.percentile(x, 90)),
            std=float(x.std()), ptp=float(np.ptp(x)),
        )
        print(f"  {k:>12s} {base:>+13.6f} {np.percentile(x,70):>+12.6f} "
              f"{np.percentile(x,80):>+12.6f} {np.percentile(x,90):>+12.6f} {np.ptp(x):>11.6f}")

    # 触发阈值：用「相对基线」的分位数
    #   标定数据里 jump 通道只有约 10% 的帧处于高位，
    #   再叠加 hold/cooldown 后实际触发率会被压到 ~4%（实测）——太少，
    #   视频里"看得出在跳"需要 10% 上下，所以取 q70 而不是 q80。
    for k in ("jump", "forward"):
        if k in out["channels"]:
            c = out["channels"][k]
            c["q_hi"] = float(np.percentile(
                df[k].to_numpy(), args.jump_q)) - c["baseline"]
    if "forward" in out["channels"]:
        c = out["channels"]["forward"]
        out["channels"]["sprint"] = dict(c)
        out["channels"]["sprint"]["q_hi"] = float(np.percentile(
            df["forward"].to_numpy(), args.sprint_q)) - c["baseline"]
    out["meta"]["jump_q"] = args.jump_q
    out["meta"]["sprint_q"] = args.sprint_q

    os.makedirs(OUT, exist_ok=True)
    with open(CALIB_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print(f"  跳阈值  = q{args.jump_q:.0f} − baseline = {out['channels'].get('jump', {}).get('q_hi', 0):+.6f}")
    print(f"  冲刺阈值= q{args.sprint_q:.0f} − baseline = {out['channels'].get('sprint', {}).get('q_hi', 0):+.6f}")
    print(f"标定 → {CALIB_JSON}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("calibrate")
    c.add_argument("--frames", default=os.path.join(HERE, "artifacts", "run2", "frames"))
    c.add_argument("--limit", type=int, default=80)
    c.add_argument("--steps", type=int, default=8)
    c.add_argument("--a", type=float, default=0.5)
    c.add_argument("--b", type=float, default=1.0)
    c.add_argument("--hfov", type=float, default=90.0)
    c.add_argument("--map-mode", choices=["fov", "linear"], default="linear")
    c.add_argument("--jump-q", type=float, default=70,
                   help="跳跃阈值分位数（越大越少跳）")
    c.add_argument("--sprint-q", type=float, default=85,
                   help="冲刺阈值分位数")
    args = ap.parse_args()
    if args.cmd == "calibrate":
        do_calibrate(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
