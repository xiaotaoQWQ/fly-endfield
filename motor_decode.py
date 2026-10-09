#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motor_decode.py — 群体解码：下行神经元活动 → 三条运动轴 → 键鼠指令。

和旧的 `sector_vote.DirectionDecoder` 的区别：
    旧的是「点名的 5 对左右 DN 取差」—— 用户要求不用这种神经元直连。
    新的是把**全部 1,332 个下行神经元**的状态投影到三条轴系数上，
    轴系数由连接组推出（见 `build_motor_axes.py`）。

三条信号（s = Σ_dn r[dn] · axis[dn]）：
    thrust  推力   → W(正) / S(负)
    strafe  横移   → D(正) / A(负)
    turn    转身   → 鼠标 X（连续，正=右转）

为什么不用绝对阈值：
    信号量级随输入分布（小地图口径 / 主画面口径）变化很大，
    固定阈值必然在换口径时失效。所以用**自适应基线 + 滑动尺度**：
        z = (s − 基线) / 尺度
    基线 = s 的慢速 EMA；尺度 = |s − 基线| 的 EMA。
    阈值定在 z 上（±1 进入 / ±0.5 退出，带滞回），就不用管绝对量级。
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
AXES = os.path.join(OUT, "motor_axes.npz")


class PopulationMotor:
    def __init__(self, axes_path: str = AXES,
                 base_tau: float = 12.0, scale_tau: float = 12.0,
                 thr_w: float = -0.50, thr_s: float = -1.60,
                 thr_strafe: float = 0.80, hyst: float = 0.30,
                 turn_gain: float = 1.5, z_clip: float = 2.5,
                 warmup: int = 15, verbose: bool = True):
        z = np.load(axes_path)
        self.dn = np.asarray(z["dn_idx"], dtype=np.int64)
        self.ax = {k: np.asarray(z[k], dtype=np.float32)
                   for k in ("thrust", "strafe", "turn")}
        self.n_axis = len(self.dn)

        self.base_tau = float(base_tau)
        self.scale_tau = float(scale_tau)
        self.thr_w = float(thr_w)
        self.thr_s = float(thr_s)
        self.thr_strafe = float(thr_strafe)
        self.hyst = float(hyst)
        self.turn_gain = float(turn_gain)
        self.z_clip = float(z_clip)
        self.warmup = int(warmup)

        self.base = {k: None for k in self.ax}
        self.var = {k: 1e-6 for k in self.ax}
        self.scale = {k: 1.0 for k in self.ax}
        self.state = {"thrust": 0, "strafe": 0}   # -1 / 0 / +1，带滞回
        self.n = 0
        self.last = {}
        if verbose:
            print(f"群体解码：{len(self.dn):,} 个下行神经元 → 三条轴")
            print(f"  W: z>{thr_w:+.2f}   S: z<{thr_s:+.2f}   "
                  f"横移: |z|>{thr_strafe:.2f}   滞回 {hyst:.2f}")

    # ------------------------------------------------------------------
    def step(self, r: np.ndarray, dt: float = 0.4) -> dict:
        s = {k: float(np.dot(r[self.dn], v)) for k, v in self.ax.items()}

        # 自适应基线 + **标准差**尺度
        # ★ 第一版用「偏差绝对值的 EMA」当尺度 —— 它约等于 0.8σ，
        #   于是 z>1 实际只有 1.25σ，**噪声就能越过阈值**（约 42% 的时间），
        #   实测表现为角色长时间在"后退+右移"。必须用真正的标准差。
        a_b = 1.0 - np.exp(-dt / max(self.base_tau, 1e-3))
        a_s = 1.0 - np.exp(-dt / max(self.scale_tau, 1e-3))
        z = {}
        for k, v in s.items():
            if self.base[k] is None:
                self.base[k] = v
                self.var[k] = 1e-6
                z[k] = 0.0
                continue
            dev = v - self.base[k]
            self.base[k] += a_b * dev
            self.var[k] = (1 - a_s) * self.var[k] + a_s * dev * dev
            self.scale[k] = float(np.sqrt(max(self.var[k], 1e-12)))
            z[k] = dev / max(self.scale[k], 1e-9)

        # ---- 非对称阈值 + 滞回 ----
        # ★ 为什么不用对称阈值：z 是零均值的，对称阈值必然给出 50/50 的
        #   前进/后退 —— 实测 W 只有 4% 的时间按着，角色几乎不走。
        #   真实果蝇是**默认在走**的，所以：
        #     W：z > thr_w(−0.5) 就按（约 70% 的时间在走）
        #     S：z < thr_s(−1.6) 才按（后退罕见，约 5%）
        #     横移：|z| > 0.8 才按（偶尔侧移，约 20%）
        #   ⚠ 「推力低 → 后退」是**设计选择**，不是生理事实：
        #     推力轴只表示「下行神经元对腿运动神经元的总驱动」，
        #     z<0 意味着驱动力低于自己的平均，并不等于"倒着走"。
        ready = self.n >= self.warmup
        # ★★ 预热期必须**按住 W**。
        #   预热的目的是收集统计量，可站着不动收集到的统计量是**退化的**：
        #   画面静止 → r 收敛到不动点 → 三条轴变成常数 → dev≈0 →
        #   scale 塌到 ~0 → z 爆到 ±5 → z_thrust=−5 永不满足"按 W" →
        #   角色继续不动 → 死锁。实测第二三遍就是这么卡死的（净位移 0）。
        #   让它先走着，世界一直在动，统计量才有意义。
        if not ready:
            self.n += 1
            self.state.update(w=True, s=False, d=False, a=False)
            self.last = dict(raw=s, z=z, state=dict(self.state), ready=False,
                             key_w=True, key_s=False, key_a=False, key_d=False,
                             turn_dx=0, turn_z=0.0, action="预热·前进")
            return self.last

        zt, zs = z["thrust"], z["strafe"]
        # 安全网：z 再怎么爆也不让它把阈值判定带飞
        zt = float(np.clip(zt, -4.0, 4.0))
        zs = float(np.clip(zs, -4.0, 4.0))
        w_on = (self.state.get("w", False) and zt > self.thr_w - self.hyst) or \
               (zt > self.thr_w)
        s_on = (self.state.get("s", False) and zt < self.thr_s + self.hyst) or \
               (zt < self.thr_s)
        d_on = (self.state.get("d", False) and zs > self.thr_strafe - self.hyst) or \
               (zs > self.thr_strafe)
        a_on = (self.state.get("a", False) and zs < -self.thr_strafe + self.hyst) or \
               (zs < -self.thr_strafe)
        if w_on:                      # W 与 S 互斥
            s_on = False
        if d_on:
            a_on = False
        self.state.update(w=w_on, s=s_on, d=d_on, a=a_on)

        self.n += 1
        # 转身限幅（预热期已在上面提前 return，这里 ready 必为真）
        tz = float(np.clip(z["turn"], -self.z_clip, self.z_clip))
        self.last = dict(
            raw=s, z=z, state=dict(self.state), ready=ready,
            key_w=w_on, key_s=s_on, key_a=a_on, key_d=d_on,
            turn_dx=int(tz * self.turn_gain * 40),
            turn_z=tz,
            action=self._describe(),
        )
        return self.last

    def _describe(self) -> str:
        st = self.state
        parts = []
        if st.get("w"):
            parts.append("前进")
        if st.get("s"):
            parts.append("后退")
        if st.get("a"):
            parts.append("左移")
        if st.get("d"):
            parts.append("右移")
        if abs(self.last.get("z", {}).get("turn", 0.0)) > 0.8:
            parts.append("转视角")
        return " + ".join(parts) if parts else "停"

    def describe_axes(self) -> str:
        z = self.last.get("z", {})
        return (f"推力 z={z.get('thrust',0):+5.2f}   "
                f"横移 z={z.get('strafe',0):+5.2f}   "
                f"转身 z={z.get('turn',0):+5.2f}")


if __name__ == "__main__":
    # 自检：喂随机状态，看三条轴有没有在动
    import time
    pm = PopulationMotor()
    rng = np.random.default_rng(0)
    n_all = 211577
    print(f"\n{'拍':>4s} {'t':>6s}  轴线")
    t0 = time.time()
    r = np.zeros(n_all, dtype=np.float32)
    for i in range(40):
        r = 0.7 * r + 0.5 * rng.normal(0, 1, n_all).astype(np.float32)
        pm.step(r, dt=0.4)
        if i % 4 == 0:
            print(f"{i:>4d} {time.time()-t0:>6.1f}  {pm.describe_axes()}")
