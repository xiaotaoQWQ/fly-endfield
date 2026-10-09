#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""combat.py — 战斗解码：pC1（攻击）与逼近回路（技能）。

设计依据：《果蝇战斗方案-交接.md》。逐条对应——

§三  攻击接**雄蝇攻击回路**（pC1 = P1 簇），技能接**逼近/逃逸回路**（LC/LPLC2）。
     理由：果蝇的逼近回路驱动的是"远离"。把恐惧接到攻击上，不是"读它"，
     是"翻译"它。两条通道来自两套功能不同的真实回路，行为上自然分得开：
     **怕就放大招，凶就上去打。**
     MaleCNS 本来就是一只雄蝇的全中枢，pC1 在里面（实测 156 个）。

§五  ★ 用**一阶差分**触发，不用活动水平。
     持续高激活不产生输出，因为它在高位的方向导数是零 —— 这一刀直接断掉
     "怕 → 打 → 锁敌拉近 → 更怕"的饱和环。饱和了就等于挂机脚本，作品就废了。
     冷却窗口保留，但从主力降级成兜底。

§六  ★ 活动状态必须**跨帧保留**。
     本项目每拍展开 10 步速率更新，上一拍的残留只有 0.5^10 ≈ 1/1024 ——
     等于每拍重新收敛到不动点，逼近信号根本攒不起来，读到的只有单帧噪声。
     解法是**读出层泄漏积分器**：不改模型（走路行为一字不动），
     只在读 pC1/LC 时做时间积分。这是"当前版本不变"前提下唯一能做到 §六 的办法。

§二  去抖三重：阈值 + 滞回 + 最小间隔（默认 300 ms）。

§八  归因：每次按键标来源、画变化率、显示"犹豫度"。
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph")

# 攻击回路：pC1 = P1 簇（雄性特异，驱动顶撞/冲撞）
ATTACK_PREFIX = "pC1"
# 逼近回路：小叶柱状细胞里的 looming 检测器
SKILL_TYPES = ("LPLC2", "LC4")


def find_combat_populations(meta_path: str | None = None):
    """从注释里找出两条回路的神经元 idx。返回 (attack_idx, skill_idx)。"""
    import pandas as pd
    p = meta_path or os.path.join(GRAPH, "graph_meta.feather")
    m = pd.read_feather(p).sort_values("idx").reset_index(drop=True)
    t = m["type"].astype(str)
    atk = m.loc[t.str.startswith(ATTACK_PREFIX), "idx"].to_numpy(np.int64)
    skl = m.loc[t.isin(SKILL_TYPES), "idx"].to_numpy(np.int64)
    return atk, skl


class CombatDecoder:
    """两条回路 → 攻击键 / 技能键。全部状态都可跨帧保留。"""

    def __init__(self, attack_idx, skill_idx, tau: float = 1.2,
                 tap_sec: float = 0.12, min_interval: float = 0.30,
                 z_enter: float = 1.0, z_exit: float = 0.5,
                 sustain_sec: float = 0.0, sustain_z: float = 0.3,
                 warmup: int = 20, verbose: bool = True):
        # ★ 阈值 1.0/0.5 是**真实数据标定**出来的，不是猜的：
        #   实跑 60 秒里攻击 z ∈ [−2.29, +1.45]（剔除预热后中位 +0.31、std 0.50），
        #   原来按合成刺激定的 1.6 一次都到不了。
        #   在真实 CSV 上回放上升沿逻辑：(1.0, 0.5) 给出攻击 2 次 + 技能 4 次 /分钟。
        #   注意这套信号**每分钟最多只能给出 5~7 次触发** —— 这是回路本身的性质，
        #   不是阈值卡得太紧。
        self.atk = np.asarray(attack_idx, np.int64)
        self.skl = np.asarray(skill_idx, np.int64)
        self.tau = float(tau)
        self.tap_sec = float(tap_sec)
        self.min_interval = float(min_interval)
        # ★ 战斗中「持续攻击」：见 _channel 里的说明。
        self.sustain_sec = float(sustain_sec)
        self.sustain_z = float(sustain_z)
        self.z_enter = float(z_enter)
        self.z_exit = float(z_exit)
        self.warmup = int(warmup)

        # ★ 跨帧保留的状态（§六）
        self.lvl = {"attack": None, "skill": None}    # 泄漏积分器
        self.prev = {"attack": 0.0, "skill": 0.0}     # 上一拍的积分值
        self.base = {"attack": None, "skill": None}   # 水平的自适应基线
        self.var = {"attack": 1e-6, "skill": 1e-6}
        self.armed = {"attack": True, "skill": True}  # 滞回：掉下来才重新武装
        self.z = {"attack": 0.0, "skill": 0.0}
        self.last_t = {"attack": -1e9, "skill": -1e9}
        self.n = 0
        self.cnt = {"fly": 0, "cooldown": 0, "script": 0}
        self.urge = 1.0
        self.last: dict = {}
        if verbose:
            print(f"战斗解码：攻击 pC1 {len(self.atk)} 个 · "
                  f"逼近 {'/'.join(SKILL_TYPES)} {len(self.skl)} 个")
            print(f"  上升沿触发 z>{z_enter:.1f}（滞回 {z_exit:.1f}）· "
                  f"按压 {tap_sec*1000:.0f}ms · 冷却 {min_interval*1000:.0f}ms")
            if sustain_sec > 0:
                print(f"  战斗中持续攻击：z>{sustain_z:.2f} 时每 "
                      f"{sustain_sec*1000:.0f}ms 一击")
            else:
                print("  战斗中持续攻击：关闭")

    def set_urge(self, urge: float):
        """提高/降低攻击欲望：把进入阈值乘上这个系数（<1 = 更爱打）。

        交接需求：**左上角出现红色叉号时增加攻击欲望** —— 那是游戏在战斗中的标志。
        实现方式是降阈值而不是直接按键：果蝇仍然要走完"上升沿"这一整套判断，
        变的只是它有多容易被点燃。
        """
        self.urge = float(urge)

    # ------------------------------------------------------------------
    def _channel(self, k: str, raw: float, dt: float, now: float,
                 combat: bool = False):
        """单通道：积分 → 水平 z → **上升沿**触发。

        ★ 为什么触发用「水平的上升沿」而不是「变化率超阈」：
          一开始按 §五 字面实现成 der > 阈值，结果**没有可用工作点**——
          实测信号导数 4e-4/s、导数噪声 2.2e-3/s，信噪比只有 0.18，
          信号被噪声埋住：阈值调低则高位平坦期乱触发，调高则上升期一次不触发。
          根因是**微分会放大噪声**。

          上升沿触发在概念上就是 §五 要的"正在快速上升的那一刻"
          （沿 = 离散导数），但抗噪好得多：水平已经被泄漏积分器平滑过。
          §五 真正要防的是「持续高激活 → 输出变成常数」——
          上升沿 + 滞回恰好解决：z 一旦站上阈值就不再产生新沿，
          必须掉回 z_exit 以下才重新武装。
        """
        # 1) 泄漏积分器（跨帧保留，§六）：让"危险"能一路垒起来
        a = 1.0 - np.exp(-dt / max(self.tau, 1e-3))
        if self.lvl[k] is None:
            self.lvl[k] = raw
            self.prev[k] = raw
            self.base[k] = raw
        self.lvl[k] += a * (raw - self.lvl[k])

        # 2) 变化率（§八 要画它；也是"沿"的连续版）
        der = (self.lvl[k] - self.prev[k]) / max(dt, 1e-3)
        self.prev[k] = self.lvl[k]

        # 3) 水平的自适应基线 + 尺度（量级随输入漂移，固定阈值必失效）
        ab = 1.0 - np.exp(-dt / max(self.tau * 4.0, 1e-3))
        dev = self.lvl[k] - self.base[k]
        self.base[k] += ab * dev
        self.var[k] = (1 - ab) * self.var[k] + ab * dev * dev
        scale = float(np.sqrt(max(self.var[k], 1e-12)))
        z = float(np.clip(dev / max(scale, 1e-9), -6.0, 6.0))
        self.z[k] = z

        # 4) 去抖三重（§二）：上升沿 + 滞回 + 最小间隔
        #    ★ 返回的是**单拍事件**（fired），不是"按住"状态：
        #      按压时长 120ms 远短于决策拍 400ms，跨拍状态机根本看不到沿。
        #      所以由执行层做一次阻塞式点击。
        ready = self.n >= self.warmup
        src, fired = "-", False
        enter = self.z_enter * self.urge      # ★ 攻击欲望：战斗标记亮起时阈值被压低
        if ready and z > enter and self.armed[k]:
            self.armed[k] = False          # 站上阈值后就不再产生新沿
            if now - self.last_t[k] >= self.min_interval:
                self.last_t[k] = now
                self.cnt["fly"] += 1
                src, fired = "fly", True
            else:
                self.cnt["cooldown"] += 1
                src = "cooldown"
        elif z < self.z_exit:
            self.armed[k] = True           # 掉下来才重新武装

        # 5) ★ 战斗中「持续攻击」（可选，sustain_sec > 0 时启用）
        #    为什么需要它：上升沿触发的频率被信号本身卡死 —— 泄漏积分 τ=1.2s
        #    决定了 z 的振荡周期，实测无论阈值怎么降，**天花板就是 ~10 次/分**
        #    （在真实 CSV 上回放验证过）。要更凶只能换机制。
        #    闸门仍然是果蝇的：**z 必须高过 sustain_z**（默认 0.3 ≈ 62 分位）
        #    才持续攻击，掉下去就停。我们定的只是节奏。
        if (not fired and combat and ready and self.sustain_sec > 0
                and z > self.sustain_z
                and now - self.last_t[k] >= self.sustain_sec):
            self.last_t[k] = now
            self.cnt["fly"] += 1
            fired, src = True, "sustain"

        return dict(lvl=float(self.lvl[k]), der=float(der), z=z,
                    fired=fired, src=src)

    # ------------------------------------------------------------------
    def step(self, r: np.ndarray, dt: float, now: float,
             combat: bool = False) -> dict:
        a = self._channel("attack", float(r[self.atk].mean()), dt, now, combat)
        s = self._channel("skill", float(r[self.skl].mean()), dt, now, combat)
        self.n += 1

        # 犹豫度（§八）：第一名与第二名的活动差 —— 差值小 → 它在犹豫
        la, ls = a["lvl"], s["lvl"]
        tot = abs(la) + abs(ls) + 1e-9
        hesit = (la - ls) / tot

        self.last = dict(
            attack=a["fired"], skill=s["fired"],
            lvl_a=a["lvl"], lvl_s=s["lvl"],
            der_a=a["der"], der_s=s["der"],
            z_a=a["z"], z_s=s["z"],
            src_a=a["src"], src_s=s["src"],
            hesit=float(hesit),
            n_atk=len(self.atk), n_skl=len(self.skl),
            ready=self.n >= self.warmup,
            cnt=dict(self.cnt),
        )
        return self.last

    def reset(self):
        self.lvl = {"attack": None, "skill": None}
        self.prev = {"attack": 0.0, "skill": 0.0}
        self.scale = {"attack": 1e-3, "skill": 1e-3}
        self.z = {"attack": 0.0, "skill": 0.0}
        self.edge = {"attack": False, "skill": False}
        self.n = 0


# ---------------------------------------------------------------- 战斗标记
class CombatMarker:
    """左上角的**红色叉号** = 游戏进入战斗状态 → 提高攻击欲望。

    ⚠ 这个标记在游戏里靠眼睛认，但程序只能靠颜色。做法是数「强红」像素：
      R 够亮、且明显高于 G/B。红色在不同光照下差异很大，所以阈值给得宽松，
      真正起作用的是**面积**（一个叉号的笔画有几十个像素）。

    `bbox` 会报告红块的包围盒 —— 第一次标定时靠它确认叉号到底在哪，
      不用靠猜。
    """

    def __init__(self, rect, min_px: int = 30, verbose: bool = True):
        self.rect = tuple(int(v) for v in rect)
        self.min_px = int(min_px)
        self.on = False
        self.red_px = 0
        self.bbox = None
        self.n = 0
        self.hits = 0
        if verbose:
            print(f"战斗标记检测：区域 {self.rect} · 强红像素 > {min_px} 判为战斗中")

    def step(self, rgb: np.ndarray) -> bool:
        a = rgb.astype(np.int16)
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        m = (r > 110) & (r - g > 50) & (r - b > 40)
        self.red_px = int(m.sum())
        self.on = self.red_px >= self.min_px
        if self.on:
            ys, xs = np.nonzero(m)
            self.bbox = (int(xs.min()), int(ys.min()),
                         int(xs.max()), int(ys.max()))
            self.hits += 1
        else:
            self.bbox = None
        self.n += 1
        return self.on

    def describe(self) -> str:
        if not self.on:
            return f"无标记（红像素 {self.red_px}）"
        return f"★ 战斗中（红像素 {self.red_px}，包围盒 {self.bbox}）"


class EvacMarker:
    """左上角的**"撤离"面板**出现 = 在副本里 → 让果蝇进入战斗状态。

    为什么不用红叉：红叉只在**已经打起来**之后才亮，而副本里要的是
      "一进副本就进入觅食状态"。撤离按钮是副本的常驻 UI，进副本就有。

    做法（UI 位置固定，所以不做全图搜索 —— 那样又慢又没必要）：
      ① 抓固定区域（实测 213×53，内容坐标 (24,45)-(236,97)）
      ② 跟预先存好的模板算**归一化相关**（NCC）
         —— 不用逐像素相等：面板会有轻微的抗锯齿/亮度差异，
            而且"按下"和"未按下"也会有变化
      ③ 再卡一道"近白填充率"，防止画面刚好有一块白墙撞上

    模板怎么来的：ef_evac_probe.py 抓一次，存 templates/evac_panel.npy。
    """

    def __init__(self, rect, tpl_path: str = None, corr_min: float = 0.55,
                 fill_min: float = 0.45, verbose: bool = True):
        self.rect = tuple(int(v) for v in rect)
        self.corr_min = float(corr_min)
        self.fill_min = float(fill_min)
        self.on = False
        self.corr = 0.0
        self.fill = 0.0
        self.n = 0
        self.hits = 0
        if tpl_path is None:
            tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "templates", "evac_panel.npy")
        self.tpl_path = tpl_path
        t = np.load(tpl_path)
        if t.ndim == 3:
            t = t.astype(np.float32).mean(axis=2)
        self.tpl = t.astype(np.float32)
        # 模板自身的均值/标准差，算 NCC 时反复用
        self._tm = float(self.tpl.mean())
        self._ts = float(self.tpl.std()) + 1e-6
        self._th, self._tw = self.tpl.shape
        if verbose:
            print(f"撤离标记检测：区域 {self.rect} · 模板 {self._tw}×{self._th} · "
                  f"相关 > {corr_min} 且 白填充 > {fill_min} 判为副本内")

    def step(self, rgb: np.ndarray) -> bool:
        a = np.asarray(rgb)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        a = a.astype(np.float32)
        # 抓到的区域可能和模板差一两个像素，裁到模板大小
        h = min(self._th, a.shape[0])
        w = min(self._tw, a.shape[1])
        g = a[:h, :w].mean(axis=2)
        t = self.tpl[:h, :w]
        gm = float(g.mean())
        gs = float(g.std()) + 1e-6
        self.corr = float(((g - gm) * (t - self._tm)).mean() / (gs * self._ts))
        r, gg, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        self.fill = float(((r > 200) & (gg > 200) & (b > 200)).mean())
        self.on = self.corr >= self.corr_min and self.fill >= self.fill_min
        if self.on:
            self.hits += 1
        self.n += 1
        return self.on

    def describe(self) -> str:
        return (f"{'★ 副本内（撤离按钮在）' if self.on else '副本外'}  "
                f"相关 {self.corr:+.3f}  白填充 {self.fill:.3f}")


class ComboMarker:
    """连携技（E）提示 —— **只在战斗中、且能用的时候才出现**。

    用户要求：战斗中出现 E 提示 **且 饥饿值到 1** → 按 E。

    实测标定（ef_combo_collect.py 采 40 帧战斗帧 + ef_combo_scan.py 扫）：
      · 提示 = **队友圆形头像 + 右下角深色 E 徽标**
      · 位置**基本固定**在 1280×720 的 (758,300)，换算内容坐标 (1516,600)
      · 但淡入淡出时会**偏移约 10px**，所以不能抓死区域，要做小范围搜索
      · 相关值：真提示 +0.606 ~ +1.000，假的最大 +0.267
        → 阈值 0.45（宁可多按一次，也别错过连携技）

    实现：抓一块带余量的区域 → 降采样 2 倍回到 1280×720 尺度（模板就是那尺度取的）
          → FFT 归一化相关 → 取最大值。FFT 是必须的：逐点搜 5~10 秒/帧，
          2.5Hz 下根本来不及（纯 numpy 扫 200 帧要半小时）。
    """

    def __init__(self, rect=None, tpl_path: str = None, corr_min: float = 0.45,
                 verbose: bool = True):
        # rect = **内容坐标**下要抓的区域（含余量）。None 则用标定值。
        self.rect = tuple(int(v) for v in rect) if rect else (1476, 560, 208, 196)
        self.corr_min = float(corr_min)
        self.on = False
        self.corr = 0.0
        self.n = 0
        self.hits = 0
        self.tpl = None
        if tpl_path is None:
            tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "templates", "combo_e.npy")
        self.tpl_path = tpl_path
        if os.path.exists(tpl_path):
            t = np.load(tpl_path)
            if t.ndim == 3:
                t = t.astype(np.float32).mean(axis=2)
            self.tpl = t.astype(np.float32)
            ready = True
        else:
            ready = False
        if verbose:
            if ready:
                print(f"连携技检测：区域 {self.rect}（内容坐标）· 模板 "
                      f"{self.tpl.shape[1]}×{self.tpl.shape[0]}"
                      f"（E 键徽标，内容尺度）· 相关 > {corr_min}")
            else:
                print(f"连携技检测：**未标定**（缺 {tpl_path}）→ 不会按 E")

    def step(self, rgb: np.ndarray) -> bool:
        if self.tpl is None:
            return False
        a = np.asarray(rgb)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        # ★★ **不降采样**。原来这里写的是 a[::2, ::2]（因为旧模板是
        #   1280 尺度取的）。实测那套模板根本是错的东西（一个圆形徽标，
        #   不是连携技的 E 键提示），而且降采样之后 30×28 的 E 徽标
        #   只剩 15×14 —— 太小，相关值噪声极大，判不出来。
        #   改成在**内容分辨率**上直接匹配：模板 30×28，判别力足够。
        g = a.astype(np.float32).mean(axis=2)
        th, tw = self.tpl.shape
        if g.shape[0] < th or g.shape[1] < tw:
            self.on = False
            return False
        t = self.tpl - self.tpl.mean()
        tn = float(np.sqrt((t * t).sum())) + 1e-6
        ones = np.ones_like(t)
        num = fftconvolve(g, t[::-1, ::-1], mode="valid")
        s1 = fftconvolve(g, ones[::-1, ::-1], mode="valid")
        s2 = fftconvolve(g * g, ones[::-1, ::-1], mode="valid")
        n = t.size
        var = np.maximum(s2 - s1 * s1 / n, 0.0)
        # ★★ 防零方差。归一化相关 = num / (std(x)·std(t))，
        #   而搜索区里只要有一块**平坦区域**（纯色/黑屏/加载画面），
        #   std(x) ≈ 0 → 分母被 1e-6 兜住 → 相关值**爆炸**。
        #   实测：录制时抓到 +744,011,840 这种数，然后 >= 0.45 判定为
        #   有提示 → E 在莫名其妙的时刻狂按。
        #   方差太小的地方直接判 0：那里没有可匹配的结构。
        den = np.sqrt(var) * tn
        sc = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)
        self.corr = float(sc.max())
        self.on = self.corr >= self.corr_min
        if self.on:
            self.hits += 1
        self.n += 1
        return self.on

    def describe(self) -> str:
        if self.tpl is None:
            return "未标定"
        return (f"{'★ 连携技可用' if self.on else '不可用'}  相关 {self.corr:+.3f}")


# ---------------------------------------------------------------- 自检
def _selftest():
    """离线自检：验证 §五 的两条要求。

    ① 活动**加速上升**时要按键（真实的逼近信号是角尺寸随时间双曲膨胀，
       不是线性斜坡 —— 第一版自检用线性斜坡，结果把设计缺陷测成了"通过"）
    ② 活动**高位平坦**时不能按键（这一条才是 §五 真正要防的饱和）
    """
    rng = np.random.default_rng(0)
    atk, skl = find_combat_populations()
    d = CombatDecoder(atk, skl, warmup=5, verbose=True)
    n_all = 211577
    r = np.zeros(n_all, dtype=np.float32)
    t = 0.0
    dt = 0.4
    presses = {"rise": 0, "flat": 0, "fall": 0}
    for i in range(150):
        t += dt
        if i < 50:                    # 逼近：加速膨胀
            phase, lvl = "rise", 0.004 * (i / 50.0) ** 2
        elif i < 100:                 # 饱和：高位平坦（贴脸了，膨胀率归零）
            phase, lvl = "flat", 0.004
        else:                         # 退开
            phase, lvl = "fall", 0.004 * (1 - (i - 100) / 50.0)
        lvl += rng.normal(0, 0.0004)  # 加噪声，别测一个完美信号
        r[:] = 0.0
        r[atk] = lvl
        r[skl] = lvl * 0.5
        if d.step(r, dt, t)["attack"]:
            presses[phase] += 1
    print("\n自检结果（攻击键按下的拍数）：")
    for k, v in presses.items():
        tag = {"rise": "加速上升期（应>0）", "flat": "高位平坦期（应≈0）",
               "fall": "下降期（应≈0）"}[k]
        print(f"  {tag:22s} {v:3d} 拍")
    print(f"\n  来源统计：{d.cnt}")
    ok = presses["rise"] >= 1 and presses["flat"] <= 1 and presses["fall"] <= 1
    print(f"\n  {'✅ 通过' if ok else '❌ 失败'}")
    return ok


if __name__ == "__main__":
    _selftest()
