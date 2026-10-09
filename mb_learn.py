#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mb_learn.py — 蘑菇体学习：三因子规则 + 饥饿内稳态 + 落盘 + 衰减。

对应交接文档：
  §九  三因子规则：**前突触活跃 且 多巴胺释放 → 那个突触的权重改变**。
       前活动的符号决定往哪改，多巴胺决定"现在改"。
       作用范围只在蘑菇体 KC→MBON 那一小块（"这个项目里唯一一块真正的新东西"）。
       奖赏/惩罚注入点按用户决定走 **DAN 簇**：PAM=奖赏，PPL1/PPL2=惩罚。
  §五  饱足感：饥饿值进食降、随时间升，同时调制
       ① 对食物线索的敏感度 ② 攻击驱动力 ③ **多巴胺注入强度**
       （"多巴胺强度 = 奖赏 × 饥饿程度"）
  §十  落盘：存**补丁**不存整个脑子；稀疏表；原子写；带版本号；
       定期快照 + 关窗前补一次。**衰减**：长期没被强化的突触回落基线。

设计取舍（写在明处）：
  · 规则形式用 dwarf 化的经典式 Δw = η · DA · (x_pre − θ)。
    θ 取 KC 活动的滑动均值 —— 这样"高于自己平均的 KC"被强化、低于的被削弱，
    天然给出符号，不需要额外假设。
  · DA 取 PAM 均值 − PPL 均值，**不分 MBON 区室**。真实蘑菇体里 DAN 是按
    区室局部投射的，这里做了简化；先跑通再谈分区室。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))

# 可塑性规则版本 —— 换规则/换学习率时必须改，否则旧补丁会被硬套
RULE_VERSION = "mb-3factor-v1"
BASE_HASH_LEN = 12


# ======================================================================
class Hunger:
    """内稳态饥饿值 h ∈ [0,1]：进食让它降，随时间慢慢回升（§五）。

    没有这一环，循环就退化成单调正反馈、又回到饱和。
    有了它才有节律：饿 → 找 → 吃 → 饱 → 停 → 又饿。
    """

    def __init__(self, h0: float = 1.0, rise: float = 0.010,
                 eat: float = 0.35, lo: float = 0.0, hi: float = 1.0):
        self.h = float(h0)
        self.rise = float(rise)     # 每秒回升
        # ★ 任务完成后的"吃饱"窗口（用户要求）：
        #     饥饿值**清空**（回到 lo）+ **10 秒内不增长** + 之后**恢复正常**。
        #
        #   语义上这就是"打赢一局 = 吃了一顿" ——
        #   比原来那个"永久冻结"说得通：吃饱了会饱一会儿，但过一阵还是会饿。
        #   用**倒计时**而不是"冻结标志 + 时间戳"：
        #     step() 只拿到 dt，拿不到绝对时间；倒计时天然对 dt 正确，
        #     而且主循环降频/卡顿时也不会算错（按实际经过的时间扣）。
        self.hold_left = 0.0
        self.eat = float(eat)       # 每次进食下降
        self.lo, self.hi = float(lo), float(hi)

    def clear(self, hold_sec: float = 10.0):
        """★ 任务完成：清空饥饿值，并在 hold_sec 秒内不增长。

        返回清空前的值（供日志）。
        """
        before = self.h
        self.h = self.lo
        self.hold_left = float(hold_sec)
        return before

    def step(self, dt: float):
        # ★ "吃饱"窗口内不回升，倒计时按实际经过的时间扣。
        if self.hold_left > 0.0:
            self.hold_left = max(0.0, self.hold_left - float(dt))
            return
        self.h = min(self.hi, self.h + self.rise * float(dt))

    def ate(self, amount: float = 1.0):
        """吃到东西。返回本次实际下降量（供日志）。"""
        before = self.h
        self.h = max(self.lo, self.h - self.eat * amount)
        return before - self.h

    # --- 三处调制（§五）---
    def odor_gain(self) -> float:
        """① 对食物线索的敏感度：越饿越敏感（0.3 倍底噪 → 1.3 倍）"""
        return 0.3 + 1.0 * self.h

    def drive_gain(self) -> float:
        """② 攻击驱动力：越饿越想上"""
        return 0.2 + 0.8 * self.h

    def da_gain(self) -> float:
        """③ 多巴胺注入强度：**吃饱了再吃也没那么爽**"""
        return 0.1 + 0.9 * self.h

    def describe(self) -> str:
        return f"h={self.h:.3f}"


# ======================================================================
class MushroomBodyLearner:
    """三因子规则，作用在 KC→MBON 的突触上。

    为什么单独拎出来算：
      W 是 211,577² 的稀疏矩阵（2600 万条边），每拍重建一次 CSR 太贵。
      而**可学习的边只有 KC→MBON 那 6 万条**，抽成一个稠密子矩阵
      (MBON × KC = 97 × 4064 ≈ 39 万个数) 直接矩阵乘，代价可以忽略。
      于是 r 的更新写成：W@r + Δ，其中 Δ 只落在 MBON 那些行上。
    """

    def __init__(self, W, kc_idx, mbon_idx, eta: float = 0.05,
                 theta_tau: float = 5.0, decay: float = 2e-4,
                 clip: float = 3.0, verbose: bool = True):
        self.kc = np.asarray(kc_idx, np.int64)
        self.mbon = np.asarray(mbon_idx, np.int64)
        self.eta = float(eta)
        self.theta_tau = float(theta_tau)
        self.decay = float(decay)
        self.clip = float(clip)

        # 只抽 KC→MBON 的边：W[mbon][:, kc] → 稠密 (n_mbon, n_kc)
        sub = W[np.ix_(self.mbon, self.kc)]
        self.w0 = np.asarray(sub.todense(), dtype=np.float32) \
            if sp.issparse(sub) else np.asarray(sub, dtype=np.float32)
        self.w = self.w0.copy()
        # 基础连接图的指纹（落盘要做版本校验）
        self.base_hash = hashlib.sha256(
            np.ascontiguousarray(self.w0).tobytes()).hexdigest()[:BASE_HASH_LEN]

        self.theta = None            # KC 活动的滑动均值
        self.da = 0.0                # 当前多巴胺（>0 奖赏 / <0 惩罚）
        self.da_pulse = 0.0          # 待注入的多巴胺脉冲（我们给的）
        self.n_updates = 0
        self.drift = 0.0             # 权重漂移量（相对基线的 L1）

        # ---- 分区多巴胺（DAN→MBON 投射）----
        # 从 out/mb_groups.npz 读实测标定值；没有就退回"所有 MBON 一样"
        # （那样就退化成原来的广播版本，学不出技能选择，但至少不崩）
        self.grp = None
        gp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "out", "mb_groups.npz")
        if os.path.exists(gp):
            z = np.load(gp, allow_pickle=False)
            if tuple(z["mbon_idx"]) == tuple(self.mbon):
                self.pam_w = z["pam_w"].astype(np.float32)
                self.ppl_w = z["ppl_w"].astype(np.float32)
                self.grp = z["grp"].astype(np.int32)
            else:
                print("  ⚠ mb_groups.npz 的 MBON 顺序对不上 → 退回均匀 DA")
        if self.grp is None:
            self.pam_w = np.ones(len(self.mbon), np.float32)
            self.ppl_w = np.ones(len(self.mbon), np.float32)
        # 归一到均值为 1：分区差异只影响**相对**分配，不改变总强度
        self.pam_w = self.pam_w / (float(self.pam_w.mean()) + 1e-9)
        self.ppl_w = self.ppl_w / (float(self.ppl_w.mean()) + 1e-9)
        self.da_v = np.zeros(len(self.mbon), np.float32)
        if verbose:
            print(f"蘑菇体学习：KC {len(self.kc):,} → MBON {len(self.mbon)}，"
                  f"可塑性边 {int((self.w0 != 0).sum()):,}")
            print(f"  η={eta}  θ时间常数={theta_tau}s  衰减={decay}  "
                  f"基线指纹={self.base_hash}  规则={RULE_VERSION}")

    # ------------------------------------------------------------------
    def deliver(self, amount: float):
        """投递一次奖赏(+)/惩罚(−)脉冲。强度已含饥饿调制（§五 ③）。"""
        self.da_pulse += float(amount)

    def step(self, r: np.ndarray, dt: float, reward_gain: float = 1.0):
        """一拍：读 DAN 活动 → 更新 KC→MBON 权重 → 返回增量矩阵的贡献行。"""
        x = np.asarray(r[self.kc], dtype=np.float32)

        # θ：KC 活动的滑动均值（给出可塑性的符号）
        if self.theta is None:
            self.theta = float(x.mean())
        else:
            a = 1.0 - np.exp(-dt / max(self.theta_tau, 1e-3))
            self.theta += a * (float(x.mean()) - self.theta)

        # 多巴胺：外部脉冲 × 饥饿强度（"吃饱了再吃也没那么爽"）
        self.da = self.da_pulse * reward_gain
        self.da_pulse = 0.0
        # ★ 按 DAN→MBON 投射把这一发多巴胺**分发到每个 MBON**。
        #   · 奖赏（正脉冲）走 PAM 通路 → 组0 拿到最多
        #   · 惩罚（负脉冲）走 PPL 通路 → 组2/3 拿到最多
        #   这是"哪个分区被强化/削弱"的来源；具体记到哪条突触上，
        #   还要再乘一个活跃门（见下面）。
        if self.da > 0:
            self.da_v = self.pam_w * self.da
        elif self.da < 0:
            self.da_v = -self.ppl_w * self.da
        else:
            self.da_v = np.zeros_like(self.pam_w)

        delta = None
        if abs(self.da) > 1e-9:
            # ★★ 三因子 × **分区**：前突触活跃 × 多巴胺 × **谁在主导行为**。
            #
            #   ⚠ 踩了两个坑，都在这里：
            #
            #   坑一（前突触活跃）：一开始写成 Δw = η·DA·(x−θ)，**连静息的突触也改**。
            #     后果：① 给"零活动状态"发奖赏时 x=0 而 θ 非零，(x−θ)<0 乘正 DA
            #     → 整片突触被统一削弱，逐试次加重直到崩溃；② 被惩罚的气味反而上升。
            #     交接 §九 原话是"前突触**活跃** 且 多巴胺释放 → 才改"，
            #     所以活跃那一半必须显式夹掉：只用 max(x−θ, 0)。
            #
            #   坑二（扫描广播）：原来是 `w += g[None, :]` —— **所有 MBON 拿到
            #     完全一样的增量**，相对模式永远不变，**学不出"该用哪个技能"**。
            #     真实蘑菇体里多巴胺是**按区室局部投射**的（PAM 只管一部分 MBON，
            #     PPL 管另一部分），所以改成按 DAN→MBON 连接给每个 MBON 自己的 DA。
            #
            #   坑三（记功给谁）：光有分区 DA 还不够 —— 那是**固定的先天偏向**
            #     （组0 永远被奖赏强化）。要让奖励记到"当时是谁在主导行为"的头上，
            #     还得乘一个**活跃门**：当时输出大的 MBON 才被改。
            #     这就是信用分配 —— 下次闻到同一个味道，主导过的那个 MBON 更强，
            #     于是它对应的技能更容易被选中。
            g_pre = np.maximum(x - self.theta, 0.0)             # (n_kc,)
            if np.any(g_pre):
                out = self.w @ x                                 # (n_mbon,) 各 MBON 当前输出
                pos = np.maximum(out, 0.0)
                # ★ 活跃门必须有**上界**。第一版用 pos/mean(pos)，无上界 ——
                #   谁分数高谁拿到更大更新 → 更强 → 正反馈失控：
                #   实测 3 次奖赏就把组0 推到 1260、漂移 44968。
                #   tanh 把它夹进 [0,1)，同时保留"谁活跃谁记功"的相对性。
                gate = np.tanh(pos / (float(pos.mean()) + 1e-6))
                da_v = self.da_v * gate                          # (n_mbon,) 每个 MBON 的 DA
                self.w += self.eta * da_v[:, None] * g_pre[None, :]
                self.n_updates += 1

        # 衰减：长期没被强化的突触往基线回落（§十 "一只会忘事的果蝇"）
        if self.decay > 0:
            self.w += (self.w0 - self.w) * min(1.0, self.decay * dt * 60.0)

        np.clip(self.w, self.w0 - self.clip, self.w0 + self.clip, out=self.w)
        # 不引入基础连接图里不存在的边
        self.w[self.w0 == 0] = 0.0
        self.drift = float(np.abs(self.w - self.w0).sum())

    def group_scores(self, r: np.ndarray) -> np.ndarray:
        """四个分区的输出 —— **技能选择就看它**。

        每个 MBON 组自己的 KC→MBON 加权和，组内取均值（组大小差很多，
        取和会让大组永远赢）。

        学到的关联会改变这些分数：被奖赏强化过的组分数上升 → 下次更容易赢
        → 它对应的技能更容易被选中。这就是"果蝇自己判断该用哪个技能"。
        """
        x = np.asarray(r[self.kc], dtype=np.float32)
        out = self.w @ x
        if self.grp is None:
            k = max(1, len(out) // 4)          # 没分区数据就均分
            return np.array([out[j * k:(j + 1) * k].mean() for j in range(4)],
                            dtype=np.float32)
        return np.array([out[self.grp == j].mean() for j in range(4)],
                        dtype=np.float32)

    def contribution(self, r: np.ndarray) -> np.ndarray:
        """把 (w − w0) 的贡献算成加在 MBON 行上的增量向量。"""
        d = self.w - self.w0
        if not d.any():
            return np.zeros(len(r), dtype=np.float32)
        out = np.zeros(len(r), dtype=np.float32)
        out[self.mbon] = d @ np.asarray(r[self.kc], dtype=np.float32)
        return out

    def stats(self) -> dict:
        d = self.w - self.w0
        return dict(n_updates=self.n_updates, da=self.da, theta=self.theta,
                    drift=self.drift, changed=int((d != 0).sum()),
                    up=int((d > 0).sum()), down=int((d < 0).sum()),
                    wmax=float(d.max()), wmin=float(d.min()))

    # ------------------------------------------------------------------
    # 落盘（§十）：存补丁，不存整个脑子
    # ------------------------------------------------------------------
    def reset(self, verbose: bool = True) -> bool:
        """**清空学习** —— 权重回到基线 w0，漂移归零。

        ★ 用户要求加这个。为什么要它：
          w0 是**连接组的原始权重**（实测出来的），不是随便一个初始值。
          清空 = 回到那只**什么都没学过的果蝇**，而不是回到一个随机状态。
          做对照实验时，这是回到起点最彻底的一档 ——
          比读一个空槽位更明确（槽位可能存的就是某个中间状态）。

        ★ 与 load() 的区别：
          load 是换成另一份记忆，reset 是忘掉一切、回到先天。
        """
        self.w[...] = self.w0
        self.drift = 0.0
        self.n_updates = 0
        # ⚠ 不要在这里动 n_reward/n_punish —— 那两个是**统计量**，
        #   记在 forager 那边（"这一轮打死了几个"），不属于突触状态。
        #   清空记忆是忘掉权重，不是假装没打过。
        if verbose:
            print("  [记忆] 已清空 —— 权重回到连接组基线，漂移归零", flush=True)
        return True

    def save(self, path: str, extra: dict | None = None):
        """原子写：先写临时文件，成功了再改名覆盖。崩在写一半就没命了。"""
        d = self.w - self.w0
        ii, jj = np.nonzero(d)
        payload = dict(
            rule=RULE_VERSION,
            base_hash=self.base_hash,
            shape=np.array(self.w.shape, np.int32),
            i=ii.astype(np.int32), j=jj.astype(np.int32),
            w=d[ii, jj].astype(np.float32),
            theta=self.theta if self.theta is not None else np.nan,
            n_updates=self.n_updates,
            extra=json.dumps(extra or {}, ensure_ascii=False),
        )
        # ⚠ 坑：np.savez_compressed 在文件名不以 .npz 结尾时会**自动追加 .npz**。
        #   一开始用 suffix=".tmp"，结果它写到了 xxx.tmp.npz，
        #   而 os.replace 搬的是那个**空的** xxx.tmp —— 落盘出来是 0 字节，补丁全丢。
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                   suffix=".npz")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:          # 传文件对象，彻底避开后缀自动追加
                np.savez_compressed(f, **payload)
            # 写后自检：0 字节或回读条数不符就当失败，别悄悄把一只果蝇的命丢了
            if os.path.getsize(tmp) == 0:
                raise IOError("补丁写出来是 0 字节")
            with np.load(tmp, allow_pickle=False) as z:
                if len(z["i"]) != len(ii):
                    raise IOError("补丁回读条数不符")
            os.replace(tmp, path)               # ★ 原子改名
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return len(ii)

    def load(self, path: str) -> bool:
        """返回是否成功加载。版本/指纹不符就**老老实实丢掉**，不硬套。"""
        if not os.path.exists(path):
            return False
        try:
            z = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"  ⚠ 补丁读不出来（{type(e).__name__}），当作新果蝇")
            return False
        if str(z["rule"]) != RULE_VERSION:
            print(f"  ⚠ 规则版本不符（补丁 {z['rule']} / 现在 {RULE_VERSION}）"
                  "→ 丢掉，不硬套")
            return False
        if str(z["base_hash"]) != self.base_hash:
            print(f"  ⚠ 基础连接图指纹不符（补丁 {z['base_hash']} / 现在 "
                  f"{self.base_hash}）→ 丢掉，不硬套")
            print("     硬套出来的会是一只精神分裂的果蝇")
            return False
        if tuple(z["shape"]) != self.w.shape:
            print("  ⚠ 形状不符 → 丢掉")
            return False
        self.w = self.w0.copy()          # 从基线重来，再叠加补丁
        self.w[z["i"], z["j"]] += z["w"]
        self.n_updates = int(z["n_updates"])
        t = float(z["theta"])
        self.theta = None if np.isnan(t) else t
        self.drift = float(np.abs(self.w - self.w0).sum())
        return True


def innate_priority(h: float, hp, combat_age: float,
                    heal_hp: float = 0.5, roles=None) -> np.ndarray:
    """**先天技能轮换** —— 返回四个技能槽的先天优先级（0~1 量级）。

    ★ 技能槽的**含义可配置**（用户要求）。原来写死成
      「1 伤害高 · 2 挂削弱 · 3 增伤 · 4 回血」，现在由 `roles` 决定：
      roles[i] 是第 i+1 个技能槽的角色。

      支持的角色（用户给的清单）：
        attack 攻击 · buff 增伤 · block 格挡 · heal 回血
        gather 聚怪 · weaken 挂削弱 · none 不用

    为什么必须有一层先天的：
      果蝇从画面上看不出技能是干什么的（没有 mod）。光靠结果去学，
      **"回血"和"格挡"永远学不出来** —— "没挨打"不是信号，
      "后续打得动"也不是信号。这是信号问题，不是算法问题。
      所以先把语义当作**本能**写进来，学习只在上面微调优先级。
    """
    hh = float(max(0.0, min(1.0, h)))
    hp_ok = isinstance(hp, float) and hp == hp
    low = hp_ok and hp < heal_hp
    # ★ 格挡：**中了段血**才用。血很少时该回血、血很满时不必防 ——
    #   格挡是**预防**，回血是**补救**，两者分工不同，不能都挂在"血低"上。
    mid = hp_ok and heal_hp <= hp < min(0.95, heal_hp + 0.25)
    early = combat_age < 4.5
    midage = 4.5 <= combat_age < 9.5

    if roles is None:
        # 老口径：1 伤害高 · 2 挂削弱 · 3 增伤 · 4 回血（保持逐位一致，便于回归对比）
        p = np.zeros(4, dtype=np.float32)
        p[0] = 0.55 + 0.45 * hh
        if early:
            p[1] = 1.45
        elif midage:
            p[2] = 1.45
        if low:
            p[3] = 1.6 + 1.5 * (1.0 - float(hp) / heal_hp)
        return p

    p = np.zeros(4, dtype=np.float32)
    for i, role in enumerate(tuple(roles)[:4]):
        if role == "attack":
            # 主攻：**底**，永远在线。越饿越想打
            p[i] = 0.55 + 0.45 * hh
        elif role == "weaken":
            # 挂削弱：开战前段给**决定性权重**。
            # ⚠ 第一版给的是 1.00/0.90（比主攻还低或持平）——结果主攻天生压过
            #   轮换，实测 31 次技能里 30 次都是主攻，轮换等于没发生。
            #   1.45 明显高于主攻上限（1.0 + 学习偏移 0.30 = 1.30）。
            p[i] = 1.45 if early else 0.0
        elif role == "buff":
            # 增伤：早段放，之后交回主攻
            p[i] = 1.45 if (early or midage) else 0.0
        elif role == "block":
            p[i] = 1.55 if mid else 0.0
        elif role == "heal":
            # ★ 回血优先级最高，而且**学不坏**：
            #   主攻上限 1.30 < 回血底 1.60。
            p[i] = (1.6 + 1.5 * (1.0 - float(hp) / heal_hp)) if low else 0.0
        elif role == "gather":
            # ★ 聚怪：开战**那一拍**放（把敌人拢到一起），晚放没意义 ——
            #   怪已经散开了。
            p[i] = 1.55 if early else 0.0
        # "none" / 未知 → 保持 0，这一槽永远不会被选中
    if not p.any():
        # 全配成 none（或全不匹配）→ 退回第一个可用槽，
        # 否则会"永远不放技能"，那是配置事故不是行为
        for i, role in enumerate(tuple(roles)[:4]):
            if role not in (None, "none"):
                p[i] = 0.60 + 0.40 * hh
                break
    return p



def select_skill(gscores: np.ndarray, h: float, hp, combat_age: float,
                 learn_gain: float = 0.30, heal_hp: float = 0.5,
                 roles=None) -> int:
    """先天优先级 + 学习微调 → 选一个技能（返回 1..4）。

    `gscores` 是蘑菇体四个分区的输出（见 MushroomBodyLearner.group_scores）。
    组 j 天生对应技能 j+1 —— 组0 是奖赏支配的那一组，对应"主攻"，
    这个对齐是刻意的：学到的正向关联直接把主攻推上去。

    ⚠ 学习偏移必须**有界**。第一版写的是 `learn_gain * g / mean|g|`，
      没有上界 —— 实测组0 被强化后偏移到 0.92，直接盖掉了"血 25% 该回血"
      （保命规则的先天差距只有 0.14）。所以改成 tanh，夹进 ±learn_gain。
    """
    p = innate_priority(h, hp, combat_age, heal_hp, roles=roles)
    if gscores is not None and learn_gain > 0:
        g = np.asarray(gscores, dtype=np.float32)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        spread = float(np.abs(g).mean()) + 1e-6
        p = p + learn_gain * np.tanh(g / (2.0 * spread))   # ★ 有界
    return int(np.argmax(p)) + 1
