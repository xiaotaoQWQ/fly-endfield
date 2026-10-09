#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""endfield_fly.py — 果蝇控制终末地角色行走（带朝向闭环与里程计）

相对 endfield_walk.py 的三处修正：

  1. **小地图标定错了**
     endfield_walk.py 写的是圆心 (188,236) 半径 143，实测应为 (228,208) 半径 118。
     x 偏 40px（=半径的 34%），采样盘整体错位并越界，
     把左侧 HUD 面板和右下游戏世界也当地形喂给了果蝇。

  2. **朝向没接进去**
     retina.encode() 的 heading_deg 一直用默认 0 —— 等于假设
     「小地图上方 = 角色正前方」。可小地图是**北向固定**的，
     于是果蝇的"正前"扇区其实指向世界正北，跟脸朝向无关。
     现在用里程计读出的位移方向当朝向（角色永远在前进 ⇒ 位移方向 = 脸朝向）。

  3. **转向幅度小到没用**
     老代码 dx = turn × 0.35 × 40 = 14px。实测 0.0647°/px ⇒ 每拍只转 0.9°，
     3Hz 下转 90° 要 33 秒。现在默认 --turn-gain 3.5（≈160px ≈ 10°/拍）。

另有卡住反射：位移≈0 且持续 3 拍 → 停止前进、用果蝇当前选的方向原地转，
转两拍再恢复。果蝇仍在选方向，只是加了一条「撞墙」反射弧。

用法（**必须管理员**，Interception 驱动）：
    python -X utf8 endfield_fly.py --seconds 300
    python -X utf8 endfield_fly.py --seconds 60 --dry     # 只决策不按键
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from collections import Counter

# ★ 兜底：把 stdout/stderr 设成 UTF-8 且**永不因编码报错**。
#   打包成 exe 之后没有 `-X utf8` 那层保护，控制台默认 GBK ——
#   程序里打印一个 ✓ 就会 UnicodeEncodeError，**一句 print 弄死整只果蝇**。
#   errors="replace" 保证最多显示成 ?，绝不抛异常。
#   （第一版把这个循环插在 `import sys` 之前，直接 NameError。）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ef_shot import grab, grab_scaled    # noqa: E402
from ef_vision import (MapOdometry, minimap_rect, MM_R,  # noqa: E402
                        content_rect)
from endfield_walk import find_endfield        # noqa: E402

user32 = ctypes.windll.user32
GRAPH = os.path.join(HERE, "graph")
OUT = os.path.join(HERE, "out")

# 实测标定：鼠标右移 1500px → 方位角 +97°（ef_steer_test.py，中位）
DEG_PER_MOUSE_PX = 0.0647

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--hz", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--turn-gain", type=float, default=3.5,
                    help="dx = turn × gain × 40；3.5 ≈ 160px ≈ 10°/拍")
    ap.add_argument("--center", choices=["none", "mean", "zscore"], default="zscore")
    ap.add_argument("--eye", choices=["minimap", "main"], default="minimap",
                    help="果蝇的『眼睛』：minimap=俯视小地图（旧）· main=主画面（第一人称）")
    ap.add_argument("--hfov", type=float, default=90.0,
                    help="主画面口径用：相机水平 FOV（估计值，未知，先试 90°）")
    ap.add_argument("--eye-map", choices=["linear", "fov"], default="linear",
                    help="linear=把 270° 铺满画面宽（所有细胞都用）· "
                         "fov=只用落在 FOV 内的细胞（几何正确但只有 1/3 工作）")
    ap.add_argument("--eye-ds", type=int, default=4,
                    help="主画面降采样倍数（抓取时就缩，省带宽）")
    ap.add_argument("--eye-halfw", type=int, default=1130,
                    help="眼睛横向半宽（像素）—— 上限由游戏 HUD 决定（≈1130）")
    ap.add_argument("--eye-crop", default="auto",
                    help="眼睛区域 x0,y0,w,h（绝对屏幕坐标）；auto=自动取对称且无 HUD 的框")
    ap.add_argument("--eye-dump", default=None,
                    help="把果蝇『看到』的画面和 drive 图存到该目录（调试用）")
    ap.add_argument("--eye-skip", default="0.41,0.08,0.59,1.0",
                    help="遮掉角色背影（裁切框内的比例 x0,y0,x1,y1）；"
                         "none=不遮。第三人称相机下角色杵在画面正中，"
                         "正好挡住 looming 最该出现的正前方")
    ap.add_argument("--pairs", default=None,
                    help="左右配对表 npz；不给就用默认的小地图口径表。"
                         "换眼睛后应指向 out/vote_pairs_main.npz")
    ap.add_argument("--eye-auto", action="store_true",
                    help="主画面失效（相机贴背→画面失去结构）时自动切回小地图口径")
    ap.add_argument("--eye-contrast", type=float, default=0.24,
                    help="切换阈值：主画面感光细胞的变异系数低于它就算「没信息」。"
                         "实测分布 q5=0.218 q25=0.265 q50=0.323 —— 取 ~q10")
    ap.add_argument("--eye-back", type=float, default=0.33,
                    help="切回主画面的阈值（必须 > --eye-contrast，形成滞回）。取 ~q50")
    ap.add_argument("--eye-dwell", type=int, default=20,
                    help="最小驻留拍数：切换后至少这么多拍不再切（默认 20 拍≈7 秒）")
    ap.add_argument("--eye-switch-stuck", action="store_true", default=True,
                    help="撞墙反射激活时也切回小地图口径（默认开）。"
                         "★ 实测对比度**测不出相机贴背** —— 角色自己的纹理很丰富，"
                         "贴背时对比度反而高(0.414)；而均匀草原会被误判(0.215)。"
                         "「推不动」这个运动学信号才是相机贴背的可靠征兆。")
    ap.add_argument("--no-jump", action="store_true")
    ap.add_argument("--no-sprint", action="store_true")
    ap.add_argument("--no-escape", action="store_true", help="关掉撞墙反射")
    # 为什么这么大：实测贴墙时原地转 20° 根本不够脱离，
    # 4 拍 × (3.5×40×2.5 px) = 350px/拍 ≈ 22.6°/拍 ≈ 90°，才拐得出去。
    ap.add_argument("--escape-ticks", type=int, default=4)
    ap.add_argument("--stuck-sec", type=float, default=3.0,
                    help="小地图连续静止这么多秒才算「卡住」（默认 3 秒）。"
                         "按时间算，不按拍数 —— 原来 3 拍在 2.5Hz 下只有 1.2 秒，"
                         "太灵敏，会在没撞墙时误触发")
    ap.add_argument("--escape-grace", type=float, default=10.0,
                    help="开局这么多秒内不触发撞墙反射（默认 10 秒）。"
                         "★ 群体解码自己有一段预热（前 ~6 秒不按移动键），"
                         "角色站着不动会被里程计判成「静止」→ 反射在开局就误触发")
    ap.add_argument("--walk-lock", type=int, default=4,
                    help="逃逸后强制前进的拍数（防止原地转死锁）")
    ap.add_argument("--escape-turn-mult", type=float, default=2.5,
                    help="逃逸时转向放大倍数")
    ap.add_argument("--snap-every", type=int, default=0,
                    help="每 N 拍存一张整屏快照（给视频用），0=不存")
    ap.add_argument("--snapdir", default=None)
    ap.add_argument("--record", default=None,
                    help="用 ffmpeg gdigrab 录屏到该文件（建议 .mkv，被强杀也不坏）")
    ap.add_argument("--record-fps", type=int, default=20)
    ap.add_argument("--record-width", type=int, default=1920)
    ap.add_argument("--record-encoder", default="h264_nvenc",
                    choices=["h264_nvenc", "h264_qsv", "h264_mf", "libx264"],
                    help="剪映自带的 ffmpeg **没有 libx264**，只有硬件编码器")
    ap.add_argument("--overlay", action="store_true",
                    help="显示实时可视化叠加（全脑活动 + 朝向 + 动作 + 轨迹）")
    ap.add_argument("--overlay-x", type=int, default=380)
    ap.add_argument("--overlay-y", type=int, default=0)
    ap.add_argument("--overlay-every", type=int, default=1,
                    help="每 N 拍刷新一次叠加（渲染约 60-100ms，太勤会拖慢闭环）")
    ap.add_argument("--dry", action="store_true", help="只决策不按键")
    ap.add_argument("--eye-hz", type=float, default=30.0,
                    help="眼睛的抓帧率。★ 必须远高于决策频率：决策只有 2Hz 时"
                         "帧间差 500ms，drive 里的时间通道（权重 1.6）会整体饱和、"
                         "光流信息全浪费。实测 564×146 单帧 18.3ms，30Hz 余量充裕。"
                         "0 = 与决策同频（旧行为）")
    ap.add_argument("--stop-key", default="end",
                    choices=["end", "f12", "pause", "home", "none"],
                    help="强制结束的按键（默认 end）。随时按下即优雅退出")
    ap.add_argument("--motor", choices=["pop", "pairs"], default="pop",
                    help="运动解码口径：pop=从连接组推的三条群体轴驱动 WASD+鼠标（新）·"
                         " pairs=旧的「点名的 5 对左右 DN 取差」（只有前进+转向）")
    ap.add_argument("--pop-turn-gain", type=float, default=1.5,
                    help="群体口径的转身增益：dx = z_turn × gain × 40")
    ap.add_argument("--pop-thr-w", type=float, default=-0.50,
                    help="推力 z 高于它就按 W（默认 −0.5，约 70%% 时间在走）")
    ap.add_argument("--pop-thr-s", type=float, default=-1.60,
                    help="推力 z 低于它才按 S（默认 −1.6，后退罕见）")
    ap.add_argument("--pop-thr-strafe", type=float, default=0.80,
                    help="横移 |z| 超过它才按 A/D")
    # ---- 战斗模式（交接文档 §二/§三/§五/§六/§八）----
    ap.add_argument("--combat", action="store_true",
                    help="启动时就打开战斗模式（默认关，运行时按 0 切换）")
    ap.add_argument("--combat-key", default="0", choices=["0", "9", "-", "="],
                    help="战斗模式开关（默认 0 键，按下沿切换）")
    ap.add_argument("--attack-key", default="mouse_left",
                    help="攻击键：mouse_left / mouse_right / 或键盘键名（默认鼠标左键）")
    ap.add_argument("--skill-key", default="1",
                    help="技能键（默认 1，按游戏内实际键位）")
    # 战斗标记（红叉 → 提高攻击欲望）
    ap.add_argument("--no-marker", action="store_true",
                    help="关闭红叉检测（默认**开启**：左上角红环亮起 = 游戏在战斗中")
    ap.add_argument("--marker-rect", default="28,47,76,97",
                    help="红环检测区 x0,y0,x1,y1（**画面坐标**，不含黑边）。★ 实测标定："
                         "非战斗 0 个红像素 / 战斗中 726 个（窗口实测 (28,127)-(76,177)）。"
                         "注意它锚定在左上角 HUD 上 —— **换分辨率必须重标**")
    ap.add_argument("--marker-min-px", type=int, default=80,
                    help="区域内强红像素超过它判为战斗中（默认 80；实测战斗 726、非战斗 0）")
    ap.add_argument("--urge", type=float, default=0.6,
                    help="战斗中攻击阈值的倍数（<1 = 更爱打；默认 0.6）")
    ap.add_argument("--escape-in-combat", action="store_true",
                    help="★ 默认**关**：红叉亮（在战斗中）时不触发撞墙反射。"
                         "战斗里被顶住/贴身缠斗时推不动是正常的，不是撞墙，"
                         "原地掉头会把贴上去的身位丢掉")
    ap.add_argument("--combat-sustain", type=float, default=0.8,
                    help="★ 战斗中「持续攻击」的间隔秒数（0=关闭，默认 0.8）。"
                         "为什么要它：上升沿触发的频率被信号本身卡死 —— 泄漏积分 "
                         "τ=1.2s 决定 z 的振荡周期，实测**无论阈值怎么降，天花板就是 "
                         "~10 次/分**（已在真实 CSV 上回放验证）；而把阈值降到 0.5 会让"
                         "自检的「高位平坦期」从 0 次涨到 5 次，即拿抗饱和能力换频率，"
                         "不划算。所以改走这条：闸门仍是果蝇的（z 要高过 "
                         "--combat-sustain-z），我们只定节奏")
    ap.add_argument("--combat-sustain-z", type=float, default=0.3,
                    help="持续攻击的闸门 z（默认 0.3 ≈ 62 分位）。调高=更挑，调低=更凶")
    ap.add_argument("--combat-yield", choices=["none", "attack"], default="none",
                    help="攻击时方向控制是否让位（交接文档 §二.2 要求提前决定）。"
                         "none=不让位，果蝇的方向控制保持生效（默认，因为本项目"
                         "底线是「它真的在控制角色」）；attack=让位，攻击时松开 "
                         "WASD 交给游戏自动寻路")
    ap.add_argument("--combat-tau", type=float, default=1.2,
                    help="战斗通道泄漏积分时间常数（秒）。这是 §六「跨帧保留」的实现")
    ap.add_argument("--combat-z", type=float, default=1.0,
                    help="战斗触发阈值（水平的 z）。★ 默认 1.0 由**真实数据标定**："
                         "实跑里攻击 z ∈ [−2.29, +1.45]（中位 +0.31/std 0.50），"
                         "按合成刺激定的 1.6 一次都到不了")
    ap.add_argument("--combat-z-exit", type=float, default=0.5,
                    help="滞回退出阈值：z 掉到它以下才重新武装。"
                         "★ 调高会让触发变稀疏 —— 实测固定 0.8 时上升沿被卡在 4~5 次")
    # ---- 觅食闭环（《果蝇攻击奖励方案·第二版》）----
    # 三层结构：饥饿值(动机) → 气味→接近→攻击(执行) → 奖赏→多巴胺→学习(价值)
    # 阶段一：一种敌人、一路布尔。气味源就用**已有的红环检测**，不需要 mod。
    ap.add_argument("--forage", action="store_true",
                    help="启用觅食闭环。气味=红环(有敌人)；攻击由**饥饿值**驱动"
                         "（不是恐惧）；打死/被打用键盘手动告诉它")
    ap.add_argument("--forage-patch", default=os.path.join(OUT, "fly_brain_patch.npz"),
                    help="学习补丁路径（稀疏表+原子写+版本号）。关掉窗口再开，"
                         "它还是同一只果蝇")
    ap.add_argument("--forage-odor", default="",
                    help="气味通道（肾小球名，如 ORN_VA2）。空=用 out/odor_pair.npz "
                         "里实测标定的那个")
    ap.add_argument("--forage-thr", type=float, default=0.45,
                    help="攻击阈值：驱动力超过它就出手（默认 0.45）。★ 由实测定："
                         "MBON 饱和后驱动力 ≈ 0.667 × 饥饿增益，阈值 0.35 时"
                         "连 h=0.5 都够得着，节律要吃到第五六顿才显出来；"
                         "0.45 让它在 h≈0.58 就停，两顿就能看见")
    ap.add_argument("--hunger-rise", type=float, default=0.030,
                    help="饥饿值回升速度 /秒（默认 0.03）。★ 必须与 --hunger-eat "
                         "平衡，否则 h 会钉死在 0、节律消失（实测踩过）")
    ap.add_argument("--hunger-eat", type=float, default=0.35,
                    help="每次进食降多少饥饿值（默认 0.35）")
    ap.add_argument("--evac", action="store_true",
                    help="★ 用左上角**「撤离」面板**判定副本内，取代红叉当战斗闸门。"
                         "红叉只在打起来之后才亮，撤离按钮是副本的常驻 UI")
    ap.add_argument("--evac-rect", default="24,45,213,53",
                    help="撤离面板的**画面坐标** (x,y,w,h)，默认 24,45,213,53"
                         "（ef_evac_probe.py 实测标定）")
    ap.add_argument("--evac-corr", type=float, default=0.55,
                    help="模板归一化相关的阈值（默认 0.55）。实测定标："
                         "面板在时 +1.000，其他 16 处最高 +0.191，余量很大")
    ap.add_argument("--auto-outcome", action="store_true",
                    help="★ 全自动识别打死/挨打，不用手按 O/P。"
                         "挨打←血条掉 或 受击红晕；打死←红环下降沿(出过手且战线够长)")
    ap.add_argument("--jump-in-combat", action="store_true",
                    help="★ 默认**关**：战斗状态（副本内）不跳跃。跳跃会打断攻击"
                         "动作、把身位带离敌人，还白吃一次落地硬直")
    ap.add_argument("--panel", action="store_true",
                    help="★ 开源版入口：先起控制台网页，在上面点「启动」才开跑。"
                         "启动前**必须检测到终末地**。隐含 --tune")
    ap.add_argument("--tune", action="store_true",
                    help="★ 起一个本地网页调参面板（127.0.0.1），跑的时候实时改，"
                         "不用重启。改完下一拍自动生效")
    ap.add_argument("--tune-port", type=int, default=8791,
                    help="调参面板端口（默认 8791）")
    ap.add_argument("--attack-hold", type=float, default=1.0,
                    help="★ 普攻按住多少秒（默认 1.0）。改长按后时长超过决策拍"
                         "（400ms），所以按下/抬起落在**不同的拍**上 —— "
                         "在一次调用里 sleep 会把整个主循环卡住")
    ap.add_argument("--skill-hunger", type=float, default=0.8,
                    help="★ 饥饿值 ≥ 它才穿插战技（默认 0.8）。饿的时候更愿意放技能，"
                         "饱了就只普攻")
    ap.add_argument("--skill-auto", action="store_true",
                    help="★ 果蝇自己选技能（1 伤害高 · 2 挂削弱 · 3 增伤 · 4 回血）。"
                         "先天轮换 + 蘑菇体分区学习微调；不开就退回永远按 --skill-key")
    # ---- 大招（长按 1/2/3/4）----
    # 设计：**久攻不下、饿到极点**时拼命扑。
    #   好玩的地方在于「战斗越久 = 越饿」—— 没吃到东西时饥饿值一直在回升，
    #   所以「久攻不下」和「极度饥饿」对果蝇来说是同一件事。
    #   它不需要「大招能量条」这个概念，它只有饿。
    # 刻意**不**在血低时放大招：生物上受伤的果蝇会回避，不会猛攻；
    #   而且那会和「血<50% 优先回血」的保命规则打架。
    ap.add_argument("--ult", action="store_true",
                    help="★ 允许放大招（长按 1/2/3/4）。条件：饥饿 ≥ ult-hunger "
                         "且 战线 ≥ ult-age 且 血 ≥ ult-hp "
                         "（久攻不下、饿到极点 —— 但不在垂死时拼命）")
    ap.add_argument("--ult-hunger", type=float, default=0.95,
                    help="大招要求的饥饿值（默认 0.95，也就是饿疯了）")
    ap.add_argument("--ult-age", type=float, default=8.0,
                    help="大招要求的战线长度秒数（默认 8）。同时也是饿到极点的来源")
    ap.add_argument("--ult-hp", type=float, default=0.40,
                    help="大招要求的**最低**血量（默认 0.40）。★ 低于它就不放 —— "
                         "受伤的果蝇该回避，不是猛攻")
    ap.add_argument("--ult-cool", type=float, default=20.0,
                    help="大招冷却秒数（默认 20）。游戏里大招有冷却，果蝇看不见，"
                         "用自己的定时器兜住，别浪费")
    ap.add_argument("--ult-hold", type=float, default=1.0,
                    help="大招按住秒数（默认 1.0）。长按才触发，所以要跨拍")
    ap.add_argument("--heal-hp", type=float, default=0.5,
                    help="血条低于它就优先放回血技能（默认 0.5）。★ 这是保命规则，"
                         "给得压得住学习偏移 —— 学不坏")
    ap.add_argument("--skill-interval", type=float, default=4.0,
                    help="战技的最小间隔秒数（默认 4.0）。战技有冷却，狂按只会空转")
    ap.add_argument("--forage-always-attack", action="store_true",
                    help="★ 默认**关**：红叉（进入战斗）是攻击的必要前提，"
                         "没进战斗不许挥刀。加上这个开关才允许它无条件出手")
    ap.add_argument("--hit-reward", type=float, default=0.1,
                    help="★ 奖励塑形（§九）：**打到**敌人给的小额奖赏，默认 0.1。"
                         "触发条件是「出手 且 红环亮着」—— 即真的在跟敌人交手，"
                         "不需要手动按。**不给饱足**（只有打死才降饥饿），"
                         "这样「打死」的总期望才明显高过「磨」")
    ap.add_argument("--hit-cooldown", type=float, default=1.0,
                    help="小额奖赏的最小间隔秒数（默认 1.0）。否则 2.5Hz 下"
                         "每拍都发，等于白送")
    ap.add_argument("--forage-eta", type=float, default=0.002,
                    help="三因子学习率。★ 实测标定：KC→MBON 的基础权重只有 "
                         "0.0000~0.0131，而 η·DA 一次就给 0.05 → 学习变成"
                         "**覆盖**而不是调制（一次惩罚把漂移推到 3000+）。"
                         "0.002 让一次事件约等于基础权重的量级")
    ap.add_argument("--forage-innate", type=float, default=0.90,
                    help="★ 先天取食反射的强度（默认 0.90）。交接 §三：执行层的"
                         "取食反射是先天的、不用学 —— 所以这一份固定给足，"
                         "学习只在它上面浮动。写 0 就退回「学习结果独占」（会崩）")
    ap.add_argument("--forage-learn", type=float, default=0.35,
                    help="学习能调制的幅度（默认 0.35）。驱动力 = "
                         "(innate + learn × 学习归一值) × 饥饿增益")
    # ---- 连携技（E）----
    # 用户要求：战斗中出现连携技提示 **且 饥饿值到 1** → 按 E。
    ap.add_argument("--combo", action="store_true",
                    help="★ 战斗中出现连携技（E）提示且饥饿值满时按 E")
    ap.add_argument("--combo-rect", default="",
                    help="连携技提示的**画面坐标** (x,y,w,h)。空着或全 0 = 未标定，"
                         "此时**绝不按 E**（免得乱按浪费冷却）")
    ap.add_argument("--combo-key", default="e", choices=["e", "q", "f", "r"],
                    help="连携技键，默认 E")
    ap.add_argument("--combo-hunger", type=float, default=0.99,
                    help="饥饿值 ≥ 它才按连携技（默认 0.99，也就是「到 1 了」）")
    # ============ 任务完成（挑战成功）============
    # ★ 用户要求：任务完成时给 50 次奖励，饥饿值不再上升。
    # ★★ 任务完成检测**默认关闭**（用户要求：误触太多，删掉）。
    #   理由见 mission.py 顶部 —— 三版判据都在录像上干净、实机翻车，
    #   根因是"靠屏幕判断游戏状态"这条路本身不可靠。
    #   代码留着（它本身是对的），要的人加 --mission 自己开。
    ap.add_argument("--mission", action="store_true",
                    help="★ 默认**关**。开启后：任务条上的「击败所有敌人」"
                         "连续消失 N 秒就判定副本结束（给 50 次奖励 + "
                         "饥饿清空 10 秒）。实测误触较多，默认不开")
    ap.add_argument("--mission-corr", type=float, default=0.50,
                    help="任务条文字的模板相关阈值。实测（428 帧）："
                         "显示 +0.63~+1.00、消失 +0.14~+0.29")
    ap.add_argument("--mission-gone", type=float, default=4.0,
                    help="任务条文字**连续消失多少秒**后判定副本结束（1~10，默认 4）。"
                         "★ 这个「连续」很关键：文字会淡入淡出，"
                         "只看下降沿会被抖动误触发")
    ap.add_argument("--no-mission", action="store_true",
                    help="（已废弃：现在默认就是关的，不用加这个）")
    ap.add_argument("--mission-reward", type=int, default=50,
                    help="任务完成时给几次奖励（默认 50 次）")
    ap.add_argument("--mission-mag", type=float, default=1.0,
                    help="每次奖励的强度")
    ap.add_argument("--mission-hold", type=float, default=10.0,
                    help="任务完成后饥饿值清空并**多少秒不增长**（默认 10 秒），"
                         "之后恢复正常增长")

    # ============ 键位与技能含义（可更换）============
    # ★ 用户要求：技能/大招/1234 键都能换，而且每个槽的**含义**也能换，
    #   可选内容 = 攻击 / 增伤 / 格挡 / 回血 / 聚怪。
    #
    #   为什么含义要能配：果蝇从画面上看不出技能是干什么的（没有 mod），
    #   只能把语义当**本能**写进去。不同配队、不同角色的技能组不一样，
    #   写死一套就只对一种配队有效。
    _ROLES = ["attack", "buff", "block", "heal", "gather", "weaken", "none"]
    for _i, _d in enumerate(("weaken", "buff", "heal", "none"), start=1):
        ap.add_argument(f"--role-{_i}", default=_d, choices=_ROLES,
                        help=f"第 {_i} 个技能槽的含义"
                             f"（attack 攻击 / buff 增伤 / block 格挡 / "
                             f"heal 回血 / gather 聚怪 / weaken 挂削弱 / none 不用）")
    for _i in range(1, 5):
        ap.add_argument(f"--key-skill{_i}", default=str(_i),
                        help=f"第 {_i} 个技能槽按哪个键（默认 {_i}）")
    ap.add_argument("--key-ult", default="1,2,3,4",
                    help="大招键序列，逗号分隔。轮换用 —— 长按同一个键放的是"
                         "那个角色的专属大招，每个角色各自有冷却，"
                         "果蝇看不出谁好了，所以轮换")
    ap.add_argument("--skill-order", default="",
                    help="先天技能轮换顺序（槽号，逗号分隔，如 4,3,1）。"
                         "空 = 用 role 决定的默认顺序")

    # ============ 对照组（消融实验）============
    # ★ 这一组不是调参，是**消融实验**：把某个脑区关掉，看行为变不变。
    #   两条底线是它得真的在控制那个角色和它得真的能走到某个地方——
    #   光靠嘴说没用，得能把系统关掉做对照。
    #   用 BooleanOptionalAction 是为了同时有 --ab-xxx 和 --no-ab-xxx 两种写法。
    _AB = argparse.BooleanOptionalAction
    ap.add_argument("--ab-connectome", action=_AB, default=True,
                    help="★ 连接组。关掉=用**随机数**代替神经网络输出 —— "
                         "这是阴性对照：如果关掉前后行为差不多，"
                         "那这 211,577 个神经元就只是装饰")
    ap.add_argument("--ab-dopamine", action=_AB, default=True,
                    help="多巴胺（奖赏/惩罚）。关掉=奖赏惩罚打不进去，学不会")
    ap.add_argument("--ab-mushroom", action=_AB, default=True,
                    help="蘑菇体学习。关掉=权重冻结（读正常，但不更新）")
    ap.add_argument("--ab-vision", action=_AB, default=True,
                    help="视觉（复眼）。关掉=瞎了，光流恒为 0")
    ap.add_argument("--ab-odor", action=_AB, default=True,
                    help="气味输入。关掉=闻不到敌人")
    ap.add_argument("--ab-hunger", action=_AB, default=True,
                    help="饥饿内稳态。关掉=饥饿值锁在 0.5")
    ap.add_argument("--ab-motor", action=_AB, default=True,
                    help="运动输出。关掉=照样算照样显示，但一个键都不按")

    ap.add_argument("--combo-corr", type=float, default=0.50,
                    # ★★ argparse 会对 help 做一次 `help % params` —— 字符串里
                    #    裸的 `%` 会被当成格式符，启动直接
                    #    `ValueError: unsupported format character`。
                    #    这个坑在 exe 上才暴露（源码跑没触发那条路径）。
                    #    要么写 `%%`，要么像我这样干脆用「百分之」把 % 去掉。
                    help="连携技模板相关阈值。★ 实测（951 帧战斗录像 × 8 种分辨率）："
                         "真提示 +0.69~+1.00、噪声底 +0.34~+0.40；"
                         "取 0.50 时每种分辨率的召回都是 97~99 个百分点，"
                         "而且「判定有提示」的帧占比在各分辨率下都是 12.3~12.8 "
                         "个百分点（真值 12.6）—— 说明没有额外误报")
    ap.add_argument("--combo-cool", type=float, default=0.0,
                    help="连携技最小间隔秒数（默认 0 = 有提示就按，无冷却）")
    ap.add_argument("--forage-decay", type=float, default=1e-4,
                    help="突触衰减（遗忘）。只增不减会漂走，跑几十次行为就崩")
    ap.add_argument("--forage-save-sec", type=float, default=30.0,
                    help="补丁定期快照间隔秒数（默认 30），关窗前还会再存一次")
    ap.add_argument("--out", default=os.path.join(OUT, "endfield_fly.csv"))
    args = ap.parse_args()

    if not args.dry and not ctypes.windll.shell32.IsUserAnAdmin():
        print("⚠ 不是管理员 —— Interception 会失败。请以管理员运行。")

    from retina import RetinaMapper
    from sector_vote import DirectionDecoder
    from action_map import ActionMapper
    from motor import MotorPlan

    print("载入连接组…")
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsr()
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = W.shape[0]
    print(f"  {n:,} 神经元 / {W.nnz:,} 条边")

    rm = RetinaMapper(os.path.join(OUT, "optic_map.feather"),
                      os.path.join(GRAPH, "graph_meta.feather"))
    rm.reset()
    # 复眼扇区掩码（面板要按扇区统计输入信号）—— 掩码只算一次
    from retina import SECTORS
    sec_names = [nm for _, _, _, nm in SECTORS]
    sec_masks = [(rm._sector_of == i) for i in range(len(SECTORS))]
    dec = DirectionDecoder(top_k=5, npz_path=args.pairs) if args.pairs \
        else DirectionDecoder(top_k=5)
    # 群体解码（用户要求：不用神经元直连）
    pmm = None
    if args.motor == "pop":
        from motor_decode import PopulationMotor
        pmm = PopulationMotor(thr_w=args.pop_thr_w, thr_s=args.pop_thr_s,
                              thr_strafe=args.pop_thr_strafe,
                              turn_gain=args.pop_turn_gain)
    amap = ActionMapper.load()

    # 战斗解码（交接文档）：pC1→攻击键，LPLC2/LC4→技能键。
    # ★ 只在 --combat 或运行时按 0 打开时才真正驱动按键；
    #   解码器本身总是建好并每拍更新（面板要显示它的状态）。
    from combat import CombatDecoder, find_combat_populations
    _atk_idx, _skl_idx = find_combat_populations()
    cdec = CombatDecoder(_atk_idx, _skl_idx, tau=args.combat_tau,
                         z_enter=args.combat_z, z_exit=args.combat_z_exit,
                         sustain_sec=args.combat_sustain,
                         sustain_z=args.combat_sustain_z)
    combat_on = bool(args.combat)

    # 战斗标记：左上角红叉 = 游戏在战斗中 → 提高攻击欲望
    cmark = None
    if not args.no_marker:
        from combat import CombatMarker
        _mr = [int(v) for v in args.marker_rect.split(",")]
        cmark = CombatMarker((_mr[0], _mr[1], _mr[2], _mr[3]),
                             min_px=args.marker_min_px)
        print(f"  战斗中 ✓ 攻击阈值 ×{args.urge}（{args.combat_z} → "
              f"{args.combat_z * args.urge:.2f}）")

    # 撤离面板 = 副本内。出现了就进战斗状态（比红叉更适合当闸门）
    emark = None
    if args.evac:
        from combat import EvacMarker
        _er = [int(v) for v in args.evac_rect.split(",")]
        emark = EvacMarker((_er[0], _er[1], _er[2], _er[3]),
                           corr_min=args.evac_corr)
        print(f"  副本判定：看左上角「撤离」面板 "
              f"({_er[0]},{_er[1]}) {_er[2]}×{_er[3]}")

    # 连携技（E）提示。区域已实测标定，ComboMarker 内部有默认值
    kmark = None
    if args.combo:
        from combat import ComboMarker
        _kr = [int(v) for v in args.combo_rect.split(",")] \
            if args.combo_rect.strip() else None
        kmark = ComboMarker(tuple(_kr) if _kr and _kr[2] > 0 else None,
                            corr_min=args.combo_corr)
        print(f"  连携技：饥饿 ≥ {args.combo_hunger} 且提示出现 → 按 "
              f"{args.combo_key.upper()}（间隔 {args.combo_cool}s，0=无冷却）")

    # ================= 觅食闭环（《攻击奖励方案·第二版》）=================
    # 三层：饥饿值(动机) → 气味→接近→攻击(执行) → 奖赏→多巴胺→学习(价值)
    # ★ 这与上面的 combat.py 是**两条不同的驱动**：
    #     combat.py = 旧的「恐惧→攻击」（交接文档说这个方向已作废）
    #     forage    = 新的「饥饿→觅食」（怕就退、饿就上）
    #   两者可以单独开，也可以对比着跑。
    forager = None
    outcome = None
    if args.forage:
        from mb_learn import Hunger, MushroomBodyLearner
        from odor_map import OdorEncoder
        _gm = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
        _gm = _gm.sort_values("idx").reset_index(drop=True)
        _cl = _gm["class"].astype("string").fillna("").astype(str)
        _ty = _gm["type"].astype("string").fillna("").astype(str)
        _ix = _gm["idx"].to_numpy()
        _glom = {}
        for _t, _i in zip(_ty[_cl == "olfactory"], _ix[_cl == "olfactory"]):
            _glom.setdefault(_t, []).append(_i)
        _glom = {k: np.array(v, np.int64) for k, v in _glom.items()}
        _orn = _ix[(_cl == "olfactory").to_numpy()]
        _mbon = _ix[(_cl == "MBON").to_numpy()]
        _kc = _ix[(_cl == "Kenyon_Cell").to_numpy()]

        enc = OdorEncoder(_glom, _orn, verbose=True)
        ch = args.forage_odor
        if not ch:
            _pp = os.path.join(OUT, "odor_pair.npz")
            if os.path.exists(_pp):
                _z = np.load(_pp, allow_pickle=False)
                ch = str(_z["chA"])
                print(f"  气味通道取自实测标定：{ch}"
                      f"（KC 重叠 Jaccard {float(_z['jaccard']):.3f}）")
            else:
                ch = enc.channels[0]
        if ch not in _glom:
            print(f"  ⚠ {ch} 不在注释表里，退回 {enc.channels[0]}")
            ch = enc.channels[0]

        _mb = MushroomBodyLearner(W, _kc, _mbon, eta=args.forage_eta,
                                  decay=args.forage_decay, verbose=True)
        _hg = Hunger(h0=1.0, rise=args.hunger_rise, eat=args.hunger_eat)
        _loaded = _mb.load(args.forage_patch)
        print(f"  补丁：{'已加载 ' + args.forage_patch if _loaded else '新果蝇（无补丁）'}"
              f"   漂移 {_mb.drift:.1f}")
        print(f"  饥饿：回升 {args.hunger_rise}/s · 每餐 −{args.hunger_eat} · "
              f"攻击阈值 {args.forage_thr}")
        outcome = None
        if args.auto_outcome:
            from auto_outcome import AutoOutcome, HP_RECT, VIG_L, VIG_R
            outcome = AutoOutcome(verbose=True)
            print(f"  ★ 全自动奖惩已开：血条 {HP_RECT} · 红晕 {VIG_L}/{VIG_R}")
        forager = dict(enc=enc, ch=ch, mb=_mb, hg=_hg, kc=_kc, mbon_idx=_mbon,
                       last_save=time.time(), n_reward=0, n_punish=0,
                       urge=0.0, mbon_val=0.0, odor=False, attack=False,
                       n_hit=0, last_hit=0.0,
                       n_skill=0, last_skill=0.0, mbout_ref=1.0,
                       n_combo=0, last_combo=0.0, combat_t0=0.0,
                       n_ult=0, last_ult=0.0, ult_holding=False,
                       ult_hold_until=0.0, ult_key="1",
                       in_dungeon=False, in_red=False)

    # ================= 后台控制台（调参 + 启动/暂停/停止）=================
    # ★ 它只做一件事：往 args 上写属性。循环里全是 `args.forage_thr` 这种读法，
    #   所以改完**下一拍自动生效** —— 一行循环逻辑都不用动。
    #   少数几个参数不在 args 上（Hunger 的 rise/eat 是启动时拷进去的），
    #   靠 sync_hooks 每拍同步一次。
    # ★ --panel 时**先把面板起来**，等你在网页上按"启动"；--tune 则直接开跑。
    def _refresh_panel_cache():
        """在主线程里刷新控制台的缓存。

        ★ 只有主线程能调用会枚举窗口的东西 —— find_endfield() 从
          tkinter 线程 / HTTP 线程调用会卡住。卡住谁谁就废：
            · HTTP 线程卡 → 接口挂到超时，每次请求还堆一条线程
            · tkinter 线程卡 → **窗口冻结、启动按钮状态不刷新**，
              表现就是"按了启动没反应"
        """
        try:
            _refresh_game_cache()  # 先探测游戏（主线程才能做）
        except Exception:
            pass                   # 没开 --panel/--tune 时这个函数不存在
        try:
            from live_tune import start_tuner
            _rf = getattr(start_tuner, "refresh", None)
            if _rf:
                _rf()
        except Exception:
            pass

    def _poll_markers_idle():
        """空闲（没在跑）时也扫一次标记，好让界面上的「位置」是**现在**的。

        ★ 不加这个的话，位置只在主循环里更新 —— 一旦暂停/一轮结束，
          界面就永远停在最后一拍的值上。实测：人在副本里，界面却显示
          「大世界」，看起来就像"副本模式没启动"。
        """
        try:
            if emark is not None:
                _r = emark.rect
                _v = emark.step(grab(x + _r[0], y + _r[1], _r[2], _r[3]))
                if forager is not None:
                    forager["in_dungeon"] = bool(_v)
                if os.environ.get("EF_DEBUG_POLL"):
                    print(f"    [poll] 撤离 @({x + _r[0]},{y + _r[1]}) "
                          f"相关 {emark.corr:+.3f} 填充 {emark.fill:.3f} "
                          f"-> {'副本内' if _v else '副本外'}", flush=True)
            if cmark is not None:
                _r = cmark.rect
                cmark.step(grab(x + _r[0], y + _r[1], _r[2], _r[3]))
                if forager is not None:
                    forager["in_red"] = bool(cmark.on)
            _refresh_panel_cache()
        except Exception as _e:
            if os.environ.get("EF_DEBUG_POLL"):
                import traceback
                print(f"    [poll] 出错: {type(_e).__name__}: {_e}", flush=True)
                traceback.print_exc()

    CTL = {"running": not args.panel, "pause": False, "stop": False,
       "load_slot": None, "reset_mem": False}
    _pause_logged = [False]        # 暂停提示只打一次（用列表当可变闭包）
    _ov_hidden = [False]           # 叠加面板当前是不是被藏起来了
    _ov_err = [0]                  # 叠加面板更新失败次数（只报前几次）
    if args.tune or args.panel:
        from live_tune import start_tuner, make_status_html, LogTap
        _log = LogTap(keep=80).install()

        _gc = [0.0, (False, "没检测过")]

        def _game_probe():
            """**真正去枚举窗口**找终末地。★ 只有主线程可以调。

            find_endfield() 会 EnumWindows。实测从 tkinter 线程 / HTTP 线程
            调用会**卡住不返回** —— 卡住的是谁，谁就废了：
              · HTTP 线程卡 → 接口一直挂到超时，还每次请求堆一条线程
              · tkinter 线程卡 → **窗口冻结、启动按钮状态不再刷新**，
                表现就是"按了启动没反应"
            这就是之前两个 bug（接口超时 / 界面卡住）的同一个根因。
            """
            w5 = find_endfield()
            if not w5:
                r = (False, "没找到 Endfield 窗口/进程")
            else:
                r = (True, f"hwnd={w5[0]} {w5[3]}×{w5[4]}")
            _gc[0], _gc[1] = time.time(), r
            return r

        def _game_check():
            """给界面 / HTTP 用的：**只读缓存，绝不自己枚举窗口**。

            缓存由主线程在 `_refresh_panel_cache()` 里定期刷新。
            ★ 如果缓存从来没填过（比如界面比主循环先起来），返回上一次的
              结果并让调用方自己看时间戳 —— 但绝不在这里现算。
            """
            return _gc[1]

        # ---- 记忆存档：存 / 读 回调 ----
        # ★ 存 = 把蘑菇体突触补丁写到 out/memory/slot_NN.npz
        #   读 = 挂到 CTL["load_slot"]，由**主循环**在自己的节拍上执行
        #   （HTTP / 界面线程绝不直接动 forager 的数据结构）
        def _mem_save(slot):
            if forager is None:
                return False
            try:
                import live_tune as _lt
                k = forager["mb"].save(_lt.slot_path(slot),
                                        extra=dict(rewards=forager["n_reward"],
                                                   punish=forager["n_punish"],
                                                   hunger=forager["hg"].h))
                print(f"  [存档] 已写入槽位 {slot}：{k:,} 条边 · "
                      f"漂移 {forager['mb'].drift:.1f}", flush=True)
                return True
            except Exception as e:
                print(f"  [存档] 槽位 {slot} 保存失败：{e}", flush=True)
                return False

        def _mem_load(slot):
            if forager is None:
                return False
            try:
                import live_tune as _lt
                forager["mb"].load(_lt.slot_path(slot))
                print(f"  [存档] 已读取槽位 {slot} · "
                      f"漂移 {forager['mb'].drift:.1f}", flush=True)
                return True
            except Exception as e:
                print(f"  [存档] 槽位 {slot} 读取失败：{e}", flush=True)
                return False

        def _mem_reset(_=None):
            # ★ 清空记忆 = 权重回到连接组基线 w0（那只**什么都没学过**的果蝇）。
            #   和 load 的区别：load 是换成另一份记忆，reset 是忘掉一切。
            if CTL is not None:
                CTL["reset_mem"] = True
            return True

        _MEM = {"save": _mem_save, "load": _mem_load, "reset": _mem_reset}

        def _refresh_game_cache():
            """主线程专用：探测一次并写进缓存。"""
            try:
                _game_probe()
            except Exception:
                pass

        def _tune_status():
            if forager is None:
                return make_status_html([("觅食闭环", "未开")])
            f = forager
            _h = f.get("auto_hp")
            _hs = f"{_h*100:.0f}%" if isinstance(_h, float) and _h == _h else "—"
            return make_status_html([
                # ★ 最上面先报「在哪」—— 副本内/外决定了它会不会出手，
                #   之前漏了这一项，看着像"副本模式没了"。
                ("位置", "★ 副本内" if f.get("in_dungeon")
                 else ("大世界·战斗中" if f.get("in_red") else "大世界")),
                ("可以出手", "是" if (f.get("in_dungeon") or f.get("in_red"))
                 else "否"),
                ("饥饿值 h", f"{f['hg'].h:.3f}"),
                ("血条", _hs),
                ("攻击驱动力", f"{f['urge']:.3f}"),
                ("蘑菇体输出", f"{f['mbon_val']:+.3f}"),
                ("权重漂移", f"{f['mb'].drift:.1f}"),
                ("闻到气味", "是" if f["odor"] else "否"),
                ("正在出手", "是" if f["attack"] else "否"),
                ("连携技相关", f"{kmark.corr:+.3f}" if kmark else "—"),
                ("自动打死", f["n_reward"]),
                ("自动挨打", f["n_punish"]),
                ("打到(小额)", f["n_hit"]),
                ("技能", f["n_skill"]),
                ("大招", f["n_ult"]),
                ("连携技", f["n_combo"]),
            ])

        def _tune_sync():
            # Hunger 的 rise/eat 是启动时从 args 拷进去的，得每拍同步
            if forager is not None:
                forager["hg"].rise = args.hunger_rise
                forager["hg"].eat = args.hunger_eat

        _panel_url = start_tuner(args, port=args.tune_port, status_fn=_tune_status,
                                 sync_hooks=[_tune_sync],
                                 ctl=CTL, game_fn=_game_check, mem=_MEM)
        # ★ 真·应用程序窗口（tkinter）。HTTP 服务保留在旁边 ——
        #   本地用窗口，手机/另一台机器用网页，两边读的是同一份状态。
        if args.panel:
            from live_tune import GROUPS, TOGGLES, SELECTS
            from app_window import start_app
            _gref = [None]

            def _on_close():
                # 右上角一关 = 整个程序退出（不然窗口没了、果蝇还在后台跑）
                CTL["stop"] = True
                CTL["running"] = False

            _app, _app_th = start_app(
                args, CTL, status_fn=_tune_status, game_fn=_game_check,
                tune={"groups": GROUPS, "toggles": TOGGLES,
                      "selects": SELECTS,
                      "args": args, "hooks": [_tune_sync]},
                on_close=_on_close, port=args.tune_port,
                log_fn=lambda: _log.tail(16), mem=_MEM)
            _gref[0] = _app
        if args.panel:
            print()
            print("=" * 62)
            print("  应用程序窗口已打开。")
            print("  在窗口里点「启动」才会开始跑。")
            print("  ★ 必须检测到终末地 —— 没开游戏的话启动按钮是灰的。")
            print(f"  （也可以在浏览器打开 {_panel_url}，手机同一 WiFi 也行）")
            print("=" * 62)
            # ★ 「等启动」**不能放这里**。此时画面区还没算出来，x/y 还是
            #   窗口原点（0,0）而不是画面原点（0,80）—— 空闲扫描会整体偏 80px，
            #   抓错位置，副本内也会显示成「大世界」。
            #   挪到画面区算好之后（见下面 _wait_for_start）。

    eye = None
    eye_rect = eye_size = None
    # 面板对象提前建：眼睛遮罩要按**面板的实际坐标**算，
    # 不能再硬编码一块固定矩形 —— 面板一搬家，硬编码的遮罩就会挖错地方
    # （实测：面板移到顶部后，旧遮罩仍按右侧位置挖，白白吃掉裁切框右 27%）。
    panel = None
    # COMBAT_H/FORAGE_H 无条件导入：眼睛遮罩要按**面板的完整高度**（含预留区）算，
    # 不管这一轮开不开叠加。
    from fly_panel import COMBAT_H, FORAGE_H
    # ★ 底部按需预留：--forage 模式下战斗条永远空着，白占 88px。
    #   两条都用才都留，否则只留用得上的那条（Overlay 尺寸启动时定死）。
    _strips = tuple(x for x, on in (("combat", args.combat),
                                    ("forage", args.forage)) if on)
    if args.overlay:
        from fly_panel import FlyPanel
        panel = FlyPanel(strips=_strips or ("combat", "forage"))
    if args.eye == "main":
        from ef_eye import MainEye
        # 自动裁切时框内本来就没有 HUD/叠加面板 → 遮罩传空表，别自找偏置
        eye = MainEye(os.path.join(OUT, "optic_map.feather"),
                      os.path.join(GRAPH, "graph_meta.feather"),
                      hfov=args.hfov, map_mode=args.eye_map, ds=1,
                      mask_rects=[] if args.eye_crop == "auto" else None)
    cal = amap.calib.get("channels", {})
    print("\n动作阈值（来自 action_calib.json）：")
    for k in ("jump", "forward", "sprint"):
        if k in cal:
            print(f"  {k:>8s}  baseline {cal[k].get('baseline', 0):+.6f}  "
                  f"q70 {cal[k].get('q70', 0):+.6f}  "
                  f"q_hi {cal[k].get('q_hi', 0):+.6f}")
    print(f"  标定来源：{amap.calib.get('meta', {}).get('source', '?')}")
    print(f"  标定口径：map_mode={amap.calib.get('meta', {}).get('map_mode', '?')} "
          f"hfov={amap.calib.get('meta', {}).get('hfov', '?')}")
    print()

    win = find_endfield()
    if not win:
        sys.exit("终末地没在运行")
    hwnd, x, y, w, h = win
    print(f"终末地窗口 ({x},{y}) {w}×{h}")
    # ★★ **画面区**（去掉上下黑边）必须最先算好 —— 眼睛裁切、小地图、HUD 遮罩
    #   全都依赖它。
    #   ★ 现在改成**实测**黑边（screen_profile.measure_content_rect），
    #     不再靠 `ch = min(h, w*9/16)` 猜 —— 换个分辨率/换个宽高比就作废。
    #     实测还有个额外好处：屏幕上压着别的窗口时（实测 DSH 窗口盖住画面底部），
    #     取"包含中心的那一段连续亮区"能把外来窗口排除掉。
    _win_h = h
    import screen_profile as _sp
    prof = _sp.build(x, y, w, h, grab_fn=grab)
    x, y, w, ch = prof.x, prof.y, prof.w, prof.h
    print(f"画面区：({x},{y}) {w}×{ch}"
          f"（窗口高 {_win_h}，上下黑边各 {(_win_h - ch) // 2}px，"
          f"{'实测' if prof.measured else '推算'}）")
    # 各标定区域现在都从 prof 换算 —— 换分辨率自动跟着走
    _evac_rc = prof.rect("evac")
    _mark_rc = prof.rect("marker")
    _combo_rc = prof.rect("combo")

    # ★ 把各检测器的矩形换成当前分辨率下的值。
    #   它们是在知道窗口尺寸**之前**建的（用的 args 里那套 2560×1440 的默认值），
    #   所以在这里统一改写 —— 比把建对象的代码整段搬下来小得多。
    if cmark is not None:
        cmark.rect = _mark_rc
    if emark is not None:
        emark.rect = _evac_rc
        _t = prof.template("evac_panel")          # 模板按当前分辨率重采样
        if _t is not None:
            emark.tpl = _t
            emark._tm = float(_t.mean())
            emark._ts = float(_t.std()) + 1e-6
            emark._th, emark._tw = _t.shape
    if kmark is not None:
        kmark.rect = _combo_rc
        _t = prof.template("combo_e")
        if _t is not None:
            kmark.tpl = _t
    # 自动奖惩的血条 / 受击红晕矩形（循环里是 `from auto_outcome import HP_RECT`，
    # 所以改模块全局变量就够了）
    import auto_outcome as _ao
    _ao.HP_RECT = prof.rect("hp")
    _ao.VIG_L = prof.rect("vig_l")
    _ao.VIG_R = prof.rect("vig_r")
    # ★★ 遮挡检查：grab() 抓的是**屏幕**，不是游戏窗口 DC
    #   （PrintWindow 对 DirectX 游戏返回全黑，拿不到）。
    #   所以任何压在游戏上面的窗口，果蝇都会当成游戏画面读进去 ——
    #   而且**不报错、只是那个功能永远不触发**。
    #   实测：聊天窗口盖住连携技区和血条 → E 永远不按 + 血量读不到。
    _sp.check_occlusion(hwnd, prof)
    print(f"  标定区域按 {prof.s:.3f} 换算："
          f"红环{cmark.rect if cmark else '—'} "
          f"撤离{emark.rect if emark else '—'} "
          f"连携{kmark.rect if kmark else '—'} "
          f"血条{_ao.HP_RECT}")

    if eye is not None:
        # ★ 眼睛必须是**以屏幕中心左右对称**的一块，否则输入本身就带方向偏置。
        #   实测：直接用整屏 + 遮罩的做法，右侧被叠加面板挖掉一大块
        #   （方位角 +54°~+135° 全没了），左边却露到 -80° ——
        #   结果方向分布 左153 : 右26（85% 往左）。
        #   所以改成对称裁切：把叠加面板/左右 HUD 全部让到框外，框内不需要遮罩。
        if args.eye_crop and args.eye_crop != "auto":
            ex0, ey0, ew, eh = (int(v) for v in args.eye_crop.split(","))
            eye_rect = (ex0, ey0, ew, eh)
        else:
            # ★★ 先算**画面区**（去掉上下黑边）—— 见 ef_vision.content_rect。
            #   实测：2560×1600 的窗口里跑的是 2560×1440 的 16:9 画面，
            #   上下各 80px 黑边。所有按 1440 标定的 HUD 坐标都必须整体下移 80px，
            #   而缩放系数要按**画面高度**算，不能按窗口高度算
            #   （按窗口算会得出 k=1.111 的假缩放，小地图圆心实测偏了 74px）。
            cx = w // 2
            half = min(cx - int(0.008 * w),          # 左边界 ≈20
                       int(0.941 * w) - cx,          # 右边界 ≈2410
                       args.eye_halfw)
            ex0 = x + cx - half
            ey0 = y + int(0.347 * ch)
            eye_rect = (ex0, ey0, 2 * half,
                        int(0.754 * ch) - int(0.347 * ch))
        eye_size = (max(160, eye_rect[2] // args.eye_ds),
                    max(90, eye_rect[3] // args.eye_ds))
        # ★ 遮罩必须在**屏幕坐标**下生成，再裁到眼睛框、缩到目标尺寸。
        #   直接把全屏比例遮罩套在裁切小图上会挖错地方（实测把方向偏置推到 95%）。
        #   裁切框本来就干净时，这里算出来自然是空遮罩。
        from ef_eye import rect_mask, HUD_RECTS_FRAC, block_any
        rects = list(HUD_RECTS_FRAC)
        if panel is not None:
            # 按面板实际位置生成遮罩（屏幕坐标 → 比例）
            # ★ 比例的分母要用**画面**尺寸 (w, ch)，不是窗口尺寸 ——
            #   面板坐标是屏幕坐标，但遮罩画布是画面尺寸，混用会整体错位。
            rects.append(((args.overlay_x - x) / w, (args.overlay_y - y) / ch,
                          (args.overlay_x - x + panel.w) / w,
                          (args.overlay_y - y + panel.h + panel.extra_h) / ch))
        _sy, _sx = eye_rect[1] - y, eye_rect[0] - x
        # ★★ 遮罩必须按**画面高度 ch** 生成，不能按窗口高 h ——
        #   黑边让它错位 80px，遮错地方（实测遮挡率从 0.3% 虚涨到 2.3%）。
        #   eye_rect 已经带了画面原点偏移，所以这里减 y 之后与 ch 画布对齐。
        _sub = rect_mask(ch, w, rects)[_sy:_sy + eye_rect[3],
                                       _sx:_sx + eye_rect[2]]
        eye_mask_hud = block_any(_sub.copy(), eye_size)   # 仅 HUD/面板
        cov_hud = 100.0 * eye_mask_hud.mean()
        # ② 角色背影：第三人称相机下它杵在画面正中，正好挡住「正前方」。
        #    用**对称的中心竖带**遮掉（对称就不会引入左右偏置）。
        if args.eye_skip and args.eye_skip != "none":
            f = [float(v) for v in args.eye_skip.split(",")]
            sy0, sy1 = int(f[1] * eye_rect[3]), int(f[3] * eye_rect[3])
            sx0, sx1 = int(f[0] * eye_rect[2]), int(f[2] * eye_rect[2])
            _sub[sy0:sy1, sx0:sx1] = True
        eye_mask = block_any(_sub, eye_size)
        cov_all = 100.0 * eye_mask.mean()

        print(f"主画面眼睛：裁切 {eye_rect}（{eye_rect[2]}×{eye_rect[3]}）"
              f" → 缩到 {eye_size[0]}×{eye_size[1]}")
        print(f"  横向对称性：左边界 {eye_rect[0]}，右边界 "
              f"{eye_rect[0]+eye_rect[2]}，屏幕中心 {x + w//2}"
              f"（{'对称 ✓' if abs((eye_rect[0]+eye_rect[2]/2) - (x+w/2)) < 4 else '不对称 ✗'}）")
        print(f"  HUD/面板遮挡 {cov_hud:.1f}%"
              f"{'（干净 ✓）' if cov_hud < 1.0 else '（⚠ 框里有 HUD，会引入偏置）'}"
              f" · 含角色遮罩后共 {cov_all:.1f}%")
        if args.eye_skip and args.eye_skip != "none":
            print(f"  已遮角色背影：裁切框内 x {f[0]:.0%}~{f[2]:.0%} · "
                  f"y {f[1]:.0%}~{f[3]:.0%}（对称，不引入左右偏置）")

    # ★★ 先算**画面区**（去掉上下黑边），再算所有 HUD 坐标。
    #   2560×1600 的窗口里跑的是 2560×1440 的 16:9 画面 + 上下各 80px 黑边。
    #   按窗口高算缩放会得到 k=1.111 的假缩放，小地图圆心实测偏 74px。
    # 画面区已在上面算好（见 hwnd, x, y, w, h 之后）
    rect_disc = minimap_rect(ch, inner=False, ox=x, oy=y)
    rect_inner = minimap_rect(ch, inner=True, ox=x, oy=y)
    # 从圆盘大图里切出内方框的偏移
    off = (rect_disc[2] - rect_inner[2]) // 2
    print(f"小地图：圆盘 {rect_disc} · 内框 {rect_inner}")

    ex = None
    ic = None
    if not args.dry:
        from interception_py import Interception
        import win_click as wclick      # ★ 鼠标按键走 SendInput（见 combat_press）
        ic = Interception()
        kb = ic.keyboard_device()
        ms = ic.mouse_device()
        print(f"Interception：键盘 {kb}，鼠标 {ms}")

        class EFExec:
            name = "endfield-interception"
            # ★ 完整 WASD：W/S 前后、A/D 左右**平移**，鼠标单独负责转视角
            KEYS = {"forward": "w", "back": "s", "left": "a", "right": "d",
                    "jump": "space", "sprint": "shift"}

            def __init__(self):
                self.held = set()

            def _sync(self, logical, want):
                now = logical in self.held
                if want and not now:
                    ic.key_down(self.KEYS[logical], kb)
                    self.held.add(logical)
                elif not want and now:
                    ic.key_up(self.KEYS[logical], kb)
                    self.held.discard(logical)

            def apply(self, plan, turn_dx):
                self._sync("forward", plan.move > 0)
                self._sync("back", plan.move < 0)
                self._sync("sprint", bool(plan.sprint and plan.move > 0))
                if turn_dx and ms:
                    ic.mouse_move(int(turn_dx), 0, ms)
                if plan.jump:
                    ic.tap("space", 0.09, kb)

            def apply_raw(self, fwd=False, back=False, left=False, right=False,
                          turn_dx=0, jump=False, sprint=False):
                """群体解码口径：直接给六个键的状态 + 鼠标位移。"""
                self._sync("forward", fwd)
                self._sync("back", back)
                self._sync("left", left)
                self._sync("right", right)
                self._sync("sprint", sprint)
                if turn_dx and ms:
                    ic.mouse_move(int(turn_dx), 0, ms)
                if jump:
                    ic.tap("space", 0.09, kb)

            def combat_press(self, key: str):
                """战斗键：做一次**阻塞式**点击。

                为什么不做跨拍状态机：按压时长 ~120ms 远短于决策拍 400ms，
                一次按压会在同一拍内开始并结束，跨拍看不到沿。

                ★★ 鼠标按键和键盘必须走**不同的注入方式**（实测定论）：
                     · 键盘 → Interception。ACE 会把 SendInput 的按键丢掉
                       （W 的验证：Interception 5.40× vs SendInput 1.02×）
                     · 鼠标按键 → **SendInput**。Interception 的鼠标按键
                       interception_send 返回 1（"成功"）但系统收不到，
                       10 个鼠标设备全试过，游戏也不响应；SendInput 一按就挥刀。
                   别把这两条统一 —— 统一到哪一边都会废掉一半输入。
                """
                if key.startswith("mouse_"):
                    wclick.click(key.split("_", 1)[1])      # ★ SendInput
                else:
                    ic.tap(key, 0.10, kb)                   # ★ Interception

            def combat_hold_down(self, key: str):
                """**按下不松** —— 大招（长按 1/2/3/4）用。

                ⚠ 和普攻长按同样的理由：按住时长（~1 秒）超过决策拍（400ms），
                  不能在一次调用里 sleep，否则整个主循环卡住。
                  按下和抬起必须落在**不同的拍**上。
                """
                if key.startswith("mouse_"):
                    wclick.down(key.split("_", 1)[1])
                else:
                    ic.key_down(key, kb)

            def combat_hold_up(self, key: str):
                """松开大招键。见 combat_hold_down()。"""
                if key.startswith("mouse_"):
                    wclick.up(key.split("_", 1)[1])
                else:
                    ic.key_up(key, kb)

            def release_movement(self):
                """只松开方向键（战斗让位时用），保留其他状态。"""
                for k in ("forward", "back", "left", "right"):
                    self._sync(k, False)

            def release_all(self):
                for k in list(self.held):
                    try:
                        ic.key_up(self.KEYS[k], kb)
                    except Exception:
                        pass
                self.held.clear()

        ex = EFExec()

    # ---- 对照组：运动输出关掉 = 一个键都不按 ----
    # ★ 用**包装**而不是在每个调用点加判断：调用点有 10 来个，
    #   漏一个就白测了（前面战斗不许跳那个 bug 就是漏了三条路径之一）。
    #   包一层能把所有出口一次堵死，而且对外接口完全不变。
    if not args.ab_motor:
        class _NoMotor:
            """把执行器包成空操作：照样算、照样显示，但一个键都不发。"""
            def __init__(self, inner):
                self._inner = inner
                self.blocked = 0

            def __getattr__(self, name):
                inner = self._inner
                # ★ 只拦「发送」类，其余（查询 / 松键）原样透传。
                #   第一版只拦了 apply/press/hold/click，**漏了 combat_press
                #   和 combat_hold_down** —— 那才是打怪真正用的通道，
                #   结果「关掉运动输出」之后它照样在打。
                #   和前面「战斗不许跳」那个 bug 同一个形状：**闸门漏了一条路径**。
                #   release_* 刻意放行：松键永远安全，卡着不放更糟。
                if name.startswith(("apply", "press", "hold", "click",
                                    "combat_")):
                    def _blocked(*a, **k):
                        self.blocked += 1
                        return None
                    return _blocked
                return getattr(inner, name)

            def __repr__(self):
                return f"<NoMotor blocked={self.blocked}>"

        ex = _NoMotor(ex)
        print("  ★ 对照组：运动输出已关闭 —— 会照常决策和显示，但一个键都不按")

    # ★ 叠加窗口要在**抢游戏焦点之前**建好：
    #   建窗口本身可能激活自己，反过来的话会把刚拿到的焦点又抢走。
    ov = None
    if args.overlay:
        from overlay import Overlay
        # ★ 高度取 **最大**（含战斗条）：Overlay.update() 尺寸不符会抛异常，
        #   而运行时按 0 切战斗模式会改变面板高度。战斗关闭时底部保持透明。
        ov = Overlay(panel.w, panel.h + panel.extra_h,
                     x=args.overlay_x, y=args.overlay_y)
        ov.pump()
        print(f"叠加面板 {panel.w}×{panel.h + panel.extra_h} @ "
              f"({args.overlay_x},{args.overlay_y})  hwnd={ov.hwnd}"
              f"（底部预留 {panel.extra_h}px：{' + '.join(panel.strips) or '无'}）")

    user32.ShowWindow(hwnd, 9)
    # ★ 把应用窗口重新推到最前 —— 上面这句恢复了全屏游戏，而叠加面板
    #   又是 TOPMOST，tkinter 窗口会被压在下面甚至从未被映射（实测
    #   IsWindowVisible=False）。必须在这些都建好之后再推一把。
    if args.panel and _app is not None:
        _app.request_show()
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.0)
    print(f"前台：{user32.GetForegroundWindow() == hwnd}")

    # ---- 高帧率眼睛线程（录屏关闭时才开；录屏会抢 GDI/CPU）----
    eye_th = None
    if eye is not None and args.eye_hz > 0:
        from ef_eye import EyeThread
        eye_th = EyeThread(eye, eye_rect, eye_size, eye_mask, hz=args.eye_hz)
        eye_th.start()
        t_wait = time.time()
        while eye_th.latest is None and time.time() - t_wait < 3.0:
            time.sleep(0.05)
        print(f"眼睛线程已启动：目标 {args.eye_hz:.0f} Hz"
              f"（首帧 {'就绪' if eye_th.latest is not None else '超时'}）")

    # ---- 强制结束键 ----
    VK = {"end": 0x23, "f12": 0x7B, "pause": 0x13, "home": 0x24}
    stop_vk = VK.get(args.stop_key, 0)
    if stop_vk:
        print(f"暂停键：{args.stop_key.upper()}（按下暂停，点「启动」接着跑）")
    else:
        print("暂停键：关闭")

    # ---- 战斗模式开关 ----
    COMBAT_VK = {"0": 0x30, "9": 0x39, "-": 0xBD, "=": 0xBB}
    combat_vk = COMBAT_VK.get(args.combat_key, 0x30)
    combat_key_down = False
    # ★ 普攻长按状态（跨拍：按下记时刻，到点再松）
    atk_holding = False
    atk_hold_until = 0.0
    print(f"战斗模式：{'开' if combat_on else '关'}"
          f"（按 {args.combat_key} 切换）· 攻击键 {args.attack_key}"
          f" · 技能键 {args.skill_key} · 方向{'让位' if args.combat_yield == 'attack' else '不让位'}")

    # ---- 录屏：**聚焦之后**才开录，免得把 UAC 弹窗/桌面录进去 ----
    rec = None
    if args.record:
        import subprocess
        # 剪映自带的 ffmpeg 构建**不含 libx264**，也不认 -crf ——
        # 只有硬件编码器。实测可用：nvenc（最好）/ qsv / mf；amf 会在本机失败。
        REC_ENC = {
            "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p1",
                           "-rc", "vbr", "-cq", "23"],
            "h264_qsv": ["-c:v", "h264_qsv", "-global_quality", "23"],
            "h264_mf": ["-c:v", "h264_mf", "-b:v", "6M"],
            "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"],
        }
        ff = None
        # ★ 不能写死剪映的版本号目录 —— 实测剪映从 11.4.0.14410 升到
        #   11.5.0.14471 后旧路径直接消失，硬编码会让录屏静默失败。
        #   所以按通配找最新版本，再退回 B 站客户端自带的那个。
        import glob as _glob
        for pat in (r"C:\Program Files\JianyingPro\Apps\*\ffmpeg.exe",
                    r"C:\Users\lanpang\AppData\Roaming\bilibili\ffmpeg\ffmpeg.exe"):
            hits = sorted(_glob.glob(pat))
            if hits:
                ff = hits[-1]
                break
        if ff is None:
            ff = "ffmpeg"
        print(f"  ffmpeg: {ff}")
        cmd = ([ff, "-hide_banner", "-loglevel", "error",
                "-f", "gdigrab", "-framerate", str(args.record_fps),
                "-offset_x", "0", "-offset_y", "0",
                "-video_size", f"{w}x{h}", "-i", "desktop",
                "-vf", f"scale={args.record_width}:-2"]
               + REC_ENC[args.record_encoder]
               + ["-pix_fmt", "yuv420p", "-y", args.record])
        log = open(args.record + ".log", "wb")
        rec = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=log)
        print(f"录屏中 → {args.record}  ({args.record_fps}fps, "
              f"{args.record_width}px 宽, {args.record_encoder})")

    total = int(args.seconds * args.hz)
    print(f"\n闭环 {args.seconds:.0f} 秒 = {total} 拍 @ {args.hz} Hz"
          f"{'（DRY）' if args.dry else ''}"
          f"{'（无撞墙反射）' if args.no_escape else ''}")
    print(f"眼睛：{'主画面（第一人称口径）' if eye is not None else '俯视小地图'}"
          + (f" · hfov {args.hfov:.0f}° · {args.eye_map} · ds{args.eye_ds}"
             if eye is not None else f" · {args.center}"))
    print()

    if args.snap_every and args.snapdir:
        os.makedirs(args.snapdir, exist_ok=True)

    # ★★ 「等启动」放在这里 —— 画面区已经算好，x/y 是**画面原点**，
    #   空闲扫描才能抓对位置。（放在面板启动那一段里会偏 80px。）
    if args.panel:
        _poll_markers_idle()
        _t_wait = time.time()
        while not CTL["running"] and not CTL["stop"]:
            _poll_markers_idle()
            time.sleep(0.25)
            if time.time() - _t_wait > 3600:
                print("等待超时（1 小时），退出")
                CTL["stop"] = True
                break
        if CTL["stop"]:
            print("收到停止请求，退出。")
            return

    # ★ 任务完成检测默认关着 —— 明确说一句，免得用户以为它在工作。
    #   靠屏幕认副本结束这条路实测误触太多（三版判据都翻车），
    #   所以从默认行为里拿掉了。要的话加 --mission。
    if not args.mission:
        print("  任务完成检测：**已关闭**")
        print("     （默认关 —— 靠屏幕认副本结束实测误触多。"
              "想要的话加 --mission 打开）")

    # ---- 技能槽 → 键 的映射（用户可配）----
    #   role-N     决定第 N 槽**是什么**（攻击/增伤/格挡/回血/聚怪/挂削弱/不用）
    #   key-skillN 决定第 N 槽**按哪个键**
    #   两件事**独立**：换配队只改键，换打法则只改含义。
    #   ★ 果蝇从画面上看不出技能是干什么的（没有 mod），只能把语义当本能写进去 ——
    #     写死一套就只对一种配队有效，所以必须可配。
    _ROLES_TUPLE = tuple(getattr(args, f"role_{i}") for i in range(1, 5))
    _KEYMAP = {i: str(getattr(args, f"key_skill{i}")) for i in range(1, 5)}
    _ROLE_CN = {"attack": "攻击", "buff": "增伤", "block": "格挡",
                "heal": "回血", "gather": "聚怪", "weaken": "挂削弱",
                "none": "不用"}
    print("  技能槽配置：" + "  ".join(
        f"{i}={_ROLE_CN.get(_ROLES_TUPLE[i-1], _ROLES_TUPLE[i-1])}→键{_KEYMAP[i]}"
        for i in range(1, 5)))
    print(f"  大招键序列：{args.key_ult}（轮换）")

    # ★★ 整帧对象：每拍抓一次屏幕，所有 HUD 从它切片。
    #   见 live_frame.py 顶部注释 —— 换来的是「同一时刻」而不是「更快」。
    from live_frame import LiveFrame
    lf = LiveFrame(prof, grab)

    # ---- 「挑战成功」检测（副本完成）----
    # ★ 用户要求：任务完成时给 50 次奖励 + 饥饿值不再上升。
    #   横幅只停留半秒到一秒多，2.5Hz 下只有一到三次机会 ——
    #   所以 MissionMarker **锁存**：看到一次就记住，这一局不再重复领奖。
    mmark = None
    if args.mission and not args.no_mission:
        from mission import MissionWatcher, reward_burst
        # ★★ 判据 = 任务条文字**连续消失 N 秒**（用户的设计）。
        #    用模板匹配「击败所有敌人」这行字 —— 文字是"任意形状"，
        #    归一化相关是**相对**的，背景亮暗怎么变都不影响。
        #    （前两版：横幅被面板遮住 ✗ / 颜色判据太容易误判 ✗）
        mmark = MissionWatcher(prof.rect("mission"),
                               tpl=prof.template("kill_text"),
                               corr_min=args.mission_corr,
                               gone_sec=args.mission_gone,
                               verbose=(not args.panel))

    odo = MapOdometry(baseline=max(0.8, 2.5 / args.hz), stuck_sec=args.stuck_sec)
    # 预热里程计（要攒够 baseline 才能出第一次读数）
    t_warm = time.time()
    while time.time() - t_warm < odo.baseline + 0.2:
        odo.step(grab(*rect_inner))
        time.sleep(0.1)

    heading = odo.heading
    # ★★ 外层循环：一轮跑完（或者按 End）**回到"等启动"状态**，而不是直接退出。
    #   用户要求：「按 end 结束时回到启动界面不是直接退出」。
    #   一次性的初始化（里程计预热、眼睛线程）留在外面，每轮要重置的状态
    #   全部搬进这个循环里 —— 不然第二轮会带着上一轮的轨迹/朝向继续跑。
    while True:
        total = int(args.seconds * args.hz) if args.seconds > 0 else 10 ** 9
        r = np.zeros(n, dtype=np.float32)
        t0 = time.time()
        _t_prev = 0.0   # 上一拍的时间戳（算这一拍实际时长用）
        rows = []
        escape_left = 0
        walk_lock = 0
        traj = []
        counts = dict(left=0, right=0, jump=0, sprint=0, esc=0, stuck=0)
        logbuf: list = []
        spikes_total = 0
        steps_total = 0
        sector_wins = [0] * 5
        using_minimap = False
        blind_ticks = 0
        good_ticks = 0
        switch_count = 0
        was_minimap = False
        last_switch = -10 ** 9
        cmd_move_prev = True
        heading = odo.heading
        if forager is not None:
            forager["hg"].h = 1.0        # 新的一轮从"饿"开始
            forager["combat_t0"] = 0.0
            forager["in_combat"] = False
        print(f"\n闭环 {args.seconds:.0f} 秒 = {total} 拍 @ {args.hz} Hz"
              f"{'（DRY）' if args.dry else ''}"
              f"{'（无撞墙反射）' if args.no_escape else ''}")
        print(f"  {'拍':>4s} {'t':>6s} {'驱动':>9s} {'margin':>10s} {'方向':>4s} "
              f"{'跳':>3s} {'冲':>3s} {'朝向':>6s} {'速度':>6s} {'卡':>3s}  动作")
        try:
            for tick in range(total):
                ts = time.time() - t0
                # 这一拍距上一拍的**实际时长**（秒）。任务完成的消失计时要用它 ——
                # 主循环降频、卡顿、暂停恢复时都能算对（不是按墙钟跳）。
                _dt_tick = max(1e-3, min(5.0, ts - _t_prev))
                _t_prev = ts

                # ---- 记忆存档：执行界面挂上来的读档请求 ----
                # ★ 界面 / HTTP 线程只写 CTL["load_slot"]，真正换权重在这里做 ——
                #   绝不让别的线程直接动 forager 的蘑菇体。
                # ---- 清空记忆（界面挂上来的请求）----
                if CTL.pop("reset_mem", False) and forager is not None:
                    try:
                        forager["mb"].reset()
                    except Exception as _e:
                        print(f"  [记忆] 清空失败：{_e}", flush=True)

                _ls = CTL.pop("load_slot", None)
                if _ls is not None and forager is not None and not args.dry:
                    try:
                        import live_tune as _lt
                        forager["mb"].load(_lt.slot_path(int(_ls)))
                        print(f"  [存档] 已读取槽位 {_ls} · "
                              f"漂移 {forager['mb'].drift:.1f}", flush=True)
                    except Exception as _e:
                        print(f"  [存档] 读档失败：{_e}", flush=True)

                # ---- End 键 = **暂停**（不是结束这一轮）----
                # ★ 用户要求："end键要的是'暂停'不是'停止'"。
                #   所以这里**不 break** —— 只把 running 置 False，
                #   然后掉进下面的暂停分支：松开按键、藏起面板、原地等。
                #   再点「启动」是**接着这一轮继续跑**，不是从头开新的一轮
                #   （t0 / 轨迹 / 计数 / 饥饿值全都保留）。
                #   想真正结束一轮、从头再来：点面板上的「停止」。
                if stop_vk and (user32.GetAsyncKeyState(stop_vk) & 0x8000):
                    if not _pause_logged[0]:
                        print(f"\n  ⏸ 检测到 {args.stop_key.upper()} 键 → 暂停"
                              f"（再按一次没反应，点「启动」继续）")
                    CTL["running"] = False

                # ---- 控制台：停止 / 暂停 ----
                # 网页上按的按钮只写标志，主循环在自己的节拍上检查 —— 从 HTTP
                # 线程里直接动主循环的数据结构迟早出事。
                if CTL["stop"]:
                    print("\n  ★ 控制台请求停止 → 退出")
                    break
                if not CTL["running"]:
                    # 暂停：**先松开所有键**再睡，不然角色会一直按着上一拍的指令
                    if ex is not None:
                        try:
                            ex.release_all()
                        except Exception:
                            pass
                    # ★ 把叠加面板藏起来 —— 主循环停了它就不刷新，留在屏幕上
                    #   既像"程序卡死"，又挡着游戏画面。
                    if ov is not None and not _ov_hidden[0]:
                        ov.set_visible(False)
                        _ov_hidden[0] = True
                    if not _pause_logged[0]:
                        print("\n  ⏸ 已暂停（控制台里点「启动」继续）")
                        _pause_logged[0] = True
                    _poll_markers_idle()      # 空闲也要让「位置」跟着画面走
                    time.sleep(0.25)
                    continue
                if _pause_logged[0]:
                    print("  ▶ 已恢复")
                    _pause_logged[0] = False
                if ov is not None and _ov_hidden[0]:
                    ov.set_visible(True)
                    _ov_hidden[0] = False

                # ---- 战斗标记扫描（红环）——★ 一次扫描，两条路都用 ----
                # ① combat.py：战斗中提高攻击欲望（阈值 ×urge）
                # ② forage 闭环：**气味源**（阶段一：一种敌人 = 红环布尔）
                # ⚠ 必须放在循环最开头。放过一次在 combat 块里、一次在它前面，
                #   两次都是 UnboundLocalError —— drive 要靠它，而 drive 在很前面。
                # ★★ **每拍只抓一次屏幕**，所有 HUD 从这一帧切片。
                #   改之前这一拍要抓 7 次（红环/撤离/连携/血条/左右红晕/小地图），
                #   两次之间隔十几毫秒 —— 而游戏是 60fps 在跑的，也就是说
                #   "血条还剩 30%"和"红环亮着"**可能不是同一时刻的状态**。
                #   战斗里 100ms 足够让血量掉一截、让连携提示出现或消失。
                #   实测耗时：整屏 87ms vs 7 次分区域 ≈105ms —— 更一致，还略快。
                _lf_ok = lf.update()

                m_red = False
                if cmark is not None:
                    _rr = cmark.rect              # 已按当前分辨率换算过
                    _mimg = lf.at(*_rr)
                    if _mimg is not None:
                        m_red = cmark.step(_mimg)
                # ★ 「撤离」面板在 = 副本内 = 战斗状态。
                #   用它当闸门比红叉合适：红叉要打起来才亮，而撤离按钮进副本就有。
                #   --evac 没开时才退回红叉。
                m_evac = False
                if emark is not None:
                    _er = emark.rect
                    _eimg = lf.at(*_er)
                    if _eimg is not None:
                        m_evac = emark.step(_eimg)
                # 连携技（E）提示：只在战斗中、且能用的时候才出现
                m_combo = False
                if kmark is not None:
                    _kr = kmark.rect              # 用标定值，不依赖命令行
                    _kimg = lf.at(*_kr)
                    if _kimg is not None:
                        m_combo = kmark.step(_kimg)
                # ---- 「挑战成功」：任务完成 ----
                # ★ 用户要求：完成时给 **50 次奖励** + 饥饿值**不再上升**。
                #
                #   为什么是"连发 50 次"而不是"发一个 50 倍的大数"：
                #     deliver() 是多巴胺脉冲，后面还跟着资格迹和权重更新。
                #     一个 50 的大数会把权重一次性顶到 clip 边界 ——
                #     学到的不是"这样打是对的"，而是"所有活跃突触都到上限"。
                #     连发是**渐进**的，更接近真实神经元的 burst firing。
                #
                #   横幅只停留半秒到一秒多，2.5Hz 下只有一到三次机会，
                #   所以 mmark 内部**锁存**：看到一次就 fire()，这一局不再领。
                m_mission = False
                if mmark is not None:
                    _mimg = lf.region("mission")
                    if _mimg is not None:
                        # ⚠ 要传这一拍的实际时长 —— 消失计时按它累加，
                        #   主循环降频/卡顿时也不会算错。
                        m_mission = mmark.step(_mimg, _dt_tick)
                if m_mission and fod is not None:
                    mmark.fire()
                    reward_burst(fod["mb"], n_times=args.mission_reward,
                                 magnitude=args.mission_mag)
                    # ★ 打赢一局 = **吃了一顿**（用户要求）：
                    #     饥饿值清空 + N 秒内不增长 + 之后**恢复正常**。
                    #   不是永久冻结 —— 吃饱了会饱一会儿，但过一阵还是会饿。
                    _h_before = fod["hg"].clear(args.mission_hold)
                    fod["n_mission"] = fod.get("n_mission", 0) + 1
                    print(f"  [{ts:6.1f}s] ★★★ 任务完成 -> "
                          f"{args.mission_reward} 次奖励 · 饥饿 "
                          f"{_h_before:.2f} → 0.00 · {args.mission_hold:.0f} 秒不涨",
                          flush=True)

                # ★ 战斗状态 = **副本内 或 红叉亮**（用户要求"全流程，大世界也可以出手"）。
                #   · 副本内（撤离面板在）→ 随时可以出手
                #   · 大世界 → 打起来了（红叉亮）才出手
                #   这样全流程都能跑：大世界遭遇战、副本清怪，同一套回路。
                #   想彻底不设闸门就加 --forage-always-attack。
                m_on = m_evac or m_red
                # ★ 把判定结果存进 forager，让状态面板能显示 —— 这两个变量是
                #   循环局部的，状态回调在别的线程里读不到，必须显式传出去。
                if forager is not None:
                    forager["in_dungeon"] = bool(m_evac)
                    forager["in_red"] = bool(m_red)
                cdec.set_urge(args.urge if m_on else 1.0)

                # 小地图也从**同一帧**切（原来这里又单独抓了一次）
                disc = lf.at(rect_disc[0] - x, rect_disc[1] - y,
                             rect_disc[2], rect_disc[3])
                if disc is None:
                    disc = grab(*rect_disc)      # 兜底：整帧没抓到时退回单抓
                inner = disc[off:off + rect_inner[2], off:off + rect_inner[3]]

                od = odo.step(inner)

                # ---- 朝向：走动时用位移方向（绝对），卡住时用转向命令航位推算 ----
                moved = (od["primed"] and od["speed"] >= odo.min_speed
                         and od["sharp"] >= 1.3)
                if moved:
                    heading = od["heading"]

                if eye is not None:
                    # ★ 主画面口径：眼睛看的是角色视角，不是小地图。
                    #   注意它**不需要 heading** —— 第一人称画面本身就是自我中心的，
                    #   感光细胞的方位角直接对应画面的横向位置。
                    #   heading 仍由小地图里程计提供（面板显示 + 撞墙判定用）。
                    if eye_th is not None:
                        got = eye_th.read()
                        if got is None:
                            continue
                        drive, contrast, age = got
                    else:
                        drive, _lum, _drv = eye.sample(grab_scaled(
                            eye_rect[0], eye_rect[1], eye_rect[2], eye_rect[3],
                            eye_size[0], eye_size[1]), mask=eye_mask)
                        contrast = eye.last_contrast

                    # ---- 对照组：视觉关掉 = 瞎了 ----
                    # 光流恒为 0：网络收不到任何视觉输入，只能靠残余驱动乱走。
                    # 这是"眼睛到底在不在起作用"的直接对照。
                    if not args.ab_vision:
                        drive = np.zeros_like(drive)
                        contrast = 0.0

                    if args.eye_auto:
                        # 滞回切换：低到 eye_contrast 以下、且连续 2 拍才切走；
                        # 回到 eye_back 以上、且连续 4 拍才切回。
                        # ★ 再加**最小驻留**：实测不加的话 120 秒里抖了 12 次（约 10 秒一次），
                        #   两种感官的统计特性完全不同，频繁换眼睛对网络不友好。
                        dwell_ok = (tick - last_switch) >= args.eye_dwell
                        # 「失效」= 画面没信息（对比度低）**或** 推不动（相机多半贴背）
                        blind_now = (contrast < args.eye_contrast) or (
                            args.eye_switch_stuck and bool(od["stuck"]) and cmd_move_prev)
                        if not using_minimap:
                            blind_ticks = blind_ticks + 1 if blind_now else 0
                            if blind_ticks >= 2 and dwell_ok:
                                using_minimap = True
                                blind_ticks = 0
                                good_ticks = 0
                                last_switch = tick
                        else:
                            good_ticks = good_ticks + 1 if not blind_now else 0
                            if good_ticks >= 4 and dwell_ok:
                                using_minimap = False
                                good_ticks = 0
                                blind_ticks = 0
                                last_switch = tick
                        if using_minimap:
                            # 相机贴背 → 主画面失去结构 → 退回看小地图
                            drive = rm.encode(disc, heading_deg=heading,
                                              center=args.center)
                else:
                    drive = rm.encode(disc, heading_deg=heading, center=args.center)
                    contrast = float("nan")

                # ---- 觅食闭环：把"气味"写进 drive，并把学到的权重增量加进 tanh ----
                # §七：游戏里没有嗅觉，这一路信号是**造出来再塞给果蝇的**。
                #      注入点 = ORN（按肾小球命名），三段代码：筛 ID → 建映射 → 每拍写 drive。
                # §七 还要求：**持续注入不脉冲** + **要有基线活动**（零基线会让网络静默，
                #      已实测：零基线下 |r| 恒为 0）。
                fod = None
                if forager is not None:
                    f_odor = bool(m_on)            # 阶段一：一种敌人 = 红环（有无敌人）
                    # ---- 对照组：气味关掉 = 闻不到敌人 ----
                    if not args.ab_odor:
                        f_odor = False
                    fg = forager
                    fg["odor"] = f_odor
                    # ★ 记录这一场战斗的开始时刻 —— 技能轮换要看"开战多久了"：
                    #   开局先挂削弱(2) → 接着增伤(3) → 之后主攻(1)。
                    #   用**上升沿**，不是电平（电平的话每一拍都会重置成 0）。
                    if m_on and not fg.get("in_combat"):
                        fg["combat_t0"] = ts
                    fg["in_combat"] = bool(m_on)
                    # ---- 对照组：饥饿关掉 = 锁在 0.5 不动 ----
                    if args.ab_hunger:
                        fg["hg"].step(1.0 / args.hz)
                    else:
                        fg["hg"].h = 0.5
                    fg["enc"].write(drive, [fg["ch"]] if f_odor else [],
                                    gain=fg["hg"].odor_gain())
                    fod = fg

                # ---- 对照组：连接组关掉 = 用随机数代替神经网络输出 ----
                # ★★ 这是**阴性对照**：如果关掉前后行为差不多，
                #    那这 211,577 个神经元就只是装饰。
                #    随机源用固定种子，保证两次对照跑的是同一串随机数（可比）。
                if not args.ab_connectome:
                    _rng = getattr(main, "_ab_rng", None)
                    if _rng is None:
                        _rng = np.random.default_rng(20261008)
                        main._ab_rng = _rng
                    # 用同样的分布尺度（网络 r 大约在 ±2），保持"驱动量级"可比
                    r = _rng.normal(0.0, 1.0, size=r.shape).astype(np.float32)
                else:
                    for _ in range(args.steps):
                        _y = W @ r + drive
                        if fod is not None:
                            _y = _y + fod["mb"].contribution(r)
                        r = (args.a * r + args.b * np.tanh(_y)).astype(np.float32)

                o = dec.step(r, dt=1.0 / args.hz)
                direction = dec.direction_name()
                turn = -1 if direction == "左" else (1 if direction == "右" else 0)
                acts = amap.step(r)
                rawj = float(acts["raw"].get("jump", float("nan")))
                rawf = float(acts["raw"].get("forward", float("nan")))

                # ---- 觅食闭环：读出 MBON → 算驱动力 → 决定要不要攻击 ----
                # 执行层（先天反射）：闻到 → 接近 → 进食(=攻击)。
                # ★ 驱动力被**饥饿值**调制（§五 ②），不是被恐惧。
                f_atk = False
                if fod is not None:
                    _m = fod["mb"]
                    # ★★ 读**蘑菇体给 MBON 的输出**，不是 MBON 自己的活动。
                    #   ⚠ 踩过：原来直接读 r[mbon_idx].mean()，结果战斗中它翻负
                    #     （实测红叉亮那 17 拍是 −0.715 ~ −0.175，全时段中位 +1.565）
                    #     → 驱动力为负 → 一次都不出手。
                    #   原因：MBON 接收的不只是 KC，还吃别的输入，视觉一变
                    #     （敌人、特效、受击红晕）它就跟着晃，根本不是嗅觉信号。
                    #   正确读法 = KC→MBON 那一块的**加权和**。KC 的输入几乎全来自
                    #     ALPN（嗅觉），所以这条路是干净的。
                    f_mbon = float((_m.w @ r[fod["kc"]]).mean())
                    # 饱和归一必须对**任意实数**安全：
                    #   ⚠ 第一版是 mbon/(mbon+1)：MBON < −1 时分母翻号，
                    #     mbon=−1.01 时直接爆到 −100（实测驱动力跑到 ±28）。
                    #   参考尺度用**运行中位数**自适应 —— 不同学习进度下 KC→MBON
                    #     的量级差很多，写死一个常数换个补丁就失灵。
                    fod["mbout_ref"] += 0.02 * (abs(f_mbon) - fod["mbout_ref"])
                    _ref = max(fod["mbout_ref"], 1e-6)
                    f_learn = f_mbon / (abs(f_mbon) + _ref)      # ∈ (−1, 1)
                    # ★★ 驱动力 = **先天取食反射 + 学习调制**。
                    #   ⚠ 踩过：原来写成 `驱动力 = 归一(蘑菇体输出) × 饥饿`，
                    #     也就是**学习结果独占**。结果 4 次惩罚就把输出压成负的
                    #     → 驱动力恒负 → 一次都不出手（实测 红叉亮 29 拍、出手 0 拍）。
                    #     那是"学会了厌食"，不是果蝇。
                    #   交接 §三 原话：执行层的取食反射**是先天的，不用学**。
                    #   所以先天那一份固定给足，学习只能在它上面上下浮动 ——
                    #     学好了更积极，学坏了变保守，但**饿极了照样会吃**。
                    f_urge = (args.forage_innate + args.forage_learn * f_learn) \
                        * fod["hg"].drive_gain()
                    # ★★ 攻击**不要求先闻到气味** —— 它是先天的取食反射（§三 执行层），
                    #   不用学，也不该等气味。
                    #   ⚠ 我第一版加了「没闻到不许吃」的闸门，结果造出**死锁**：
                    #     气味源 = 红环，而红环只在**战斗中**才亮；战斗又要先攻击才会开始。
                    #     要闻到得先战斗，要战斗得先闻到 —— 它永远开不出第一枪。
                    #   正确的顺序是：**主动扑上去 → 吃到 → 才知道是什么味道**。
                    #   （--forage-need-odor 可以把那条闸门调回来做对照）
                    # ★★ 红叉（= 进入战斗）是攻击的**必要前提** —— 用户明确要求：
                    #   "在左上角的红叉出现之前不要让果蝇做出攻击"。
                    #   没进战斗就别挥刀（免得对着空气打，看着像挂机脚本）。
                    #   战斗由敌人主动进攻、或人来发起。
                    #   （--forage-always-attack 可以关掉这个闸门）
                    f_atk = bool(f_urge > args.forage_thr)
                    if not args.forage_always_attack:
                        f_atk = f_atk and m_on
                    fod["mbon_val"] = f_mbon
                    fod["urge"] = f_urge
                    fod["attack"] = f_atk
                    # ★ 顺序要紧（§九 时间对齐）：气味已经在上面写进 drive、
                    #   这一拍 r 里已经含了气味响应 —— 此时再消费奖赏脉冲，
                    #   "气味在先、奖赏在后"才对得上。反了会学不会（已实测）。
                    # 对照组：蘑菇体学习关掉 = 权重冻结（读取正常，但不再更新）。
                # 和「多巴胺关掉」的区别：那个是信号本身没了，
                # 这个是信号还在但不写进去—— 用来区分学不会和没得学。
                if args.ab_mushroom:
                    _m.step(r, dt=1.0 / args.hz, reward_gain=fod["hg"].da_gain())

                # ---- 群体解码：下行神经元 → 三条轴 → WASD + 鼠标 ----
                pm = None
                if pmm is not None:
                    pm = pmm.step(r, dt=1.0 / args.hz)
                    # 面板/日志用的方向标签（由转身轴的符号给）
                    tz = pm["turn_z"]
                    direction = "右" if tz > 0.8 else ("左" if tz < -0.8 else "未定")
                    turn = 1 if direction == "右" else (-1 if direction == "左" else 0)

                # ---- 战斗模式：0 键切换（按下沿，不是电平）----
                _cd = bool(user32.GetAsyncKeyState(combat_vk) & 0x8000)
                if _cd and not combat_key_down:
                    combat_on = not combat_on
                    cdec.reset()
                    print(f"  [{ts:6.1f}s] 战斗模式 → {'开' if combat_on else '关'}")
                combat_key_down = _cd

                # （红环扫描已在循环开头做过，这里直接用 m_on）
                cm = cdec.step(r, dt=1.0 / args.hz, now=ts,
                                 combat=(combat_on and m_on))
                yield_dir = (combat_on and args.combat_yield == "attack"
                             and (cm["attack"] or cm["skill"]))

                # 放电统计：速率模型没有离散脉冲，所以用「显著高于整体的神经元数」当代理量。
                # 阈值取 μ+2σ（标准离群口径）。
                # ★ 一开始用的是 2.5×均值 —— 但 r 的分布很集中时 max < 2.5×mean，
                #   会一直数出 0，读数就废了。μ+2σ 对分布形状不敏感。
                # ⚠ 必须放在 rows.append **之前**：rows 里要记这几个值。
                rmean = float(r.mean())
                rstd = float(r.std())
                rmax = float(r.max())
                spikes = int(np.count_nonzero(r > rmean + 2.0 * rstd))
                spikes_total += spikes
                steps_total += args.steps

                # ---- 撞墙反射 ----
                # ★ 前提门：只有**命令了移动键**时，"没动"才等于"撞墙"。
                #   群体解码允许果蝇主动「停」或只转视角（鼠标转镜头时角色原地不动），
                #   这两种情况位移天然为 0 —— 不加这个门就会把"主动停"误判成"撞墙"，
                #   实测逃逸占比高达 24~31%，而且用户观察到"没撞墙也在触发"。
                cmd_move = True
                if pm is not None:
                    cmd_move = bool(pm["key_w"] or pm["key_s"]
                                    or pm["key_a"] or pm["key_d"])
                cmd_move_prev = cmd_move
                stuck_now = bool(od["stuck"]) and cmd_move
                # ★★ 进了战斗（红叉亮）就不触发撞墙反射 —— 用户明确要求。
                #   理由：战斗里被敌人顶住、或贴身缠斗时，"推不动"是正常的，
                #   不是撞墙；这时原地掉头会把好不容易贴上去的身位丢掉。
                esc = False
                if (not args.no_escape and ts >= args.escape_grace
                        and not (m_on and not args.escape_in_combat)):
                    if escape_left > 0:
                        esc = True
                        escape_left -= 1
                        if escape_left == 0:
                            walk_lock = args.walk_lock
                    elif walk_lock > 0:
                        walk_lock -= 1
                    elif stuck_now:
                        esc = True
                        escape_left = max(0, args.escape_ticks - 1)

                esc_mult = args.escape_turn_mult if esc else 1.0
                if pm is not None:
                    # ===== 群体解码口径：三条轴直接驱动 WASD + 鼠标 =====
                    if esc:
                        # 撞墙反射：松开所有移动键，原地转
                        turn = turn if turn else 1
                        turn_dx = turn * args.turn_gain * 40 * esc_mult
                        if ex is not None:
                            ex.apply_raw(turn_dx=turn_dx)
                        action_txt = "WALL-ESC turn " + ("R" if turn > 0 else "L")
                    else:
                        turn_dx = pm["turn_dx"]
                        # ★ 战斗状态下**不许跳**（用户要求）。跳跃会打断攻击动作、
                        #   把身位带离敌人，还会浪费一次落地硬直。
                        jump = ((not args.no_jump) and bool(acts["jump"])
                                and not (m_on and not args.jump_in_combat))
                        sprint = (not args.no_sprint) and bool(acts["sprint"])
                        if ex is not None:
                            if yield_dir:
                                # §二.2 决定：攻击时方向控制**让位**（交给游戏自动寻路）
                                ex.release_movement()
                            else:
                                ex.apply_raw(pm["key_w"], pm["key_s"],
                                             pm["key_a"], pm["key_d"],
                                             turn_dx, jump=jump, sprint=sprint)
                        keys = "".join(k for k, on in (("W", pm["key_w"]),
                                                       ("A", pm["key_a"]),
                                                       ("S", pm["key_s"]),
                                                       ("D", pm["key_d"])) if on)
                        # ★ 按键**不写进动作文字** —— 面板上已经有专门的
                        #   按键徽标列了，写两遍反而乱（用户要求：按键要标出来，
                        #   但不是混在动作描述里）。这里只说做了什么。
                        action_txt = "MOVE" if keys else "IDLE"
                    plan = MotorPlan(
                        move=(1 if pm["key_w"] else (-1 if pm["key_s"] else 0)),
                        turn=turn,
                        # ★ 战斗状态下不许跳（用户要求）—— 和上面 apply_raw 那条同样的闸门
                        jump=(False if (args.no_jump or esc
                                        or (m_on and not args.jump_in_combat))
                              else bool(acts["jump"])),
                        sprint=(False if (args.no_sprint or esc) else bool(acts["sprint"])),
                        turn_amount=pm["turn_z"])
                else:
                    # ===== 旧口径：点名的 5 对左右 DN 取差 =====
                    turn_dx = turn * args.turn_gain * 40 * esc_mult
                    plan = MotorPlan(
                        move=0 if esc else 1,
                        turn=turn,
                        jump=(False if (args.no_jump or esc
                                        or (m_on and not args.jump_in_combat))
                              else acts["jump"]),
                        sprint=(False if (args.no_sprint or esc) else acts["sprint"]),
                        turn_amount=acts["turn"])
                    # ★ 旧口径的 describe() 是中文，指令列表要英文（用户要求）
                    action_txt = ("FWD " if plan.move > 0 else
                                  ("BACK " if plan.move < 0 else "IDLE ")) + \
                                 ("L" if plan.turn < 0 else ("R" if plan.turn > 0 else "-")) + \
                                 ("  SPACE" if plan.jump else "") + \
                                 ("  SHIFT" if plan.sprint else "")
                    if ex is not None:
                        ex.apply(plan, turn_dx)

                # ---- 战斗按键（交接文档）。★ 只有 combat_on 时才真的发出去 ----
                # ★★ 所有真正发出去的按键都记进 pressed_keys ——
                #    实时指令列表要靠它显示"这一拍到底按了什么"。
                #    用**英文大写**（用户要求）：W A S D SPACE SHIFT
                #    LMB RMB SK1..SK4 ULT1..ULT4 E
                pressed_keys = []
                if combat_on and ex is not None and cm["ready"]:
                    if cm["attack"]:
                        ex.combat_press(args.attack_key)
                        pressed_keys.append("LMB" if "mouse" in str(args.attack_key)
                                            else str(args.attack_key).upper())
                    if cm["skill"]:
                        ex.combat_press(args.skill_key)
                        pressed_keys.append("SK" + str(args.skill_key))
                if combat_on:
                    # 中文不再混进指令列表 —— 改用英文短标签
                    if cm["attack"]:
                        action_txt += "  ATK"
                    if cm["skill"]:
                        action_txt += "  SKL"

                # ---- 觅食闭环：攻击 + 手动奖赏/惩罚键 ----
                if fod is not None:
                    # 执行层：驱动力超阈值 → 出手（"进食"）。
                    # ★★ 普攻改成了**长按**（--attack-hold，默认 1 秒）。
                    #   1 秒 > 决策拍 400ms，所以不能在一次调用里 sleep ——
                    #   那样整个主循环（走路、挨打判定、连携技）都会被卡住。
                    #   按下和抬起拆到**不同的拍**上：按下记时刻，到点再松。
                    if ex is not None:
                        _mouse_atk = str(args.attack_key).startswith("mouse_")
                        if f_atk:
                            if _mouse_atk:
                                if not atk_holding:
                                    wclick.down(args.attack_key.split("_", 1)[1])
                                    atk_holding = True
                                    atk_hold_until = ts + args.attack_hold
                            else:
                                # 键盘键没法长按（Interception 的 tap 本身有沿），
                                # 退回点按；要长按的话把 --attack-key 改成 mouse_left
                                ex.combat_press(args.attack_key)
                                pressed_keys.append("LMB" if "mouse" in str(args.attack_key) else str(args.attack_key).upper())
                        # 抬起：不管这一拍还在不在打，到点就松（不然会一直按住）
                        if atk_holding and ts >= atk_hold_until:
                            if _mouse_atk:
                                wclick.up(args.attack_key.split("_", 1)[1])
                            atk_holding = False
                        if f_atk:
                            # ★ 技能选择。--skill-auto 时用「先天轮换 + 学习微调」：
                            #   1 伤害高 · 2 挂削弱 · 3 增伤 · 4 回血（用户给的语义当本能）
                            #   蘑菇体的四个分区（DAN→MBON 投射聚出来的）给它做微调。
                            #   否则退回老行为：永远按 --skill-key。
                            _sk = None
                            if ts - fod["last_skill"] >= args.skill_interval:
                                if args.skill_auto:
                                    from mb_learn import select_skill
                                    _slot = select_skill(
                                        _m.group_scores(r), fod["hg"].h,
                                        fod.get("auto_hp"),
                                        ts - fod["combat_t0"],
                                        heal_hp=args.heal_hp,
                                        roles=_ROLES_TUPLE)
                                    # ★ 返回的是**槽号**（1..4），不是键 ——
                                    #   键由 --key-skillN 决定。这样哪个槽什么含义
                                    #   和按哪个键是两件独立可配的事。
                                    _sk = _KEYMAP.get(_slot, args.skill_key)
                                elif fod["hg"].h >= args.skill_hunger:
                                    _sk = args.skill_key
                            if _sk is not None:
                                ex.combat_press(str(_sk))
                                pressed_keys.append(f"SK{_sk}")
                                fod["last_skill"] = ts
                                fod["n_skill"] += 1
                                if args.skill_auto:
                                    _gs = _m.group_scores(r)
                                    _hps = fod.get("auto_hp")
                                    _hps = f"{_hps*100:.0f}%" if isinstance(
                                        _hps, float) and _hps == _hps else "—"
                                    print(f"  [{ts:6.1f}s] ★ 技能 {_sk}"
                                          f"  （h={fod['hg'].h:.2f} 血={_hps} "
                                          f"战线={ts - fod['combat_t0']:.1f}s "
                                          f"分区={np.round(_gs, 1)}）")
                            # ★★ 大招：长按 1/2/3/4，**久攻不下、饿到极点**时放。
                            #   好玩的地方：战斗越久 = 越饿（没吃到东西时饥饿值一直
                            #   在回升），所以"久攻不下"和"极度饥饿"对果蝇来说是同一
                            #   件事 —— 它不需要"大招能量条"这个概念，它只有饿。
                            #   刻意**不**在血低时放：生物上受伤的果蝇会回避，不是猛攻，
                            #   而且那会和"血<50% 优先回血"的保命规则打架。
                            if args.ult and ex is not None and not fod["ult_holding"]:
                                _age = ts - fod["combat_t0"]
                                _hp = fod.get("auto_hp")
                                # ★ 收紧：**血条读不到就不放大招**。
                                #   第一版写的是"读不到就当满足"，结果第一次大招就是在
                                #   血=nan 时放的 —— 那时候它可能已经血很少了。
                                #   宁可漏放一次（20 秒后还有机会），也不要在垂死时拼命。
                                _hpok = (isinstance(_hp, float) and _hp == _hp
                                         and _hp >= args.ult_hp)
                                if (f_atk and fod["hg"].h >= args.ult_hunger
                                        and _age >= args.ult_age and _hpok
                                        and ts - fod["last_ult"] >= args.ult_cool):
                                    # ★★ 大招键 **1→2→3→4 轮换**。
                                    #
                                    #   用户要求："大招现在只放 1 的，我想当 1 没有大招时
                                    #   要能放 234 的大招。"
                                    #
                                    #   原来是走 select_skill()（蘑菇体分区选技能）——
                                    #   但那个逻辑是"选当前最该用的**技能**"，
                                    #   它几乎永远选 1（主攻），于是大招永远长按 1。
                                    #
                                    #   而大招和技能不是一回事：**长按同一个键放的是
                                    #   那个角色的专属大招，每个角色**各自**有冷却。**
                                    #   果蝇从画面上看不出哪个角色的冷却好了，所以
                                    #   最实在的办法就是**轮换** —— 每次大招换下一个键，
                                    #   1 在冷却时就轮到 234 了。
                                    #
                                    #   轮换一圈 4 次 × 20 秒冷却 = 80 秒，
                                    #   比一般大招冷却长，转到谁基本都能放出来。
                                    # ★ 可配：--key-ult "1,2,3,4"
                                    _ULT_KEYS = tuple(
                                        k.strip() for k in args.key_ult.split(",")
                                        if k.strip()) or ("1",)
                                    _uk = _ULT_KEYS[fod["n_ult"] % len(_ULT_KEYS)]
                                    ex.combat_hold_down(_uk)
                                    pressed_keys.append(f"ULT{_uk}")
                                    fod["ult_key"] = _uk
                                    fod["ult_holding"] = True
                                    fod["ult_hold_until"] = ts + args.ult_hold
                                    fod["last_ult"] = ts
                                    fod["n_ult"] += 1
                                    print(f"  [{ts:6.1f}s] ★★ 大招 → 长按 {_uk} "
                                          f"{args.ult_hold}s  （h={fod['hg'].h:.2f} "
                                          f"血={_hp if not isinstance(_hp,float) else round(_hp,2)} "
                                          f"战线={_age:.1f}s 第 {fod['n_ult']} 次）")
                            # 到点松开大招键（跨拍，和普攻长按同一个理由）
                            if fod["ult_holding"] and ts >= fod["ult_hold_until"]:
                                # 松开不记：按下去那一拍已经记过 ULTx 了，记两次会看着像按了两次
                                ex.combat_hold_up(fod["ult_key"])
                                fod["ult_holding"] = False

                            # ★ 连携技（E）：出现提示就按。
                            #   用户要求："E 键没有冷却时间，有就可以放"。
                            #   所以冷却默认 0 —— 提示还在就每拍都按（2.5Hz）。
                            #   原来默认 1.5 秒，会出现"提示亮着但按不了"的空窗。
                            if (m_combo
                                    and fod["hg"].h >= args.combo_hunger
                                    and ts - fod["last_combo"] >= args.combo_cool):
                                pressed_keys.append(str(args.combo_key).upper())
                                ex.combat_press(args.combo_key)
                                fod["last_combo"] = ts
                                fod["n_combo"] += 1
                                # ★ 诊断：连着按很多次还不生效 → 大声报出来。
                                #   连携技**没有冷却**（用户要求"有就可以放"），
                                #   所以按错键时的症状就是"每拍都按、永远不生效"。
                                #   实测这个 bug 出现过：combo_key 被滚轮改成 q，
                                #   于是 7 秒里连按 11 次、日志刷屏，
                                #   但没人看得出问题在哪。
                                #   做法：连续按满 4 次且间隔都很短时，警告一次。
                                _cnow = fod.setdefault("combo_streak", {"n": 0, "t": 0.0})
                                if ts - _cnow["t"] < 2.0:
                                    _cnow["n"] += 1
                                else:
                                    _cnow["n"] = 1
                                _cnow["t"] = ts
                                if _cnow["n"] == 4:
                                    print(f"  ⚠ 连携技连按 4 次（键 "
                                          f"「{args.combo_key}」）提示仍不消失 —— "
                                          f"多半是**键位设错了**。"
                                          f"游戏里的连携技提示写的是它要求的键，"
                                          f"如果 combo_key 不是那个键，按了也没用。"
                                          f"面板最上面那行「键位」能看到当前值。",
                                          flush=True)
                                print(f"  [{ts:6.1f}s] ★ 连携技 → 按 "
                                      f"{args.combo_key.upper()}"
                                      f"  （饥饿 {fod['hg'].h:.2f}，"
                                      f"第 {fod['n_combo']} 次）")
                    # 指令列表一律英文（用户要求）
                    action_txt += ("  EAT-ATK" if f_atk else "  HUNGRY")

                    # ★ 奖励塑形（§九）：打到就给**小额**奖赏。
                    #   条件 = 出手 且 红环亮着 —— 后者是"真的在跟敌人交手"的代理，
                    #   不是对着空气挥刀。不给饱足：只有打死才降饥饿，
                    #   这样"杀死"的期望才明显高过"磨"（否则它会停在"打到"这一层）。
                    if (f_atk and m_on and args.hit_reward > 0
                            and ts - fod["last_hit"] >= args.hit_cooldown):
                        if args.ab_dopamine:
                            fod["mb"].deliver(+args.hit_reward * fod["hg"].da_gain())
                        fod["last_hit"] = ts
                        fod["n_hit"] += 1

                    # ---- 全自动奖惩：自己看画面认打死/挨打 ----
                    # 挨打 ← 血条相对下降 或 受击红晕
                    # 打死 ← 红环下降沿（这一段出过手、且战线够长）
                    if outcome is not None:
                        from auto_outcome import HP_RECT as _HPR, VIG_L as _VL, VIG_R as _VR
                        # ★ 血条和左右红晕也从**同一帧**切 —— 这样"血量掉了"
                        #   和"红晕亮了"是同一时刻的判定，不会自相矛盾。
                        _hp = lf.at(*_HPR)
                        _vl = lf.at(*_VL)
                        _vr = lf.at(*_VR)
                        if _hp is None:      # 整帧没抓到时退回单抓
                            _hp = grab(x + _HPR[0], y + _HPR[1], _HPR[2], _HPR[3])
                        if _vl is None:
                            _vl = grab(x + _VL[0], y + _VL[1], _VL[2], _VL[3])
                        if _vr is None:
                            _vr = grab(x + _VR[0], y + _VR[1], _VR[2], _VR[3])
                        _o = outcome.step(ts, bool(m_on), bool(f_atk),
                                          hp_img=_hp, vig_left=_vl, vig_right=_vr)
                        fod["auto_hp"] = _o["hp"]
                        fod["auto_vig"] = _o["vig"]
                        if _o["reward"] > 0:
                            _g = fod["hg"].da_gain()
                            if args.ab_dopamine:
                                fod["mb"].deliver(+_o["reward"] * _g)
                            _eaten = fod["hg"].ate(1.0)
                            fod["n_reward"] += 1
                            print(f"  [{ts:6.1f}s] ★ 自动判定**打死** → 奖赏 "
                                  f"×{_g:.2f}  饥饿 {fod['hg'].h + _eaten:.2f}→"
                                  f"{fod['hg'].h:.2f}  （{_o['why']}）")
                        if _o["punish"] > 0:
                            _g = fod["hg"].da_gain()
                            if args.ab_dopamine:
                                fod["mb"].deliver(-_o["punish"] * _g)
                            fod["n_punish"] += 1
                            print(f"  [{ts:6.1f}s] ✕ 自动判定**挨打** → 惩罚 "
                                  f"×{_g:.2f}  （{_o['why']}）")

                    # （O/P 手动键已去掉 —— 全自动识别已经跑通，手按那套不再需要。
                    #   要临时补一刀就加 --auto-outcome 之外再想办法，别再挂手动键，
                    #   否则面板上要多两行说明，还会让人以为程序没认出来。）

                    # ★ h 是**状态**不是指令 —— 已经在「果蝇当前状态」那块显示，
                    #   不要再往指令列表里塞（用户要求：状态另外写一个位置）。
                    #   pass

                    # 定期快照（§十：别每帧写；30 秒一次 + 关窗前补一次）
                    if ts - fod["last_save"] > args.forage_save_sec:
                        try:
                            _n = fod["mb"].save(args.forage_patch,
                                                extra=dict(t=round(ts, 1),
                                                           rewards=fod["n_reward"],
                                                           punish=fod["n_punish"]))
                            fod["last_save"] = ts
                            print(f"  [{ts:6.1f}s] 补丁快照 {_n:,} 条边  "
                                  f"漂移 {fod['mb'].drift:.1f}")
                        except Exception as _e:
                            print(f"  ⚠ 补丁快照失败：{type(_e).__name__}: {_e}")

                if not moved:
                    heading = (heading + turn_dx * DEG_PER_MOUSE_PX) % 360.0

                rows.append(dict(tick=tick, t=round(ts, 2),
                                 drive=float(drive[rm._pr_idx].mean()),
                                 margin=float(o["margin"]), direction=direction,
                                 jump=bool(plan.jump), sprint=bool(plan.sprint),
                                 heading=round(heading, 1),
                                 speed=round(float(od["speed"]), 2),
                                 sharp=round(float(od["sharp"]), 2),
                                 stuck=stuck_now, esc=esc,
                                 x=round(odo.x, 1), y=round(odo.y, 1),
                                 move=plan.move, turn=turn,
                                 raw_jump=rawj, raw_forward=rawf,
                                 spikes=spikes, r_mean=round(rmean, 5),
                                 r_std=round(rstd, 5), r_max=round(rmax, 4),
                                 eye_contrast=round(float(contrast), 4)
                                 if contrast == contrast else "",
                                 eye_used=("minimap" if using_minimap else
                                           ("main" if eye is not None else "minimap")),
                                 z_thrust=round(pm["z"]["thrust"], 3) if pm else "",
                                 z_strafe=round(pm["z"]["strafe"], 3) if pm else "",
                                 z_turn=round(pm["z"]["turn"], 3) if pm else "",
                                 # ---- 战斗（交接文档 §八：归因要能查）----
                                 combat=bool(combat_on),
                                 cm_atk=bool(cm["attack"]), cm_skl=bool(cm["skill"]),
                                 cm_z_a=round(cm["z_a"], 3), cm_z_s=round(cm["z_s"], 3),
                                 cm_der_a=round(cm["der_a"], 5),
                                 cm_lvl_a=round(cm["lvl_a"], 5),
                                 cm_lvl_s=round(cm["lvl_s"], 5),
                                 cm_hesit=round(cm["hesit"], 3),
                                 cm_src_a=cm["src_a"], cm_src_s=cm["src_s"],
                                 # ---- 觅食闭环 ----
                                 f_odor=bool(forager['odor']) if forager else False,
                                 f_hunger=round(forager['hg'].h, 4) if forager else '',
                        # ★ 任务完成后的吃饱倒计时（秒）。>0 说明正在不涨。
                        f_hold=round(forager['hg'].hold_left, 2) if forager else 0.0,
                        f_nmission=(forager.get('n_mission', 0) if forager else 0),
                                 f_urge=round(forager['urge'], 4) if forager else '',
                                 f_atk=bool(forager['attack']) if forager else False,
                                 f_mbon=round(forager['mbon_val'], 4) if forager else '',
                                 f_drift=round(forager['mb'].drift, 2) if forager else '',
                                 f_hit=(forager['n_hit'] if forager else 0),
                                 f_skill=(forager['n_skill'] if forager else 0),
                        f_auto=(outcome is not None),
                        f_hp=(forager.get('auto_hp') if forager else None),
                        f_vig=(forager.get('auto_vig') if forager else None),
                                 f_reward=(forager['n_reward'] if forager else 0),
                                 f_punish=(forager['n_punish'] if forager else 0),
                                 cm_mark=bool(m_on),
                                 cm_evac=bool(m_evac) if emark is not None else False,
                                 cm_evac_corr=(round(emark.corr, 3) if emark else ''),
                        cm_red=(bool(m_red) if cmark is not None else False),
                                 cm_urge=round(cdec.urge, 2),
                                 cm_red_px=(cmark.red_px if cmark else 0),
                                 action=("FLEE " if esc else "") + action_txt))
                if using_minimap and not was_minimap:
                    switch_count += 1
                    why = []
                    if contrast == contrast and contrast < args.eye_contrast:
                        why.append(f"对比度 {contrast:.3f}<{args.eye_contrast}")
                    if args.eye_switch_stuck and stuck_now:
                        why.append("推不动")
                    print(f"  {tick:>4d} {ts:>6.1f}  ★ 切回小地图口径（{' / '.join(why) or '?'}）")
                elif was_minimap and not using_minimap:
                    switch_count += 1
                    print(f"  {tick:>4d} {ts:>6.1f}  ★ 切回主画面口径"
                          f"（对比度 {contrast:.3f}，且能动）")
                was_minimap = using_minimap
                if tick % 2 == 0 or plan.jump or plan.sprint or esc:
                    print(f"  {tick:>4d} {ts:>6.1f} {drive[rm._pr_idx].mean():>9.5f} "
                          f"{o['margin']:>+10.5f} {direction:>4s} "
                          f"{'跳' if plan.jump else '  ':>3s} "
                          f"{'冲' if plan.sprint else '  ':>3s} "
                          f"{heading:>5.0f}° {od['speed']:>6.2f} "
                          f"{'卡' if od['stuck'] else '  ':>3s}  "
                          f"{'FLEE ' if esc else ''}{action_txt}")

                if args.snap_every and args.snapdir and tick % args.snap_every == 0:
                    Image.fromarray(grab(x, y, w, h)).save(
                        os.path.join(args.snapdir, f"f{tick:05d}.jpg"), quality=88)

                # 导出「果蝇看到了什么」—— 第 3 拍和最后一拍各存一次
                if (args.eye_dump and eye is not None
                        and tick in (2, total - 1) and eye.last_small is not None):
                    os.makedirs(args.eye_dump, exist_ok=True)
                    tag = "early" if tick == 2 else "late"
                    Image.fromarray(eye.last_small.astype(np.uint8)).resize(
                        (eye.last_small.shape[1] * 2, eye.last_small.shape[0] * 2),
                        Image.NEAREST).save(
                        os.path.join(args.eye_dump, f"eye_{tag}_input.png"))
                    dv = (np.clip(eye.last_drive * 2.2, 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(dv).resize(
                        (dv.shape[1] * 2, dv.shape[0] * 2), Image.NEAREST).save(
                        os.path.join(args.eye_dump, f"eye_{tag}_drive.png"))
                    mk = (eye.last_mask * 255).astype(np.uint8)
                    Image.fromarray(mk).resize(
                        (mk.shape[1] * 2, mk.shape[0] * 2), Image.NEAREST).save(
                        os.path.join(args.eye_dump, f"eye_{tag}_mask.png"))

                # ---- 实时可视化叠加 ----
                if ov is not None and tick % max(1, args.overlay_every) == 0:
                    if moved:
                        traj.append((odo.x, odo.y))
                        if len(traj) > 4000:
                            traj = traj[::2]
                    if direction == "左":
                        counts["left"] += 1
                    elif direction == "右":
                        counts["right"] += 1
                    if plan.jump:
                        counts["jump"] += 1
                    if plan.sprint:
                        counts["sprint"] += 1
                    if esc:
                        counts["esc"] += 1
                    if stuck_now:
                        counts["stuck"] += 1

                    # ★ 这一拍**真正按下去的键**，给实时指令列表显示用。
                    #   合起来三路：移动键（WASD）+ 战斗键（普攻/技能/连携/大招）
                    #   + 状态键（SPACE/SHIFT）。全部英文大写（用户要求）。
                    #
                    #   ⚠ 这里踩过一个坑：插这一段的那次补丁被前面的
                    #     `'_tick_keys' not in s` 守卫挡掉了 —— 因为**同一次补丁的
                    #     前一步**刚在下面加了 `keys=list(_tick_keys)`，
                    #     守卫误判成"已经有了"，于是定义没插、用它的地方倒是插了，
                    #     启动直接 NameError。**多步补丁的守卫要看整个补丁的最终状态，
                    #     不能一步一看。**
                    _tick_keys = []
                    if pm:
                        for _k, _on in (("W", pm.get("key_w")), ("A", pm.get("key_a")),
                                        ("S", pm.get("key_s")), ("D", pm.get("key_d"))):
                            if _on:
                                _tick_keys.append(_k)
                    _tick_keys += pressed_keys
                    if plan.jump:
                        _tick_keys.append("SPACE")
                    if plan.sprint:
                        _tick_keys.append("SHIFT")

                    logbuf.append(dict(
                        t=round(ts, 1),
                        dir=(">" if turn > 0 else ("<" if turn < 0 else "-")),
                        # 群体解码口径下这一列显示**转身轴的 z**（旧口径才显示 margin）
                        margin=(round(pm["z"]["turn"], 3) if pm
                                else round(float(o["margin"]), 4)),
                        pop=bool(pm),
                        dx=int(turn_dx),
                        # ★ 指令列表全部用英文按键名（用户要求）。
                        #   中文只留在**状态区**，不混进指令流。
                        action=("WALL-ESC turn" if esc else action_txt),
                        jump=bool(plan.jump), sprint=bool(plan.sprint), esc=bool(esc),
                        keys=list(_tick_keys)))
                    if len(logbuf) > 60:
                        del logbuf[:-60]

                    panel.bv.adapt_range(r)

                    # ---- 输入信号：复眼 5 扇区的驱动 ----
                    pv = drive[rm._pr_idx]
                    svals = [float(pv[m].mean()) if m.any() else 0.0
                             for m in sec_masks]
                    sector_wins[int(np.argmax(svals))] += 1
                    in_sectors = list(zip(sec_names, svals))

                    # ---- 输出信号：下行神经元通道（刻度用滑动窗口分位数，
                    #      不能用标定文件里的 —— 那个会因为分布漂移而失真） ----
                    out_chans = []
                    for key, nm, use, thk in (
                            ("forward", "前进", True, "sprint_hi"),
                            ("turn", "转向", True, "turn"),
                            ("jump", "跳跃", True, "jump")):
                        if key not in amap.calib.get("channels", {}):
                            continue
                        q50 = amap.hist_q(key, 50)
                        q90 = amap.hist_q(key, 90)
                        if q50 is None or q90 is None:
                            continue
                        if thk == "jump":
                            thr = float(amap.last_thr.get("jump", q90))
                        elif thk == "sprint_hi":
                            thr = float(amap.last_thr.get("sprint_hi", q90))
                        else:
                            thr = q90                     # 转向没有触发阈值，用 q90 当量程
                        v = float(acts["raw"].get(key, 0.0))
                        out_chans.append(dict(key=key, name=nm, v=v, q50=q50,
                                              thr=thr, over=bool(v > thr)))

                    # ★★ 叠加面板的更新**绝不能弄死果蝇**。
                    #   实测：按 End 回到启动界面、再点启动之后，第二轮的
                    #   第一拍 ov.update() 抛异常 → 整个进程崩掉 →
                    #   用户看到的就是「点了启动没反应」（窗口一闪没了）。
                    #   这只果蝇的正事是开车，画不出面板不该让它停摆。
                    try:
                        ov.update(panel.render(dict(
                            r=r, heading=heading, direction=direction,
                            margin=float(o["margin"]), speed=float(od["speed"]),
                            drive=float(drive[rm._pr_idx].mean()),
                            jump=bool(plan.jump), sprint=bool(plan.sprint), esc=esc,
                            stuck=stuck_now, move=plan.move > 0,
                            turn_dx=int(turn_dx), action=action_txt,
                            tick=tick, counts=counts, log=logbuf,
                            running=True, exec="Interception",
                            z_thrust=(pm["z"]["thrust"] if pm else None),
                            z_strafe=(pm["z"]["strafe"] if pm else None),
                            z_turn=(pm["z"]["turn"] if pm else None),
                            key_w=(pm["key_w"] if pm else None),
                            key_s=(pm["key_s"] if pm else None),
                            key_a=(pm["key_a"] if pm else None),
                            key_d=(pm["key_d"] if pm else None),
                            combat_on=bool(combat_on),
                            cm_atk=bool(cm["attack"]), cm_skl=bool(cm["skill"]),
                            cm_z_a=cm["z_a"], cm_z_s=cm["z_s"],
                            cm_der_a=cm["der_a"], cm_der_s=cm["der_s"],
                            cm_hesit=cm["hesit"],
                            cm_src_a=cm["src_a"], cm_src_s=cm["src_s"],
                            cm_cnt=cm["cnt"], cm_n_atk=cm["n_atk"], cm_n_skl=cm["n_skl"],
                            # ---- 觅食闭环 ----
                            forage_on=(forager is not None),
                            f_odor=(forager['odor'] if forager else False),
                            f_evac=(bool(m_evac) if emark is not None else False),
                            f_hunger=(forager['hg'].h if forager else 0.0),
                            f_urge=(forager['urge'] if forager else 0.0),
                            f_thr=args.forage_thr,
                            f_attack=(forager['attack'] if forager else False),
                            f_reward=(forager['n_reward'] if forager else 0),
                            f_punish=(forager['n_punish'] if forager else 0),
                            f_drift=(forager['mb'].drift if forager else 0.0),
                            f_mbon=(forager['mbon_val'] if forager else 0.0),
                            f_ch=(forager['ch'] if forager else ''),
                            cm_mark=bool(m_on), cm_urge=cdec.urge,
                            cm_red_px=(cmark.red_px if cmark else 0),
                            spikes=spikes, spikes_total=spikes_total,
                            steps_total=steps_total, steps=args.steps,
                            integral=float(r.sum()),
                            r_mean=rmean, r_std=rstd, r_max=rmax,
                            in_sectors=in_sectors, sector_wins=sector_wins,
                            out_chans=out_chans,
                            dist=float(np.hypot(odo.x, odo.y)),
                            yaw=0.0, pitch=0.30)))
                    except Exception as _ov_e:
                        _ov_err[0] += 1
                        if _ov_err[0] <= 3:
                            print(f"  ⚠ 叠加面板更新失败（第 {_ov_err[0]} 次，"
                                  f"不影响操控）：{type(_ov_e).__name__}: {_ov_e}",
                                  flush=True)
                    ov.pump()

                nxt = t0 + (tick + 1) / args.hz
                sl = nxt - time.time()
                if sl > 0:
                    time.sleep(sl)
            # ---- 一轮结束 ----
            # 面板模式下**不退出**，回到"等启动"；非面板模式（纯命令行）保持老行为。
            print(f"\n  [轮次] 循环退出（跑完 {tick + 1}/{total} 拍）")
            if not args.panel or CTL["stop"]:
                print(f"  [轮次] → 真退出：panel={args.panel} "
                      f"stop={CTL['stop']}")
                break
            CTL["running"] = False
            if ex is not None:
                try:
                    ex.release_all()
                    print("  [轮次] 已松开所有按键")
                except Exception as _e:
                    print(f"  [轮次] 松键出错（不影响）：{_e}")
            if ov is not None:
                ov.set_visible(False)       # 同上：别留一块冻住的面板
                _ov_hidden[0] = True
                print("  [轮次] 叠加面板已隐藏")
            print("\n" + "=" * 62)
            print("  ■ 本轮结束，已回到启动界面。")
            print("    在窗口里点「启动」再来一轮。")
            print("    想彻底退出：点窗口上的「停止」，或者关掉窗口。")
            print("=" * 62, flush=True)
            _t_idle = time.time()
            _n_idle = 0
            while not CTL["running"] and not CTL["stop"]:
                # ★ 空闲也要续上：刷新游戏检测缓存 + 扫标记
                #   （少了这行，界面上的「位置」会停在最后一拍，
                #     而且启动按钮的可点状态可能一直是旧的）
                _poll_markers_idle()
                time.sleep(0.25)
                _n_idle += 1
                if _n_idle % 8 == 0:            # 每 2 秒报一次，别刷屏
                    print(f"  [等启动] {time.time() - _t_idle:5.0f}s  "
                          f"running={CTL['running']} stop={CTL['stop']}",
                          flush=True)
                if time.time() - _t_idle > 3600:
                    print("  [等启动] 超时 1 小时，退出")
                    CTL["stop"] = True
                    break
            print(f"  [等启动] 结束：running={CTL['running']} "
                  f"stop={CTL['stop']}  等了 {time.time() - _t_idle:.1f}s",
                  flush=True)
            if CTL["stop"]:
                print("  [轮次] stop=True → 真退出")
                break
            print("  [轮次] → 开始新一轮\n", flush=True)
        except KeyboardInterrupt:
            print("\n  中断")
            CTL["stop"] = True
        finally:
            # ★ 退出前**一定要松开长按的键** —— 长按是跨拍的，程序被 End
            #   打断时可能正按着；不松的话游戏里那个键会一直摁住。
            if not args.dry and forager is not None and forager.get("ult_holding"):
                try:
                    import win_click as _wc2
                    _k = str(forager.get("ult_key", "1"))
                    if _k.startswith("mouse_"):
                        _wc2.up(_k.split("_", 1)[1])
                    else:
                        ic.key_up(_k, kb)          # ★ 键盘走 Interception
                    print(f"  大招键 {_k} 已松开")
                except Exception as _e:
                    print(f"  ⚠ 松大招失败：{_e}")
            if not args.dry and locals().get("atk_holding"):
                try:
                    import win_click as _wc
                    _wc.up(str(args.attack_key).split("_", 1)[1])
                    print("  普攻键已松开")
                except Exception as _e:
                    print(f"  ⚠ 松普攻失败：{_e}")
            # ★ 关窗前补一次补丁（§十："别每帧写，定期快照 + 关窗前补一次"）。
            #   不做的话，这一轮学到的全丢，下次开窗就是一只新果蝇。
            if forager is not None:
                try:
                    _n = forager["mb"].save(args.forage_patch,
                                            extra=dict(final=True,
                                                       rewards=forager["n_reward"],
                                                       punish=forager["n_punish"]))
                    print(f"  补丁已存：{_n:,} 条边 · 漂移 "
                          f"{forager['mb'].drift:.1f} · 奖赏 {forager['n_reward']} "
                          f"惩罚 {forager['n_punish']} → {args.forage_patch}")
                except Exception as _e:
                    print(f"  ⚠ 关窗前存补丁失败：{type(_e).__name__}: {_e}")
            if eye_th is not None:
                eye_th.stop()
                if eye_th.fps:
                    print(f"  眼睛线程收尾：实测 {eye_th.fps:.1f} Hz / "
                          f"{eye_th.frame_id} 帧"
                          + (f" · ⚠ {eye_th.error}" if eye_th.error else ""))
            if ex is not None:
                ex.release_all()
            if ic is not None:
                ic.close()
            if ov is not None:
                ov.close()
            if rec is not None:
                # 先礼后兵：给 ffmpeg 发 'q' 让它把文件尾写完整，超时才强杀
                try:
                    rec.stdin.write(b"q")
                    rec.stdin.flush()
                    rec.wait(timeout=20)
                    print(f"  录屏已收尾 → {args.record}")
                except Exception:
                    rec.terminate()
                    print(f"  录屏被强杀（文件可能少了尾巴）→ {args.record}")

        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False, encoding="utf-8")
        print()
        print("=" * 78)
        print(f"  {len(df)} 拍 / {time.time()-t0:.0f} 秒")
        print(f"  方向分布：{dict(Counter(df['direction']))}")
        print(f"  跳跃 {int(df['jump'].sum())} 次 · 冲刺 {int(df['sprint'].sum())} 次 "
              f"· 逃逸 {int(df['esc'].sum())} 次")
        print(f"  感光驱动均值：{df['drive'].mean():.5f}")
        mv = df[df['speed'] >= 1.5]
        print(f"  行进拍占比 {len(mv)}/{len(df)} ({100*len(mv)/max(len(df),1):.0f}%) "
              f"· 速度中位 {df['speed'].median():.2f} px/s")
        if args.eye_auto:
            nm = int((df["eye_used"] == "minimap").sum())
            print(f"  眼睛切换：共 {switch_count} 次 · 走小地图口径的拍 {nm}/{len(df)}"
                  f" ({100*nm/max(len(df),1):.0f}%)")
            ec = pd.to_numeric(df["eye_contrast"], errors="coerce").dropna()
            if len(ec):
                print(f"  主画面对比度：中位 {ec.median():.3f} · "
                      f"5分位 {ec.quantile(0.05):.3f} · 最低 {ec.min():.3f}")
        print(f"  轨迹累计 ({odo.x:+.1f}, {odo.y:+.1f}) 地图像素 "
              f"· 净位移 {np.hypot(odo.x, odo.y):.1f} px")
        print(f"  日志 → {args.out}")


if __name__ == "__main__":
    main()
