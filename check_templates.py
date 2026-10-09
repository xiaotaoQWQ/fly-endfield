# -*- coding: utf-8 -*-
"""自检：模板重采样后，必须真的能用。

★ 这个自检是有来历的：
  combo_e 的尺度曾被写成 1280×720，重采样把它从 64×58 放大成 128×116。
  而 ComboMarker.step() 内部会把画面降采样 2 倍再匹配，所以它要的就是
  64×58 —— 放大后的模板比降采样后的搜索区还大，撞上 step() 里
  `if g.shape[0] < th: return False` 的防线，**永远返回 False**。
  表现是"果蝇不会按 E"，而且**没有任何报错**。

  这类"尺寸对不上就静默失效"的 bug 只能靠自检抓。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_profile import Profile, TEMPLATES, REGIONS  # noqa: E402

FAIL = 0


def check(cond, msg):
    global FAIL
    print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL += 1


print("=" * 64)
print("  模板重采样自检")
print("=" * 64)

# ---- ① 参考分辨率下，重采样后的模板必须和文件里的**一模一样** ----
p = Profile(0, 80, 2560, 1440, measured=True, verbose=False)
for name in TEMPLATES:
    f = os.path.join("templates", name + ".npy")
    if not os.path.exists(f):
        check(False, f"{name}: 模板文件不存在")
        continue
    t0 = np.load(f)
    if t0.ndim == 3:
        t0 = t0.astype(np.float32).mean(axis=2)
    t1 = p.template(name)
    check(t0.shape == t1.shape,
          f"{name}: 2560×1440 下重采样后 {t1.shape[1]}×{t1.shape[0]} "
          f"应等于文件里的 {t0.shape[1]}×{t0.shape[0]}")

# ---- ② combo_e 必须能塞进降采样后的搜索区 ----
print()
print("  ComboMarker 的尺寸约束：")
nx, ny, nw, nh = REGIONS["combo"]
rw, rh = 2560, 1440
region_w, region_h = int(nw * rw), int(nh * rh)
g_w, g_h = region_w // 2, region_h // 2          # step() 里 a[::2, ::2]
t = p.template("combo_e")
check(g_h >= t.shape[0] and g_w >= t.shape[1],
      f"搜索区 {region_w}×{region_h} "
      f"必须 ≥ 模板 {t.shape[1]}×{t.shape[0]}")
print(f"       （不满足的话 step() 会 return False —— 静默失效，不报错）")

# ---- ③ 其他分辨率下也不能倒挂 ----
print()
print("  其他分辨率：")
for w, h in ((1920, 1080), (2560, 1080), (3840, 2160), (1600, 900)):
    pp = Profile(0, 0, w, h, measured=True, verbose=False)
    rc = pp.rect("combo")
    gw, gh = rc[2] // 2, rc[3] // 2
    tt = pp.template("combo_e")
    ok = gh >= tt.shape[0] and gw >= tt.shape[1]
    print(f"    {w}×{h}: 区域 {rc[2]}×{rc[3]} → 降采样 {gw}×{gh}   "
          f"模板 {tt.shape[1]}×{tt.shape[0]}   {'OK' if ok else 'FAIL'}")

print()
print("=" * 64)
print(f"  {'全部通过 ✓' if FAIL == 0 else f'{FAIL} 项失败 ✗'}")
print("=" * 64)
sys.exit(1 if FAIL else 0)
