# -*- coding: utf-8 -*-
"""离线对比：ONNX 的 combo_active vs 旧的 E 徽标模板，在 428 帧上逐帧比。

这是接入 ONNX 后最该做的一件事 ——
只看"ONNX 能不能跑"没意义，要看**它和旧方案在哪分歧**，
然后判断哪个才是对的。

方法：
  · 旧方案：`combat.ComboMarker` + `templates/combo_e.npy`，
    搜索区用 `screen_profile.REGIONS["combo"]`（内容坐标 → 折半到 1280）
  · 新方案：`onnx_detect.OnnxDetect` 的 `combo_active`

输出：
  · 两者一致的帧数
  · 只有 ONNX 说有的帧
  · 只有模板说有的帧
  · 两者都说有的帧
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from combat import ComboMarker                    # noqa: E402
from onnx_detect import OnnxDetect                 # noqa: E402
from screen_profile import Profile                 # noqa: E402

REC = os.path.join(HERE, "..", "_handoff", "rec")


def newest_rec():
    ds = sorted(glob.glob(os.path.join(REC, "*")), key=os.path.getmtime, reverse=True)
    for d in ds:
        if glob.glob(os.path.join(d, "full_*.jpg")):
            return d
    return None


def main():
    d = newest_rec()
    if not d:
        print("找不到录制")
        return 1
    fulls = sorted(glob.glob(os.path.join(d, "full_*.jpg")))
    print(f"录制 {os.path.basename(d)}  {len(fulls)} 帧")
    print()

    # 内容区按 2560×1440 标定，录制是 1280×720 → 折半
    prof = Profile(0, 0, 2560, 1440, measured=True, verbose=False)
    rx, ry, rw, rh = prof.rect("combo")
    rx, ry, rw, rh = rx // 2, ry // 2, rw // 2, rh // 2
    print(f"旧模板搜索区（1280 尺度）({rx},{ry}) {rw}×{rh}")

    kmark = ComboMarker((rx, ry, rw, rh), verbose=False)
    kmark.corr_min = 0.50
    # ★ 模板本体是 2560 尺度的（30x28），而这里喂的是 1280 裁图
    #   不缩的话模板比实际大一倍 → 相关值全部偏低 ✗
    import numpy as _np
    from PIL import Image as _Im
    _t = kmark.tpl
    kmark.tpl = _np.asarray(
        _Im.fromarray(_t.astype('uint8')).resize(
            (max(1, _t.shape[1] // 2), max(1, _t.shape[0] // 2)),
            _Im.LANCZOS)).astype('float32')
    det = OnnxDetect(conf=0.35, verbose=False)
    if not det.ok:
        print("ONNX 不可用")
        return 1
    print(f"模板 {kmark.tpl.shape[1]}×{kmark.tpl.shape[0]} · 阈值 {kmark.corr_min}")
    print()

    both, only_onnx, only_tpl, neither = [], [], [], []
    rows = []
    for p in fulls:
        n = int(os.path.basename(p)[5:9])
        im = Image.open(p).convert("RGB")
        a = np.asarray(im)

        # 旧模板
        sub = a[ry:ry + rh, rx:rx + rw]
        tpl_hit = bool(kmark.step(sub))
        tpl_corr = kmark.corr

        # ONNX
        st = det.step(a)
        onnx_hit = bool(st["combo_active"])
        rows.append((n, tpl_hit, onnx_hit, tpl_corr, st["n_det"]))

        if tpl_hit and onnx_hit:
            both.append(n)
        elif onnx_hit:
            only_onnx.append(n)
        elif tpl_hit:
            only_tpl.append(n)
        else:
            neither.append(n)

    print("=== 对比结果 ===")
    print(f"  两者都说有      {len(both):4d} 帧  {both[:20]}" + ("…" if len(both) > 20 else ""))
    print(f"  只有 ONNX 说有  {len(only_onnx):4d} 帧  {only_onnx[:20]}" + ("…" if len(only_onnx) > 20 else ""))
    print(f"  只有模板说有    {len(only_tpl):4d} 帧  {only_tpl[:20]}" + ("…" if len(only_tpl) > 20 else ""))
    print(f"  都说没有        {len(neither):4d} 帧")
    print()
    print(f"  一致率 {(len(both)+len(neither))/len(fulls)*100:.1f}%")

    # 模板相关值分布
    tpl_corrs = np.array([r[3] for r in rows])
    print()
    print(f"  === 旧模板的相关值分布 ===")
    print(f"    中位 {np.median(tpl_corrs):+.3f}")
    print(f"    p90  {np.percentile(tpl_corrs,90):+.3f}")
    print(f"    最大 {tpl_corrs.max():+.3f}")
    print(f"    ≥0.5 的帧 {int((tpl_corrs>=0.5).sum())} / {len(fulls)}")

    # 只看"有提示"的帧，模板的相关值是多少
    if both or only_tpl:
        hit_idx = [i for i, r in enumerate(rows) if r[1]]
        print()
        print("  === 模板说「有」的帧，ONNX 怎么说 ===")
        for i in hit_idx[:15]:
            n, t, o, c, nd = rows[i]
            print(f"    {n:04d}: 模板✓ corr={c:+.3f}  ONNX={'✓' if o else '✗'}  ({nd}框)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
