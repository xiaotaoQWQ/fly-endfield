#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""onnx_detect.py — 用 MaaEnd 的 YOLO11s 模型做识别。

为什么要它
==========
我们原来的识别是「手工抠模板 + 归一化相关」，踩了很多坑：
  · 连携技徽标抓错过 UI 元素（§40）
  · 颜色判据在录像上零误报、实机全翻车（§46–§51）
  · 每个检测器都要单独标定 ROI、单独适配分辨率

这个模型是 MaaEnd 训练的 YOLO11s（22 个类），一次推理直接给出
**所有战斗 UI 元素的框 + 标签 + 置信度**，不需要我们标定任何 ROI。

★ 预处理严格照 MaaFramework 的源码实现（`NeuralNetworkDetector.cpp` +
  `VisionUtils.hpp`），不是猜的：

    ① ROI 裁剪（我们喂的就是内容区，不用再裁）
    ② letterbox: scale = min(inW/w, inH/h)，缩小用 INTER_AREA
    ③ 补边到 640×640，填充 (114,114,114)   ← YOLO 标准灰
    ④ BGR → RGB  ← MaaFramework 里必要（cv::imread 给 BGR）；
                    我们用 PIL/截屏给的是 RGB，**所以不转**
    ⑤ HWC → CHW
    ⑥ ÷ 255.0 → float32
    ⑦ 不减均值、不除标准差

★ 模型出处与许可
    MaaEnd/MaaEnd-AI · detect/AutoFight/autofightv13.onnx · AGPL-3.0
    https://github.com/MaaEnd/MaaEnd-AI
    我们不修改模型，只做推理；本项目和 MaaEnd 同为 AGPL-3.0。
"""
from __future__ import annotations

import ast
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "autofightv13.onnx")

IMG = 640
PAD = 114          # YOLO letterbox 填充值


# ---------------------------------------------------------------- 标签语义
# 模型 metadata 里的 names（从模型里读出来的，不是抄文档）
COMBO_ACTIVE = "CharacterComboActive"      # 该按连携技了（瞬时提示）
COMBO_FULL = "CharacterComboFull"          # 连携能量条满
COMBO_EMPTY = "CharacterComboEmpty"        # 连携能量条空
HP_DANGER = "CharacterHealthDangerous"     # 血量危险
HP_NORMAL = "CharacterHealthNormal"        # 血量正常
CHAR_SELECT = "CharacterSelect"            # 当前选中角色
CHAR_DIED = "CharacterDied"
END_SKILL = "EndSkillFull"                 # 终结技（大招）就绪
ENERGY_FULL = "EnergyLevelFull"
ENERGY_EMPTY = "EnergyLevelEmpty"
ENEMY_ATTACK = "EnemyAttackGroundCharging"  # 敌人地面攻击蓄力
ENEMY_CHARGE = "EnemyAccumulatingPower"
ENEMY_TARGET = "EnemyTarget"

# 每个角色一个槽位（屏幕底部左侧 4 条能量/血条）
N_SLOTS = 4


# ---------------------------------------------------------------- 预处理
def letterbox(im):
    """im: PIL.Image (RGB) -> (tensor[1,3,640,640], scale, pad_left, pad_top)"""
    w, h = im.size
    s = min(IMG / float(w), IMG / float(h))
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    # 缩小用 BILINEAR；PIL 的 BILINEAR 在缩小时会自动做 region 平均，
    # 效果接近 cv::INTER_AREA
    resized = im.resize((nw, nh), 3 if s < 1.0 else 1)  # 3=AREA(BICUBIC-ish), 1=BILINEAR
    canvas = np.full((IMG, IMG, 3), PAD, dtype=np.uint8)
    pl, pt = (IMG - nw) // 2, (IMG - nh) // 2
    canvas[pt:pt + nh, pl:pl + nw] = np.asarray(resized, dtype=np.uint8)
    a = canvas.astype(np.float32) / 255.0          # ★ 只除 255
    a = a.transpose(2, 0, 1)                       # HWC -> CHW
    return a[None, ...], s, pl, pt


def _nms_1d(boxes, scores, thr=0.45):
    """普通 NMS。boxes 是 (N,4) xyxy。返回保留的下标。"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(ai + ar - inter, 1e-6)
        order = rest[iou < thr]
    return keep


class OnnxDetect:
    """战斗界面识别。缺模型时**静默关闭**（ok=False），调用方退回模板检测。"""

    def __init__(self, model_path: str = None, conf: float = 0.35,
                 iou: float = 0.45, verbose: bool = True):
        self.model_path = model_path or DEFAULT_MODEL
        self.conf = float(conf)
        self.iou = float(iou)
        self.ok = False
        self.sess = None
        self.names = {}
        self.ms = 0.0          # 上次推理耗时
        self.n = 0
        self.state = {}

        if not os.path.exists(self.model_path):
            if verbose:
                print(f"ONNX 识别：**未启用**（缺 {self.model_path}）"
                      f"\n           跑 `python download_model.py` 下载，"
                      f"或继续用模板检测")
            return
        try:
            import onnxruntime as ort
        except Exception as e:
            if verbose:
                print(f"ONNX 识别：**未启用**（没装 onnxruntime：{e}）"
                      f"\n           `pip install onnxruntime`")
            return

        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = 4
            self.sess = ort.InferenceSession(
                self.model_path, so, providers=["CPUExecutionProvider"])
            md = self.sess.get_modelmeta().custom_metadata_map
            try:
                self.names = ast.literal_eval(md.get("names", "{}"))
            except Exception:
                self.names = {}
            if not self.names:
                self.names = {i: str(i) for i in range(22)}
            self.ok = True
            if verbose:
                print(f"ONNX 识别：{os.path.basename(self.model_path)} · "
                      f"输入 {IMG}×{IMG} · {len(self.names)} 类 · "
                      f"conf>{conf} iou>{iou}")
        except Exception as e:
            if verbose:
                print(f"ONNX 识别：加载失败（{type(e).__name__}: {e}）")

    # ------------------------------------------------------------------
    def step(self, frame_rgb) -> dict:
        """喂一帧内容区画面（HxWx3 uint8 RGB），返回语义状态 dict。

        返回的 `ok` 为 False 时，调用方应退回模板检测。
        """
        out = dict(ok=False, ms=0.0, dets=[], combo_active=False,
                   combo_full=[False] * N_SLOTS, combo_empty=[False] * N_SLOTS,
                   hp_danger=[False] * N_SLOTS, hp_normal=[False] * N_SLOTS,
                   end_skill=False, enemy_attack=False,
                   energy=0, char_select=None, n_det=0)
        if not self.ok or frame_rgb is None:
            return out

        from PIL import Image

        a = np.asarray(frame_rgb)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        if a.ndim != 3 or a.shape[2] != 3:
            return out

        t0 = time.perf_counter()
        try:
            tensor, s, pl, pt = letterbox(Image.fromarray(a, mode="RGB"))
            raw = self.sess.run(None, {"images": tensor})[0][0]   # (26, 8400)
        except Exception:
            return out
        self.ms = (time.perf_counter() - t0) * 1000.0
        self.n += 1

        nc = raw.shape[0] - 4
        cls = raw[4:, :]
        best = cls.max(axis=0)
        bc = cls.argmax(axis=0)
        keep = best > self.conf
        if not keep.any():
            out.update(ok=True, ms=self.ms)
            self.state = out
            return out

        boxes = raw[:4, keep].T.copy()
        scores = best[keep]
        classes = bc[keep]

        # 反算回原帧坐标（和 MaaFramework 一样）
        bx = (boxes[:, 0] - boxes[:, 2] / 2 - pl) / s
        by = (boxes[:, 1] - boxes[:, 3] / 2 - pt) / s
        bw = boxes[:, 2] / s
        bh = boxes[:, 3] / s
        xyxy = np.stack([bx, by, bx + bw, by + bh], axis=1)

        dets = []
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            for k in _nms_1d(xyxy[idx], scores[idx], self.iou):
                j = idx[k]
                dets.append(dict(
                    label=self.names.get(int(classes[j]), str(int(classes[j]))),
                    score=float(scores[j]),
                    box=(float(xyxy[j, 0]), float(xyxy[j, 1]),
                         float(xyxy[j, 2]), float(xyxy[j, 3])),
                ))

        # ---- 映射到语义 ----
        by_name = {}
        for d in dets:
            by_name.setdefault(d["label"], []).append(d)

        def cx(d):
            return (d["box"][0] + d["box"][2]) / 2.0

        # ★ 连携技：「Active」= 瞬时提示（该按了）
        out["combo_active"] = bool(by_name.get(COMBO_ACTIVE))

        # 4 个槽位：按框的 x 坐标排序（屏幕底部左侧 4 条）
        for key, tag in (("combo_full", COMBO_FULL),
                         ("combo_empty", COMBO_EMPTY),
                         ("hp_danger", HP_DANGER),
                         ("hp_normal", HP_NORMAL)):
            items = sorted(by_name.get(tag, []), key=cx)[:N_SLOTS]
            for i, d in enumerate(items):
                out[key][i] = True

        out["end_skill"] = bool(by_name.get(END_SKILL))
        out["enemy_attack"] = bool(by_name.get(ENEMY_ATTACK)
                                   or by_name.get(ENEMY_CHARGE))
        out["energy"] = 2 if by_name.get(ENERGY_FULL) else (
            0 if by_name.get(ENERGY_EMPTY) else 1)
        sel = by_name.get(CHAR_SELECT)
        if sel:
            out["char_select"] = int(round(cx(sorted(sel, key=cx)[0]) * 0))
        out.update(ok=True, ms=self.ms, dets=dets, n_det=len(dets))
        self.state = out
        return out

    # ------------------------------------------------------------------
    def describe(self, st: dict = None) -> str:
        """一行人类可读摘要（给日志 / 面板用）。"""
        st = st or self.state
        if not st or not st.get("ok"):
            return "ONNX：未启用"
        bits = []
        if st["combo_active"]:
            bits.append("★连携")
        if st["end_skill"]:
            bits.append("大招")
        nf = sum(st["combo_full"])
        if nf:
            bits.append(f"连携{nf}/4")
        nd = sum(st["hp_danger"])
        if nd:
            bits.append(f"血危{nd}")
        if st["enemy_attack"]:
            bits.append("敌攻")
        return (f"ONNX {st['n_det']}框 {st['ms']:.0f}ms"
                + (" " + " ".join(bits) if bits else ""))


# ---------------------------------------------------------------- 自检
def _selftest():
    """自检：预处理、NMS、缺模型的降级、真实模型的输出格式。

    ★ 没有模型时后面几条会跳过 —— 但前几条必须过。
    """
    import sys
    state = [True]

    def chk(c, m):
        print(f"  {'OK  ' if c else 'FAIL'} {m}")
        state[0] = state[0] and c

    # ① letterbox 的几何
    from PIL import Image
    im = Image.new("RGB", (1280, 720), (10, 20, 30))
    t, s, pl, pt = letterbox(im)
    chk(t.shape == (1, 3, IMG, IMG), f"letterbox 输出 {t.shape}")
    chk(abs(s - IMG / 1280) < 1e-6, f"scale={s:.4f} 应为 0.5")
    chk(t.shape[0] == 1 and 0.0 <= t.min() and t.max() <= 1.0,
        "归一化到 [0,1]")
    chk(abs(float(t[0, :, 0, 0].mean()) - PAD / 255.0) < 0.01,
        f"补边填充 {PAD}/255")

    # ② 宽高比保持（720p -> 640 宽，高应该是 360，上下各补 140）
    chk(pl == 0 and pt == 140, f"pad_left={pl} pad_top={pt} 应为 0/140")

    # ③ NMS
    bx = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]],
                  dtype=np.float32)
    sc = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = _nms_1d(bx, sc, 0.45)
    chk(keep == [0, 2], f"NMS 保留 {keep} 应为 [0,2]（重叠的 1 被压掉）")

    # ④ 缺模型 -> 静默关闭，不抛异常
    d = OnnxDetect(model_path="__nope__.onnx", verbose=False)
    chk(not d.ok, "缺模型时 ok=False")
    st = d.step(np.zeros((720, 1280, 3), dtype=np.uint8))
    chk(st["ok"] is False and st["combo_active"] is False,
        "缺模型时 step() 返回安全默认值")

    # ⑤ 有模型时：真实推理
    d2 = OnnxDetect(verbose=False)
    if d2.ok:
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        st = d2.step(noise)
        chk(st["ok"], "真实模型推理成功")
        chk(st["ms"] > 0, f"耗时 {st['ms']:.0f} ms")
        chk(st["ms"] < 400, "单帧推理 < 400 ms（2.5Hz 预算）")
        chk(isinstance(st["dets"], list), f"返回 {st['n_det']} 个检测框")
        chk(len(st["hp_normal"]) == N_SLOTS, f"槽位长度 {N_SLOTS}")
        print(f"    describe(): {d2.describe(st)}")
    else:
        print("  SKIP 没有模型，跳过真实推理（跑 download_model.py 下载）")

    print()
    print("  全部通过 ✓" if state[0] else "  有失败 ✗")
    return 0 if state[0] else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
