#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_optic_map.py — 给感光细胞建眼面坐标（复现 Fly64 的三级映射）

为什么需要这个：
  交接文档第六节要求「按复眼几何映射到感光细胞位置」，但注释表里感光细胞
  只有左右（somaSide 覆盖 0.6%，rootSide 99.9%），做不出 5 扇区。
  Fly64 用官方 optic-column 数据解决了，本脚本复现它的做法。

三级映射（严格照 fly64/data.py:168-195）：
  ① R7/R8：直接在 optic-column 表里查到所属柱 → hex 坐标 (h1,h2)
  ② R1-R6：注释表里没有柱分配 → 取「与已分配柱的 L1 连接最强」的那个 L1 的柱
  ③ 还没落位的：按 bodyId 排序，均匀铺在 48×32 网格上

输出：optic_map.feather  列 = bodyId, idx, type, side, h1, h2, px_x, px_y, mapping_level
      mapping_level: 1=官方柱分配  2=连接组推导  0=均匀铺开（无空间信息）
"""
import io
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import scipy.sparse as sp


DATA = r"E:\终末地\_handoff\data"
GRAPH = r"E:\终末地\_handoff\graph"
OUT = r"E:\终末地\_handoff\out"
XLSX = os.path.join(DATA, "optic", "optic-column-type-assignments-v1.0.xlsx")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# 感光细胞（MaleCNS 的写法）
PR_PAT = r"^(R1-R6|R[78][pyd]?|R[78]|R[78]_unclear|R7R8_unclear|HBeyelet)$"
# Fly64 只取这三类做视觉输入（不含 *_unclear / HBeyelet）
VISUAL_TYPES = {"R1-R6", "R7", "R8"}   # 这里用 flywireType 口径


def optic_columns(path):
    """解析 optic-column 表 → {bodyId: (side, h1, h2)}

    照 fly64/data.py:25-49。列 A 是柱名 ME_<LR>_col_<h1>_<h2>，
    B=L1、C=R7、E=R8 是柱内对应细胞的 bodyId（-99 表示该柱没有这种细胞）。
    """
    result = {}
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(e.itertext())
                   for e in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        for sheet in (1, 2):
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
            for row in root.findall("s:sheetData/s:row", NS)[1:]:
                cells = {}
                for c in row:
                    v = c.find("s:v", NS)
                    if v is not None:
                        cells[re.sub(r"\d", "", c.attrib["r"])] = (
                            strings[int(v.text)] if c.get("t") == "s" else v.text)
                match = re.fullmatch(r"ME_([LR])_col_(\d+)_(\d+)", cells.get("A", ""))
                if match:
                    side, h1, h2 = match.groups()
                    for col in ("B", "C", "E"):
                        try:
                            body = int(cells.get(col, -99))
                            if body > 0:
                                result[body] = (side, int(h1), int(h2))
                        except (ValueError, TypeError):
                            pass
    return result


def main():
    meta = pd.read_feather(os.path.join(GRAPH, "graph_meta.feather"))
    n = len(meta)
    print(f"神经元 {n:,}")

    columns = optic_columns(XLSX)
    print(f"optic-column 表解析出 {len(columns):,} 个 bodyId 的柱分配")
    sides = pd.Series([v[0] for v in columns.values()]).value_counts().to_dict()
    h1s = [v[1] for v in columns.values()]
    h2s = [v[2] for v in columns.values()]
    print(f"  左右分布 {sides}   h1 范围 {min(h1s)}~{max(h1s)}   h2 范围 {min(h2s)}~{max(h2s)}")

    # ---------------- 视觉神经元（用 flywireType，照 Fly64 口径）----------------
    fw = meta["flywireType"].astype(str).to_numpy()
    ty = meta["type"].astype(str).to_numpy()
    is_visual = np.isin(fw, list(VISUAL_TYPES)) | np.isin(ty, list(VISUAL_TYPES))
    visual = np.flatnonzero(is_visual)
    print(f"\n视觉输入神经元（R1-6/R7/R8 口径）：{len(visual):,}")
    print("  按 flywireType 构成：")
    print(pd.Series(fw[visual]).value_counts().head(10).to_string())

    # 侧别
    side_arr = meta["rootSide"].astype(str).to_numpy()
    bad = ~np.isin(side_arr, ["L", "R"])
    side_arr[bad] = meta["somaSide"].astype(str).to_numpy()[bad]
    inst = meta["instance"].astype(str).to_numpy()
    inst_side = np.array([s[-1] if isinstance(s, str) and s.endswith(("_L", "_R")) else "?"
                          for s in inst])
    bad = ~np.isin(side_arr, ["L", "R"])
    side_arr[bad] = inst_side[bad]

    body_ids = meta["bodyId"].to_numpy()

    # ---------------- ① 官方柱分配 ----------------
    mapping_level = np.zeros(len(visual), dtype=np.uint8)
    h1_arr = np.full(len(visual), -1, dtype=np.int32)
    h2_arr = np.full(len(visual), -1, dtype=np.int32)
    for m, neuron in enumerate(visual):
        loc = columns.get(int(body_ids[neuron]))
        if loc:
            _, h1, h2 = loc
            h1_arr[m], h2_arr[m] = h1, h2
            mapping_level[m] = 1
    print(f"\n① 官方柱分配命中：{int((mapping_level==1).sum()):,}")

    # ---------------- ② R1-R6：连到已分配柱的 L1 ----------------
    W = sp.load_npz(os.path.join(GRAPH, "graph_W_raw.npz")).tocsc()
    known_idx = np.flatnonzero(np.isin(body_ids, np.fromiter(columns.keys(), dtype=np.int64,
                                                             count=len(columns))))
    # 只看已知柱的 L1 细胞
    l1_idx = np.array([i for i in known_idx
                       if str(meta["type"].iloc[i]) == "L1" or str(meta["flywireType"].iloc[i]) == "L1"],
                      dtype=np.int64)
    print(f"② 已分配的 L1 细胞：{len(l1_idx):,}")

    need = np.flatnonzero(mapping_level == 0)
    if len(l1_idx) and len(need):
        sub = np.abs(W[visual[need]][:, l1_idx]).tocsr()      # (need, l1)
        print(f"   连接矩阵子块 {sub.shape}  非零 {sub.nnz:,}")
        for r, m in enumerate(need):
            a, b = sub.indptr[r], sub.indptr[r + 1]
            if b > a:
                best = l1_idx[sub.indices[a + int(np.argmax(sub.data[a:b]))]]
                loc = columns.get(int(body_ids[best]))
                if loc:
                    _, h1, h2 = loc
                    h1_arr[m], h2_arr[m] = h1, h2
                    mapping_level[m] = 2
    print(f"   连接组推导命中：{int((mapping_level==2).sum()):,}")

    # ---------------- ③ 剩下的均匀铺开 ----------------
    n_left = int((mapping_level == 0).sum())
    for side, x0 in (("L", 0), ("R", 32)):
        members = np.array([m for m in range(len(visual)) if side_arr[visual[m]] == side],
                           dtype=np.int64)
        ordered = members[np.argsort(body_ids[visual[members]], kind="stable")]
        for rank, m in enumerate(ordered):
            if mapping_level[m] == 0:
                h1_arr[m] = rank % 32 + 1
                h2_arr[m] = rank * 48 // max(len(ordered), 1) + 1
                mapping_level[m] = 0
    print(f"③ 无空间信息、均匀铺开：{int((mapping_level==0).sum()):,}")

    # ---------------- 仿射投影到 48×64 像素网格（照 Fly64:188-189）----------------
    px_x = np.clip(np.round((h2_arr - 1) * 47 / 38), 0, 47).astype(np.int32)
    px_y = (np.where(side_arr[visual] == "L", 0, 32)
            + np.clip(np.round((h1_arr - 1) * 31 / 35), 0, 31)).astype(np.int32)

    out = pd.DataFrame({
        "bodyId": body_ids[visual],
        "idx": visual,
        "type": ty[visual],
        "flywireType": fw[visual],
        "side": side_arr[visual],
        "h1": h1_arr,
        "h2": h2_arr,
        "px_x": px_x,
        "px_y": px_y,
        "mapping_level": mapping_level,
    })
    out.to_feather(os.path.join(OUT, "optic_map.feather"))
    print(f"\n已写出 {os.path.join(OUT, 'optic_map.feather')}   {len(out):,} 行")
    print()
    print("落位质量：")
    print(out["mapping_level"].value_counts().sort_index()
          .rename({1: "① 官方柱分配", 2: "② 连接组推导", 0: "③ 无线索铺开"}).to_string())
    print()
    print("px_x/px_y 分布（应落在 0..47 / 0..63）：")
    print(f"  px_x {px_x.min()}~{px_x.max()}   px_y {px_y.min()}~{px_y.max()}")
    print()
    print("左右两眼的角覆盖（照 retina.py 的公式）：")
    for s, e0 in (("L", -135.0), ("R", -8.5)):
        sel = side_arr[visual] == s
        if sel.sum():
            horiz = (px_y[sel] % 32) / 31
            az = e0 + horiz * 143.5
            el = 72 - px_x[sel] / 47 * 144
            print(f"  {s}眼 {int(sel.sum()):>6,} 个：方位角 {az.min():7.1f}° ~ {az.max():7.1f}°"
                  f"   仰角 {el.min():7.1f}° ~ {el.max():7.1f}°")


if __name__ == "__main__":
    main()
