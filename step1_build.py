#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step1_build.py — 第 1 步：把连接组跑起来（不带游戏）

做什么：
  1. 读连接矩阵（1.5 亿行）→ 按 (pre, post) 聚合成 contact_count
  2. 读递质表定正负号：ACh 兴奋(+)，GABA/谷氨酸/组胺 抑制(-)，其余默认 +
  3. 建边权： W[post, pre] = sign × sqrt(contact_count)，再按目标节点归一化
  4. 存成 .npz，供后续反复使用（这一步慢，只做一次）

产出：
  graph_W.npz    稀疏矩阵（CSR, float32）
  graph_meta.npz 神经元索引表：bodyId、type、superclass、somaSide、nt、sign…
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.feather as feather


DATA = r"E:\终末地\_handoff\data"
OUT = r"E:\终末地\_handoff\graph"
os.makedirs(OUT, exist_ok=True)

W_F = DATA + r"\connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANN_F = DATA + r"\body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_F = DATA + r"\body-neurotransmitters-male-cns-v1.0.feather"

INHIBITORY = {"gaba", "glutamate", "histamine"}
t00 = time.time()


def log(msg):
    print(f"[{time.time() - t00:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 注释表
log("读注释表…")
ann = pd.read_feather(ANN_F, columns=["bodyId", "type", "flywireType", "superclass", "somaSide",
                                      "rootSide", "status", "instance", "subclass", "class"])
ann["bodyId"] = ann["bodyId"].astype("int64")
log(f"  神经元 {len(ann):,}")

# ---------------------------------------------------------------- 递质表
log("读递质表…")
nt = pd.read_feather(NT_F, columns=["body", "consensus_nt", "ground_truth", "predicted_nt",
                                    "predicted_nt_confidence"])
nt["body"] = nt["body"].astype("int64")
nt = nt.drop_duplicates(subset="body", keep="first")
log(f"  递质记录 {len(nt):,}")

ann = ann.merge(nt, left_on="bodyId", right_on="body", how="left").drop(columns=["body"])
nt_use = ann["consensus_nt"].fillna(ann["predicted_nt"])
ann["nt"] = nt_use.fillna("unclear")
ann["sign"] = np.where(ann["nt"].str.lower().isin(INHIBITORY), -1.0, 1.0).astype("float32")
log("  递质分布：")
for k, v in ann["nt"].value_counts().items():
    log(f"    {k:16s} {v:8,}  ({100.0*v/len(ann):5.1f}%)   sign={'-' if k.lower() in INHIBITORY else '+'}")
log(f"  正号 {(ann['sign'] > 0).sum():,}   负号 {(ann['sign'] < 0).sum():,}")

# 索引表：行号 = 神经元在 ann 里的位置
n = len(ann)
body2idx = pd.Series(np.arange(n, dtype=np.int32), index=ann["bodyId"].to_numpy())

# ---------------------------------------------------------------- 连接矩阵
log("读连接矩阵（1.5 亿行，分块聚合）…")
f = feather.read_table(W_F, memory_map=True)
rows_total = f.num_rows
log(f"  行数 {rows_total:,}")

CH = 10_000_000
pre_parts, post_parts, cnt_parts = [], [], []
for start in range(0, rows_total, CH):
    sub = f.slice(start, min(CH, rows_total - start))
    d = sub.to_pandas()
    d["body_pre"] = d["body_pre"].astype("int64")
    d["body_post"] = d["body_post"].astype("int64")

    # 只保留两端都在注释表里的边（约 88%）
    m = d["body_pre"].isin(body2idx.index) & d["body_post"].isin(body2idx.index)
    dd = d.loc[m]
    g = dd.groupby(["body_pre", "body_post"], sort=False, observed=True)["weight"].sum()
    pre_parts.append(g.index.get_level_values(0).to_numpy(dtype="int64"))
    post_parts.append(g.index.get_level_values(1).to_numpy(dtype="int64"))
    cnt_parts.append(g.to_numpy(dtype="float32"))
    del d, dd, g, sub, m
    log(f"  …{min(start + CH, rows_total):,}/{rows_total:,}  保留边对 {sum(len(x) for x in cnt_parts):,}")

pre = np.concatenate(pre_parts)
post = np.concatenate(post_parts)
cnt = np.concatenate(cnt_parts).astype("float32")
del pre_parts, post_parts, cnt_parts
log(f"聚合后 (pre,post) 对：{len(pre):,}")

# 全局去重合并（跨分块的同一对）
log("跨分块合并同一 (pre,post) 对…")
df = pd.DataFrame({"pre": pre, "post": post, "cnt": cnt})
df = df.groupby(["pre", "post"], sort=False, observed=True)["cnt"].sum().reset_index()
log(f"  合并后 {len(df):,} 条边")

# ---------------------------------------------------------------- 建矩阵
log("建稀疏矩阵 W[post, pre]…")
pi = df["pre"].map(body2idx).to_numpy()
qi = df["post"].map(body2idx).to_numpy()
c = df["cnt"].to_numpy()
sign = ann["sign"].to_numpy()
# 边权：sign × sqrt(contact_count)
w = sign[pi] * np.sqrt(c)
W = sp.coo_matrix((w, (qi, pi)), shape=(n, n)).tocsr()
W.sum_duplicates()
log(f"  W 形状 {W.shape}   非零 {W.nnz:,}   密度 {100.0*W.nnz/(n*n):.4f}%")

# 按目标节点归一化：每个 post 的入边权除以该 post 入边绝对值之和
log("按目标节点归一化…")
colsum = np.asarray(np.abs(W).sum(axis=0)).ravel()
colsum[colsum == 0] = 1.0
D = sp.diags(1.0 / colsum)
W = (D @ W).tocsr().astype("float32")
log(f"  归一化后 nnz {W.nnz:,}   |值| 最大 {np.abs(W.data).max():.4f}")

# ---------------------------------------------------------------- 落盘
log("存盘…")
sp.save_npz(os.path.join(OUT, "graph_W.npz"), W)
meta = ann[["bodyId", "type", "flywireType", "superclass", "somaSide", "rootSide", "status",
            "nt", "sign", "instance", "subclass", "class"]].copy()
meta["idx"] = np.arange(n, dtype=np.int32)
meta.to_feather(os.path.join(OUT, "graph_meta.feather"))
log(f"  {os.path.join(OUT, 'graph_W.npz')}")
log(f"  {os.path.join(OUT, 'graph_meta.feather')}")

# ---------------------------------------------------------------- 体检
log("体检：点名神经元在矩阵里的连接")
tgt = meta[meta["type"].isin(["DNg100", "DNa02", "DNg13", "DNp01", "DNp10"])]
for _, r in tgt.sort_values(["type", "somaSide"]).iterrows():
    i = int(r["idx"])
    out_deg = W[i, :].nnz
    in_deg = W[:, i].nnz
    log(f"  {r['bodyId']:>10d} {r['type']:>8s} {r['somaSide']}  出边(下游) {out_deg:>7,}   入边(上游) {in_deg:>7,}")

log(f"总耗时 {time.time() - t00:.0f}s   完成")
