#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""download_model.py — 下载 ONNX 识别模型（MaaEnd 的 YOLO11s）。

模型不在仓库里（36 MB），要单独下。这个脚本负责下 + 校验。

★ 为什么不用 raw.githubusercontent.com
    这台机器上它不通（HTTP 000）。改用 GitHub API + `Accept: application/vnd.github.raw`，
    会 302 到 S3，能通。

★ 为什么下载要那么久
    实测约 60 KB/s，36 MB 要 **10 分钟**。所以默认 `--max-time 1800`，
    而且**下载完必须校验字节数** —— 中途超过 max-time 会被截断成半截文件，
    而 onnxruntime 加载半截文件会报一个完全看不出原因的错。

用法：
    python -X utf8 download_model.py
    python -X utf8 download_model.py --force     # 已存在也重下
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models", "autofightv13.onnx")

# MaaEnd/MaaEnd-AI · AGPL-3.0
URL = ("https://api.github.com/repos/MaaEnd/MaaEnd-AI/contents/"
       "detect/AutoFight/autofightv13.onnx")
EXPECT_SIZE = 37960805      # 37,960,805 B —— 必须精确匹配


def download(dest: str, url: str, expect: int) -> bool:
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)

    req = urllib.request.Request(url, headers={
        "User-Agent": "fly-endfield",
        "Accept": "application/vnd.github.raw",
    })

    print(f"  下载 {expect:,} B（约 10 分钟，60 KB/s）…")
    t0 = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r, \
                open(tmp, "wb") as f:
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got % (5 * 1024 * 1024) < 256 * 1024:
                    print(f"    {got / 1e6:6.1f} / {expect / 1e6:.1f} MB "
                          f"（{time.time() - t0:.0f}s）")
    except Exception as e:
        print(f"  ✗ 下载失败：{type(e).__name__}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False

    size = os.path.getsize(tmp)
    if size != expect:
        print(f"  ✗ 大小不符：{size:,} B，应为 {expect:,} B（被截断了？）")
        os.remove(tmp)
        return False

    os.replace(tmp, dest)
    print(f"  ✓ 完成 {size:,} B  用时 {time.time() - t0:.0f} 秒")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    args = ap.parse_args()

    print("ONNX 识别模型（MaaEnd YOLO11s，AGPL-3.0）")
    print(f"  目标 {DEST}")

    if os.path.exists(DEST) and not args.force:
        size = os.path.getsize(DEST)
        if size == EXPECT_SIZE:
            print(f"  ✓ 已存在且大小正确（{size:,} B），跳过")
            return 0
        print(f"  ⚠ 已存在但大小不符（{size:,} B），重下")

    if not download(DEST, URL, EXPECT_SIZE):
        print("\n  失败。可以手动下载：")
        print(f"    {URL}")
        print(f"  存到 {DEST}")
        return 1

    # 校验能加载
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        s = ort.InferenceSession(DEST, so, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0]
        print(f"  ✓ 能加载：输入 {inp.name} {inp.shape}")
        return 0
    except Exception as e:
        print(f"  ⚠ 下载完成但加载失败：{e}")
        print("    （可能 onnxruntime 没装：pip install onnxruntime）")
        return 1


if __name__ == "__main__":
    sys.exit(main())
