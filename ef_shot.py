# -*- coding: utf-8 -*-
"""终末地截图小工具：全屏 GDI BitBlt 抓一张 PNG。

用法：
    python -X utf8 ef_shot.py <输出png> [x y w h]
不给区域就抓全屏（自动取主显示器分辨率）。
"""
import ctypes
import sys
from ctypes import wintypes

import numpy as np
from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# ★ 必须设 DPI 感知，否则拿到的是虚拟化坐标（本机 125% → 2048×1152 而不是 2560×1440），
#   跟 endfield_walk.py 抓到的图分辨率对不上，标定就会整体错位。
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def grab(x=0, y=0, w=None, h=None):
    if w is None:
        w = user32.GetSystemMetrics(0)
    if h is None:
        h = user32.GetSystemMetrics(1)
    hdc = user32.GetDC(0)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    gdi32.BitBlt(mem, 0, 0, w, h, hdc, x, y, SRCCOPY)

    bi = BITMAPINFO()
    bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bi.bmiHeader.biWidth = w
    bi.bmiHeader.biHeight = -h          # 负数 = 自上而下
    bi.bmiHeader.biPlanes = 1
    bi.bmiHeader.biBitCount = 32
    bi.bmiHeader.biCompression = BI_RGB

    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), DIB_RGB_COLORS)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(0, hdc)

    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return arr[:, :, 2::-1].copy()      # BGRA -> RGB


def grab_scaled(x, y, w, h, dw, dh):
    """抓取并**缩放**到 dw×dh（StretchBlt，缩放交给 GDI 驱动做）。

    为什么要它：全屏 2560×1440 抓一次约 50ms（14.7MB 传输），
    回来还要再降采样一遍。果蝇的眼睛只要 640×360 这个量级 ——
    让 GDI 先缩好再传，两头都省。
    """
    HALFTONE = 4
    hdc = user32.GetDC(0)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, dw, dh)
    old = gdi32.SelectObject(mem, bmp)
    gdi32.SetStretchBltMode(mem, HALFTONE)     # HALFTONE 画质好，避免跨步混叠
    gdi32.SetBrushOrgEx(mem, 0, 0, None)       # HALFTONE 模式要求设一次
    gdi32.StretchBlt(mem, 0, 0, dw, dh, hdc, x, y, w, h, SRCCOPY)

    bi = BITMAPINFO()
    bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bi.bmiHeader.biWidth = dw
    bi.bmiHeader.biHeight = -dh
    bi.bmiHeader.biPlanes = 1
    bi.bmiHeader.biBitCount = 32
    bi.bmiHeader.biCompression = BI_RGB

    buf = ctypes.create_string_buffer(dw * dh * 4)
    gdi32.GetDIBits(mem, bmp, 0, dh, buf, ctypes.byref(bi), DIB_RGB_COLORS)
    gdi32.SelectObject(mem, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(0, hdc)

    arr = np.frombuffer(buf, dtype=np.uint8).reshape(dh, dw, 4)
    return arr[:, :, 2::-1].copy()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "ef_shot.png"
    if len(sys.argv) >= 6:
        x, y, w, h = (int(v) for v in sys.argv[2:6])
        img = grab(x, y, w, h)
    else:
        img = grab()
    Image.fromarray(img).save(out)
    print(f"已保存 {out}  {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
