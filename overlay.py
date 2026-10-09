#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""overlay.py — 一个「永远不抢焦点」的置顶叠加窗口（纯 Win32，不用 tkinter）。

为什么不用 tkinter：
    这台机器上 tkinter 会**硬崩**（exit 0xC0000005，之前做剪贴板时踩过）。

为什么必须不抢焦点：
    游戏一旦失焦，Interception 的输入就送不到，整个闭环当场作废。
    而且我们的感知是靠 BitBlt 抓屏的，任何盖住小地图采样区的东西
    都会污染果蝇的输入。

做法：
    WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT
    · NOACTIVATE  → 点它、显示它都不会激活它
    · TRANSPARENT → 鼠标事件穿透
    每帧用 UpdateLayeredWindow 把一张 RGBA 图推上去。

用法（自检：显示一段动画，自己截图验证，然后退出）：
    python -X utf8 overlay.py --test
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# ★ 必须显式声明 argtypes/restype：
#   否则 LPARAM（64 位）会被当成 32 位 int，回调里直接 OverflowError。
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                  wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = ctypes.c_longlong
user32.CreateWindowExW.restype = wintypes.HWND
user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                wintypes.UINT, wintypes.UINT, wintypes.UINT]
user32.UpdateLayeredWindow.restype = wintypes.BOOL

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
SW_SHOWNOACTIVATE = 4
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
BI_RGB = 0
DIB_RGB_COLORS = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_byte), ("BlendFlags", ctypes.c_byte),
                ("SourceConstantAlpha", ctypes.c_byte),
                ("AlphaFormat", ctypes.c_byte)]


WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_longlong, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSEX(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


class Overlay:
    """置顶、不抢焦点、鼠标穿透的叠加层。"""

    _cls_registered = False
    _cls_name = "DshFlyOverlayWnd"

    def __init__(self, w: int, h: int, x: int = 1900, y: int = 300,
                 title: str = "fly-overlay"):
        self.w, self.h = int(w), int(h)
        self.x, self.y = int(x), int(y)
        self.hwnd = None
        self._hdc_src = None
        self._hbmp = None
        self._old = None
        self._buf = None
        self._closed = False

        hinst = kernel32.GetModuleHandleW(None)

        if not Overlay._cls_registered:
            self._wndproc = WNDPROC(self._proc)      # 必须留引用，否则被 GC
            wc = WNDCLASSEX()
            wc.cbSize = ctypes.sizeof(WNDCLASSEX)
            wc.style = 0
            wc.lpfnWndProc = self._wndproc
            wc.hInstance = hinst
            wc.hCursor = user32.LoadCursorW(None, 32512)   # IDC_ARROW
            wc.lpszClassName = Overlay._cls_name
            if not user32.RegisterClassExW(ctypes.byref(wc)):
                err = kernel32.GetLastError()
                if err != 1410:                        # 1410 = 类已存在
                    raise ctypes.WinError(err)
            Overlay._cls_registered = True

        ex = (WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_NOACTIVATE
              | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
        self.hwnd = user32.CreateWindowExW(
            ex, Overlay._cls_name, title, WS_POPUP,
            self.x, self.y, self.w, self.h, None, None, hinst, None)
        if not self.hwnd:
            raise ctypes.WinError(kernel32.GetLastError())

        # 建一次 DIB，之后每帧复用
        screen = user32.GetDC(None)
        self._hdc_src = gdi32.CreateCompatibleDC(screen)
        bi = BITMAPINFO()
        bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.bmiHeader.biWidth = self.w
        bi.bmiHeader.biHeight = -self.h          # 自上而下
        bi.bmiHeader.biPlanes = 1
        bi.bmiHeader.biBitCount = 32
        bi.bmiHeader.biCompression = BI_RGB
        self._hbmp = gdi32.CreateDIBSection(
            screen, ctypes.byref(bi), DIB_RGB_COLORS,
            ctypes.byref(ctypes.c_void_p()), None, 0)
        user32.ReleaseDC(None, screen)
        if not self._hbmp:
            raise ctypes.WinError(kernel32.GetLastError())
        self._old = gdi32.SelectObject(self._hdc_src, self._hbmp)
        self._bi = bi

        self.pump()
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        try:
            # 只改 Z 序、不激活
            user32.SetWindowPos(self.hwnd, -1, self.x, self.y, self.w, self.h,
                                0x0010 | 0x0040)   # SWP_NOACTIVATE | SWP_SHOWWINDOW
        except Exception:
            pass

    # ---------------------------------------------------------------- 消息
    def _proc(self, hwnd, msg, wp, lp):
        if msg == 0x0002:            # WM_DESTROY
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wp, lp)

    def pump(self):
        """把窗口消息抽干（我们跑在别人的主循环里，不能阻塞）。"""
        msg = wintypes.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    # ---------------------------------------------------------------- 更新
    def update(self, rgba):
        """rgba: (h,w,4) uint8。alpha=0 的地方完全透明（也是鼠标穿透区）。"""
        import numpy as np
        a = np.asarray(rgba, dtype=np.uint8)
        if a.shape[0] != self.h or a.shape[1] != self.w:
            raise ValueError(f"尺寸不符：给的是 {a.shape[:2]}，窗口是 {(self.h, self.w)}")

        # UpdateLayeredWindow 要**预乘 alpha** 的 BGRA
        af = a[..., 3:4].astype(np.uint16)
        bgra = np.empty((self.h, self.w, 4), dtype=np.uint8)
        bgra[..., 0] = (a[..., 2].astype(np.uint16) * af[..., 0] // 255).astype(np.uint8)
        bgra[..., 1] = (a[..., 1].astype(np.uint16) * af[..., 0] // 255).astype(np.uint8)
        bgra[..., 2] = (a[..., 0].astype(np.uint16) * af[..., 0] // 255).astype(np.uint8)
        bgra[..., 3] = a[..., 3]
        self._buf = np.ascontiguousarray(bgra)

        gdi32.SetDIBits(self._hdc_src, self._hbmp, 0, self.h,
                        self._buf.ctypes.data_as(ctypes.c_void_p),
                        ctypes.byref(self._bi), DIB_RGB_COLORS)

        screen = user32.GetDC(None)
        pt_dst = wintypes.POINT(self.x, self.y)
        size = wintypes.SIZE(self.w, self.h)
        pt_src = wintypes.POINT(0, 0)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        ok = user32.UpdateLayeredWindow(
            self.hwnd, screen, ctypes.byref(pt_dst), ctypes.byref(size),
            self._hdc_src, ctypes.byref(pt_src), 0, ctypes.byref(blend), ULW_ALPHA)
        user32.ReleaseDC(None, screen)
        if not ok:
            raise ctypes.WinError(kernel32.GetLastError())

    def set_visible(self, on: bool):
        """显示 / 隐藏面板。

        ★ 为什么需要：面板是主循环画的。主循环一停（暂停 / 一轮结束），
          面板就冻在屏幕上一动不动 —— 看起来就像"程序卡死了"，
          而且它还挡着游戏画面。所以空闲时**必须把它藏起来**。
        """
        try:
            ctypes.windll.user32.ShowWindow(self.hwnd, 5 if on else 0)
        except Exception:
            pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._old:
                gdi32.SelectObject(self._hdc_src, self._old)
            if self._hbmp:
                gdi32.DeleteObject(self._hbmp)
            if self._hdc_src:
                gdi32.DeleteDC(self._hdc_src)
            if self.hwnd:
                user32.DestroyWindow(self.hwnd)
        except Exception:
            pass


# -------------------------------------------------------------------- 自检
def _test():
    import numpy as np
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from brain_view import BrainView
    from ef_shot import grab

    W, H = 620, 880
    ov = Overlay(W, H, x=1900, y=300)
    bv = BrainView(w=W - 28, h=520)

    rng = np.random.default_rng(1)
    r = np.zeros(bv.n_all, dtype=np.float32)
    hot = rng.choice(bv.idx_ok, 30000, replace=False)
    r[hot] = rng.random(30000).astype(np.float32) * 0.1
    bv._hi = float(np.percentile(r[bv.idx_ok], 99.0))

    print(f"叠加窗口 {W}×{H} @ (1900,300)  hwnd={ov.hwnd}")
    t0 = time.time()
    frames = 0
    try:
        while time.time() - t0 < 8.0:
            t = time.time() - t0
            brain = bv.render(r, yaw=0.6 * np.sin(t * 0.7), pitch=0.35)
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[..., :3] = 12
            rgba[..., 3] = 235
            rgba[92:92 + brain.shape[0], 14:14 + brain.shape[1], :3] = brain
            ov.update(rgba)
            ov.pump()
            frames += 1
            time.sleep(0.1)
    finally:
        full = grab()                 # 抓全屏，验证 BitBlt 能不能抓到叠加层
        Image.fromarray(full).save(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shots", "overlay_test.png"))
        ov.close()

    print(f"  {frames} 帧 / 8 秒  →  {frames/8:.0f} fps")
    print("  已截图 shots/overlay_test.png（能否看到面板 = BitBlt 抓不抓得到叠加层）")


if __name__ == "__main__":
    _test()
