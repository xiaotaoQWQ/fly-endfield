#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live_tune_cc.py — 危机合约专用调试台（独立于 live_tune.py）。

为什么单独一个：原版 `live_tune.py` 是给「日常跑图」用的，参数范围是按
探索/觅食调过的（比如技能最小间隔下限 0.5s、大招冷却下限 5s）。
**危机合约要的是「压满输出」**：所有冷却都应该能拉到 0 —— 宁可空转也要在
能放的那一瞬放出去。所以这里单开一套：

  · 所有冷却类参数的下限全部 = 0
  · 一键「全部冷却归零」
  · 实时状态卡（接管 / 连携成-败 / 闪避三路 / 挨打 / 饥饿 / 血 / Hz）
  · 参数改动**立即生效**（主循环每拍读 args，不需要重启）

用法（由 endfield_fly.py 启动）：
    python -X utf8 endfield_fly.py ... --cc-debug --cc-port 9121
    浏览器打开 http://127.0.0.1:9121
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# (属性名, 标签, 最小, 最大, 步长, 是否整数, 说明)
# ★ 冷却类的 min 一律 0
PARAMS = [
    ("combo_cool", "连携最小间隔 s", 0.0, 10.0, 0.1, 0, "0 = 有提示就按"),
    ("combo_cool_full", "连携兜底节奏 s", 0.0, 5.0, 0.1, 0, "0 = 关（默认关）"),
    ("combo_repeat", "多角色追加按间隔 s", 0.0, 3.0, 0.1, 0, "0 = 只按上升沿"),
    ("combo_repeat_max", "一次提示最多连按", 1.0, 8.0, 1.0, 1, "默认 4 = 队伍人数"),
    ("combo_hunger", "连携要求饥饿", 0.0, 1.0, 0.05, 0, "0 = 不看饥饿"),
    ("dodge_cool", "闪避最小间隔 s", 0.0, 5.0, 0.1, 0, "0 = 能闪就闪（注意体力）"),
    ("dodge_z", "闪避·looming 阈值", 0.0, 3.0, 0.1, 0, "越小越敏感"),
    ("dodge_flash_red", "闪避·中心闪红阈值", 0.0, 120.0, 5.0, 0, "越小越敏感"),
    ("skill_interval", "技能最小间隔 s", 0.0, 20.0, 0.1, 0, "0 = 每拍都试"),
    ("ult_cool", "大招冷却 s", 0.0, 120.0, 1.0, 0, "0 = 一有机会就放"),
    ("ult_hold", "大招按住 s", 0.2, 3.0, 0.1, 0, ""),
    ("attack_hold", "普攻按住 s", 0.05, 3.0, 0.05, 0, "跨拍长按"),
    ("hit_cooldown", "小额奖赏冷却 s", 0.0, 5.0, 0.1, 0, "0 = 每拍都给"),
    ("punish_cool", "挨打惩罚冷却 s", 0.0, 5.0, 0.1, 0, "auto_outcome 用"),
    ("forage_thr", "攻击阈值", 0.0, 1.5, 0.01, 0, "越小越爱出手"),
    ("skill_hunger", "技能要求饥饿", 0.0, 1.0, 0.05, 0, ""),
    ("ult_hunger", "大招要求饥饿", 0.0, 1.0, 0.05, 0, ""),
    ("heal_hp", "回血线", 0.0, 1.0, 0.05, 0, "血低于它就放回血"),
    ("stuck_sec", "判卡住 秒", 0.5, 20.0, 0.5, 0, ""),
]
PNAMES = {p[0] for p in PARAMS}
# 一键归零：只动「冷却/间隔」类
ZERO_KEYS = ["combo_cool", "combo_cool_full", "dodge_cool", "skill_interval",
             "ult_cool", "hit_cooldown", "punish_cool"]

_PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>危机合约调试台</title>
<style>
body{background:#0d1016;color:#dfe6f2;font:14px/1.5 "Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:16px}
h1{font-size:18px;color:#ffcf8f;margin:0 0 4px}
.sub{color:#7b879b;font-size:12px;margin-bottom:12px}
.card{background:#161b24;border:1px solid #232c3a;border-radius:8px;padding:12px;margin-bottom:12px}
.row{display:grid;grid-template-columns:190px 1fr 64px;gap:10px;align-items:center;padding:3px 0}
.row label{color:#b9c6da;font-size:13px}
.row input[type=range]{width:100%}
.val{font-family:Consolas,monospace;color:#7fe3a8;text-align:right}
.btn{background:#22303f;border:1px solid #35506b;color:#cfe3ff;border-radius:6px;padding:8px 14px;margin-right:8px;cursor:pointer;font-size:13px}
.btn:hover{background:#2c3d50}
.btn.warn{background:#4a2020;border-color:#7a3131;color:#ffc9c9}
.btn.go{background:#1d3b2a;border-color:#2f6b47;color:#b9ffcf}
#st{font-family:Consolas,monospace;font-size:13px;color:#a9e0ff;white-space:pre-wrap}
.k{color:#7b879b}
</style></head><body>
<h1>危机合约 · 调试台</h1>
<div class="sub">参数改动立即生效（主循环每拍读 args）。<span id="conn">连接中…</span></div>
<div class="card"><div id="st">读取状态…</div></div>
<div class="card">
  <button class="btn go" onclick="zero()">一键全部冷却归零</button>
  <button class="btn" onclick="reload()">重新读取</button>
  <button class="btn warn" onclick="loc(0)">关掉闪避</button>
  <button class="btn" onclick="loc(1)">恢复闪避</button>
</div>
<div class="card" id="sliders"></div>
<script>
const P = __PARAMS__;
function fmt(v){ return (Math.round(v*1000)/1000).toString(); }
function build(){
  const box = document.getElementById("sliders");
  box.innerHTML = "<div style='color:#7b879b;font-size:12px;margin-bottom:6px'>参数（拖动即生效）</div>";
  P.forEach(p=>{
    const [k,label,lo,hi,step,isint,tip] = p;
    const d = document.createElement("div"); d.className="row";
    d.innerHTML = `<label title="${tip||""}">${label}</label>` +
      `<input type="range" min="${lo}" max="${hi}" step="${step}" id="s_${k}">` +
      `<span class="val" id="v_${k}"></span>`;
    box.appendChild(d);
    const s = d.querySelector("input");
    s.addEventListener("input", ()=>{ document.getElementById("v_"+k).textContent = fmt(s.value); });
    s.addEventListener("change", ()=>{ setp(k, s.value); });
  });
}
async function api(path, body){
  const r = await fetch(path, {method: body?"POST":"GET", headers:{"Content-Type":"application/json"}, body: body?JSON.stringify(body):undefined});
  return await r.json();
}
async function reload(){ const j = await api("/api/params");
  P.forEach(p=>{ const s=document.getElementById("s_"+p[0]); if(s && j[p[0]]!==undefined){ s.value=j[p[0]]; document.getElementById("v_"+p[0]).textContent=fmt(s.value);} }); }
async function setp(k,v){ await api("/api/set", {k:k, v:parseFloat(v)}); document.getElementById("conn").textContent="已设置 "+k; }
async function zero(){ const j = await api("/api/zero", {}); document.getElementById("conn").textContent = "已归零："+j.changed.join(", "); await reload(); }
async function loc(on){ await api("/api/dodge", {on:on}); document.getElementById("conn").textContent = on?"闪避已开":"闪避已关"; }
async function tick(){ try{ const j = await api("/api/status"); document.getElementById("st").textContent = j.text || JSON.stringify(j); document.getElementById("conn").textContent="已连接"; }catch(e){ document.getElementById("conn").textContent="连接断开"; } }
build(); reload(); tick(); setInterval(tick, 1000);
</script></body></html>"""


def start_cc_debug(args, port: int = 9121, status_fn=None, verbose: bool = True):
    """起危机合约调试台。返回网址。参数改动直接写回 args（白名单 + 夹紧）。"""
    page = _PAGE.replace("__PARAMS__", json.dumps(PARAMS, ensure_ascii=False))
    state = {"args": args, "status": status_fn, "lock": threading.Lock()}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):      # 不往控制台刷访问日志
            return

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except Exception:
                pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if u.path == "/api/params":
                a = state["args"]
                return self._send(200, json.dumps({p[0]: getattr(a, p[0], None) for p in PARAMS}))
            if u.path == "/api/status":
                try:
                    st = state["status"]() if state["status"] else {}
                except Exception as e:
                    st = {"text": f"状态回调出错：{e}"}
                return self._send(200, json.dumps(st, ensure_ascii=False))
            return self._send(404, "{}")

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                body = {}
            a = state["args"]
            if u.path == "/api/set":
                k = str(body.get("k", ""))
                if k not in PNAMES:
                    return self._send(400, json.dumps({"err": "not allowed"}))
                p = next(x for x in PARAMS if x[0] == k)
                lo, hi, isint = p[2], p[3], p[5]
                try:
                    v = float(body.get("v"))
                except Exception:
                    return self._send(400, json.dumps({"err": "bad value"}))
                v = max(lo, min(hi, v))
                if isint:
                    v = int(round(v))
                setattr(a, k, v)
                return self._send(200, json.dumps({"ok": True, "k": k, "v": v}))
            if u.path == "/api/zero":
                changed = []
                for k in ZERO_KEYS:
                    setattr(a, k, 0 if k != "combo_repeat" else 0)
                    changed.append(k)
                return self._send(200, json.dumps({"ok": True, "changed": changed}))
            if u.path == "/api/dodge":
                on = bool(body.get("on"))
                a.no_dodge = not on
                return self._send(200, json.dumps({"ok": True, "dodge": on}))
            return self._send(404, "{}")

    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{int(port)}"
    if verbose:
        print(f"  ★ 危机合约调试台：{url}（冷却全部可调到 0）")
    return url
