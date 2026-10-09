#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live_tune.py — 后台实时调参面板（本地网页，零依赖）。

设计上只做了一件事：**HTTP 处理器直接写 argparse.Namespace**。

    为什么这样就够了：endfield_fly.py 的循环里全是 `args.forage_thr` 这种读法，
    只要在跑的时候把 args 上那个属性改掉，**下一拍自动就是新值** ——
    一行循环逻辑都不用动。这是这个面板能做得这么小的唯一原因。

    少数几个参数不在 args 上（比如 Hunger 对象的 rise/eat 是启动时拷进去的），
    靠 `sync_hooks` 里注册的小函数在每拍同步一次。

用法：
    from live_tune import start_tuner
    url = start_tuner(args, status_fn=我的状态回调, sync_hooks=[...])
    print(url)          # http://127.0.0.1:8791/

页面：打开就是一组滑块，拖完自动提交。左边是当前值，右边是实时状态读数。

安全：只监听 127.0.0.1，只认白名单里的参数名 —— 不接受任意属性名，
      免得网页上随便发个字符串就把 args 写坏。
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- 参数白名单
# (属性名, 中文标签, 最小, 最大, 步长, 说明)
GROUPS = [
    ("觅食驱动", [
        ("forage_thr", "攻击阈值", 0.0, 1.5, 0.01,
         "驱动力超过它就出手。调高=更挑，调低=更凶"),
        ("forage_innate", "先天取食反射", 0.0, 2.0, 0.01,
         "★ 这一份是本能，不用学。写 0 会退化成「学习独占」并崩掉"),
        ("forage_learn", "学习调制幅度", 0.0, 1.0, 0.01,
         "学习只能在先天那份上面浮动这么多"),
        ("forage_eta", "学习率 η", 0.0, 0.02, 0.0005,
         "★ 基础突触权重只有 0.0000~0.0131。给大了就是覆盖，不是学习"),
        ("forage_decay", "突触衰减", 0.0, 0.001, 0.00005,
         "遗忘。只增不减会漂走，跑几十次行为就崩"),
    ]),
    ("饥饿内稳态", [
        ("hunger_rise", "饥饿回升 /秒", 0.0, 0.2, 0.005,
         "★ 必须和「每餐下降」平衡，否则 h 会钉死在 0、节律消失"),
        ("hunger_eat", "每餐下降", 0.0, 1.0, 0.05,
         "吃一口降多少"),
    ]),
    ("攻击执行", [
        ("attack_hold", "普攻按住 秒", 0.05, 3.0, 0.05,
         "★ 超过决策拍(0.4s)就会跨拍，按下和抬起落在不同拍上"),
        ("heal_hp", "回血线", 0.0, 1.0, 0.05,
         "血低于它就放技能4。这是保命规则，学不坏"),
        ("skill_interval", "技能最小间隔 秒", 0.5, 20.0, 0.5,
         "技能有冷却，狂按只会空转"),
    ]),
    ("技能轮换", [
        ("ult_age", "大招：要求战线 秒", 0.0, 90.0, 1.0,
         "★ 战斗越久=越饿。这个值同时也是「饿到极点」的来源"),
    ]),
    ("奖励塑形", [
        ("hit_reward", "打到的小额奖赏", 0.0, 1.0, 0.05,
         "★ 必须远小于「打死」的大奖，否则它会停在打到那一层"),
        ("hit_cooldown", "小额奖赏冷却 秒", 0.0, 5.0, 0.1,
         "否则 2.5Hz 下每拍都发，等于白送"),
    ]),
    ("连携技 E", [
        ("combo_hunger", "要求饥饿", 0.0, 1.0, 0.05, "饥饿满才按 E"),
        ("combo_cool", "最小间隔 秒", 0.0, 10.0, 0.5, ""),
        ("combo_corr", "模板相关阈值", 0.2, 0.9, 0.05,
         "实测真提示 +0.53~+1.00，假的最大 +0.27"),
    ]),
    ("大招（长按 1/2/3/4）", [
        ("ult_hunger", "要求饥饿", 0.0, 1.0, 0.05,
         "★ 饿疯了才放。它没有能量条这个概念，它只有饿"),
        ("ult_hp", "最低血量", 0.0, 1.0, 0.05,
         "★ 低于它就不放 —— 受伤的果蝇该回避，不是猛攻"),
        ("ult_cool", "冷却 秒", 5.0, 120.0, 5.0, ""),
        ("ult_hold", "按住 秒", 0.2, 3.0, 0.1, "长按才触发大招"),
    ]),
    ("任务完成（★ 默认关，要 --mission 才生效）", [
        ("mission_gone", "文字消失判定 秒", 1.0, 10.0, 0.5,
         "★ 任务条上的「击败所有敌人」**连续消失这么久**就结算："
         "给 50 次奖励 + 饥饿值清空 10 秒。"
         "这个「连续」很关键 —— 文字会淡入淡出，"
         "只看下降沿会被抖动误触发"),
        ("mission_reward", "奖励次数", 1.0, 200.0, 1.0,
         "结算时连发多少次多巴胺（默认 50）"),
        ("mission_hold", "吃饱 秒", 0.0, 60.0, 1.0,
         "结算后饥饿值清空并**多少秒不涨**（默认 10），之后恢复正常增长"),
        ("mission_corr", "文字相关阈值", 0.30, 0.90, 0.05,
         "实测：文字显示 +0.63~+1.00、消失 +0.14~+0.29，0.50 干净分离"),
    ]),
    ("移动", [
        ("stuck_sec", "判定卡住 秒", 0.5, 20.0, 0.5, "撞墙反射的触发门槛"),
        ("escape_grace", "开局免反射 秒", 0.0, 60.0, 1.0, ""),
        ("urge", "战斗中攻击阈值倍率", 0.1, 2.0, 0.05,
         "红叉亮时把阈值乘上它（越小越凶）"),
    ]),
]

WHITELIST = {p[0] for _, ps in GROUPS for p in ps}

# ---------------------------------------------------------------- 对照组开关
# ★ 为什么要有这一组：
#   这个项目的两条底线是"**它得真的在控制那个角色**（不能是脚本在走、
#   果蝇只做装饰）"和"它得真的能走到某个地方"。
#   光靠嘴说没用 —— **得能把某个脑区关掉，看行为变不变。**
#
#   所以这一组不是"调参"，是"**消融实验**"：
#   关掉某个系统，如果行为完全不变，那说明那个系统根本没在起作用 ——
#   那时候"果蝇在开车"就是假的。
#
#   ★★ 最关键的是「连接组」那一项：关掉之后用**随机数**代替神经网络的输出。
#      如果关掉前后行为差不多，那 211,577 个神经元就是装饰。
#      （对照实验的设计原则：**阴性对照**要能真的把处理去掉。）
#
# (属性名, 标签, 默认开, 说明)
TOGGLES = [
    ("ab_connectome", "连接组（真实神经元）", True,
     "★★★ 关掉 = 用随机数代替神经网络输出。这是**阴性对照** —— "
     "如果关掉前后行为差不多，那这 211,577 个神经元就只是装饰"),
    ("ab_dopamine", "多巴胺（奖赏 / 惩罚）", True,
     "关掉 = 奖赏惩罚打不进去，学不会。打死了不知道高兴，挨打了不知道躲"),
    ("ab_mushroom", "蘑菇体学习", True,
     "关掉 = 权重冻结（读取正常，但不再更新）。和上面那个的区别："
     "这个是「不改了」，上面是「信号本身没了」"),
    ("ab_vision", "视觉（复眼输入）", True,
     "关掉 = 瞎了。光流恒为 0，它只能靠残余驱动乱走"),
    ("ab_odor", "气味输入", True,
     "关掉 = 闻不到敌人（红环）。觅食驱动会掉一截"),
    ("ab_hunger", "饥饿内稳态", True,
     "关掉 = 饥饿值锁在 0.5 不动。行为失去「饿→扑→吃→饱」的节律"),
    ("ab_motor", "运动输出（按键）", True,
     "关掉 = 照样算、照样显示，但**一个键都不按**。"
     "用来区分「决策不对」还是「执行没生效」"),
]

TOGGLE_NAMES = {t[0] for t in TOGGLES}
WHITELIST |= TOGGLE_NAMES

# ---------------------------------------------------------------- 下拉选择
# ★ 键位和技能含义是**离散选择**，不是滑块 —— 用下拉框。
#   ★★ 两件事刻意分开（用户要求）：
#     · role_N      第 N 槽**是什么**（攻击/增伤/格挡/回血/聚怪/挂削弱/不用）
#     · key_skillN  第 N 槽**按哪个键**
#   换配队只改键，换打法则只改含义。
#
#   为什么含义要能配：果蝇从画面上看不出技能是干什么的（没有 mod），
#   只能把语义当**本能**写进去。不同配队、不同角色的技能组不一样，
#   写死一套就只对一种配队有效。
# ★ 只放名字，**不带解释**（用户要求）。
#   做法写在代码注释里就够了 —— 界面上每个选项后面挂一句
#   "主攻，永远在线"，读下来又长又吵，选的时候也帮不上忙。
#   想查的话看 mb_learn.innate_priority 的注释。
ROLE_OPTS = [
    ("attack", "攻击"),
    ("buff", "增伤"),
    ("block", "格挡"),
    ("heal", "回血"),
    ("gather", "聚怪"),
    ("weaken", "挂削弱"),
    ("none", "不用"),
]
KEY_OPTS = ["1", "2", "3", "4", "5", "6",
            "q", "e", "r", "f", "g", "x", "c", "v"]
ATK_OPTS = ["mouse_left", "mouse_right", "mouse_middle",
            "1", "2", "3", "4", "e", "q", "f", "r", "g"]
COMBO_OPTS = ["e", "q", "f", "r", "g", "x", "c"]

# (属性名, 标签, 选项, 说明)。属性名以 __hdr_ 开头的是分节标题，不是输入项。
SELECTS = [
    ("__hdr_role", "技能槽含义", None, ""),
    ("role_1", "技能① 含义", ROLE_OPTS, ""),
    ("role_2", "技能② 含义", ROLE_OPTS, ""),
    ("role_3", "技能③ 含义", ROLE_OPTS, ""),
    ("role_4", "技能④ 含义", ROLE_OPTS, ""),
    ("__hdr_key", "按键映射", None, ""),
    ("key_skill1", "技能① 键", KEY_OPTS, ""),
    ("key_skill2", "技能② 键", KEY_OPTS, ""),
    ("key_skill3", "技能③ 键", KEY_OPTS, ""),
    ("key_skill4", "技能④ 键", KEY_OPTS, ""),
    ("attack_key", "普通攻击 键", ATK_OPTS, "打怪用的键，鼠标或键盘都行"),
    ("combo_key", "连携技 键", COMBO_OPTS, "出现提示就按，无冷却"),
    ("key_ult", "大招 键序列", ["1,2,3,4", "4,3,2,1", "1,2", "3,4", "1", "2"],
     "轮换用 —— 长按同一个键放的是那个角色的专属大招，"
     "每个角色各自有冷却，果蝇看不出谁好了，所以轮换"),
]

SELECT_NAMES = {s[0] for s in SELECTS if not s[0].startswith("__hdr")}
WHITELIST |= SELECT_NAMES

_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>果蝇控制台</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{background:#0b0d12;color:#e8eef7;font:14px/1.6 -apple-system,"Segoe UI",
      "Microsoft YaHei",sans-serif;margin:0;padding:16px 20px 60px}
 h1{font-size:17px;margin:0 0 4px}
 .sub{color:#7c8aa0;font-size:12px;margin-bottom:14px}
 .wrap{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}
 .cols{flex:1 1 620px;min-width:520px}
 .side{flex:1 1 320px;min-width:300px;position:sticky;top:16px}
 fieldset{border:1px solid #1e2634;border-radius:8px;margin:0 0 14px;padding:10px 14px 14px}
 legend{color:#8fa3bd;font-size:12px;padding:0 6px}
 .row{display:grid;grid-template-columns:150px 62px 1fr;gap:10px;
      align-items:center;padding:3px 0}
 .row label{font-size:12.5px}
 .row .val{font-family:Consolas,monospace;color:#ffd479;text-align:right;font-size:12.5px}
 input[type=range]{width:100%;accent-color:#4ea1ff}
 .hint{grid-column:1/4;color:#6b7a90;font-size:11px;margin:-2px 0 4px 150px}
 .st{background:#111722;border:1px solid #1e2634;border-radius:8px;padding:12px 14px}
 .st div{display:flex;justify-content:space-between;font-size:12.5px;padding:2px 0}
 .st b{font-family:Consolas,monospace;color:#7ee787;font-weight:400}
 .banner{border-radius:8px;padding:10px 14px;font-size:13px;margin-bottom:12px}
 .ok{background:#16281a;border:1px solid #2c5233;color:#7ee787}
 .bad{background:#2a1717;border:1px solid #5a2c2c;color:#ff8f8f}
 .warn{background:#2a2417;border:1px solid #5a4a2c;color:#ffd479}
 .ctl{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
 button{background:#1b2433;color:#cfe0f5;border:1px solid #2b3a52;border-radius:6px;
        padding:9px 18px;cursor:pointer;font-size:13.5px;font-weight:600}
 button:hover:not(:disabled){background:#243044}
 button:disabled{opacity:.32;cursor:not-allowed}
 button.go{background:#17351f;border-color:#2f6b3d;color:#8ef0a4}
 button.pause{background:#3a2e14;border-color:#7a6224;color:#ffd479}
 button.stop{background:#3a1a1a;border-color:#7a3232;color:#ff9b9b}
 select{background:#1b2433;color:#cfe0f5;border:1px solid #2b3a52;border-radius:6px;
        padding:9px 10px;font-size:13px}
 .toast{position:fixed;right:18px;bottom:18px;background:#16281a;border:1px solid #2c5233;
        color:#7ee787;padding:8px 14px;border-radius:6px;opacity:0;transition:.2s;font-size:12.5px}
 .toast.show{opacity:1}
</style></head><body>
<h1>果蝇全脑驾驶终末地 · 控制台</h1>
<div class="sub">参数拖动即生效（下一拍）。<b>必须检测到终末地才能启动。</b></div>

<div id="banner" class="banner warn">正在检测终末地…</div>

<div class="ctl">
  <button id="bstart" class="go" onclick="ctl('start')">▶ 启动</button>
  <button id="bpause" class="pause" onclick="ctl('pause')">⏸ 暂停</button>
  <button id="bstop" class="stop" onclick="ctl('stop')">■ 停止</button>
  <select id="secs" title="运行时长">
    <option value="180">3 分钟</option>
    <option value="360" selected>6 分钟</option>
    <option value="600">10 分钟</option>
    <option value="1800">30 分钟</option>
    <option value="0">不限时</option>
  </select>
</div>

<div class="wrap">
 <div class="cols" id="cols"></div>
 <div class="side">
  <div class="st" id="st">连接中…</div>
  <div style="margin-top:14px;border-top:1px solid #1e2634;padding-top:12px">
   <div style="color:#8fa3bd;font-size:12px;margin-bottom:6px">
     记忆存档（最多 10 个 · 点数字存，右键读/删）</div>
   <div id="memslots" style="display:flex;flex-wrap:wrap;gap:4px"></div>
   <div id="memhint" style="color:#6b7a90;font-size:11px;margin-top:6px"></div>
  </div>
  <div style="margin-top:12px"><button onclick="location.reload()">刷新页面</button></div>
 </div>
</div>
<div class="toast" id="toast"></div>
<script>
const G = __GROUPS__, T = __TOGGLES__, S = __SELECTS__;
function build(){
  const c = document.getElementById('cols');
  // ★ 对照组（消融实验）放最前面 —— 它比调参重要
  const fs0 = document.createElement('fieldset');
  fs0.innerHTML = '<legend>对照组（消融实验）</legend>';
  for (const [k, lab, def, hint] of T){
    const r = document.createElement('div'); r.className='row';
    r.innerHTML = '<label style="color:#ffd479">'+lab+'</label>'+
      '<span></span>'+
      '<label style="display:flex;align-items:center;gap:8px;cursor:pointer">'+
      '<input type="checkbox" id="t_'+k+'" style="width:auto;accent-color:#7ee787">'+
      '<span style="font-size:11px;color:#6b7a90">开</span></label>';
    fs0.appendChild(r);
    if (hint){ const h=document.createElement('div'); h.className='hint';
               h.textContent=hint; fs0.appendChild(h); }
    r.querySelector('input').addEventListener('change',
      e => set(k, e.target.checked ? 1 : 0));
  }
  c.appendChild(fs0);

  // ---- 键位 / 技能含义（下拉框）----
  const fsS = document.createElement('fieldset');
  fsS.innerHTML = '<legend>键位与技能含义</legend>';
  for (const [k, lab, opts, hint] of S){
    if (k.startsWith('__hdr_')){          // 分节标题
      const h = document.createElement('div');
      h.textContent = lab;
      h.style.cssText = 'color:#8fa3bd;font-size:12px;margin:10px 0 2px';
      fsS.appendChild(h);
      continue;
    }
    const r = document.createElement('div'); r.className='row';
    let html = '<label>'+lab+'</label><span></span>'+
               '<select id="s_'+k+'" style="width:100%">';
    for (const o of opts){
      const val = (o instanceof Array) ? o[0] : o;
      const txt = (o instanceof Array) ? o[1] : o;
      html += '<option value="'+val+'">'+txt+'</option>';
    }
    html += '</select>';
    r.innerHTML = html;
    fsS.appendChild(r);
    if (hint){ const h=document.createElement('div'); h.className='hint';
               h.textContent=hint; fsS.appendChild(h); }
    r.querySelector('select').addEventListener('change',
      e => set(k, e.target.value));
  }
  c.appendChild(fsS);

  for (const [title, ps] of G){
    const fs = document.createElement('fieldset');
    fs.innerHTML = '<legend>'+title+'</legend>';
    for (const [k,lab,lo,hi,step,hint] of ps){
      const r = document.createElement('div'); r.className='row';
      r.innerHTML = '<label>'+lab+'</label>'+
        '<span class="val" id="v_'+k+'">-</span>'+
        '<input type="range" id="s_'+k+'" min="'+lo+'" max="'+hi+'" step="'+step+'">';
      fs.appendChild(r);
      if (hint){ const h=document.createElement('div'); h.className='hint';
                 h.textContent=hint; fs.appendChild(h); }
      const s = r.querySelector('input');
      s.addEventListener('input', ()=>{ document.getElementById('v_'+k).textContent =
        (+s.value).toFixed(dec(step)); });
      s.addEventListener('change', ()=> set(k, +s.value));
    }
    c.appendChild(fs);
  }
}
function dec(step){ const t=String(step); const i=t.indexOf('.'); return i<0?0:t.length-i-1; }
async function pull(){
  try{
    const j = await (await fetch('/api/get')).json();
    for (const k in j.params){
      const cb = document.getElementById('t_'+k);
      if (cb){ if (document.activeElement!==cb) cb.checked = !!j.params[k]; continue; }
      const se = document.getElementById('s_'+k);
      if (se && se.tagName === 'SELECT'){
        if (document.activeElement!==se) se.value = j.params[k];
        continue;
      }
      const s=document.getElementById('s_'+k); if(!s) continue;
      if (document.activeElement!==s) s.value = j.params[k];
      document.getElementById('v_'+k).textContent = (+j.params[k]).toFixed(dec(s.step));
    }
    document.getElementById('st').innerHTML = j.status;
    if (j.slots) renderSlots(j.slots);
    const b = document.getElementById('banner');
    b.className = 'banner ' + (j.game ? 'ok' : 'bad');
    b.innerHTML = j.game
      ? '已检测到终末地 <span style="opacity:.7">('+j.gameinfo+')</span>　'
        + (j.running ? '<b>运行中</b>' : '已停止')
      : '没有检测到终末地 —— <b>启动已禁用</b>。请先打开游戏，再刷新本页。';
    document.getElementById('bstart').disabled = !j.game || j.running;
    document.getElementById('bpause').disabled = !j.running;
    document.getElementById('bstop').disabled = !j.running;
  }catch(e){ document.getElementById('st').textContent='连不上 ('+e+')'; }
}
async function set(k,v){
  const r = await fetch('/api/set', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({k:k,v:v})});
  const j = await r.json();
  toast(j.ok ? (k+' = '+v) : ('拒绝: '+j.msg));
}
async function ctl(a){
  const body = {a:a};
  if (a==='start') body.seconds = +document.getElementById('secs').value;
  const r = await fetch('/api/ctl', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const j = await r.json();
  toast(j.ok ? ('已'+(a==='start'?'启动':a==='pause'?'暂停':'停止')) : ('失败: '+j.msg));
  setTimeout(pull, 400);
}
let tt;
function toast(m){ const e=document.getElementById('toast');
  e.textContent=m; e.classList.add('show'); clearTimeout(tt);
  tt=setTimeout(()=>e.classList.remove('show'),1400); }
function renderSlots(slots){
  const box = document.getElementById('memslots'); box.innerHTML='';
  for (const s of slots){
    const b = document.createElement('button');
    b.textContent = s.slot;
    b.title = s.empty ? ('槽位 '+s.slot+' 空 —— 点一下存入')
                      : ('槽位 '+s.slot+' · '+s.mtime+' · '+(s.n_edges||'?')+' 条边\n点=存 右键=读/删');
    b.style.padding='4px 9px';
    b.style.fontSize='12px';
    if (!s.empty){ b.style.background='#17351f'; b.style.borderColor='#2f6b3d'; }
    b.onclick = () => mem('save', s.slot);
    b.oncontextmenu = (e) => {
      e.preventDefault();
      if (s.empty) return;
      if (confirm('读取槽位 '+s.slot+'？（会换掉当前记忆）')) mem('load', s.slot);
    };
    box.appendChild(b);
  }
  const n = slots.filter(s=>!s.empty).length;
  document.getElementById('memhint').textContent =
    n + ' / ' + slots.length + ' 个已存' + (n? '   （右键槽位 = 读取）':'');
}
async function mem(a, slot){
  const r = await fetch('/api/mem', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify({a:a, slot:slot})});
  const j = await r.json();
  toast(j.ok ? ('槽位 '+slot+' 已'+(a==='save'?'保存':a==='load'?'请求读取':'删除')) : ('失败: '+j.msg));
  setTimeout(pull, 300);
}
build(); pull(); setInterval(pull, 1000);
</script></body></html>
"""


def start_tuner(args, port: int = 8791, status_fn=None, sync_hooks=None,
                verbose: bool = True, ctl: dict = None, game_fn=None,
                mem=None):
    """起一个本地控制台（调参 + 启动/暂停/停止 + 游戏检测）。返回网址。

    args       —— argparse.Namespace。白名单里的属性会被读写。
    status_fn  —— 无参函数，返回一段 HTML（实时状态读数）。可为 None。
    sync_hooks —— 每拍要同步一次的小函数列表（把 args 推到别的对象上）。
    ctl        —— 控制字典，约定 running / pause / stop 三个键。
                  ★ 只写标志，**不直接操作主循环** —— 主循环在自己的节拍上
                    检查。从 HTTP 线程里去动主循环的数据结构迟早出事。
    game_fn    —— 无参函数，返回 (在跑吗, 说明)。**启动按钮只在它为真时可用**，
                  后端也拦一道 —— 这是"必须检测到终末地才能运行"那条要求的落点：
                  前端禁用只是提示，后端不拦的话随便发个请求就绕过去了。
    """
    state = {"args": args, "status": status_fn, "hooks": sync_hooks or [],
             "ctl": ctl if ctl is not None else {}, "game": game_fn,
             # ★ 记忆存档回调：{"save": fn(slot)->bool, "load": fn(slot)->bool}
             #   做成回调而不是在这里直接动 forager —— HTTP 线程不碰主循环的数据结构。
             "mem": mem or {},
             # ★ 状态缓存。理由见 do_GET：HTTP 线程里**绝不能**调会阻塞的东西
             #   （find_endfield 要枚举窗口），否则请求会挂住，
             #   而且每次请求一条线程，很快就堆成几十条。
             "cache": {}, "lock": threading.Lock()}
    page = (_PAGE.replace("__GROUPS__", json.dumps(GROUPS, ensure_ascii=False))
                  .replace("__TOGGLES__", json.dumps(TOGGLES, ensure_ascii=False))
                  .replace("__SELECTS__", json.dumps(SELECTS, ensure_ascii=False)))

    def refresh_status():
        """算一次状态并放进缓存。**由主循环/界面线程调用**，不要在 HTTP 线程里调。"""
        st = ""
        if state["status"]:
            try:
                st = state["status"]()
            except Exception as e:
                st = f"状态回调出错：{e}"
        g_ok, g_info = False, ""
        if state["game"]:
            try:
                g_ok, g_info = state["game"]()
            except Exception as e:
                g_info = f"检测出错：{e}"
        with state["lock"]:
            state["cache"] = {"status": st, "game": bool(g_ok),
                              "gameinfo": g_info, "ts": time.time()}

    def snapshot() -> dict:
        out = {}
        for k in WHITELIST:
            v = getattr(args, k, None)
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, str):
                out[k] = v                 # ★ 下拉框存的是字符串（键名/角色名）
            elif isinstance(v, (int, float)):
                out[k] = round(float(v), 6)
        return out

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                      # 别把控制台刷满

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/api/get"):
                # ★ 只读缓存，**不在这个线程里做任何会阻塞的事**。
                #   （find_endfield 要枚举窗口；以前在这里直接调它，
                #     和界面线程抢，请求会挂住，线程越堆越多。）
                with state["lock"]:
                    cch = dict(state["cache"])
                if not cch or time.time() - cch.get("ts", 0) > 3.0:
                    refresh_status()          # 缓存过期才自己算一次
                    with state["lock"]:
                        cch = dict(state["cache"])
                st = cch.get("status", "")
                g_ok, g_info = cch.get("game", False), cch.get("gameinfo", "")
                c = state["ctl"]
                self._send(200, json.dumps(
                    {"params": snapshot(), "status": st,
                     "game": bool(g_ok), "gameinfo": g_info,
                     "running": bool(c.get("running", False)),
                     "slots": list_slots(), "maxslots": MAX_SLOTS},
                    ensure_ascii=False),
                    "application/json; charset=utf-8")
            else:
                self._send(200, page)

        def do_POST(self):
            # ---------- 记忆存档：存 / 读 / 删 ----------
            if self.path.startswith("/api/mem"):
                mem = state.get("mem") or {}
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    d = json.loads(self.rfile.read(n) or b"{}")
                    act, slot = d.get("a"), int(d.get("slot", 0))
                except Exception as e:
                    self._send(200, json.dumps({"ok": False, "msg": str(e)}))
                    return
                if not (1 <= slot <= MAX_SLOTS):
                    self._send(200, json.dumps(
                        {"ok": False, "msg": f"槽位要在 1~{MAX_SLOTS} 之间"}))
                    return
                try:
                    if act == "save":
                        fn = mem.get("save")
                        if fn is None:
                            self._send(200, json.dumps(
                                {"ok": False, "msg": "这一轮没有可存的记忆"}))
                            return
                        ok = bool(fn(slot))
                        self._send(200, json.dumps(
                            {"ok": ok, "msg": "" if ok else "存失败（看日志）"}))
                    elif act == "load":
                        # ★ 读档要主循环来做（会换掉蘑菇体权重），
                        #   这里只挂一个待办，主循环下一拍执行。
                        state["ctl"]["load_slot"] = slot
                        self._send(200, json.dumps({"ok": True}))
                    elif act == "reset":
                        # ★ 清空记忆：挂待办，主循环执行（会换掉蘑菇体权重）
                        state["ctl"]["reset_mem"] = True
                        self._send(200, json.dumps({"ok": True}))
                    elif act == "delete":
                        pth = slot_path(slot)
                        if os.path.exists(pth):
                            os.remove(pth)
                        self._send(200, json.dumps({"ok": True}))
                    else:
                        self._send(200, json.dumps({"ok": False, "msg": "未知动作"}))
                except Exception as e:
                    self._send(200, json.dumps({"ok": False, "msg": str(e)}))
                return
            # ---------- 启动 / 暂停 / 停止 ----------
            if self.path.startswith("/api/ctl"):
                c = state["ctl"]
                if not c:
                    self._send(200, json.dumps({"ok": False, "msg": "没有控制通道"}))
                    return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    a = json.loads(self.rfile.read(n) or b"{}").get("a")
                except Exception as e:
                    self._send(200, json.dumps({"ok": False, "msg": str(e)}))
                    return
                print(f"  [HTTP] /api/ctl {a}", flush=True)
                if a == "start":
                    # ★★ 后端也拦一道：没有终末地就不许开。
                    #   前端把按钮禁掉只是提示，绕过前端直接发请求就失效了。
                    if state["game"]:
                        ok, info = state["game"]()
                        if not ok:
                            self._send(200, json.dumps(
                                {"ok": False, "msg": f"没有检测到终末地（{info}）"}))
                            return
                    c["running"] = True
                    c["pause"] = False
                    c["stop"] = False
                    self._send(200, json.dumps({"ok": True}))
                elif a == "pause":
                    c["running"] = False        # 主循环看到就停手，但不退出
                    self._send(200, json.dumps({"ok": True}))
                elif a == "stop":
                    c["running"] = False
                    c["stop"] = True            # 主循环看到就真的退出去
                    self._send(200, json.dumps({"ok": True}))
                else:
                    self._send(200, json.dumps({"ok": False, "msg": "未知动作"}))
                return
            if not self.path.startswith("/api/set"):
                self._send(404, "no")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n) or b"{}")
                k, v = d.get("k"), d.get("v")
                # ★ 只认白名单 —— 不接受任意属性名，免得网页上随便发个
                #   字符串就把 args 写坏（比如把 attack_key 写成数字）
                if k not in WHITELIST:
                    self._send(200, json.dumps({"ok": False, "msg": "不在白名单"}))
                    return
                lo = hi = None
                for _, ps in GROUPS:
                    for p in ps:
                        if p[0] == k:
                            lo, hi = p[2], p[3]
                # ★ 开关写 bool，滑块写 float。
                #   写错类型不会立刻出错（1.0 也是真值），但状态回调
                #   序列化出来会变成 1.0 而不是 true，界面上的勾会跳。
                if k in TOGGLE_NAMES:
                    setattr(state["args"], k, bool(v))
                elif k in SELECT_NAMES:
                    # ★ 下拉框：必须是选项之一。**不做类型转换、也不夹范围** ——
                    #   字符串选项没有越界的概念，只校验在不在表里。
                    opts = []
                    for _n, _lab, _o, _h in SELECTS:
                        if _n == k and _o:
                            opts = [x[0] if isinstance(x, tuple) else x for x in _o]
                    sv = str(v)
                    if opts and sv not in opts:
                        self._send(200, json.dumps(
                            {"ok": False, "msg": f"{k} 只能是 {opts}"}))
                        return
                    setattr(state["args"], k, sv)
                else:
                    v = float(v)
                    if lo is not None:
                        v = max(lo, min(hi, v))
                    setattr(state["args"], k, v)
                for h in state["hooks"]:
                    try:
                        h()
                    except Exception:
                        pass
                self._send(200, json.dumps({"ok": True, "v": v}))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "msg": str(e)}))

    refresh_status()          # 先填一次，免得第一个请求等到超时
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["refresh"] = refresh_status
    url = f"http://127.0.0.1:{port}/"
    # ★ 把刷新函数挂成属性 —— 界面线程可以定期调它，让 HTTP 端读到的
    #   永远是一份新鲜缓存，而不是每次请求都自己去枚举窗口。
    start_tuner.refresh = refresh_status
    refresh_status()
    if verbose:
        print(f"调参面板：{url}   （关了窗口就没了）")
    return url


def make_status_html(pairs):
    """把 [(标签, 值), ...] 拼成状态读数 HTML。"""
    rows = "".join(f"<div><span>{a}</span><b>{b}</b></div>" for a, b in pairs)
    return rows or "<div>—</div>"


# ---------------------------------------------------------------- 记忆存档
# ★ 果蝇学到的东西（蘑菇体 KC→MBON 稀疏补丁）存成一个个槽位。
#   为什么要 10 个而不是 1 个：**做对照实验要能回到同一个起点**。
#   跑一轮 → 存下来 → 关掉某个脑区再跑一轮：两次从同一份记忆出发，
#   差异才归因得到那个脑区上。只有一个存档的话，第二轮是从第一轮
#   学到的记忆开始的 ——「关掉的到底是脑区还是记忆」就分不清了。
MAX_SLOTS = 10


def mem_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "memory")
    os.makedirs(d, exist_ok=True)
    return d


def slot_path(n: int) -> str:
    return os.path.join(mem_dir(), f"slot_{int(n):02d}.npz")


def list_slots() -> list:
    """列出 10 个槽位。空的返回 empty=True，有内容的带元信息。"""
    out = []
    for i in range(1, MAX_SLOTS + 1):
        p = slot_path(i)
        if not os.path.exists(p):
            out.append(dict(slot=i, empty=True))
            continue
        try:
            st = os.stat(p)
            meta = {}
            try:
                z = np.load(p, allow_pickle=True)
                if "extra" in z:
                    e = z["extra"]
                    meta = dict(e.item()) if getattr(e, "shape", None) == () else {}
            except Exception:
                pass
            rec = dict(slot=i, empty=False, size=st.st_size,
                       mtime=time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime)))
            for k in ("rewards", "punish", "n_edges", "final"):
                if k in meta:
                    rec[k] = meta[k]
            out.append(rec)
        except Exception as e:
            out.append(dict(slot=i, empty=True, err=str(e)))
    return out


class LogTap:
    """把程序自己的 print 输出抄一份到环形缓冲里，给界面显示。

    为什么需要：出问题时只能说"卡住了 / 不动了"，而真正的原因全在
    stdout 里 —— 那些输出只有控制台窗口有，界面里看不到。
    装上这个，界面底部就能直接看到最近几十行，不用再去翻控制台。
    """

    def __init__(self, keep: int = 80):
        from collections import deque
        self.buf = deque(maxlen=keep)
        self._old = None

    def install(self):
        import sys
        if self._old is not None:
            return self
        self._old = sys.stdout
        tap = self

        class _T:
            def write(self, s):
                try:
                    if s:
                        for ln in str(s).splitlines():
                            if ln.strip():
                                tap.buf.append(ln.rstrip())
                except Exception:
                    pass
                try:
                    return tap._old.write(s)
                except Exception:
                    return len(s)

            def flush(self):
                try:
                    tap._old.flush()
                except Exception:
                    pass

            def __getattr__(self, k):
                return getattr(tap._old, k)

        sys.stdout = _T()
        return self

    def tail(self, n: int = 14):
        return list(self.buf)[-n:]

