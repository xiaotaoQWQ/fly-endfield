# 07 · 外部方案调研：MaaEnd / MaaFramework

> 结论先行：**"换成 MaaEnd 的识别"在技术上不成立** ——
> 不是因为不好，而是因为它的识别**绑定它自己的画面假设**，搬不过来。
> 但它的**判据思路**确实比我们的强，值得借。
>
> 调研时间 2026-10-28。

---

## 一、它们是什么（核实过的事实）

| | [MaaEnd](https://github.com/MaaEnd/MaaEnd) | [MaaFramework](https://github.com/MaaXYZ/MaaFramework) |
|---|---|---|
| 定位 | 「明日方舟：终末地」自动化工具（完整应用）| 基于图像识别的自动化框架 |
| 许可 | **AGPL-3.0** | LGPL-3.0 |
| 语言 | C++ / Python | C++ |
| Stars | 4222 | 5082 |
| 仓库大小 | **252 MB** | — |
| 活跃度 | 活跃（2026-10 仍在推）| 活跃 |

**许可兼容**：我们也是 AGPL-3.0 ✓ 可以复用代码和资源 ✓

**MaaEnd 的规模**：`assets/resource/image/` 下 **2295 个模板**、
57 个任务分类（AutoFight / InScene / Common / …）✓

---

## 二、★ 为什么"换进来"不成立

### 2.1 `maafw` 装不上 Python 3.14

```
maafw 5.14.3   →  只有旧版 cpXX wheel，无 cp314   ✗
```

我们的环境是 **Python 3.14.7** ✓ —— `pip install maafw` 直接失败 ✗
**A 方案（整体集成 MaaFramework）被环境卡死。**

### 2.2 它是框架，不是识别库

MaaFramework 的主流程是**任务流水线（JSON pipeline）** ✓：

```json
"__AutoFightRecognitionComboUsable": {
    "recognition": "ColorMatch",
    "roi": [28, 657, 56, 4],
    "lower": [240, 180, 180], "upper": [255, 255, 255],
    "connected": true, "count": 30
}
```

**没有"给我一个模板，返回一个相关系数"这种 API** ✗ ——
它要你描述一整段任务 ✓。

而我们的循环是 **2.5 Hz「这一拍给我 5 个布尔」** ✓ —— **架构不匹配** ✗

### 2.3 它的输入层用不了

MaaFramework 有自己的 controller ✓ —— 但终末地的 ACE 会把标准注入路径
的键盘事件丢掉 ✓（见 04 篇）。要接我们的 Interception 得实现
`CustomControlUnit` ✓ —— 又是一层重写 ✓。

### 2.4 它的参数不能直接搬（★ 实测）

拿 MaaEnd `AutoFight/Recognition.json` 里调好的两条，在我们的录制上跑：

| 节点 | MaaEnd 的参数 | 我们的 428 帧实测 |
|---|---|---|
| `ComboUsable` | `roi[28,657,56,4]` · 白 `[240,180,180]~[255,255,255]` · count≥30 | **8 帧里只命中 1 帧** ✗ |
| `HasEnemy` | `roi[280,150,750,370]` · 红 `[240,40,80]~[255,80,120]` · count≥100 | **全部为 0** ✗ |

**原因（推断）**：ROI 假设的分辨率 / **游戏内 UI 缩放设置** / 游戏版本 /
场景（副本内 vs 大地图）都与我们不同 ✓。

**结论**：它的参数是**给它自己那套画面**调的 ✓ —— 搬到我们这里要**重新标定** ✓，
那就和"自己做"没有本质区别 ✗。

---

## 三、★ 但它的判据思路确实更强（这是真收获）

### 3.1 它用什么做识别

`AutoFight/Recognition.json` 的构成：

| 类型 | 数量 | 用途 |
|---|---:|---|
| **ColorMatch** | **7** | 能量条 · 敌人血条 · 连携技可用 / 冷却 |
| TemplateMatch | 6 | 干员图标 · 终结技 · 战技图标 · 连携技提示 |
| **OCR** | **2** | 干员等级、干员栏 —— **只用在文字上** |
| **NeuralNetworkDetect** | 2 | 敌人攻击、整体画面（ONNX）|
| Or / And | 3 | 组合逻辑 |

**★ 最重要的一条：MaaEnd 自己也不用 OCR 判断游戏状态。**

OCR 只用在"读干员等级 / 读干员栏"这种**真正的文字**上 ✓
状态判断（有敌人 / 连携可用 / 能量几格）全部走 **ColorMatch** ✓。

**所以"用 OCR 替换识别"这个方向是错的** ✓ ——
我们也试了 Windows 自带 OCR ✓ 在这类小字上读不稳 ✓（`4/5` 时好时坏 ✓）。

### 3.2 三个可以立刻借的东西

**① 用「最大连通域面积」代替「颜色占比」**

MaaEnd 的 `connected: true` + `count` 是
「**最大连通块**的像素数 ≥ N」✓ 不是「整块区域里红像素占比」✓

**为什么更强**：零散的同色噪点（伤害数字、特效）成不了大连通块 ✓
所以它天然抗噪 ✓ —— 而我们现在用的是**占比** ✓ 区域里任何同色东西都算进去 ✗

> 这正是「任务完成检测」三版翻车的根因之一（见 03 篇 3.6 节）✓

**② 连携技用「一条细白条」而不是「E 徽标模板」**

MaaEnd 的 `ComboUsable`：`roi [28,657,56,4]` —— **56×4 的一小条** ✓
判据是"这条里有没有 ≥30 个连通的近白色像素" ✓

**为什么更简单**：不用管徽标长什么样 ✓ 不用管分辨率重采样 ✓
**只需要标定一个位置 + 一个颜色区间** ✓

**③ `green_mask`**

模板匹配时把**绿色通道当遮罩** ✓ —— 只匹配形状不匹配颜色 ✓
**抗背景变色** ✓（同一个图标在战斗/非战斗状态下底色会变 ✓）

---

## 四、下一步可以做什么（按性价比排）

| | 做什么 | 收益 | 成本 |
|---|---|---|---|
| **①** | **把颜色判据从「占比」改成「最大连通域」** | 高 · 直接修抗噪 | 低 · 一个函数 |
| **②** | 连携技重标定成「细白条 + 连通域」 | 高 · 比 E 徽标稳 | 中 · 要重新标 ROI |
| **③** | 模板匹配加 `green_mask` | 中 · 抗背景 | 低 · 一个参数 |
| **④** | 复用 MaaEnd 的 `Common/Button/*` 模板 | 中 · 省抠图 | 低 · 但要重标位置 |
| ⑤ | 整体集成 MaaFramework | — | **不可行**（无 cp314 wheel）|
| ✗ | OCR 判状态 | — | **方向错**（MaaEnd 自己也不用）|

---

## 五、留下的参考数据

`_maa_probe/AutoFight/` 下保留了：
- `Recognition.json`（9809 B）—— **调好的 ROI 和颜色区间，值得逐条读**
- `Action.json`（5115 B）
- 6 个模板 PNG（`Combo` / `Skill` / `EndSkill` / `CharacterBar` / `EnemyTarget` / `MenuOperators`）

> ⚠ 这些是 MaaEnd 的 AGPL-3.0 内容 ✓ ——
> 如果复用，**必须保留 AGPL 署名** ✓ 这和我们的许可相容 ✓

---

## 六、★ 深挖：MaaEnd 的连携技判断到底怎么做的

（2026-10-28 追加，读了它的 Go 源码和模型仓库之后）

### 6.1 一句话

**一个 37.9 MB 的 ONNX 检测模型（YOLO 系），在每一帧上把整个战斗界面
的 UI 元素全部框出来并打标签，其中一个标签叫 CharacterComboActive。**

不是颜色、不是模板、不是 OCR。

### 6.2 完整链路

`
① 每帧整屏 → autofightv13.onnx（detect/AutoFight/）
                    ↓
② 输出一堆检测框：[{label, box, score}, ...]
   标签包括：角色头像 / 血条 / 能量等级 / 连携可用 / 敌人 / 敌人攻击预警 …
                    ↓
③ GetCharacterComboActive()
      = hasLabelInFrames(CharacterComboActive, 1, false)
                    ↓
④ if params.EnableCombo && GetCharacterComboActive():
        enqueueAction(ActionCombo)
                    ↓
⑤ ctx.RunAction(__AutoFightActionComboClick, ...)
     → key: 69   ← E 键
`

**关键代码**（gent/go-service/autofight/）：

`go
// screenanalyzer.go
func (sa *ScreenAnalyzer) GetCharacterComboActive() bool {
    return sa.hasLabelInFrames(LabelCharacterComboActive, 1, false)
}

// autofight.go:689
if params.EnableCombo && screenAnalyzer.GetCharacterComboActive() {
    enqueueAction(fightAction{action: ActionCombo})
}

// autofight.go:842
case ActionCombo:
    ctx.RunAction(__AutoFightActionComboClick, maa.Rect{600, 320, 80, 80}, ", nil)

// Action.json
__AutoFightActionComboClick: { action: ClickKey, key: 69 }   // E
`

### 6.3 模型在哪

MaaEnd/MaaEnd-AI（**git submodule**，AGPL-3.0，2.2 GB）：

| 模型 | 大小 | 用途 |
|---|---:|---|
| **detect/AutoFight/autofightv13.onnx** | **37.9 MB** | **战斗检测（连携/能量/敌人/预警）** |
| detect/ProtocolSpace/best.onnx | 10.6 MB | 协议空间 |
| ocr/det.onnx + ocr/rec.onnx | 9.9 + 21.1 MB | PaddleOCR 系 |
| map/cls.onnx | 23.2 MB | 地图分类 |

### 6.4 ★ 它有个很聪明的防抖

`go
func (sa *ScreenAnalyzer) hasLabelInFrames(label string, n int, unused bool, region ...maa.Rect) bool {
    // 在最近 n 帧里找这个标签，每帧最多算一次
    ...
    // 如果总帧数超过 1 且匹配帧数不超过总帧数的一半，认为不可靠，防止 yolo 识别异常
    if total > 1 && matchedFrames*2 <= total {
        return false
    }
}
`

**「少数服从多数」** —— 单帧误检不算数 ✓。
这正好对付我们那个老问题：**单帧检测到就触发，等于把识别噪声直接变成行为** ✓。

### 6.5 ⚠ 顺带纠正一处：Recognition.json 里那条 ColorMatch 是**死代码**

Recognition.json 里写着：

`json
__AutoFightRecognitionComboUsable: {
    recognition: ColorMatch,
    roi: [28, 657, 56, 4],
    lower: [240, 180, 180], upper: [255, 255, 255],
    connected: true, count: 30
}
`

看起来很合理 ✓ —— 但**实测：它在 utofight.go 和 screenanalyzer.go 里
完全没有被引用**（grep ComboUsable|ComboNotice|ComboCooldown → 0 处命中）✗

**结论**：那些是**旧版残留**（或给 ADB/安卓版本用的），PC 版实际走的是 ONNX 模型。

> 这条很重要 —— 我一开始就是被这段看起来很像的 JSON 误导了 ✗
> **验证要看被谁调用，不是定义了没有。**

### 6.6 对我们的可迁移性

| 件 | 能搬吗 |
|---|---|
| 那个 37.9 MB 的 ONNX 模型 | ⚠ 能下 ✓ 但要在 **Python 3.14** 上跑推理 —— onnxruntime 有 cp314 wheel ✓，**不需要 maafw** ✓ |
| 模型的**标签集** | ✓ 可以读（它的 label 定义在 screenanalyzer.go 的 const 里）|
| hasLabelInFrames 的**多数帧防抖** | ✓ **纯逻辑，能直接抄** |
| ComboUsable 那条 ColorMatch | ✗ **死代码，别抄** |

**最小可行方案**：onnxruntime（14.6 MB）+ utofightv13.onnx（37.9 MB）
→ 自己写一个轻量推理包装（不用 maafw）→ 拿 CharacterComboActive 标签 ✓

**但要注意**：模型的输入预处理（尺寸、归一化、letterbox）必须和它训练时一致 ✓
—— 这部分在 maafw 的 NeuralNetworkDetect 里 ✓ 要么读它的 C++ 源码，要么试错 ✓

---

## 七、★★ 关键核查：MaaFramework 到底怎么注入 / 怎么抓屏

（2026-10-29 追加，读了 `MaaWin32ControlUnit` 的源码）

### 7.1 输入：**它也用的是 Interception**

`source/MaaWin32ControlUnit/Input/` 目录：

| 文件 | 大小 | 方式 |
|---|---:|---|
| **`InterceptionInput.cpp`** | **20 KB** | ★ **Interception 驱动** |
| `AnchoredTouchInput.cpp` | 49 KB | 锚定鼠标 |
| `MessageInput.cpp` | 51 KB | PostMessage（后台）|
| `LegacyEventInput.cpp` | 6 KB | keybd_event / mouse_event |
| `BackgroundManagedKeyInput.cpp` | 12 KB | 后台按键 |
| `SeizeInput.cpp` | 8 KB | — |

接口（`InterceptionInput.h`）：

```cpp
class InterceptionInput : public InputBase
{
    explicit InterceptionInput(HWND hwnd);
    virtual bool click(int x, int y) override;
    virtual bool swipe(int x1, int y1, int x2, int y2, int duration) override;
    virtual bool click_key(int key) override;
    virtual bool key_down(int key) override;
    virtual bool key_up(int key) override;
    virtual bool scroll(int dx, int dy) override;
};
```

> ★ **重要修正**：我在 04 篇说「我们选 Interception 是对的」——
> **MaaFramework 的选择也是一样的**，所以**这一块它并不比我们强**。
> 它只是实现更完整（多了 swipe / scroll / input_text）。
>
> **我们现有的 `interception_py.py` 已经够用**，不需要换。

### 7.2 抓屏：DXGI，**但解决不了遮挡**

`source/MaaWin32ControlUnit/Screencap/`：

| 文件 | 大小 | 方式 |
|---|---:|---|
| `DesktopDupScreencap.cpp` | 11 KB | DXGI 桌面复制 |
| **`DesktopDupWindowScreencap.cpp`** | 4 KB | **DXGI + 裁到窗口矩形** |
| `FramePoolScreencap.cpp` | 16 KB | DXGI FramePool |
| `PrintWindowScreencap.cpp` | 5 KB | PrintWindow |
| `GdiScreencap.cpp` | 2 KB | GDI BitBlt（我们在用）|
| `ScreenDCScreencap.cpp` | 3 KB | 屏幕 DC |

`DesktopDupWindowScreencap`：

```cpp
class DesktopDupWindowScreencap : public DesktopDupScreencap {
    std::optional<cv::Mat> screencap() override;
    RECT get_window_client_rect_screen() const;    // 窗口客户区在屏幕上的位置
    RECT get_output_desktop_coordinates() const;
};
```

**它就是「复制整个桌面，然后裁剪到窗口矩形」**——
**所以遮挡问题依然存在**（窗口被盖住，抓到的还是盖住它的那种内容）。

> **「换 MaaFramework 就解决遮挡」是错的。**
> 但它**确实更快**（DXGI 是 GPU 直通，比 GDI BitBlt 快一个量级）。

### 7.3 ★ ONNX 预处理：完全可复现

`source/MaaFramework/Vision/NeuralNetworkDetector.cpp`：

```cpp
// ① letterbox：按比例缩放，保持宽高比（cv::INTER_AREA）
const double scale = std::min(inW / roi_w, inH / roi_h);
cv::resize(image, image, resized_size, 0, 0, cv::INTER_AREA);

// ② 补边到输入尺寸，填充 (114,114,114)  ← YOLO 标准灰
cv::copyMakeBorder(..., cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

// ③ 转张量
std::vector<float> input = image_to_tensor(image);
```

`source/MaaFramework/Vision/VisionUtils.hpp`：

```cpp
inline static std::vector<float> image_to_tensor(const cv::Mat& image)
{
    cv::Mat src = image.clone();
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);      // BGR → RGB
    cv::Mat chw = hwc_to_chw(src);                   // HWC → CHW
    cv::Mat chw_32f;
    chw.convertTo(chw_32f, CV_32F, 1.0 / 255.0);     // ÷255 → [0,1]
    ...
}
```

**完整链（7 步）**：

```
① ROI 裁剪
② letterbox: scale = min(inW/w, inH/h)，INTER_AREA 缩放
③ 补边到输入尺寸，填 (114,114,114)
④ BGR → RGB
⑤ HWC → CHW
⑥ ÷ 255.0 → float32
⑦ ★ 不减均值、不除标准差 —— 就是纯 /255
```

**后处理**（YOLO 标准）：

```cpp
int x = (center_x - w / 2 - pad_left) / scale;
int y = (center_y - h / 2 - pad_top) / scale;
```

**这 7 步用 PIL + numpy 就能复现，约 30 行，不需要 cv2。**

### 7.4 ★★ 最终结论：不需要 fork MaaEnd

```
        果蝇的脑（我们的，Python）
              ↓ 决策
        我们的执行层（Interception，已验证）
              ↑ 感知
        它的 ONNX 模型（autofightv13.onnx，37.9 MB）
              ↑
        onnxruntime（14.6 MB，cp314 有 wheel）
```

| 想要的 | 怎么做 | 需要改 MaaEnd 吗 |
|---|---|---|
| **用它的识别** | `onnxruntime` + `autofightv13.onnx` + 复现 7 步预处理 | **不需要** |
| **用它的执行** | 不需要 —— 它和我们一样用 Interception，我们已有 | 不需要 |
| **决策是果蝇** | 我们的主循环，原样 | 不需要 |
| **可视化** | 我们的叠加面板，原样 | 不需要 |

**整个方案只需要新增一个 `onnx_detect.py`，约 100 行。**

**依赖**：`onnxruntime`（14.6 MB，cp314 wheel）+ 模型（37.9 MB）

**唯一待确认**：模型的输入尺寸（`inW × inH`）和输出格式
（YOLOv8 的 `[1, 4+nc, N]`？）—— 要下载模型跑一次才知道。

---

## 八、★★★ 实测：模型跑通了（2026-10-29）

### 8.1 下载

`raw.githubusercontent.com` 在这台机器上**不通**（HTTP 000）。
用 GitHub API + raw accept 头可以 ✓：

```bash
curl -L -H "User-Agent: probe" -H "Accept: application/vnd.github.raw" \
  -o autofightv13.onnx \
  https://api.github.com/repos/MaaEnd/MaaEnd-AI/contents/detect/AutoFight/autofightv13.onnx
```

**注意**：36.2 MB 下载要 **10 分钟**（约 60 KB/s），
第一次用 `--max-time 600` 会被截断成 29.1 MB ✗ —— 要 `--max-time 900 --retry 3` ✓
校验：**必须是 37,960,805 B** ✓

### 8.2 模型结构

```
输入   images    [1, 3, 640, 640]   float32
输出   output0   [1, 26, 8400]      float32

producer:  pytorch
task:      detect
imgsz:     [640, 640]
opset:     17
nms:       False          ← ★ 要自己做 NMS
end2end:   False
description: Ultralytics YOLO11s model trained on ./NeuralNetworkDetect/AutoFight.v13i.yolov11
license:   AGPL-3.0
```

`26 = 4 (bbox cx,cy,w,h) + 22 (类别)`
`8400 = (640/8)² + (640/16)² + (640/32)² = 6400 + 1600 + 400`

### 8.3 ★ 22 个标签（从 metadata 直接读的，不是猜的）

```
 0  CharacterComboActive        连携技**提示激活**（该按 E 了）
 1  CharacterComboEmpty         连携能量条空
 2  CharacterComboFull          连携能量条满
 3  CharacterDied               角色阵亡
 4  CharacterHealthDangerous    血量危险
 5  CharacterHealthNormal       血量正常
 6  CharacterLevel              等级（换算血量百分比用）
 7  CharacterSelect             当前选中的角色
 8  EndSkillFull                终结技（大招）就绪
 9  EndSkillFullADB             （安卓版）
10  EnemyAccumulatingPower      敌人蓄力
11  EnemyAttackGroundCharging   敌人地面攻击蓄力
... （完整 22 个见模型 metadata）
```

### 8.4 实测检测结果（我们自己的录制，1280×720 帧）

`full_0040`：

| 标签 | 置信度 | 框 (x,y) w×h |
|---|---:|---|
| CharacterHealthNormal | 0.927 | (180,661) 62×7 |
| CharacterHealthNormal | 0.923 | (24,661) 63×7 |
| CharacterHealthNormal | 0.922 | (102,660) 63×8 |
| CharacterHealthNormal | 0.919 | (258,661) 62×7 |
| MenuOperators | 0.919 | (1223,19) 38×38 |
| **EnergyLevelFull** | 0.910 | (531,649) 72×9 |
| **CharacterComboFull** | 0.908 | (24,654) 63×7 |
| CharacterSelect | 0.905 | (48,598) 16×12 |

**★ `CharacterComboActive`（该按 E）出现在 `full_0022` / `full_0040`：**

```
框 (771,277) 27×20     ← 屏幕中偏右，就是那个 E 提示徽标
置信度 0.777 ~ 0.865
```

换算到内容坐标（2560×1440）：**(1542, 554) 54×40**
我们手标的 E 徽标在 **(1468, 637)** —— **同一个元素，我们的标定有偏差** ✓

**★ `CharacterComboFull`（能量条满）在底部左侧 4 条：**

```
(24,654) 62×6   (102,654) 61×6   (179,654) 61×6   (257,654) 61×6
```

**和 MaaEnd 那条 `ComboUsable` 的 ROI `[28,657,56,4]` 几乎完全重合** ✓
→ **所以那条 ColorMatch 不是死代码，是"被 ONNX 取代的旧实现"** ✓
（我在第 6.5 节说"死代码"不准确，这里更正 ✓）

### 8.5 速度（428 帧，CPU 单线程）

```
中位   96.2 ms
均值   99.2 ms
p95   121.5 ms
最大  131.6 ms

→ 2.5 Hz 预算 400 ms，占用 24.0%
→ 推理 96 ms + 整屏截图 87 ms = 183 ms，还有 217 ms 余量 ✓
```

### 8.6 428 帧的标签命中统计（阈值 0.35，按检测框计数）

| 标签 | 帧数 | 占比 |
|---|---:|---:|
| EnergyLevelEmpty | 282 | 65.9% |
| CharacterHealthNormal | 214 | 50.0% |
| **CharacterComboFull** | **196** | **45.8%** |
| **EndSkillFull** | **183** | **42.8%** |
| CharacterComboEmpty | 112 | 26.2% |
| EnergyLevelFull | 96 | 22.4% |
| CharacterSelect | 54 | 12.6% |
| CharacterLevel | 23 | 5.4% |
| EnemyTarget | 17 | 4.0% |
| EnemyAttackGroundDodge | 7 | 1.6% |
| **CharacterComboActive** | **6** | **1.4%** |

> `CharacterComboActive` 只有 1.4% ✓ —— 因为它是**瞬时的"该按了"提示** ✓
> 这正好符合"连携技提示只短暂出现"的观察 ✓

### 8.7 ★ 结论：这条路完全可行

| | 状态 |
|---|---|
| 模型能下载 | ✅ 37,960,805 B |
| onnxruntime 能装（cp314） | ✅ 1.31.1 |
| 预处理可复现 | ✅ 7 步，PIL+numpy |
| 输出能解析 | ✅ YOLOv11 `[1,26,8400]` + 自己做 NMS |
| **检测质量** | ✅ **远超我们手抠的模板**（一次给出 15 个框 + 标签 + 置信度 + 坐标） |
| 速度 | ✅ 96 ms / 400 ms 预算 |
| 需要 maafw 吗 | **不需要** ✓ |
| 需要 fork MaaEnd 吗 | **不需要** ✓ |

**下一步**：写 `onnx_detect.py`（约 150 行），接进主循环，
用它替代 `combat.py` 里的模板检测（红环 / 撤离 / 连携 / 血条）。

**依赖新增**：`onnxruntime==1.31.1`（14.6 MB）+ `autofightv13.onnx`（36.2 MB）

---

## 九、★★ 已接入：`onnx_detect.py`（2026-10-29）

### 9.1 新增文件

| 文件 | 作用 |
|---|---|
| **`onnx_detect.py`** | ONNX 推理模块（预处理 + NMS + 语义映射 + 自检）|
| **`download_model.py`** | 下载模型 + 校验字节数 + 验证能加载 |
| `models/autofightv13.onnx` | 模型本体（36.2 MB，**不进仓库**）|
| `requirements.txt` | 依赖清单（含 `onnxruntime>=1.31`）|

### 9.2 API

```python
from onnx_detect import OnnxDetect

det = OnnxDetect()            # 缺模型时 det.ok = False，静默降级
st  = det.step(frame_rgb)     # frame = 整帧内容区 HxWx3 uint8
# st["ok"]            bool   推理成功
# st["ms"]            float  耗时
# st["dets"]          list   [{label, score, box}, ...]
# st["combo_active"]  bool   ★ 连携技提示激活（该按 E 了）
# st["combo_full"]    [bool]*4   4 个角色的连携能量条满
# st["combo_empty"]   [bool]*4
# st["hp_danger"]     [bool]*4   血量危险
# st["hp_normal"]     [bool]*4
# st["end_skill"]     bool   ★ 终结技就绪
# st["enemy_attack"]  bool   敌人攻击蓄力
# st["energy"]        int    0=空 1=中 2=满
```

### 9.3 接入点（`endfield_fly.py`）

```
① 参数
   --onnx / --no-onnx      默认开；缺模型自动退回模板检测
   --onnx-conf 0.35        置信度阈值（实测真检测 0.77~0.93）
   --onnx-model PATH       模型路径

② 初始化（lf 之后）
   onnx_det = OnnxDetect(conf=args.onnx_conf)
   onnx_det.ok 为 False 时静默降级

③ 主循环（每拍一次，**必须在所有检测之前**）
   st_onnx = onnx_det.step(lf.img)
   → 写进 forager["onnx_*"]，面板/HTML 状态可读

④ 连携技判断（唯一被替换的一条）
   if st_onnx["ok"]:  m_combo = st_onnx["combo_active"]
   else:              m_combo = kmark.step(...)     # 退回模板
```

**为什么只替换连携技**：那一块用模板/颜色反复翻车（§40 抓错 UI，
§46–51 三版判据全翻车）。其它检测（红环 / 撤离 / 小地图）现在能用，
**不动就是不动**。

**模型没有的类**：红环、撤离面板、小地图 —— 这些继续用模板 ✓

### 9.4 踩的两个坑

**① 代码顺序错了** → `UnboundLocalError: st_onnx`

第一版把 ONNX 块插到了连携技判断**后面**（因为它锚在「任务完成」那段），
而连携技判断要用 `st_onnx` ✗

**教训**：插代码时要看**依赖方向**，不能只看锚点就近。

**② `fod` 还没绑定** → `UnboundLocalError: fod`

ONNX 块在循环靠前的位置，而 `fod`（觅食状态）是后面才绑定的 ✗
改用 `forager`（循环外就建好的）✓

### 9.5 性能

```
推理中位   96 ms
整屏截图   87 ms
合计      183 ms  /  400 ms 预算（2.5 Hz）→ 占用 46%
实测闭环  8.9 秒 18 拍 = 2.1 Hz（比 2.5 Hz 略慢，因为多了推理）
```

**可以接受**，但如果以后识别项变多，要考虑把推理放到独立线程
（和眼睛线程一样）。

### 9.6 打包

`build_exe.py`：

```python
DATA_DIRS = [
    ("graph", "graph"),
    ("templates", "templates"),
    ("tools", "tools"),
    ("models", "models"),      # ★ 36 MB ONNX 模型
]
```

打包时会打印是否包含模型；缺模型会警告「会退回模板检测」。

### 9.7 降级路径（重要）

**每一个环节都有降级**，任何一环缺失都不影响启动：

| 缺什么 | 后果 |
|---|---|
| `models/*.onnx` | `onnx_detect.ok=False` → 连携技退回模板检测 |
| `onnxruntime` | 同上 |
| `graph/` | **启动失败**（这个是真必需）|

---

## 十、★★★ 决定性验证：ONNX vs 旧模板，逐帧对比（2026-10-29）

### 10.1 方法

拿 428 帧真实录制，同一帧同时跑两套识别：

| 方案 | 实现 |
|---|---|
| 旧 | `combat.ComboMarker` + `templates/combo_e.npy`（30×28 → 缩到 15×14），相关 > 0.50 |
| 新 | `onnx_detect.OnnxDetect` 的 `combo_active`（模型 conf > 0.35）|

### 10.2 结果

```
两者都说有      2 帧  [0022, 0040]
只有 ONNX 说有  0 帧              ← ★ 没有一帧是 ONNX 独有的
只有模板说有    5 帧  [0025, 0032, 0046, 0049, 0050]
都说没有       421 帧
一致率 98.8%
```

**旧模板的相关值分布**：

```
中位   +0.366
p90    +0.435
最大   +0.645
≥0.5  只有 7 / 428 帧
```

### 10.3 ★ 人工核验：模板 5 次全是误报

把争议帧的搜索区裁出来看（`shots/combo_dispute.png`）：

| 帧 | 模板 | ONNX | 框里实际有什么 |
|---|---|---|---|
| 0022 | ✓ corr +0.616 | ✓ | **真的有 E 徽标** |
| 0040 | ✓ corr +0.606 | ✓ | **真的有 E 徽标** |
| 0025 | ✓ corr +0.514 | ✗ | **空的**（纯背景）|
| 0032 | ✓ corr +0.516 | ✗ | **空的** |
| 0046 | ✓ corr +0.529 | ✗ | **空的** |
| 0049 | ✓ corr +0.645 | ✗ | **Windows 桌面**（截图时切出去了）|
| 0050 | ✓ corr +0.582 | ✗ | **Windows 桌面** |

**结论：ONNX 全对，模板 5 次全错。**

### 10.4 这解释了历史上所有连携技 bug

| 症状 | 根因 |
|---|---|
| "特定角色 E 不释放" | 模板抓错了 UI 元素（§40）|
| **"连携技连按 11 次"** | **模板在 0.51~0.65 的边缘值上乱触发** |

那 5 次误报的相关值都在 **0.514~0.645** —— 刚好压过 0.50 阈值。
阈值再高一点会漏真提示，再低一点误报更多。
**这就是"用手工模板 + 单一相关值"这种判据的天花板。**

### 10.5 教训

**"一致率 98.8%" 听起来很好，但它是骗人的。**

428 帧里只有 2 帧真有提示 —— 所以"全说没有"就有 99.5% 准确率 ✗
**在这种类别极端不平衡的问题上，一致率/准确率完全不能用来评价判据。**

真正该看的是：
1. **误报率**（假阳）—— 模板 5/426 = 1.2%，ONNX 0/426 = **0%**
2. **漏报率**（假阴）—— 两者都是 0（ONNX 检出了全部 2 帧真提示）
3. **人工核验分歧帧** —— 这是唯一能定谁对的办法

> ⚠ **必做的一步：把分歧帧裁出来亲眼看。**
> 只看数字永远不知道谁对 —— 光看"一致率 98.8%"甚至会得出
> "模板也挺好"的错误结论。

### 10.6 顺带一个坑

第一次跑对比时，模板的相关值最高只有 0.346 ✗ —— 因为
**模板本体是 2560 尺度的 30×28，而喂进去的是 1280 裁图** ✗
模板比实际大了一倍 → 相关值全线偏低。

缩到 15×14 之后才正常（最大 0.645）✓

**模板的尺度必须和搜索区的尺度一致** —— 这和 §40.2 里
"step() 降采样"是同一个问题的两个方向。

---

## 11. 撞墙逃逸：三次翻车换来的正确做法（2026-10-29）

### 11.1 用户需求

> "只是想让果蝇知道前面有障碍，会绕路，不至于在一个卡很久，
>  这样观赏性很不会（好）。"

**不是导航** —— 只要能绕开、别卡太久。所以**不需要** MaaEnd 那套
640 KB C++ 的 navmesh（见第 12 节）。

### 11.2 原始实现为什么不管用

```
[73.0s] 40°  卡  MOVE          ← 撞上
[73.8s]      卡  WALL-ESC R    ← 转
[74.2s] 86°  卡  WALL-ESC R
[74.6s] 109° 卡  WALL-ESC R
[75.4s] 109°     MOVE          ← ★ 转完就退出，又朝目标走 → 回撞
```

`escape_ticks` 是**固定拍数**：转 3 拍就把控制权还给大脑，
而大脑立刻按趋光/趋味把方向拉回原处 → **反复撞同一堵墙**。

### 11.3 第一次修：改成「累计位移够了才退出」→ **引入死锁**

```python
if esc_run is None and stuck_now:
    esc_run = 0.0
if esc_run is not None:
    esc_run += od["dist"]
    if esc_run >= args.escape_dist:   # 40 地图像素
        esc_run = None
    else:
        esc = True
```

**但逃逸分支是这么写的**：

```python
if esc:
    # 撞墙反射：松开所有移动键，原地转
    ex.apply_raw(turn_dx=turn_dx)      # ← 一个移动键都不按！
```

**原地转 → `dist` 恒为 0 → 条件永远不满足 → esc 永久为真 → 无限转圈。**

用户看到的现象：**「会自己站在原地视野围着人物转」**

> ★★ **教训：改退出条件之前，先确认这个量在对应分支里会不会变。**
> `dist` 在"只转不走"的分支里恒为 0 —— 条件写得再对也成立不了。

### 11.4 第二次修：边转边走 + 超时

```
第 1 段（--escape-turn-sec 0.6s）：原地转，瞄新方向
第 2 段：转 + 按 W，朝新方向走
超时（--escape-max-sec 2.5s）：反向，试试另一边
```

### 11.5 第三次修：斜向走 + **不能清零进度**

**斜向**：原来加了个「连卡两次才斜向」的门槛，模拟显示要卡 **11 秒**才生效 ——
太慢。改成**恒斜向**（W+A 或 W+D，由 `esc_flip` 决定）——
被挡住的果蝇本来就会斜着挤，墙角里正前方堵着、斜着能出去。

**进度不能被清零**：模拟抓到

```
2.8s acc=25.0  ESC walk 斜D
3.2s acc= 0.0  ESC反向(-1)   ← ★ 25px 进度被清了
```

超时分支原本写 `esc_run = 0.0` —— **把已积累的位移丢掉了** →
永远到不了 `escape_dist` → 无限「反向 → 反向」循环。

**只重置计时器 `esc_t0`，进度 `esc_run` 必须保留。**

### 11.6 最终行为（离线模拟）

```
0.4s  ESC turn(斜D)      ← 先转 0.6 秒瞄方向
1.2s  ESC walk 斜D       ← 开始边转边走
2.8s  acc=25.0
3.2s  ESC反向(-1) acc保留 ← 2.5 秒超时，反向但保留进度
4.0s  acc=35.0  ESC walk 斜A
4.4s  ★走开了            ← 累计 40px → 退出
```

**4.4 秒脱困**，而不是无限转。

### 11.7 参数

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--escape-turn-sec` | 0.6 | 第 1 段只转多久 |
| `--escape-max-sec` | 2.5 | 最长逃逸，到点反向 |
| `--escape-dist` | 40.0 | 累计走够多少地图像素算走开了 |
| `--stuck-sec` | 3.0 | 静止多久算卡住 |

**保留的两个门控**（没动）：
- 只有**命令了移动键**时"没动"才算撞墙（主动停 / 只转视角不算）
- **战斗中不触发**（被敌人顶住是正常的）

### 11.8 这一轮的方法论收获

**三次翻车全部靠「离线模拟 + 读日志时间线」抓到，没有一次是靠上机试出来的。**

- 第一次（只转不走 → 死锁）：读逃逸分支的代码就看出来了
- 第二次（门槛太慢）：离线模拟 20 拍就暴露
- 第三次（清零进度）：离线模拟打到 acc=25→0 的跳变

**结论：改状态机之前先写一个 20 行的离线模拟。**
比上机跑 300 秒快 1000 倍，而且能看见每一拍。

### 11.9 最终实测数据（2026-10-29 第四轮）

| 轮次 | 标「卡」 | 测不准「!」 | WALL-ESC |
|---|---:|---:|---:|
| 原始 | **28.6%** | —— | **48%** |
| 修 1（三态判定，sharp>1.3） | 2.6% | 43.9% | 7.1% |
| **修 2（sharp>1.0）** | **4.9%** | **0%** | **13.6%** |

**修 2 之后测速 100% 可信**（`!` 占 0%）✓

**「卡」从 2.6% 升到 4.9% 是正常的** ——
上一轮的 2.6% 里有一部分是「测不准被跳过」所以偏低；
4.9% 才是真实的撞墙率。

**关键证据**：判定那一拍的 `speed = 1.51`，刚好压着 `min_speed = 1.5` ✓
说明阈值是准的（实测走速约 7 地图像素/秒）。

### 11.10 这一轮的根本教训（写给以后的自己）

**四次翻车，三次是同一类错误：把「没有数据」当成「数据说没有」。**

| # | 现象 | 错误 |
|---|---|---|
| 1 | 连携技一次提示按 4 次 | 电平触发当成边沿触发 |
| 2 | 大招一次没放 | `_hpok` 恒 False（数据源根本没开）|
| 3 | **站着原地转** | 只转不走 → `dist` 恒 0 → 「累计位移 ≥ 40」永不成立 |
| 4 | **没撞墙也触发** | `sharp` 低（小地图在转）→ 「测不准」被判成「没在动」|

**判据写出来之前，先问三句话**：

1. **这个量在这个分支里会不会变？**（`dist` 在只转不走的分支里恒 0）
2. **这个量读不到的时候，代码会当成什么？**（`sharp` 低 → 被当成"静止"）
3. **「没有」和「不知道」分开了吗？**（对 3 个状态的判定写成两分支 = 有 bug）

**还有一条方法论**：三次翻车全部是**离线模拟 + 读日志时间线**抓到的，
一次都不是上机试出来的。**改状态机先写 20 行离线模拟**，
比上机跑 300 秒快 1000 倍，而且能看见每一拍。
