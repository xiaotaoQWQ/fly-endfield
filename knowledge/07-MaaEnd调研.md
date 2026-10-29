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
