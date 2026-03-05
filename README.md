# MY_YOLO: 基于 YOLO11 的 2D 时空序列包裹翻滚检测框架 (DL + 后处理双引擎)

## 1. 项目简介
my_yolo 是一个针对 Ultralytics YOLO11 框架的深度定制扩展，旨在解决传统目标检测模型缺乏“时间记忆”的痛点。本项目聚焦于 2D 图像序列中的包裹动态状态分析，精准判别连续 5 帧图像中包裹是否发生“翻滚 (Rolling)”动作。

本项目在技术演进上包含两个核心模块：
1. **端到端深度学习架构（实验与探索）**：在 YOLO 底层非侵入式地植入 1D 卷积时序头，探索神经网络跨帧捕捉特征梯度波动的能力。
2. **时序解耦后处理引擎（实际生产运行）**：针对极小样本场景下的过拟合物理定律，通过原生空间分割搭配几何运动学规则，实现高鲁棒性的生产级落地。

## 2. 核心特性
* **无缝时序组装 (Temporal Sequence Pipeline)**：支持将连续单帧 2D 图像拼装为时间窗口为 5 ($t\_window=5$) 的动作序列输入。
* **一维时序翻滚头 (1D Temporal Roll Head)**：在主干网络 C3k2 模块后方并联独立动作大脑，利用张量折叠与 1D 卷积捕捉 256 维语义特征的时序节律。
* **非侵入式 Monkey-patching**：定制 `TemporalLoss` 处理高类别不平衡，同时在验证系统中注入 `ratio_pad` 容错补丁，确保时序数据在 YOLO 原生管线中平稳运行不崩溃。
* **零样本泛化后处理 (Zero-Shot Temporal Post-processing)**：在实际运行中，摒弃黑盒依赖，采用基于长宽比反转、Mask 面积畸变的纯物理逻辑门，彻底根除小样本带来的过拟合问题。

## 3. 深度学习架构：端到端时序探索 (Deep Learning Engine)
在探索阶段，我们成功构建了时空联合网络，赋予了 YOLO 感知时间的能力：

* **特征截取与压缩**：在网络第 22 层 (Layer 22) 截取特征，获取维度为 [5, 256, 20, 20] 的高浓度语义特征图。使用 `AdaptiveAvgPool2d(1)` 将空间信息“压扁”，重构为 [5, 256] 的时间轴矩阵。
* **动作判决逻辑**：通过 `kernel_size=3` 的 `nn.Conv1d` 在时间轴上滑动，计算相邻帧之间的梯度差异。最后由全连接层 `Linear(64, 1)` 裁决输出翻滚 Logit。
* **抗不平衡机制**：定制了 `TemporalLoss`，引入极高的正样本惩罚权重 (`pos_weight`)，结合禁用时序破坏性增强 (`mosaic=0.0`, `mixup=0.0`)，强制模型在时空域学习。

## 4. 小样本物理限制与生产环境解耦 (Paradigm Shift)
尽管端到端时序网络在工程管线上完美闭环，但在实际应用中，我们遭遇了深度学习在小样本任务中的**物理限制极限**：

* **维度灾难与死记硬背**：在仅有 120 张图（折合约 24 个动作序列）的极限工况下，极其庞大的特征维度导致模型迅速掉入局部最优陷阱，不再学习“动作规律”，而是死记硬背像素坐标。
* **捷径学习 (Shortcut Learning)**：面对极度稀缺的正样本，模型发现“无脑输出负面预测”即可获得极低 Loss。

**工程折衷与降维打击：** 我们果断进行了架构解耦。让 YOLO11n-seg 回归其最擅长的**空间特征提取（抠图）**，而将跨帧的判断逻辑交接给**可解释性极强的后处理算法群**。

## 5. 实际运行：后处理逻辑的卓越设计 (Production Post-Processing)
本项目的生产环境核心在于脚本目录下的一系列评估组件。它们将“翻滚”动作降维打击为精确的几何运算，彻底摆脱了数据量的束缚：

1. **五帧联动评估 (`evaluate_5frames.py`)**：专门针对 $t\_window=5$ 的动作序列，提取原生 YOLO 的预测坐标与 Mask 轮廓，构建虚拟生命周期。
2. **几何拓扑畸变分析**：单纯的平移只会改变位置，而翻滚会引发 Mask 面积的剧烈收缩/扩张，以及 Bounding Box 长宽比例的非线性反转。
3. **确定性物理逻辑门**：后处理采用硬性阈值进行研判。只要动作符合物理学上的翻转规律，无论光照与背景如何变化，系统均能实现近乎 100% 的准确拦截与召回，生成详尽的 `evaluation_report.csv` 诊断记录。

## 6. 快速开始

### 6.1 安装依赖
```bash
pip install -r requirements.txt

```

### 6.2 深度学习时序网络训练（探索环境）

用于复现与研究 1D Conv 时序头的训练过程：

```bash
python scripts/train_roll.py

```

> **注意**：必须严格遵循 `batch=5` 对齐时间窗口，并开启 `freeze=22` 保护主干网络，配置安全的空间增强 (`fliplr=0.5, degrees=10.0`)。

### 6.3 基于后处理的生产级推理与评估（实际运行）

加载原生 YOLO 分割权重，利用后处理引擎对 5 帧图像序列进行多维度动作判别：

```bash
# 执行单序列核心推理逻辑
python scripts/evaluate_5frames.py

# 批量执行评估并生成 evaluation_report.csv
python scripts/batch_evalustes.py

```

## 7. 项目目录结构

```text
.
├── configs/                    # YOLO 训练与数据配置文件
│   ├── roll_data.yaml
│   ├── train.yaml
│   └── ...
├── my_yolo/                    # 🚀 核心框架组件 (DL探索环境)
│   ├── data/                   # 自定义时序数据加载
│   ├── engine/
│   │   ├── trainer.py          # 时序分割训练器
│   │   └── validator.py        # 兼容时序的定制验证器 (含 ratio_pad 补丁)
│   ├── models/yolo/            # 包含 TemporalRollHead 及 1D Conv 逻辑
│   └── utils/
│       ├── instances.py
│       └── losses.py           # 定制 TemporalLoss (高惩罚权重)
├── scripts/                    # 🚀 训练与实际生产后处理脚本
│   ├── train_roll.py           # 端到端 DL 模型训练入口
│   ├── debug_inference.py      # 模型单步诊断工具
│   ├── evaluate_5frames.py     # 基于几何/拓扑分析的单序列翻滚判决器 (后处理核心)
│   ├── batch_evalustes.py      # 批量序列跑批与报告生成器
│   └── interactive_eval.py     # 交互式结果确认与可视化
├── only_p_debug/               # 评估日志与可视化产出目录
│   ├── evaluation_report.csv
│   └── *_debug.jpg             # 诊断过程帧
├── test_5_frames/              # 样板测试序列目录
└── yolo11n-seg.pt              # 基础空间分割预训练权重

