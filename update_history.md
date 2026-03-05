### 2/12
#    尝试了光流分类与空间估计技术两种未成功
#    转而使用朝向和角度 优势：或许可以避免传送带移动带来的视角转换造成的估计错误

package_id=7
status=roll
method=feature_similarity
min_similarity=0.830647
pairs_similarities=(0,1)=0.9939,(1,2)=0.9969,(2,3)=0.9997,(3,4)=0.9998,(4,5)=0.9480,(5,6)=0.9832,(6,7)=0.9905,(7,8)=0.9874,(8,9)=0.9759,(0,2)=0.9923,(0,4)=0.9919,(0,9)=0.8306
  check_color_img_2026-01-30_19:00:34.033.jpg angle=29.1°
  check_color_img_2026-01-30_19:00:34.133.jpg angle=28.3°
  check_color_img_2026-01-30_19:00:34.233.jpg angle=28.4°
  check_color_img_2026-01-30_19:00:34.333.jpg angle=28.7°
  check_color_img_2026-01-30_19:00:34.996.jpg angle=28.8°
  check_color_img_2026-01-30_18:20:54.330.jpg angle=31.7°
  check_color_img_2026-01-30_18:20:54.402.jpg angle=33.0°
  check_color_img_2026-01-30_18:20:54.502.jpg angle=32.4°
  check_color_img_2026-01-30_18:20:54.602.jpg angle=33.1°

# 日志分类错误 -》函数中忽略了timediff作为负数的影响
 

13时
package_id=4
status=stable
method=fusion (angle_weight=0.8)
method=feature_similarity
min_similarity=0.980111
pairs_similarities=(0,1)=0.9981,(1,2)=0.9924,(2,3)=0.9949,(3,4)=0.9947,(0,2)=0.9949,(0,4)=0.9801
max_angle_change=2.70°
angle_threshold=6.0°
angles=['120.8°', '120.5°', '117.9°', '118.5°', '121.2°']
  check_color_img_2026-01-30_18:27:23.037.jpg angle=120.8°
  check_color_img_2026-01-30_18:27:23.149.jpg angle=120.5°
  check_color_img_2026-01-30_18:27:23.249.jpg angle=117.9°
  check_color_img_2026-01-30_18:27:23.289.jpg angle=118.5°
  check_color_img_2026-01-30_18:27:23.389.jpg angle=121.2°

package_id=2
status=stable
method=fusion (angle_weight=0.8)
method=feature_similarity
min_similarity=0.949925
pairs_similarities=(0,1)=0.9946,(1,2)=0.9953,(2,3)=0.9899,(3,4)=0.9893,(4,5)=0.9499,(0,2)=0.9886,(0,4)=0.9612,(0,5)=0.9709
max_angle_change=2.58°
angle_threshold=6.0°
angles=['116.5°', '117.3°', '114.7°', '116.1°', '114.8°', '113.6°']
  check_color_img_2026-01-30_18:22:08.711.jpg angle=116.5°
  check_color_img_2026-01-30_18:22:08.799.jpg angle=117.3°
  check_color_img_2026-01-30_18:22:08.899.jpg angle=114.7°
  check_color_img_2026-01-30_18:22:08.999.jpg angle=116.1°
  check_color_img_2026-01-30_18:22:09.099.jpg angle=114.8°
  check_color_img_2026-01-30_18:22:09.107.jpg angle=113.6°

两张图在roll中误判为stable，但是角度上的参数已经无法更大，同样的权重方面影响也不大（因为两4图min同样很高），判断为难以预测件


目前参数：

# 特征层索引 (0-23, 推荐试验: 4, 6, 9, 10, 13, 16, 19)
feature_layer: 13

# 相似度阈值
# 高于 threshold_stable -> Stable
# 低于 threshold_roll -> Roll
# 两者之间 -> Suspect
threshold_stable: 0.96
threshold_roll: 0.85

# 同一包裹的时间间隔阈值(秒), 超过则视为新包裹
time_gap_seconds: 0.7

# 跨帧比较：除相邻帧外，第一帧还与第几帧比较（1-based，如 3=第3帧）
# 相机约 0.1s 间隔时，跨多帧可放大翻滚信号
first_vs_frames: [3, 5]  # 第一帧 vs 第三帧、第一帧 vs 第五帧
first_vs_last: true      # 是否加入第一帧 vs 最后一帧（捕获全程变化）

# 模型推理参数
model:
  imgsz: 640
  conf: 0.5
  iou: 0.75
  retina_masks: true

# 角度判断配置（平移不变，适合传送带场景）
use_angle: true          # 启用角度判断
angle_threshold: 6.0    # 角度翻滚阈值（度）
angle_weight: 0.8



2/14
model：

import torch
import torch.nn as nn
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import SegmentationModel

class TemporalRollHead(nn.Module):
    def __init__(self, in_channels=256, t_window=5):
        super().__init__()
        self.t = t_window
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_seq = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * t_window, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        bt, c, h, w = x.shape
        b = bt // self.t
        x = self.pool(x).view(bt, c)
        x = x.view(b, self.t, c).transpose(1, 2)
        return self.conv_seq(x)

class ROLL_SegmentationModel(SegmentationModel):
    def __init__(self, cfg='yolo11n-seg.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.roll_head = TemporalRollHead(in_channels=256, t_window=5)
        self.roll_pred = None

    def forward(self, x, *args, **kwargs):
        # 截取特征
        feat = x
        for i in range(11): feat = self.model[i](feat)
        self.roll_pred = self.roll_head(feat)
        return super().forward(x, *args, **kwargs)

class ROLL_YOLO(YOLO):
    @property
    def task_map(self):
        # 【关键修复】使用绝对导入消除 VSCode 红线和运行错误
        from my_yolo.engine.trainer import TemporalSegmentationTrainer
        from my_yolo.engine.validator import TemporalSegmentationValidator
        return {
            'segment': {
                'model': ROLL_SegmentationModel,
                'trainer': TemporalSegmentationTrainer,
                'validator': TemporalSegmentationValidator,
            }
        }

train:
#!/usr/bin/env python3
"""
训练脚本 (最终稳定版)
功能：自动适配路径 + 强制单进程 + 开启验证
"""
import sys
import os
import yaml
from pathlib import Path
import warnings

# 忽略一些无关紧要的警告
warnings.filterwarnings("ignore")

# 1. 环境路径配置
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"📂 项目根目录: {project_root}")

try:
    from my_yolo import ROLL_YOLO
    print("✅ 成功导入 ROLL_YOLO 模型类")
except ImportError as e:
    print(f"\n❌ 无法导入 my_yolo 模块: {e}")
    sys.exit(1)


def train():
    # =========================================================
    # 1. 自动路径适配逻辑
    # =========================================================
    cfg_path = project_root / 'configs' / 'roll_data.yaml'
    dataset_dir_name = 'datasets/roll_detection'
    
    # 查找数据集真实目录
    possible_roots = [
        project_root / dataset_dir_name,
        project_root.parent / dataset_dir_name
    ]
    
    dataset_root = None
    for p in possible_roots:
        if p.exists():
            dataset_root = p
            break
            
    if not dataset_root:
        print(f"❌ 严重错误: 未找到数据集目录 '{dataset_dir_name}'")
        return

    print(f"✅ 定位到数据集: {dataset_root}")
    
    # =========================================================
    # 2. 生成配置文件
    # =========================================================
    print(f"🛠️  生成配置文件: {cfg_path}")
    
    cfg_dict = {
        'path': str(dataset_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 4,
        'names': ['BAG', 'BOX', 'MAIL', 'ROBOT'],
        'train_json': str((dataset_root / 'instances_Train.json').resolve()),
        'val_json': str((dataset_root / 'instances_Val.json').resolve()),
        
        # 自定义参数
        'temporal_window': 5,
        'time_gap': 2.0,
        'roll_loss_weight': 1.0,
        'temporal_consistency': 0.5,
        'feature_layer': 15,
        
        # 关闭增强 (时序数据通常不适合随机马赛克)
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    # 确保目录存在并写入
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg_dict, f, allow_unicode=True)
        
    print(f"✅ 配置已就绪。")

    # =========================================================
    # 3. 开始训练
    # =========================================================
    
    model_name = 'yolo11n-seg.pt' 
    print(f"📦 使用模型: {model_name}")
    model = ROLL_YOLO(model_name)
    
    try:
        model.train(
            data=str(cfg_path),            
            epochs=300,                    
            
            # 【重要】降低 Batch Size 防止显存溢出 (OOM)
            batch=2,                       
            
            imgsz=640,                     
            
            # 【核心修改】强制单进程加载
            # 这可以彻底解决 "Bus error" (共享内存不足) 和 "ConnectionResetError"
            workers=0, 
            
            device='0',                    
            project=str(project_root / 'results'), 
            name='roll_exp_final',
            
            mosaic=0.0,
            
            # 【开启验证】现在有了自定义 Validator，可以安全开启验证了
            val=True, 
            save=True,
        )
    except Exception as e:
        print(f"\n❌ 训练发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    train()

dataset:
import json
import torch
from pathlib import Path
from collections import defaultdict
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import LOGGER

class TemporalYOLODataset(YOLODataset):
    def __init__(self, *args, temporal_window=5, coco_json_path=None, data=None, **kwargs):
        self.temporal_window = temporal_window
        self.coco_json_path = coco_json_path
        self.sequence_map = []
        super().__init__(*args, data=data, **kwargs)
        
    def get_labels(self):
        self.use_segments = True 
        labels = super().get_labels()
        if self.coco_json_path and Path(self.coco_json_path).exists():
            self._build_temporal_sequences(labels)
        return labels

    def _build_temporal_sequences(self, labels):
        LOGGER.info(f"🔍 正在建立时序关联标注: {self.coco_json_path}")
        with open(self.coco_json_path, 'r') as f:
            data = json.load(f)
        
        img_info_map = {img['file_name']: img for img in data['images']}
        parcel_groups = defaultdict(list)
        
        for idx, label in enumerate(labels):
            img_path = Path(label['im_file']).name
            if img_path in img_info_map:
                info = img_info_map[img_path]
                pid = str(info.get('parcel_id', info.get('id', 'unknown')))
                ts = info.get('timestamp', info.get('_timestamp', 0))
                # 兼容处理 bool 和 int
                raw_rolling = info.get('rolling', 0)
                is_rolling = 1.0 if raw_rolling is True or raw_rolling == 1 else 0.0
                parcel_groups[pid].append({'index': idx, 'timestamp': ts, 'rolling': is_rolling})
        
        self.sequence_map = [None] * len(labels)
        for pid, frames in parcel_groups.items():
            frames.sort(key=lambda x: x['timestamp'])
            indices = [f['index'] for f in frames]
            
            for i, f in enumerate(frames):
                start = max(0, i - self.temporal_window + 1)
                window = indices[start : i + 1]
                if len(window) < self.temporal_window:
                    window = [window[0]] * (self.temporal_window - len(window)) + window
                
                self.sequence_map[f['index']] = {
                    'indices': window, 
                    'rolling_label': f['rolling']
                }

    def __getitem__(self, index):
        seq_info = self.sequence_map[index]
        # 【核心修改】加载 5 帧图片，并保留每帧各自的 labels (bboxes, masks, cls)
        temporal_items = [super(TemporalYOLODataset, self).__getitem__(i) for i in seq_info['indices']]
        
        imgs = torch.stack([item['img'] for item in temporal_items])
        
        # 以序列的最后一帧作为主样本，但要携带整个序列的标签信息
        base_item = temporal_items[-1].copy()
        base_item['img'] = imgs
        base_item['roll_gt'] = torch.tensor([seq_info['rolling_label']], dtype=torch.float32)
        
        # 将 5 帧的所有标签传回，供 Trainer 展平
        # 排除掉 item 里的 img 本身以节省内存
        sanitized_temporal = []
        for item in temporal_items:
            it = item.copy()
            if 'img' in it: del it['img']
            sanitized_temporal.append(it)
        
        base_item['temporal_data'] = sanitized_temporal
        return base_item


trainer:
import torch
import torch.nn.functional as F
from copy import copy
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from my_yolo.engine.validator import TemporalSegmentationValidator
from my_yolo.data.dataset import TemporalYOLODataset
from ultralytics.utils import LOGGER

class TemporalSegmentationTrainer(SegmentationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        data_cfg = self.data if hasattr(self, 'data') else {}
        return TemporalYOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=mode == "train", hyp=self.args, rect=False,
            stride=int(self.stride), pad=0.5, data=data_cfg,
            temporal_window=5,
            coco_json_path=data_cfg.get('train_json' if mode == 'train' else 'val_json')
        )

    def get_validator(self):
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'roll_loss'
        return TemporalSegmentationValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        # 1. 提取真值
        if 'roll_gt' in batch:
            if isinstance(batch['roll_gt'], list):
                self.current_roll_gt = torch.stack(batch['roll_gt']).to(self.device).float()
            else:
                self.current_roll_gt = batch['roll_gt'].to(self.device).float()

        # 2. 图片展平 [B, 5, 3, 640, 640] -> [B*5, 3, 640, 640]
        inputs = batch['img']
        if inputs.ndim == 5:
            b, t, c, h, w = inputs.shape
            batch['img'] = inputs.view(b * t, c, h, w)
            
            # 3. 【核心修复】平铺标签，解决 train_batch*.jpg 错位
            if 'temporal_data' in batch:
                all_cls, all_bboxes, all_masks, all_batch_idx = [], [], [], []
                
                # 遍历 Batch 里的每个序列
                for b_idx, seq in enumerate(batch['temporal_data']):
                    # 遍历序列里的每一帧
                    for t_idx, frame_label in enumerate(seq):
                        flat_idx = b_idx * t + t_idx # 计算展平后的新图像索引
                        if 'cls' in frame_label and len(frame_label['cls']) > 0:
                            num_obj = len(frame_label['cls'])
                            all_cls.append(frame_label['cls'])
                            all_bboxes.append(frame_label['bboxes'])
                            if 'masks' in frame_label: all_masks.append(frame_label['masks'])
                            all_batch_idx.append(torch.full((num_obj,), flat_idx))
                
                # 覆盖 Batch 里的旧标签，使其与展平后的图片 1:1 对应
                if all_cls:
                    batch['cls'] = torch.cat(all_cls).to(self.device)
                    batch['bboxes'] = torch.cat(all_bboxes).to(self.device)
                    batch['batch_idx'] = torch.cat(all_batch_idx).to(self.device)
                    if all_masks: batch['masks'] = torch.cat(all_masks).to(self.device)
            
            # 扩展元数据
            for k in ['im_file', 'ori_shape', 'resized_shape']:
                if k in batch: batch[k] = [v for v in batch[k] for _ in range(t)]
        
        return super(SegmentationTrainer, self).preprocess_batch(batch)

    def criterion(self, preds, batch):
        loss, loss_items = super().criterion(preds, batch)
        m = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(m, 'roll_pred') and hasattr(self, 'current_roll_gt'):
            # 【修复点】不再使用固定值，而是计算 BCE 损失
            actual_bce = F.binary_cross_entropy(m.roll_pred, self.current_roll_gt)
            roll_weight = 2.0 
            
            loss += actual_bce * roll_weight
            
            # 记录到日志的必须是计算后的动态值
            log_val = actual_bce.detach() * roll_weight
            loss_items = torch.cat([loss_items, log_val.unsqueeze(0)])
            
        return loss, loss_items

validator:
"""
时序分割验证器 - 最终稳定版
功能：
1. 解决 'TemporalSegmentationValidator' object has no attribute 'model' 报错。
2. 解决 'list' object has no attribute 'to' 报错，确保 masks 为拼接好的 Tensor。
3. 通过重构 batch_idx 解决展平后的索引对齐问题，消除 IndexError。
4. 保持 roll_loss 的验证统计与 2.0 权重。
"""
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOGGER

class TemporalSegmentationValidator(SegmentationValidator):
    """
    自定义验证器：确保 5 帧时序数据在验证环节的标签 1:1 对齐，并兼容 YOLO 损失计算
    """
    
    def preprocess(self, batch):
        """
        验证前预处理：将 [B, 5, ...] 展平，并同步重构标签。
        核心修复：将 cls、bboxes 和 masks 全部转换为拼接好的 Tensor。
        """
        inputs = batch['img'] # 形状 [B, 5, 3, 640, 640]
        
        if inputs.ndim == 5:
            b, t, c, h, w = inputs.shape
            
            # 1. 展平图像张量 -> [B*T, 3, 640, 640]
            batch['img'] = inputs.view(b * t, c, h, w)
            
            # 2. 同步重构所有真值标签
            if 'temporal_data' in batch:
                all_cls = []
                all_bboxes = []
                all_masks = []
                new_batch_idx = []
                
                # 遍历原始 Batch 中的每个包裹序列
                for b_idx, seq_labels in enumerate(batch['temporal_data']):
                    # 遍历序列中的每一帧 (5帧)
                    for t_idx, frame_label in enumerate(seq_labels):
                        # 计算该帧在展平后的全局图像索引 (0 到 B*T-1)
                        flat_img_idx = b_idx * t + t_idx
                        
                        # 获取该帧原始标签 (确保在正确设备上)
                        f_cls = frame_label['cls'].to(self.device)
                        f_bboxes = frame_label['bboxes'].to(self.device)
                        f_masks = frame_label['masks'].to(self.device)
                        
                        if len(f_cls) > 0:
                            all_cls.append(f_cls)
                            all_bboxes.append(f_bboxes)
                            all_masks.append(f_masks)
                            # 构建新的 batch_idx 映射到展平后的 flat_img_idx
                            new_batch_idx.append(torch.full((len(f_cls),), flat_img_idx, device=self.device, dtype=torch.long))
                
                # 拼接为 Tensor 以支持原生 YOLO 损失函数计算
                if all_cls:
                    batch['cls'] = torch.cat(all_cls)
                    batch['bboxes'] = torch.cat(all_bboxes)
                    batch['masks'] = torch.cat(all_masks)
                    batch['batch_idx'] = torch.cat(new_batch_idx)
                else:
                    # 处理空 Batch 情况
                    batch['cls'] = torch.zeros(0, device=self.device)
                    batch['bboxes'] = torch.zeros((0, 4), device=self.device)
                    batch['masks'] = torch.zeros((0, h, w), device=self.device)
                    batch['batch_idx'] = torch.zeros(0, device=self.device, dtype=torch.long)

            # 3. 扩展元数据以匹配图像数量 (im_file, ori_shape 等)
            for key in ['im_file', 'ori_shape', 'resized_shape']:
                if key in batch:
                    batch[key] = [item for item in batch[key] for _ in range(t)]
            
            # 4. 处理 ratio_pad
            if 'ratio_pad' in batch:
                rp = batch['ratio_pad']
                if isinstance(rp, torch.Tensor):
                    batch['ratio_pad'] = rp.repeat_interleave(t, dim=0)
                else:
                    batch['ratio_pad'] = [s for s in rp for _ in range(t)]

            # 5. 处理翻滚标签 roll_gt
            if 'roll_gt' in batch:
                if isinstance(batch['roll_gt'], list):
                    self.current_roll_gt = torch.stack(batch['roll_gt']).to(self.device).float()
                else:
                    self.current_roll_gt = batch['roll_gt'].to(self.device).float()

        # 6. 调用基类方法
        # 使用 super().preprocess 确保执行标准分割验证器的 float 转换和归一化
        return super().preprocess(batch)

    def init_metrics(self, model):
        super().init_metrics(model)
        # 在验证指标列表中加入自定义的 roll_loss
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'roll_loss'

    def update_metrics(self, preds, batch):
        """
        更新指标，此时 batch 已经是展平且对齐的格式
        """
        # 【关键修复】防御性获取模型实例，解决 AttributeError
        model = getattr(self, 'model', None)
        if model is None and hasattr(self, 'trainer'):
            model = getattr(self.trainer, 'model', None)
        
        if model:
            # 处理多显卡并行 (DDP) 情况下的模型封装
            m = model.module if hasattr(model, 'module') else model
            
            # 计算验证集的 roll_loss
            if hasattr(m, 'roll_pred') and hasattr(self, 'current_roll_gt'):
                roll_loss_val = F.binary_cross_entropy(m.roll_pred, self.current_roll_gt)
                # 记录验证损失，固定权重为 2.0
                self.roll_loss = roll_loss_val.detach() * 2.0
        else:
            # 如果实在找不到模型，设置一个默认值防止绘图逻辑报错
            self.roll_loss = torch.tensor(0.0, device=self.device)
            
        return super().update_metrics(preds, batch)






根据你的统计结果：50个正样本 vs 70个负样本。这说明你的数据集其实非常均衡（正样本占比约 42%），这对于训练来说是极好的消息。

那么，为什么 roll_loss 还会“飞速下降”呢？真相只有一个：严重的过拟合（Overfitting）。
你的总标注只有 120 条，对于 RTX 4090 D 和 YOLOv11 这样强大的组合来说，这 120 张图就像 120 个单词一样，模型在 3-5 轮内就把它们死记硬背下来了，而不是在学习翻滚的规律。

针对你要求的“C 的具体步骤”，我们需要通过以下三步“组合拳”，把模型从“死记硬背”拉回到“逻辑推理”：

步骤 1：修正 losses.py 中的权重
既然样本是均衡的，我们之前为了对抗想象中的不平衡而设置的 pos_weight=5.0 就太重了，这会加速 Loss 的坍塌。

修改位置： my_yolo/utils/losses.py

Python
# 将 pos_weight 从 5.0 改为更科学的 1.4（根据你的统计结果 70/50 = 1.4）
self.roll_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.4], device=device))
步骤 2：在 model.py 中加入“干扰项”（Dropout）
为了防止 4090 D 强大的算力瞬间“背下”这 120 张图，我们需要在翻滚头里加入 Dropout。它会在训练时随机关掉一半的神经元，强迫模型不能依赖某几张图的特定像素，而是寻找更通用的特征。

修改位置： my_yolo/models/yolo/model.py

Python
class TemporalRollHead(nn.Module):
    def __init__(self, in_channels=256, t_window=5):
        super().__init__()
        # ... 前面代码不变 ...
        self.conv_seq = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * t_window, 64),
            nn.ReLU(),
            # 🚀 插入这一行：强迫模型进行“泛化”学习
            nn.Dropout(0.5), 
            nn.Linear(64, 1) 
        )
步骤 3：调整训练策略（降速增稳）
由于数据集极小（仅 120 个样本），默认的学习率太高了，模型会像赛车一样直接冲出赛道。

修改位置： train_roll.py

Python
    model.train(
        # ... 原有参数 ...
        lr0=0.001,      # 🚀 关键：将起始学习率降低 10 倍（从 0.01 改为 0.001）
        warmup_epochs=5.0, # 🚀 增加预热轮数，让模型慢慢适应这 120 张图
        # ...
    )
💡 为什么这样做能救回 roll_loss？
拒绝死记硬背：Dropout 像是在考试时遮住了一部分课本，逼模型去理解翻滚的原理，而不是记下第几张图是翻滚。

梯度更平滑：降低 lr0 能让权重更新更细腻，避免 Loss 像之前那样“高台跳水”。

权重对齐：使用 1.4 的 pos_weight 符合你数据集的真实物理分布，Loss 的下降会反映真实的准确率提升



层索引: 0 | 模块: Conv
层索引: 1 | 模块: Conv
层索引: 2 | 模块: C3k2
层索引: 3 | 模块: Conv
层索引: 4 | 模块: C3k2
层索引: 5 | 模块: Conv
层索引: 6 | 模块: C3k2
层索引: 7 | 模块: Conv
层索引: 8 | 模块: C3k2
层索引: 9 | 模块: SPPF
层索引: 10 | 模块: C2PSA
层索引: 11 | 模块: Upsample
层索引: 12 | 模块: Concat
层索引: 13 | 模块: C3k2
层索引: 14 | 模块: Upsample
层索引: 15 | 模块: Concat
层索引: 16 | 模块: C3k2
层索引: 17 | 模块: Conv
层索引: 18 | 模块: Concat
层索引: 19 | 模块: C3k2
层索引: 20 | 模块: Conv
层索引: 21 | 模块: Concat
层索引: 22 | 模块: C3k2
层索引: 23 | 模块: Segment
