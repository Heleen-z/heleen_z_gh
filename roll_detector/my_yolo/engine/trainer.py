import json
from pathlib import Path
import os
import torch
import copy
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from my_yolo.engine.validator import TemporalSegmentationValidator
from my_yolo.data.dataset import TemporalYOLODataset
from my_yolo.models.yolo.model import ROLL_SegmentationModel 

class TemporalSegmentationTrainer(SegmentationTrainer):
    
    def set_model_attributes(self):
        super().set_model_attributes()
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'roll_loss'
        self.roll_label_map = None 

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ROLL_SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'roll_loss'
        return TemporalSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, 
            args=copy.deepcopy(self.args), _callbacks=self.callbacks
        )

    # 🚨🚨🚨 就是这里！把之前不小心漏掉的方法加回来了！
    def build_dataset(self, img_path, mode, batch=None):
        # 🚀 从环境变量读取路径，绕过参数校验
        json_path = os.environ.get('COCO_TRAIN_JSON') if mode == 'train' else os.environ.get('COCO_VAL_JSON')
        t_window = int(os.environ.get('TEMPORAL_WINDOW', 5))
        
        return TemporalYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            data=self.data,
            mode=mode,            # 现在 __init__ 会正确处理它了
            temporal_window=t_window,
            coco_json_path=json_path
        )

    def _init_roll_label_map(self):
        """提前解析 JSON，建立 [图片名 -> 翻滚标签] 的查找表"""
        self.roll_label_map = {}
        data_cfg = self.data if hasattr(self, 'data') else {}
        for key in ['train_json', 'val_json']:
            json_path = data_cfg.get(key)
            if json_path and Path(json_path).exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                for img in data.get('images', []):
                    raw_rolling = img.get('rolling', 0)
                    is_rolling = 1 if raw_rolling is True or str(raw_rolling).lower() == 'true' or raw_rolling == 1 else 0
                    self.roll_label_map[img['file_name']] = is_rolling

    def preprocess_batch(self, batch):
        if self.roll_label_map is None:
            self._init_roll_label_map()
        
        roll_gts = []
        for path in batch['im_file']:
            filename = Path(path).name
            roll_gts.append(self.roll_label_map.get(filename, 0))
        batch['roll_gt'] = torch.tensor(roll_gts, device=self.device).long()

        inputs = batch['img']
        if inputs.ndim == 5:
            b, t, c, h, w = inputs.shape
            batch['img'] = inputs.view(b * t, c, h, w)
            
            # 🌟 绝杀修复：从 temporal_data 完美展开所有 5 帧的真实标签 🌟
            if 'temporal_data' in batch:
                all_cls, all_bboxes, all_masks, new_batch_idx = [], [], [], []
                for b_idx, seq in enumerate(batch['temporal_data']):
                    for t_idx, frame in enumerate(seq):
                        flat_idx = b_idx * t + t_idx  # 算出在 10 张图中的绝对索引
                        f_cls = frame.get('cls')
                        if f_cls is not None and len(f_cls) > 0:
                            all_cls.append(f_cls)
                            all_bboxes.append(frame.get('bboxes'))
                            # 兼容 Ultralytics 的 Masks 对象或纯 Tensor
                            f_mask = frame.get('masks')
                            if hasattr(f_mask, 'data'):
                                all_masks.append(f_mask.data)
                            elif f_mask is not None:
                                all_masks.append(f_mask)
                                
                            new_batch_idx.append(torch.full((len(f_cls),), flat_idx, dtype=torch.long))

                if all_cls:
                    batch['cls'] = torch.cat(all_cls).to(self.device)
                    batch['bboxes'] = torch.cat(all_bboxes).to(self.device)
                    batch['batch_idx'] = torch.cat(new_batch_idx).to(self.device)
                    if all_masks:
                        batch['masks'] = torch.cat(all_masks).to(self.device)
            
            # 复制图片名等元数据，防止画图报错
            for k in ['im_file', 'ori_shape', 'resized_shape']:
                if k in batch: 
                    batch[k] = [v for v in batch[k] for _ in range(t)]
            
            return super().preprocess_batch(batch)
            
        return super().preprocess_batch(batch)