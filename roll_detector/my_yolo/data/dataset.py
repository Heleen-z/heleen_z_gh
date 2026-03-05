import json
import torch
from pathlib import Path
from collections import defaultdict
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import LOGGER

class TemporalYOLODataset(YOLODataset):
    def __init__(self, *args, temporal_window=5, coco_json_path=None, data=None, **kwargs):
        # 1. 弹出 BaseDataset 不认识的参数，防止 TypeError
        self.mode = kwargs.pop('mode', 'train') 
        
        # 2. 保存你自定义的属性
        self.temporal_window = temporal_window
        self.coco_json_path = coco_json_path
        self.sequence_map = []
        
        # 3. 将剩余合规参数传给父类
        super().__init__(*args, data=data, **kwargs)
        
    def get_labels(self):
        self.use_segments = True 
        labels = super().get_labels()
        if self.coco_json_path and Path(self.coco_json_path).exists():
            self._build_temporal_sequences(labels)
        return labels

    def _build_temporal_sequences(self, labels):
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
                raw_rolling = info.get('rolling', 0)
                is_rolling = 1.0 if raw_rolling is True or str(raw_rolling).lower() == 'true' or raw_rolling == 1 else 0.0
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
                self.sequence_map[f['index']] = {'indices': window, 'rolling_label': f['rolling']}

    # 修改 my_yolo/data/dataset.py 中的 __getitem__
    def __getitem__(self, index):
        seq_info = self.sequence_map[index]
    
    # 获取时序窗口内的所有原始项
        temporal_items = [super(TemporalYOLODataset, self).__getitem__(i) for i in seq_info['indices']]
    
    # 重点：深拷贝最后一帧（基准帧）的所有元数据
        import copy
        base_item = copy.deepcopy(temporal_items[-1])
    
    # 堆叠图片并覆盖
        base_item['img'] = torch.stack([item['img'] for item in temporal_items])
    
    # 注入你的自定义标签
        base_item['roll_gt'] = torch.tensor([seq_info['rolling_label']], dtype=torch.float32)
    
    # 💡 确保这些关键字段存在且类型正确
    # base_item 已经包含了父类生成的 'ratio_pad', 'ori_shape', 'resized_shape'
    
        return base_item
    
    