import torch
import torch.nn.functional as F
from ultralytics.models.yolo.segment.val import SegmentationValidator

class TemporalSegmentationValidator(SegmentationValidator):
    def preprocess(self, batch):
        inputs = batch['img']
        if inputs.ndim == 5:
            b, t, c, h, w = inputs.shape
            batch['img'] = inputs.view(b * t, c, h, w)
            if 'temporal_data' in batch:
                all_cls, all_bboxes, all_masks, new_batch_idx = [], [], [], []
                for b_idx, seq in enumerate(batch['temporal_data']):
                    for t_idx, frame in enumerate(seq):
                        flat_idx = b_idx * t + t_idx
                        f_cls = frame['cls'].to(self.device)
                        if len(f_cls) > 0:
                            all_cls.append(f_cls); all_bboxes.append(frame['bboxes'].to(self.device))
                            all_masks.append(frame['masks'].to(self.device))
                            new_batch_idx.append(torch.full((len(f_cls),), flat_idx, device=self.device, dtype=torch.long))
                if all_cls:
                    batch['cls'], batch['bboxes'], batch['masks'] = torch.cat(all_cls), torch.cat(all_bboxes), torch.cat(all_masks)
                    batch['batch_idx'] = torch.cat(new_batch_idx)
            for k in ['im_file', 'ori_shape', 'resized_shape']:
                if k in batch: batch[k] = [v for v in batch[k] for _ in range(t)]
            if 'ratio_pad' in batch:
                batch['ratio_pad'] = [s for s in batch['ratio_pad'] for _ in range(t)]
            
            # 🌟 修复 1：同时兼容 list 和 tuple，并将真实的分类标签转为整型 (long)
            if 'roll_gt' in batch:
                rg = batch['roll_gt']
                if isinstance(rg, (list, tuple)):
                    self.current_roll_gt = torch.stack(rg).to(self.device).long().view(-1)
                else:
                    self.current_roll_gt = rg.to(self.device).long().view(-1)
                    
        return super().preprocess(batch)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'roll_loss'
        self.roll_loss = torch.tensor(0.0, device=self.device)

    def update_metrics(self, preds, batch):
        # 🚀 增加容错防崩溃保护
        # 如果 batch 里没有 ratio_pad（比如时序自定义数据），就强行塞一个默认值 (比例1.0，补边0.0)
        if "ratio_pad" not in batch:
            batch["ratio_pad"] = [((1.0, 1.0), (0.0, 0.0))] * len(batch["img"])
            
        return super().update_metrics(preds, batch)