"""
自定义评估指标
"""

import torch
import numpy as np
from ultralytics.utils.metrics import SegmentMetrics


class CustomSegmentMetrics(SegmentMetrics):
    """
    自定义分割评估指标，添加翻滚检测指标
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roll_metrics = RollMetrics()
    
    def process(self, tp, conf, pred_cls, target_cls, roll_preds=None, roll_targets=None):
        """处理预测结果"""
        super().process(tp, conf, pred_cls, target_cls)
        
        # 处理翻滚预测
        if roll_preds is not None and roll_targets is not None:
            self.roll_metrics.process(roll_preds, roll_targets)
    
    def results_dict(self):
        """返回结果字典"""
        results = super().results_dict()
        
        # 添加翻滚指标
        roll_results = self.roll_metrics.compute()
        results['roll_metrics'] = roll_results
        
        return results


class RollMetrics:
    """翻滚检测评估指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.predictions = []
        self.targets = []
        self.confidences = []
    
    def process(self, preds, targets):
        """处理一批预测结果"""
        # preds: [B, 3] 翻滚预测概率
        # targets: [B] 真实标签
        
        pred_classes = preds.argmax(dim=1).cpu().numpy()
        pred_conf = preds.max(dim=1)[0].cpu().numpy()
        target_classes = targets.cpu().numpy()
        
        self.predictions.extend(pred_classes.tolist())
        self.targets.extend(target_classes.tolist())
        self.confidences.extend(pred_conf.tolist())
    
    def compute(self):
        """计算评估指标"""
        if not self.predictions:
            return {
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
            }
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 基础分类指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision_macro': precision_score(targets, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(targets, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(targets, preds, average='macro', zero_division=0),
        }
        
        # 每个类别的指标
        for i, name in enumerate(['stable', 'suspect', 'roll']):
            mask = targets == i
            if mask.sum() > 0:
                metrics[f'{name}_precision'] = precision_score(
                    targets == i, preds == i, zero_division=0
                )
                metrics[f'{name}_recall'] = recall_score(
                    targets == i, preds == i, zero_division=0
                )
        
        return metrics
