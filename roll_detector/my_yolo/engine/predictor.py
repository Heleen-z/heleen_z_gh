"""
时序分割预测器
"""

from ultralytics.models.yolo.segment.predict import SegmentationPredictor


class TemporalSegmentationPredictor(SegmentationPredictor):
    """时序分割预测器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def construct_results(self, preds, img, orig_img, img_path):
        """构建结果"""
        results = super().construct_results(preds, img, orig_img, img_path)
        
        # 添加翻滚预测到结果
        if len(preds) > 3:
            roll_preds = preds[3]
            for result, roll_pred in zip(results, roll_preds):
                result.roll_preds = roll_pred
        
        return results
