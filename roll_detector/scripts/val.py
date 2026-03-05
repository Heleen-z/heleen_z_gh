#!/usr/bin/env python3
"""
验证脚本
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from my_yolo import ROLL_YOLO


def validate():
    """验证模型性能"""
    
    # 加载训练好的模型
    model = ROLL_YOLO('results/roll_detection_exp/weights/best.pt')
    
    # 验证
    metrics = model.val(
        data='configs/train.yaml',
        batch=16,
        imgsz=640,
        conf=0.25,
        iou=0.6,
        device='0',
        save_json=True,
        save_hybrid=True,
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("验证结果")
    print("="*50)
    print(metrics)


if __name__ == '__main__':
    validate()
