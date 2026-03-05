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