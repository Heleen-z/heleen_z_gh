import torch
import sys
import os
from pathlib import Path

# --- 1. 强制本地路径优先 ---
current_file = Path(__file__).resolve()
project_root = str(current_file.parent.parent) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from my_yolo.models.yolo.model import ROLL_YOLO

def inspect():
    model_path = "/workspaces/2d_roll/roll_detector/results/roll_final_training_env11/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到文件: {model_path}")
        return

    print(f"📦 正在通过 ROLL_YOLO 引擎解析: {model_path}")
    
    # 🚀 直接利用你的 ROLL_YOLO 加载，它会自动处理权重映射
    yolo = ROLL_YOLO(model_path)
    
    # 获取 state_dict
    weights = yolo.model.state_dict()
    
    # 2. 搜索翻滚头相关的键
    roll_keys = [k for k in weights.keys() if 'roll_head' in k]

    print("-" * 50)
    if not roll_keys:
        print("❌ 警告：在当前加载的模型中没有发现任何 'roll_head' 权重！")
        print("💡 结论：权重文件 best.pt 并不包含翻滚检测分支。")
    else:
        print(f"✅ 成功提取到 {len(roll_keys)} 个翻滚头权重项！")
        
        # 3. 统计关键层的数值
        # 尝试查找 conv_seq 里的权重层
        sample_key = next((k for k in roll_keys if 'weight' in k), roll_keys[0])
        w = weights[sample_key]
        
        print(f"📊 抽样层: {sample_key}")
        print(f"📊 权重均值: {w.mean().item():.6f}")
        print(f"📊 权重标准差: {w.std().item():.6f}")
        
        if w.std().item() < 1e-4:
            print("⚠️ 风险提示：权重数值几乎没有波动，说明该层并未学到有效信息。")
        else:
            print("✨ 权重数值分布正常，说明该分支已成功存入训练后的参数。")
    print("-" * 50)

if __name__ == "__main__":
    inspect()