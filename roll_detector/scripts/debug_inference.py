import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path

# --- 1. 强制本地路径优先 ---
current_file = Path(__file__).resolve()
project_root = str(current_file.parent.parent) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from my_yolo.models.yolo.model import ROLL_YOLO
    print(f"✅ 成功从本地加载模型类: {project_root}")
except ImportError as e:
    print(f"❌ 导入失败，请检查路径: {project_root}")
    raise e

# --- 2. Letterbox 预处理逻辑 ---
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

# --- 3. 诊断核心逻辑 ---
def debug_test():
    # 路径锁定为 v12
    model_path = "/workspaces/2d_roll/roll_detector/results/roll_final_training_env18/weights/best.pt"
    img_dir = "/workspaces/2d_roll/datasets/roll_detection/images/train"
    
    # 加载模型
    yolo = ROLL_YOLO(model_path)
    model = yolo.model
    
    # 🚀 核心实验：切换到训练模式推理
    # 这样做是为了绕过 BN 层在 eval 模式下可能存在的“坏账”统计量
    model.train() 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 准备 Sequence 5 正样本
    test_files = [
        "check_color_img_2026-01-30_17-18-21.576.jpg",
        "check_color_img_2026-01-30_17-18-21.622.jpg",
        "check_color_img_2026-01-30_17-18-21.722.jpg",
        "check_color_img_2026-01-30_17-18-21.822.jpg",
        "check_color_img_2026-01-30_17-18-21.982.jpg"
    ]

    frames = []
    for f_name in test_files:
        img_p = os.path.join(img_dir, f_name)
        img = cv2.imread(img_p)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox(img)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        frames.append(img)

    img_tensor = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0).to(device)

    # 4. 推理与特征探针
    with torch.no_grad():
        # 🚀 修正：在 train 模式下，结果是通过返回值直接拿到的
        # 按照你的 model.py: return x, roll_pred
        _, logit = model(img_tensor) 
        
        prob = torch.sigmoid(logit).item()

    print("-" * 50)
    if hasattr(model, 'roll_feat_debug'):
        feat = model.roll_feat_debug
        print(f"🔍 Layer 22 特征均值: {feat.mean().item():.4f}")
    
    print(f"📊(强制实时统计模式) 结果:")
    print(f"   预测 Logit: {logit.item():.4f}")
    print(f"   预测概率: {prob:.4%}")
    print("-" * 50)
    
    if prob > 0.1: # 只要概率比 0.007% 有明显提升
        print("💡 结论：找到了！概率提升说明 BN 层的全局统计量（坏账）是导致推理失效的主因。")
    else:
        print("💡 结论：即便使用实时统计，概率依然极低。说明训练时的特征映射与当前输入存在更深层的偏移。")

if __name__ == "__main__":
    debug_test()