import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path

# 1. 注册框架
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
import my_yolo.models.yolo.model 
from ultralytics import YOLO

# --- 配置 ---
MODEL_PATH = "/workspaces/2d_roll/roll_detector/results/roll_final_training_env11/weights/best.pt"
SOURCE_DIR = "/workspaces/2d_roll/datasets/ann_data/images/Train"
#SOURCE_DIR = "../only_p/roll"
# 根据 YOLO11-Seg 的结构，40个通道中：0-3是坐标，4-n是分类
# 我们需要确认这 8400 个点中哪个点的“包裹感”最强
CLASSES = ["BAG", "BOX", "MAIL"] 
SEQ_LEN = 5

def main():
    model = YOLO(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    img_files = sorted(list(Path(SOURCE_DIR).glob("check_color_img_*")))
    print(f">>> 开始检测（3通道模式匹配权重）...")

    for i in range(0, len(img_files) - SEQ_LEN + 1, SEQ_LEN):
        batch = img_files[i : i + SEQ_LEN]
        
        # 预处理：保持 3 通道
        frames = []
        for p in batch:
            img = cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (640, 640))
            frames.append(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1)))
        
        input_data = torch.from_numpy(np.stack(frames, axis=0)).to(device) # [5, 3, 640, 640]

        with torch.no_grad():
            # 1. 运行推理：results 拿到的是基础分割结果，roll_pred 藏在 model.model 里
            results = model.model(input_data) 
            
            # 2. 🚀 获取翻滚预测（这是二分类，输出只有一个值）
            # 由于你用了 5 帧，roll_pred 的形状应该是 [1, 1]
            roll_logit = model.model.roll_pred 
            roll_prob = torch.sigmoid(roll_logit).item() 

            # 3. 基础分割结果解析（只需处理最后一帧，即 index 0，因为我们做了特征切片）
            # results[0] 形状通常是 [1, 38+, 8400]
            preds = results[0] 
            
            # 找到包裹类别（BAG, BOX, MAIL）最高分的点
            # 索引 4:7 对应 CLASSES = ["BAG", "BOX", "MAIL"]
            pkg_scores = preds[0, 4:7, :] 
            max_score, best_point = torch.max(pkg_scores.max(dim=0)[0], dim=0)
            pkg_type = CLASSES[torch.argmax(pkg_scores[:, best_point]).item()]

            # 4. 判定
            is_rolling = roll_prob > 0.6  # 设定翻滚阈值

        # 5. 输出
        status = "【翻滚中】" if is_rolling else "【状态正常】"
        print(f"时间戳: {batch[0].name.split('img_')[-1]}")
        print(f"  └─ 识别结果: {pkg_type} | 状态: {status} (置信度: {roll_prob:.2%})")
        print("-" * 50)

if __name__ == "__main__":
    main()