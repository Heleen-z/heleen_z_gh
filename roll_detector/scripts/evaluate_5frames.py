import sys
from pathlib import Path

# ================= 关键修复 1 =================
# 将项目根目录强制加入环境变量，让 Python 认识 'my_yolo'
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 必须在此处导入你的自定义类，接管模型加载过程
try:
    from my_yolo.models.yolo.model import ROLL_YOLO
except ImportError:
    print("❌ 无法导入 my_yolo，请确认脚本运行路径在 roll_detector 下。")
    sys.exit(1)
# ===============================================

import cv2
import numpy as np
import os

# ================= 物理阈值配置 =================
DISPLACEMENT_TH = 10.0  
ANGLE_DIFF_TH = 5.0     
# ===============================================

def get_obb_info(mask_points):
    """通过拟合椭圆获取更稳定的物理主轴角度和质心"""
    contour = np.array(mask_points, dtype=np.float32)
    # 拟合椭圆需要至少 5 个点，Mask 点阵通常有几十个，完全满足
    if len(contour) >= 5:
        (cx, cy), (axes_length), angle = cv2.fitEllipse(contour)
        # fitEllipse 返回的角度范围是 [0, 180)，更符合人类直觉
    else:
        # 极端情况降级（防崩溃）
        rect = cv2.minAreaRect(contour)
        (cx, cy), _, angle = rect
    return cx, cy, angle

def analyze_5_frames(model_path, frames_dir):
    print(f"Loading model from: {model_path}")
    
    # ================= 关键修复 2 =================
    # 不再使用原生的 YOLO，而是使用你的 ROLL_YOLO 类！
    # 放心，你在 _predict_once 里的逻辑会在推理时自动跳过时序梯度计算
    model = ROLL_YOLO(model_path)
    # ===============================================
    
    valid_exts = ['.jpg', '.jpeg', '.png']
    image_paths = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
                   if os.path.splitext(f)[1].lower() in valid_exts]
    image_paths.sort() 
    
    if len(image_paths) != 5:
        print(f"⚠️ 警告：找到了 {len(image_paths)} 张图片，期望是 5 张！")
    
    frame_data = []
    
    for idx, img_path in enumerate(image_paths):
        print(f"Processing frame {idx+1}: {os.path.basename(img_path)}")
        
        # 调低一点置信度，看看模型到底输出了什么
        results = model.predict(source=img_path, conf=0.25, verbose=False)[0]

        annoataed_frame = results.plot()
        cv2.imwrite(os.path.join(frames_dir, f"annotated_frame_{idx+1}.jpg"), annoataed_frame)
        
        # 优先寻找分割 Mask
        if results.masks is not None and len(results.masks) > 0:
            mask_points = results.masks.xy[0] 
            cx, cy, theta = get_obb_info(mask_points)
            frame_data.append({'frame': idx+1, 'cx': cx, 'cy': cy, 'theta': theta})
            print(f"   -> ✅ 找到轮廓! 坐标: ({cx:.1f}, {cy:.1f}), 角度: {theta:.1f}°")
            
        # 如果没有 Mask，但找到了 Box (降级情况)
        elif results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes.xywh[0].cpu().numpy() 
            cx, cy, w, h = box
            print(f"   -> ⚠️ 只找到了水平框(Box)，没有轮廓(Mask)！坐标: ({cx:.1f}, {cy:.1f})")
            frame_data.append({'frame': idx+1, 'cx': cx, 'cy': cy, 'theta': 0.0})
            
        else:
            print("   -> ❌ 什么都没检测到！")
            frame_data.append(None)
            
    # 后处理：综合 5 帧数据判断是否翻滚
    valid_data = [d for d in frame_data if d is not None]
    
    if len(valid_data) < 2:
        print("\n🤔 有效帧不足，无法计算运动状态。")
        return
        
    first_frame = valid_data[0]
    last_frame = valid_data[-1]
    
    dx = last_frame['cx'] - first_frame['cx']
    dy = last_frame['cy'] - first_frame['cy']
    displacement = np.sqrt(dx**2 + dy**2)
    
    d_theta = abs(last_frame['theta'] - first_frame['theta'])
    if d_theta > 45:
        d_theta = 90 - d_theta
        
    print("\n" + "="*40)
    print("📊 5帧综合运动学分析报告:")
    print(f"   总位移   : {displacement:.2f} 像素")
    print(f"   总角度差 : {d_theta:.2f} 度")
    
    is_moving = displacement > DISPLACEMENT_TH
    is_rotating = d_theta > ANGLE_DIFF_TH
    
    if is_moving and is_rotating:
        print("🚨 最终判定 : 翻滚 (ROLLING)!")
    elif is_moving and not is_rotating:
        print("➡️ 最终判定 : 平移滑动 (SLIDING)")
    else:
        print("🛑 最终判定 : 静止 (STATIC)")
    print("="*40 + "\n")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    
    # 指向你的自定义训练权重
    MODEL_PATH = "/workspaces/2d_roll/ckpts/yolo_11m_20260204_part.pt"
    FRAMES_DIR = root_dir / "test_5_frames"   
    
    analyze_5_frames(str(MODEL_PATH), str(FRAMES_DIR))