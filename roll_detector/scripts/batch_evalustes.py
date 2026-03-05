import sys
import os
import cv2
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import math

# ================= 关键修复 1：环境变量 =================
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from my_yolo.models.yolo.model import ROLL_YOLO
except ImportError:
    print("❌ 无法导入 my_yolo，请确认脚本运行路径在 roll_detector 下。")
    sys.exit(1)
# ========================================================

# ================= 物理阈值配置 =================
DISPLACEMENT_TH = 10.0  
# 外轮廓判定阈值：真正翻滚的角度累加通常大于25度
ANGLE_DIFF_TH = 25.0     
# 🌟 新增：内部特征匹配旋转阈值。因为 ORB 极其精准，只要它说转了，那就是真转了，阈值可以设得非常严谨
TEX_ANGLE_DIFF_TH = 15.0  
TIME_GAP_THRESHOLD = 1.0  
# ===============================================

def parse_timestamp_from_filename(filename):
    """从文件名中提取时间戳，兼容冒号和横杠"""
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}[:\-]\d{2}[:\-]\d{2}\.\d{3})", filename)
    if not match:
        return None
    time_str = match.group(1)
    try:
        return datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%S.%f")
    except ValueError:
        pass
    try:
        return datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S.%f")
    except ValueError:
        return None

def group_images_by_time(folder_path):
    """读取文件夹，按时间戳排序并分组"""
    valid_exts = ['.jpg', '.jpeg', '.png']
    image_data = []
    for f in os.listdir(folder_path):
        if os.path.splitext(f)[1].lower() in valid_exts:
            dt = parse_timestamp_from_filename(f)
            if dt is not None:
                image_data.append({"file": f, "time": dt, "path": os.path.join(folder_path, f)})
                
    image_data.sort(key=lambda x: x["time"])
    if not image_data: return []

    packages, current_package = [], [image_data[0]]
    for i in range(1, len(image_data)):
        delta = (image_data[i]["time"] - image_data[i-1]["time"]).total_seconds()
        if delta > TIME_GAP_THRESHOLD:
            packages.append(current_package)
            current_package = [image_data[i]]
        else:
            current_package.append(image_data[i])
    if current_package: packages.append(current_package)
    return packages

def get_obb_info(mask_points):
    """通过拟合椭圆获取稳定的物理主轴角度"""
    contour = np.array(mask_points, dtype=np.float32)
    if len(contour) >= 5:
        (cx, cy), _, angle = cv2.fitEllipse(contour)
    else:
        rect = cv2.minAreaRect(contour)
        (cx, cy), _, angle = rect
        if angle < 0: angle += 180
    return cx, cy, angle % 180

def analyze_package_sequence(model, image_paths, package_id, output_dir):
    """结合 YOLO 轮廓追踪 与 ORB 内部特征点匹配的融合分析"""
    print(f"\n📦 开始分析 [包裹 {package_id}] - 共 {len(image_paths)} 帧")
    frame_data = []
    
    # 初始化 ORB 特征提取器和匹配器 (工业级光照鲁棒性)
    orb = cv2.ORB_create(nfeatures=200)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    prev_kp, prev_des, prev_img = None, None, None

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"  -> 处理: {filename}")
        
        results = model.predict(source=img_path, conf=0.25, retina_masks=True, verbose=False)[0]
        
        if results.masks is not None and len(results.masks) > 0:
            mask_points = results.masks.xy[0] 
            cx, cy, obb_theta = get_obb_info(mask_points)
            
            # --- 🌟 工业级特征提取与跨帧仿射变换 ---
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 只提取包裹区域的掩码
            blank_mask = np.zeros_like(gray)
            cv2.fillPoly(blank_mask, [np.array(mask_points, dtype=np.int32)], 255)
            
            # 在包裹表面寻找 ORB 关键点和描述子
            kp, des = orb.detectAndCompute(gray, mask=blank_mask)
            
            step_tex_rotation = 0.0 # 当前帧相比上一帧的内部旋转角度
            annotated_frame = results.plot(boxes=False, masks=True)
            
            if prev_des is not None and des is not None and len(prev_des) > 5 and len(des) > 5:
                # 匹配当前帧和上一帧的特征点
                matches = matcher.match(prev_des, des)
                # 按照距离（匹配质量）排序，只取前 30 个最可靠的点对
                matches = sorted(matches, key=lambda x: x.distance)[:30]
                
                if len(matches) >= 5: # 至少需要5个点对才能进行可靠的几何变换计算
                    pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    pts_curr = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # 🚀 核心数学计算：求解二维仿射变换矩阵（自动剔除错误匹配）
                    # 它会算出从上一帧到这一帧，这块表面“平移+旋转+缩放”的最优解
                    matrix, inliers = cv2.estimateAffinePartial2D(pts_prev, pts_curr, method=cv2.RANSAC)
                    
                    if matrix is not None:
                        # 从仿射矩阵中提取出极其精确的纯物理旋转角度
                        # 矩阵格式为 [[cos(θ)*s, -sin(θ)*s, tx], [sin(θ)*s, cos(θ)*s, ty]]
                        step_tex_rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                        
                        # (可视化) 画出特征点和它们跨帧移动的轨迹！
                        for i, match in enumerate(matches):
                            if inliers[i][0] == 1: # 只画出被算法认定为“靠谱”的有效点
                                pt1 = (int(pts_prev[i][0][0]), int(pts_prev[i][0][1]))
                                pt2 = (int(pts_curr[i][0][0]), int(pts_curr[i][0][1]))
                                cv2.circle(annotated_frame, pt2, 4, (0, 0, 255), -1) # 当前位置：红点
                                cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 2) # 运动轨迹：黄线

            # 更新 prev 变量，供下一帧使用
            prev_kp, prev_des = kp, des
            
            save_path = os.path.join(output_dir, f"pkg{package_id}_frame{idx+1}_debug.jpg")
            cv2.imwrite(save_path, annotated_frame)
            
            frame_data.append({
                'frame': idx+1, 'cx': cx, 'cy': cy, 
                'obb_theta': obb_theta, 'step_tex_rot': step_tex_rotation
            })
        else:
            print("     ❌ 未检测到有效轮廓！")
            frame_data.append(None)
            
    # === 运动学综合分析 ===
    valid_data = [d for d in frame_data if d is not None]
    if len(valid_data) < 2: return
        
    first_frame, last_frame = valid_data[0], valid_data[-1]
    
    # 1. 计算总位移
    displacement = np.sqrt((last_frame['cx'] - first_frame['cx'])**2 + (last_frame['cy'] - first_frame['cy'])**2)
    
    # 2. 计算外轮廓累加旋转 (防180度陷阱)
    acc_obb_theta = 0.0
    for i in range(1, len(valid_data)):
        diff = abs(valid_data[i]['obb_theta'] - valid_data[i-1]['obb_theta']) % 180
        acc_obb_theta += (diff if diff <= 90 else 180 - diff)
        
    # 3. 🌟 计算绝对纹理旋转 (直接累加跨帧求出的步长旋转角)
    acc_tex_theta = sum(abs(d['step_tex_rot']) for d in valid_data)
            
    print("-" * 40)
    print(f"📊 [包裹 {package_id}] 运动学与 ORB面朝向报告:")
    print(f"   总位移           : {displacement:.2f} 像素")
    print(f"   外轮廓累加旋转   : {acc_obb_theta:.2f} 度")
    print(f"   内部ORB纹理旋转  : {acc_tex_theta:.2f} 度 (纯物理表面旋转)")
        
    # === 终极融合判定逻辑 ===
    is_moving = displacement > DISPLACEMENT_TH
    # 采用高精度 ORB 匹配后，纹理旋转累加如果超过 15 度，几乎 100% 是在翻滚！
    is_rolling = (acc_obb_theta > ANGLE_DIFF_TH) or (acc_tex_theta > TEX_ANGLE_DIFF_TH) 
    
    if is_moving and is_rolling:
        print("🚨 最终判定 : 翻滚 (ROLLING)! [ORB特征强关联支持]")
    elif is_moving and not is_rolling:
        print("➡️ 最终判定 : 平移滑动 (SLIDING)")
    else:
        print("🛑 最终判定 : 静止 (STATIC)")
    print("=" * 40)

if __name__ == "__main__":
    MODEL_PATH = root_dir / "results" / "roll_final_training_env18" / "weights" / "best.pt"  
    INPUT_DIR = "/workspaces/2d_roll/datasets/ann_data/images/Train"
    OUTPUT_DIR = root_dir / "only_p_debug"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🕒 正在扫描并解析时间戳...")
    packages = group_images_by_time(str(INPUT_DIR))
    print(f"✅ 扫描完毕！共识别到 {len(packages)} 个包裹序列。\n")
    
    if len(packages) > 0:
        model = ROLL_YOLO(str(MODEL_PATH))
        for i, pkg in enumerate(packages):
            paths = [item["path"] for item in pkg]
            analyze_package_sequence(model, paths, i+1, str(OUTPUT_DIR))