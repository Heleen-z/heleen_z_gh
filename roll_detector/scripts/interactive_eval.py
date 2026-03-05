import sys
import os
import cv2
import numpy as np
import re
import csv
from datetime import datetime
from pathlib import Path
import math

# ================= 关键修复：环境变量 =================
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
ANGLE_DIFF_TH = 25.0     
TEX_ANGLE_DIFF_TH = 15.0  
TIME_GAP_THRESHOLD = 1.0  
# ===============================================

def parse_timestamp_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}[:\-]\d{2}[:\-]\d{2}\.\d{3})", filename)
    if not match: return None
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
    valid_exts = ['.jpg', '.jpeg', '.png']
    image_data = []
    if not os.path.exists(folder_path):
        print(f"❌ 找不到输入文件夹: {folder_path}")
        return []
        
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
    contour = np.array(mask_points, dtype=np.float32)
    if len(contour) >= 5:
        (cx, cy), _, angle = cv2.fitEllipse(contour)
    else:
        rect = cv2.minAreaRect(contour)
        (cx, cy), _, angle = rect
        if angle < 0: angle += 180
    return cx, cy, angle % 180

def get_algorithm_prediction(model, image_paths):
    """运行核心算法，带有高级传感器置信度融合逻辑"""
    orb = cv2.ORB_create(nfeatures=200)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    prev_kp, prev_des = None, None
    frame_data = []
    annotated_frames = []

    for img_path in image_paths:
        results = model.predict(source=img_path, conf=0.25, retina_masks=True, verbose=False)[0]
        img = cv2.imread(img_path)
        
        if results.masks is not None and len(results.masks) > 0:
            mask_points = results.masks.xy[0] 
            cx, cy, obb_theta = get_obb_info(mask_points)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blank_mask = np.zeros_like(gray)
            cv2.fillPoly(blank_mask, [np.array(mask_points, dtype=np.int32)], 255)
            kp, des = orb.detectAndCompute(gray, mask=blank_mask)
            
            step_tex_rotation = 0.0 
            orb_valid = False  
            
            annotated = results.plot(boxes=False, masks=True)
            
            if prev_des is not None and des is not None and len(prev_des) > 5 and len(des) > 5:
                matches = matcher.match(prev_des, des)
                matches = sorted(matches, key=lambda x: x.distance)[:30]
                
                if len(matches) >= 5: 
                    pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    pts_curr = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    matrix, inliers = cv2.estimateAffinePartial2D(pts_prev, pts_curr, method=cv2.RANSAC)
                    
                    if matrix is not None:
                        step_tex_rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                        orb_valid = True  
                        
                        for i, match in enumerate(matches):
                            if inliers is not None and inliers[i][0] == 1:
                                pt1 = (int(pts_prev[i][0][0]), int(pts_prev[i][0][1]))
                                pt2 = (int(pts_curr[i][0][0]), int(pts_curr[i][0][1]))
                                cv2.circle(annotated, pt2, 4, (0, 0, 255), -1) 
                                cv2.line(annotated, pt1, pt2, (0, 255, 255), 2) 

            prev_kp, prev_des = kp, des
            annotated_frames.append(annotated)
            frame_data.append({'cx': cx, 'cy': cy, 'obb_theta': obb_theta, 'step_tex_rot': step_tex_rotation, 'orb_valid': orb_valid})
        else:
            annotated_frames.append(img)
            frame_data.append(None)
            
    valid_data = [d for d in frame_data if d is not None]
    if len(valid_data) < 2:
        return "UNKNOWN (Lack of valid frames)", annotated_frames, {}
        
    first_frame, last_frame = valid_data[0], valid_data[-1]
    displacement = np.sqrt((last_frame['cx'] - first_frame['cx'])**2 + (last_frame['cy'] - first_frame['cy'])**2)
    
    acc_obb_theta = 0.0
    for i in range(1, len(valid_data)):
        diff = abs(valid_data[i]['obb_theta'] - valid_data[i-1]['obb_theta']) % 180
        acc_obb_theta += (diff if diff <= 90 else 180 - diff)
        
    acc_tex_theta = sum(abs(d['step_tex_rot']) for d in valid_data)
    valid_orb_steps = sum(1 for d in valid_data if d.get('orb_valid', False))
            
    is_moving = displacement > DISPLACEMENT_TH
    
    if valid_orb_steps >= 2:
        is_rolling = acc_tex_theta > TEX_ANGLE_DIFF_TH
    else:
        is_rolling = acc_obb_theta > ANGLE_DIFF_TH
    
    pred = "ROLLING" if (is_moving and is_rolling) else ("SLIDING" if is_moving else "STATIC")
    details = {"disp": displacement, "obb_rot": acc_obb_theta, "tex_rot": acc_tex_theta}
    
    return pred, annotated_frames, details

def create_combined_view(pred, frames, details):
    """将多帧图片水平拼接，并在顶部添加信息面板"""
    target_width = 300
    resized_frames = []
    for f in frames:
        h, w = f.shape[:2]
        target_height = int(h * (target_width / w))
        resized_frames.append(cv2.resize(f, (target_width, target_height)))
        
    combo_img = np.hstack(resized_frames)
    
    bar_height = 100
    info_bar = np.zeros((bar_height, combo_img.shape[1], 3), dtype=np.uint8)
    
    color = (0, 0, 255) if pred == "ROLLING" else ((0, 255, 0) if pred == "SLIDING" else (255, 255, 255))
    
    title = f"AI Prediction: {pred}"
    stats = f"Disp: {details.get('disp', 0):.1f}px | OBB Rot: {details.get('obb_rot', 0):.1f} deg | ORB Rot: {details.get('tex_rot', 0):.1f} deg"
    instruction = "Press [y] Correct, [n] Wrong, [q] Quit in terminal."
    
    cv2.putText(info_bar, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(info_bar, stats, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(info_bar, instruction, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    return np.vstack((info_bar, combo_img))

if __name__ == "__main__":
    # ================= 路径配置 =================
    MODEL_PATH = root_dir / "results" / "roll_final_training_env18" / "weights" / "best.pt"  
    INPUT_DIR = Path("/workspaces/2d_roll/datasets/all_p")  
    OUTPUT_DIR = root_dir / "only_p_debug"                  
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 欢迎进入交互式评估系统")
    print("="*60)
    
    # 🌟 新增：强制要求输入版本号与修改日志
    version = input("👉 请输入本次评估的版本号 (例如 v1.1, 直接回车默认为 default): ").strip()
    if not version:
        version = "default"
        
    # 动态生成特定版本的 CSV，并设定一个总的历史日志文件
    EVAL_IMG_PATH = OUTPUT_DIR / "current_eval.jpg"
    CSV_REPORT_PATH = OUTPUT_DIR / f"evaluation_report_{version}.csv"
    HISTORY_LOG_PATH = OUTPUT_DIR / "evaluation_history.log"
    
    # 如果是一个全新的版本，要求输入修改内容并记录在案
    if not CSV_REPORT_PATH.exists():
        notes = input("👉 请输入本次代码/参数的修改内容 (例如 '优化了ORB判定逻辑'): ").strip()
        with open(HISTORY_LOG_PATH, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] 版本: {version} | 修改内容: {notes}\n")
    else:
        print(f"\n💡 发现版本号 [{version}] 已有评估记录，将跳过已评数据，继续进度...")
    # ============================================

    print(f"\n📁 数据源: {INPUT_DIR}")
    print(f"📄 当前写入报告: {CSV_REPORT_PATH.name}")
    print("="*60)
    
    packages = group_images_by_time(str(INPUT_DIR))
    if len(packages) == 0:
        print("💡 请检查 INPUT_DIR 路径是否正确，或文件夹内是否有图片。")
        sys.exit(0)
        
    model = ROLL_YOLO(str(MODEL_PATH))
    
    # 续传逻辑读取
    evaluated_ids = set()
    if CSV_REPORT_PATH.exists():
        with open(CSV_REPORT_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) 
            for row in reader:
                if row: evaluated_ids.add(row[0])
                
    csv_file = open(CSV_REPORT_PATH, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    if not evaluated_ids:
        csv_writer.writerow(['Package_ID', 'First_Image', 'Prediction', 'User_Eval_Correct'])
        csv_file.flush()

    correct_count = 0
    total_eval = 0

    for i, pkg in enumerate(packages):
        pkg_id = f"pkg_{i+1}"
        if pkg_id in evaluated_ids:
            continue 
            
        paths = [item["path"] for item in pkg]
        print(f"\n⏳ 正在分析 {pkg_id} ({len(paths)} 帧) ...")
        
        pred, frames, details = get_algorithm_prediction(model, paths)
        
        combined_img = create_combined_view(pred, frames, details)
        cv2.imwrite(str(EVAL_IMG_PATH), combined_img)
        
        print("-" * 40)
        print(f"📦 当前: {pkg_id} | AI 判定: [{pred}]")
        print(f"👀 请在侧边栏双击查看: {EVAL_IMG_PATH}")
        
        while True:
            user_input = input("👉 该判定是否正确？[y]=对, [n]=错, [q]=退出: ").strip().lower()
            if user_input in ['y', 'n', 'q']:
                break
            print("输入无效，请重新输入 'y', 'n' 或 'q'。")
            
        if user_input == 'q':
            print("⏹️ 已中断评估。")
            break
            
        is_correct = (user_input == 'y')
        if is_correct: correct_count += 1
        total_eval += 1
        
        csv_writer.writerow([pkg_id, os.path.basename(paths[0]), pred, is_correct])
        csv_file.flush()
        
    csv_file.close()
    
    print("\n" + "="*50)
    print(f"🎉 评估完成 / 结算报告 (版本: {version})")
    print("="*50)
    
    final_total = 0
    final_correct = 0
    if CSV_REPORT_PATH.exists():
        with open(CSV_REPORT_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                final_total += 1
                if row['User_Eval_Correct'] == 'True':
                    final_correct += 1
                    
    if final_total > 0:
        accuracy = (final_correct / final_total) * 100
        print(f"📊 总评估包裹数 : {final_total}")
        print(f"✅ AI 判断正确数 : {final_correct}")
        print(f"❌ AI 判断错误数 : {final_total - final_correct}")
        print(f"🎯 整体正确率 (Accuracy) : {accuracy:.2f}%")
        print(f"📁 详细记录已保存在: {CSV_REPORT_PATH.name}")
    else:
        print("未产生任何评估记录。")