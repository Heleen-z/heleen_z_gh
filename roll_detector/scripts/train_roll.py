import sys
import os
from pathlib import Path
from ultralytics import YOLO  
# --- 拦截 Ultralytics 的联网检查 ---
import ultralytics.utils.checks
# 将 check_font 替换为一个什么都不做的空函数
ultralytics.utils.checks.check_font = lambda *args, **kwargs: True
# 顺便拦截模型权重的自动下载（防止它找不到 yolo11n-seg.pt 时又卡死）
import ultralytics.utils.downloads
ultralytics.utils.downloads.safe_download = lambda *args, **kwargs: print("⚠️ 已禁用自动下载，请确保本地已有必要文件")
# -------------------------------

# 1. 定义项目根目录
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 2. 导入自定义模型类
try:
    from my_yolo.models.yolo.model import ROLL_YOLO
    print("✅ 成功加载自定义 ROLL_YOLO 框架")
except ImportError:
    print("❌ 导入失败：请确认 my_yolo 文件夹在项目根目录下")
    sys.exit(1)

def run_train():
    cfg_path = project_root / 'configs' / 'roll_data.yaml'
    
    # 🚀 使用环境变量存储路径，绕过 YOLO 参数校验
    os.environ['COCO_TRAIN_JSON'] = "/workspaces/2d_roll/datasets/roll_detection/instances_Train.json"
    os.environ['COCO_VAL_JSON'] = "/workspaces/2d_roll/datasets/roll_detection/instances_Val.json"
    os.environ['TEMPORAL_WINDOW'] = "5"

    model = ROLL_YOLO("yolo11n-seg.pt") 
    
    # model.train 只放官方允许的参数
    model.train(
        data=str(cfg_path),
        epochs=10,
        lr0=0.001,
        batch=5,
        imgsz=640,
        device=0,
        freeze=22,
        project=str(project_root / 'results'),
        name='roll_final_training_env',
        workers=0,
        mosaic=0.0,
        mixup=0.0,
        val=False,
        plots=False,
        fliplr=0.5,      # 50% 概率左右翻转
        degrees=10.0,    # 随机轻微旋转 10 度
        translate=0.1,   # 画面随机平移 10%
        scale=0.2        # 画面随机缩放 20%
    )

if __name__ == "__main__":
    run_train()