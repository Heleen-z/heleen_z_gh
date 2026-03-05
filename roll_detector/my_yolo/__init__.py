# my_yolo/__init__.py
"""
My YOLO 扩展模块初始化
该文件负责将深层目录下的模型类暴露给外部调用
"""
import sys

# 1. 检查 ultralytics 是否安装
try:
    import ultralytics
except ImportError:
    print("❌ 错误: 未找到 ultralytics 库。请在虚拟环境中运行: pip install ultralytics")
    raise

# 2. 导出模型类
# 根据你的实际结构: my_yolo/models/yolo/model.py
# 我们使用相对导入从深层目录获取类
try:
    from .models.yolo.model import ROLL_YOLO
except ImportError as e:
    # 详细的错误提示，帮助定位问题
    print(f"❌ 导入错误: 无法从 my_yolo.models.yolo.model 导入 ROLL_YOLO")
    print(f"   Python 报错信息: {e}")
    print("   请确认文件 my_yolo/models/yolo/model.py 是否存在且没有语法错误。")
    
    # 最后的挽救措施：防止彻底崩溃，尝试回退查找（万一文件被移动到了根目录）
    try:
        from .model import ROLL_YOLO
        print("   ⚠️ 警告: 在 my_yolo 根目录找到了 model.py，已回退使用该版本。")
    except ImportError:
        # 如果到处都找不到，抛出原始错误
        raise e

__all__ = ['ROLL_YOLO']