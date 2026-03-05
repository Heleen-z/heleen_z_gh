#!/usr/bin/env python3
"""
数据集切分脚本 (最终全能版)
功能：
1. 基于时间戳聚类 (Gap < 1.0s) 进行包裹分组
2. 自动生成 Parcel ID (如 "001")
3. 【新增】将 COCO 标注转换为 YOLO 格式 (.txt) 并保存到 labels/ 目录
4. 递归搜索图片路径
"""
import json
import shutil
import random
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# ================= 配置区域 =================
# 原始数据路径
SOURCE_ROOT = Path("datasets/ann_data")
SOURCE_IMG_DIR = SOURCE_ROOT / "images"
SOURCE_ANN_DIR = SOURCE_ROOT / "annotations"

# 目标输出路径
TARGET_ROOT = Path("datasets/roll_detection")
TARGET_IMG_TRAIN = TARGET_ROOT / "images/train"
TARGET_IMG_VAL = TARGET_ROOT / "images/val"
TARGET_LBL_TRAIN = TARGET_ROOT / "labels/train"
TARGET_LBL_VAL = TARGET_ROOT / "labels/val"

# 切分比例
VAL_RATIO = 0.2
SEED = 42
TIME_GAP_THRESHOLD = 1.0

# YOLO 类别名称映射 (必须与 train.yaml 中的 names 顺序一致)
# 如果 JSON 中的 category_name 不在这里，将被忽略或报错
YOLO_NAMES = ['BAG', 'BOX', 'MAIL', 'ROBOT']
# ===========================================

def parse_timestamp_from_filename(filename):
    """从文件名解析时间戳"""
    pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)"
    match = re.search(pattern, filename)
    if match:
        time_str = match.group(1)
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S.%f")
            return dt.timestamp()
        except ValueError:
            pass
    return 0.0

def find_image_file(filename, search_dir):
    """递归查找图片文件"""
    p_obj = Path(filename)
    if p_obj.is_absolute() and p_obj.exists():
        return p_obj

    direct_path = search_dir / filename
    if direct_path.exists():
        return direct_path
        
    if not hasattr(find_image_file, "_dir_cache"):
        print(f"🔍 正在建立文件索引 (递归搜索 {search_dir})...")
        cache = {}
        for p in search_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                cache[p.name.lower()] = p
        find_image_file._dir_cache = cache
        print(f"   索引建立完成，共找到 {len(cache)} 张图片文件。")
    
    return find_image_file._dir_cache.get(Path(filename).name.lower())

def convert_to_yolo_format(ann, img_width, img_height, cls_id):
    """
    将 COCO 分割转换为 YOLO 格式
    YOLO Segment: <class-index> <x1> <y1> <x2> <y2> ... (归一化 0-1)
    """
    segmentation = ann.get('segmentation', [])
    if not segmentation:
        return None
        
    # 处理多边形 (polygon)
    # COCO segmentation 可能是 [[x1, y1, x2, y2, ...], [poly2...]]
    # YOLO 通常只取最大的一个轮廓，或者保存多行
    # 这里我们把每个多边形都作为一行保存
    
    yolo_lines = []
    
    for poly in segmentation:
        # 归一化
        normalized_poly = []
        for i in range(0, len(poly), 2):
            x = poly[i] / img_width
            y = poly[i+1] / img_height
            
            # 裁剪到 [0, 1] 范围
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            
            normalized_poly.append(f"{x:.6f}")
            normalized_poly.append(f"{y:.6f}")
            
        if len(normalized_poly) >= 6: # 至少3个点
            line = f"{cls_id} " + " ".join(normalized_poly)
            yolo_lines.append(line)
            
    return yolo_lines

def split_dataset():
    print(f"🚀 开始处理数据集 (含 YOLO 标签转换)...")
    
    if not SOURCE_ANN_DIR.exists():
        print(f"❌ 错误: 找不到标注目录 {SOURCE_ANN_DIR}")
        return
        
    json_files = list(SOURCE_ANN_DIR.glob("*.json"))
    if not json_files:
        print(f"❌ 错误: 未找到 .json 文件")
        return
    
    src_json_path = json_files[0]
    print(f"📄 读取标注文件: {src_json_path}")
    
    with open(src_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 1. 建立类别映射 (Name -> ID)
    cat_id_map = {}
    print("🏷️  类别映射:")
    for cat in coco_data.get('categories', []):
        name = cat['name']
        # 简单匹配：忽略大小写
        match_idx = -1
        for i, yolo_name in enumerate(YOLO_NAMES):
            if name.upper() == yolo_name.upper():
                match_idx = i
                break
        
        if match_idx != -1:
            cat_id_map[cat['id']] = match_idx
            print(f"   '{name}' (ID {cat['id']}) -> YOLO Class {match_idx} ({YOLO_NAMES[match_idx]})")
        else:
            print(f"   ⚠️ 警告: 类别 '{name}' 不在 train.yaml 的 names 列表中，将被忽略！")

    # 2. 建立 Annotation 索引
    img_to_anns = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        img_to_anns[ann['image_id']].append(ann)

    # 3. 收集并清洗图片
    print("🧹 收集并清洗图片数据...")
    valid_images = []
    if not SOURCE_IMG_DIR.exists(): return

    for img in tqdm(coco_data['images'], desc="解析图片"):
        real_path = find_image_file(img['file_name'], SOURCE_IMG_DIR)
        if real_path is None: continue
            
        img['original_file_name'] = img['file_name']
        img['file_name'] = real_path.name
        img['_real_path'] = real_path 
        
        # 时间戳解析
        ts = img.get('timestamp', 0)
        if ts == 0: ts = parse_timestamp_from_filename(img['file_name'])
        img['_timestamp'] = ts
        
        valid_images.append(img)

    # 4. 时间戳聚类
    print("⏳ 正在根据时间戳进行聚类分组 (Gap < 1.0s)...")
    valid_images.sort(key=lambda x: x['_timestamp'])
    
    parcel_groups = defaultdict(list)
    if not valid_images: return

    current_parcel_idx = 1
    current_group_id = f"{current_parcel_idx:03d}"
    parcel_groups[current_group_id].append(valid_images[0])
    
    for i in range(1, len(valid_images)):
        curr, prev = valid_images[i], valid_images[i-1]
        diff = curr['_timestamp'] - prev['_timestamp']
        
        if 0 <= diff < TIME_GAP_THRESHOLD and curr['_timestamp'] > 0:
            parcel_groups[current_group_id].append(curr)
        else:
            current_parcel_idx += 1
            current_group_id = f"{current_parcel_idx:03d}"
            parcel_groups[current_group_id].append(curr)

    # 5. 准备目录
    for d in [TARGET_IMG_TRAIN, TARGET_IMG_VAL, TARGET_LBL_TRAIN, TARGET_LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    # 6. 切分与生成
    all_parcels = list(parcel_groups.keys())
    random.seed(SEED)
    random.shuffle(all_parcels)
    
    num_val = int(len(all_parcels) * VAL_RATIO)
    val_parcels = set(all_parcels[:num_val])
    
    splits = {
        'train': {'images': [], 'annotations': [], 'categories': coco_data.get('categories', [])},
        'val':   {'images': [], 'annotations': [], 'categories': coco_data.get('categories', [])}
    }
    
    # 记录每个图片最终所属的集，用于分配 annotation 到 json
    img_id_to_split = {}
    
    print("🚚 正在分发文件并生成 YOLO 标签...")
    for pid in tqdm(all_parcels, desc="处理包裹"):
        split_name = 'val' if pid in val_parcels else 'train'
        img_target_dir = TARGET_IMG_VAL if split_name == 'val' else TARGET_IMG_TRAIN
        lbl_target_dir = TARGET_LBL_VAL if split_name == 'val' else TARGET_LBL_TRAIN
        
        for img_info in parcel_groups[pid]:
            real_src = img_info.pop('_real_path')
            img_info['parcel_id'] = pid
            
            # 1. 拷贝图片
            dst_img = img_target_dir / img_info['file_name']
            if not dst_img.exists() or dst_img.stat().st_size != real_src.stat().st_size:
                shutil.copy2(real_src, dst_img)
            
            # 2. 生成 YOLO TXT 标签
            anns = img_to_anns.get(img_info['id'], [])
            txt_lines = []
            
            for ann in anns:
                cid = ann['category_id']
                if cid not in cat_id_map: continue
                
                yolo_cls = cat_id_map[cid]
                lines = convert_to_yolo_format(ann, img_info['width'], img_info['height'], yolo_cls)
                if lines:
                    txt_lines.extend(lines)
            
            # 保存 TXT
            txt_filename = Path(img_info['file_name']).with_suffix('.txt')
            with open(lbl_target_dir / txt_filename, 'w') as f:
                f.write('\n'.join(txt_lines))

            # 3. 准备 JSON 数据 (用于时序 Dataset 读取)
            if 'original_file_name' in img_info: del img_info['original_file_name']
            
            splits[split_name]['images'].append(img_info)
            img_id_to_split[img_info['id']] = split_name

    # 7. 分发 JSON Annotations (仅用于 Dataset.py 逻辑，YOLO 训练用 txt)
    print("📝 保存 JSON 辅助文件...")
    for ann in coco_data.get('annotations', []):
        if ann['image_id'] in img_id_to_split:
            s_name = img_id_to_split[ann['image_id']]
            splits[s_name]['annotations'].append(ann)
            
    with open(TARGET_ROOT / "instances_Train.json", 'w') as f:
        json.dump(splits['train'], f)
    with open(TARGET_ROOT / "instances_Val.json", 'w') as f:
        json.dump(splits['val'], f)
        
    print(f"✅ 处理完成！数据集已就绪: {TARGET_ROOT}")
    print(f"   包含图片: {sum(len(splits[k]['images']) for k in splits)}")
    print(f"   包含标签: {len(list(TARGET_LBL_TRAIN.glob('*.txt'))) + len(list(TARGET_LBL_VAL.glob('*.txt')))}")

if __name__ == '__main__':
    split_dataset()
