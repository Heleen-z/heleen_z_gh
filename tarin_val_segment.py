#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分割脚本：将COCO格式数据分割为训练集和验证集
用法: python split_dataset.py
"""

import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ==================== 参数配置区（根据你的实际情况修改） ====================
# 输入路径
INPUT_JSON = Path("/workspaces/2d_roll/ann_data/annotations/instances_Train.json")  # 原始COCO标注文件
INPUT_IMAGES_DIR = Path("/workspaces/2d_roll/ann_data/images")  # 原始图片目录

# 输出路径
OUTPUT_DIR = Path("/workspaces/2d_roll/datasets/roll_detection")  # 输出数据集根目录
TRAIN_JSON = OUTPUT_DIR / "instances_Train.json"  # 训练集标注文件
VAL_JSON = OUTPUT_DIR / "instances_Val.json"      # 验证集标注文件
TRAIN_IMG_DIR = OUTPUT_DIR / "images" / "train"   # 训练图片目录
VAL_IMG_DIR = OUTPUT_DIR / "images" / "val"       # 验证图片目录

# 分割比例
TRAIN_RATIO = 0.8  # 训练集比例 (80%)
VAL_RATIO = 0.2    # 验证集比例 (20%)

# 随机种子（保证可复现，可修改或删除）
RANDOM_SEED = 42
# =========================================================================

def split_dataset():
    """
    将COCO格式的数据集分割为训练集和验证集
    保持图片和标注的对应关系，按图片级别进行分割
    """
    random.seed(RANDOM_SEED)
    
    # 1. 检查输入文件是否存在
    if not INPUT_JSON.exists():
        print(f"❌ 错误: 标注文件不存在: {INPUT_JSON}")
        print(f"   请检查路径是否正确，或修改脚本顶部的 INPUT_JSON 路径")
        return
    
    if not INPUT_IMAGES_DIR.exists():
        print(f"❌ 错误: 图片目录不存在: {INPUT_IMAGES_DIR}")
        print(f"   请检查路径是否正确，或修改脚本顶部的 INPUT_IMAGES_DIR 路径")
        return
    
    # 2. 加载原始COCO标注
    print(f"📂 正在加载标注文件: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    info = coco_data.get('info', {})
    licenses = coco_data.get('licenses', [])
    
    print(f"   原始数据: {len(images)}张图片, {len(annotations)}个标注, {len(categories)}个类别")
    
    if len(images) == 0:
        print("❌ 错误: 没有找到任何图片信息")
        return
    
    # 3. 按图片ID分组标注（确保同一张图片的标注不会分散到不同集合）
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann['image_id']].append(ann)
    
    # 4. 随机打乱图片顺序并分割
    image_indices = list(range(len(images)))
    random.shuffle(image_indices)
    
    split_idx = int(len(image_indices) * TRAIN_RATIO)
    train_indices = image_indices[:split_idx]
    val_indices = image_indices[split_idx:]
    
    print(f"\n📊 数据集分割:")
    print(f"   训练集: {len(train_indices)}张图片 ({TRAIN_RATIO*100:.0f}%)")
    print(f"   验证集: {len(val_indices)}张图片 ({VAL_RATIO*100:.0f}%)")
    
    # 5. 创建输出目录
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 创建输出目录: {OUTPUT_DIR}")
    
    # 6. 分割图片和标注
    train_images = []
    val_images = []
    train_annotations = []
    val_annotations = []
    
    # 处理训练集
    print(f"\n🚀 正在处理训练集...")
    copied_count = 0
    for idx, img_idx in enumerate(train_indices):
        img_info = images[img_idx]
        filename = img_info['file_name']
        src_path = INPUT_IMAGES_DIR / filename
        dst_path = TRAIN_IMG_DIR / filename
        
        # 复制图片
        if src_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"   ⚠️ 复制失败 {filename}: {e}")
                continue
        else:
            print(f"   ⚠️ 警告: 图片不存在 {src_path}")
            continue
        
        # 收集图片信息和对应标注
        train_images.append(img_info)
        train_annotations.extend(anns_by_image.get(img_info['id'], []))
        
        # 每100张或最后一张打印进度
        if (idx + 1) % 100 == 0 or idx == len(train_indices) - 1:
            print(f"   进度: {idx + 1}/{len(train_indices)} ({copied_count}张已复制)")
    
    # 处理验证集
    print(f"\n🚀 正在处理验证集...")
    copied_count = 0
    for idx, img_idx in enumerate(val_indices):
        img_info = images[img_idx]
        filename = img_info['file_name']
        src_path = INPUT_IMAGES_DIR / filename
        dst_path = VAL_IMG_DIR / filename
        
        # 复制图片
        if src_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"   ⚠️ 复制失败 {filename}: {e}")
                continue
        else:
            print(f"   ⚠️ 警告: 图片不存在 {src_path}")
            continue
        
        # 收集图片信息和对应标注
        val_images.append(img_info)
        val_annotations.extend(anns_by_image.get(img_info['id'], []))
        
        if (idx + 1) % 100 == 0 or idx == len(val_indices) - 1:
            print(f"   进度: {idx + 1}/{len(val_indices)} ({copied_count}张已复制)")
    
    # 7. 保存分割后的JSON文件（保持COCO格式）
    train_data = {
        "info": info,
        "licenses": licenses,
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    
    val_data = {
        "info": info,
        "licenses": licenses,
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }
    
    with open(TRAIN_JSON, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 保存训练集标注: {TRAIN_JSON}")
    print(f"   包含: {len(train_images)}张图片, {len(train_annotations)}个标注")
    
    with open(VAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"💾 保存验证集标注: {VAL_JSON}")
    print(f"   包含: {len(val_images)}张图片, {len(val_annotations)}个标注")
    
    # 8. 打印最终统计和目录结构
    print_final_stats(train_images, val_images, OUTPUT_DIR)

def print_final_stats(train_images, val_images, output_dir):
    """打印最终统计信息"""
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    
    # 统计实际复制的文件数
    train_copied = len(list(train_img_dir.glob("*"))) if train_img_dir.exists() else 0
    val_copied = len(list(val_img_dir.glob("*"))) if val_img_dir.exists() else 0
    
    print(f"\n" + "="*60)
    print(f"✅ 数据集分割完成!")
    print(f"="*60)
    print(f"📂 输出目录结构:")
    print(f"   {output_dir}/")
    print(f"   ├── instances_Train.json  ({len(train_images)}张图片元数据)")
    print(f"   ├── instances_Val.json    ({len(val_images)}张图片元数据)")
    print(f"   └── images/")
    print(f"       ├── train/            ({train_copied}张实际图片)")
    print(f"       └── val/              ({val_copied}张实际图片)")
    
    # 验证数据完整性
    print(f"\n🔍 数据完整性检查:")
    
    # 检查重复
    train_names = {img['file_name'] for img in train_images}
    val_names = {img['file_name'] for img in val_images}
    overlap = train_names & val_names
    
    if overlap:
        print(f"   ⚠️ 警告: 训练集和验证集存在 {len(overlap)} 张重复图片!")
    else:
        print(f"   ✓ 训练集和验证集无重复")
    
    # 检查标注分布
    print(f"   ✓ 训练集标注数: {sum(len([a for a in json.load(open(output_dir/'instances_Train.json'))['annotations'] if a['image_id']==img['id']]) for img in train_images)}")
    print(f"   ✓ 验证集标注数: {sum(len([a for a in json.load(open(output_dir/'instances_Val.json'))['annotations'] if a['image_id']==img['id']]) for img in val_images)}")

if __name__ == "__main__":
    split_dataset()
