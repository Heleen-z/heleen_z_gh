#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D/3D Data Format Converter
Preserve original timestamp digits, only adjust format symbols and prefixes
Output timestamp format: YYYY-MM-DD_HH:MM:SS.mmm (STRICT)
"""

import os
import sys
import json
import shutil
import re
import platform
import locale
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
    ImageTK_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    ImageTK_AVAILABLE = False
    Image = None
    ImageTk = None


# Global encoding flag - auto-detect Chinese support
USE_CHINESE = True

def check_encoding_support():
    """Check if Chinese characters are supported in the terminal"""
    global USE_CHINESE
    try:
        test_str = "中文"
        test_str.encode(sys.stdout.encoding if hasattr(sys.stdout, 'encoding') else 'utf-8')
        if platform.system() == 'Windows':
            if sys.stdout.encoding in ['cp437', 'cp1252', 'ascii']:
                USE_CHINESE = False
        else:
            if 'UTF-8' not in locale.getpreferredencoding().upper():
                USE_CHINESE = False
    except (UnicodeEncodeError, AttributeError):
        USE_CHINESE = False
    return USE_CHINESE

check_encoding_support()

def T(zh_text, en_text):
    """Translation helper - return Chinese or English based on encoding support"""
    return zh_text if USE_CHINESE else en_text


class TimestampError(ValueError):
    """Timestamp parsing error with detailed information"""
    pass


class DataConverter:
    """Data Converter Main Class"""
    
    CATEGORIES_2D = [
        {"id": 1, "name": "BAG", "supercategory": ""},
        {"id": 2, "name": "BOX", "supercategory": ""},
        {"id": 3, "name": "MAIL", "supercategory": ""},
        {"id": 4, "name": "ROBOT", "supercategory": ""}
    ]
    
    CATEGORIES_3D = [
        {"id": 1, "name": "bag", "supercategory": ""},
        {"id": 2, "name": "box", "supercategory": ""},
        {"id": 3, "name": "mail", "supercategory": ""},
        {"id": 4, "name": "bubble_bag", "supercategory": ""},
        {"id": 5, "name": "box_in_bag", "supercategory": ""},
        {"id": 6, "name": "bubble_box", "supercategory": ""},
        {"id": 7, "name": "object_round", "supercategory": ""},
        {"id": 8, "name": "object_broken", "supercategory": ""},
        {"id": 9, "name": "object_irregular", "supercategory": ""},
        {"id": 10, "name": "debris", "supercategory": ""},
        {"id": 11, "name": "object_special", "supercategory": ""}
    ]
    
    # STRICT output format: YYYY-MM-DD_HH:MM:SS.mmm
    OUTPUT_TIMESTAMP_FORMAT = "{year}-{month}-{day}_{hour}:{minute}:{second}.{millisecond}"
    
    def __init__(self):
        self.source_path = None
        self.output_path = None
        self.mode = None
    
    def analyze_timestamp_error(self, filename, reason=""):
        """
        Analyze and return detailed timestamp error message
        Args:
            filename: Filename being processed
            reason: Specific error reason
        Returns: Detailed error message string
        """
        msg = f"\n{'='*60}\n"
        msg += f"[{T('TIMESTAMP ERROR', 'TIMESTAMP ERROR')}]\n"
        msg += f"{'='*60}\n"
        msg += f"{T('Filename', 'Filename')}: {filename}\n"
        msg += f"{T('Error', 'Error')}: {reason if reason else T('Invalid timestamp format', 'Invalid timestamp format')}\n\n"
        msg += f"{T('Expected timestamp format', 'Expected timestamp format')}: YYYY-MM-DD_HH:MM:SS.mmm\n"
        msg += f"{T('Example', 'Example')}: 2025-09-28_18:20:37.275\n\n"
        msg += f"{T('Timestamp position guide', 'Timestamp position guide')}:\n"
        msg += f"  YYYY - 4-digit year\n"
        msg += f"  MM   - 2-digit month (01-12)\n"
        msg += f"  DD   - 2-digit day (01-31)\n"
        msg += f"  HH   - 2-digit hour (00-23)\n"
        msg += f"  MM   - 2-digit minute (00-59)\n"
        msg += f"  SS   - 2-digit second (00-59)\n"
        msg += f"  mmm  - 3-digit millisecond (000-999)\n\n"
        msg += f"{T('Note', 'Note')}: {T('Separators can be any character, but output will use strict format', 'Separators can be any character, but output will use strict format')}\n"
        msg += f"{'='*60}\n"
        return msg
    
    def extract_timestamp(self, filename):
        """
        Extract timestamp from filename - STRICT parsing
        Format: YYYYxMMxDDxHHxMMxSSxmmm where x is any single character separator
        
        Args:
            filename: Filename to parse
        Returns:
            tuple: (year, month, day, hour, minute, second, millisecond)
        Raises:
            TimestampError: If timestamp format is invalid
        """
        # Find all possible start positions (4-digit year)
        candidates = []
        for match in re.finditer(r'\d{4}', filename):
            start = match.start()
            if start + 23 <= len(filename):  # Need at least 23 chars for full timestamp
                candidates.append(start)
        
        if not candidates:
            raise TimestampError(
                self.analyze_timestamp_error(
                    filename, 
                    T("No 4-digit year found in filename", "No 4-digit year found in filename")
                )
            )
        
        # Try each candidate position
        last_error = ""
        for start_idx in candidates:
            try:
                return self._parse_timestamp_at_position(filename, start_idx)
            except ValueError as e:
                last_error = str(e)
                continue
        
        # All positions failed
        raise TimestampError(
            self.analyze_timestamp_error(
                filename,
                f"{T('Tried', 'Tried')} {len(candidates)} {T('positions, all failed', 'positions, all failed')}. {last_error}"
            )
        )
    
    def _parse_timestamp_at_position(self, filename, start_idx):
        """
        Parse timestamp from specific position
        Format: YYYYxMMxDDxHHxMMxSSxmmm (x = any single char separator)
        """
        def get_slice(start, length, desc):
            end = start + length
            if end > len(filename):
                raise ValueError(f"{T('Unexpected end at position', 'Unexpected end at position')} {start}, {T('expected', 'expected')} {desc}")
            return filename[start:end]
        
        def expect_digit(s, pos, desc):
            if not s.isdigit():
                raise ValueError(f"{T('Position', 'Position')} {pos}: {T('expected', 'expected')} {desc}, {T('got', 'got')} '{s}'")
            return s
        
        pos = start_idx
        
        # Year: 4 digits
        year = expect_digit(get_slice(pos, 4, "year"), pos, "4-digit year")
        pos += 4
        
        # Separator 1
        pos += 1
        
        # Month: 2 digits
        month = expect_digit(get_slice(pos, 2, "month"), pos, "2-digit month")
        if not (1 <= int(month) <= 12):
            raise ValueError(f"{T('Invalid month', 'Invalid month')}: {month}")
        pos += 2
        
        # Separator 2
        pos += 1
        
        # Day: 2 digits
        day = expect_digit(get_slice(pos, 2, "day"), pos, "2-digit day")
        if not (1 <= int(day) <= 31):
            raise ValueError(f"{T('Invalid day', 'Invalid day')}: {day}")
        pos += 2
        
        # Separator 3
        pos += 1
        
        # Hour: 2 digits
        hour = expect_digit(get_slice(pos, 2, "hour"), pos, "2-digit hour")
        if not (0 <= int(hour) <= 23):
            raise ValueError(f"{T('Invalid hour', 'Invalid hour')}: {hour}")
        pos += 2
        
        # Separator 4
        pos += 1
        
        # Minute: 2 digits
        minute = expect_digit(get_slice(pos, 2, "minute"), pos, "2-digit minute")
        if not (0 <= int(minute) <= 59):
            raise ValueError(f"{T('Invalid minute', 'Invalid minute')}: {minute}")
        pos += 2
        
        # Separator 5
        pos += 1
        
        # Second: 2 digits
        second = expect_digit(get_slice(pos, 2, "second"), pos, "2-digit second")
        if not (0 <= int(second) <= 59):
            raise ValueError(f"{T('Invalid second', 'Invalid second')}: {second}")
        pos += 2
        
        # Separator 6
        pos += 1
        
        # Millisecond: 3 digits
        millisecond = expect_digit(get_slice(pos, 3, "millisecond"), pos, "3-digit millisecond")
        
        return (year, month, day, hour, minute, second, millisecond)
    
    def format_timestamp(self, timestamp_tuple):
        """
        Format timestamp tuple to STRICT output format: YYYY-MM-DD_HH:MM:SS.mmm
        
        Args:
            timestamp_tuple: (year, month, day, hour, minute, second, millisecond)
        Returns:
            str: Formatted timestamp string
        """
        year, month, day, hour, minute, second, millisecond = timestamp_tuple
        return f"{year}-{month}-{day}_{hour}:{minute}:{second}.{millisecond}"
    
    def get_image_files_recursive(self, folder, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        """Recursively get all image files in folder and subfolders"""
        files = []
        folder_path = Path(folder)
        if not folder_path.exists():
            return files
        
        for ext in extensions:
            files.extend(folder_path.rglob(f"*{ext}"))
            files.extend(folder_path.rglob(f"*{ext.upper()}"))
        
        # Deduplicate and sort
        seen = set()
        unique_files = []
        for f in sorted(files):
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        
        return unique_files
    
    def find_json_files_recursive(self, folder):
        """Recursively find all JSON files in folder"""
        folder_path = Path(folder)
        if not folder_path.exists():
            return []
        return sorted(folder_path.rglob("*.json"))
    
    def load_json(self, json_path):
        """Load JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_json(self, data, output_path):
        """Save JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # ==================== 1. 2D Processing ====================
    
    def process_2d(self, source_folder, output_folder):
        """
        Process 2D data - handle check* images
        Output format: check_color_img_YYYY-MM-DD_HH:MM:SS.mmm.jpg
        """
        print(f"\n{'='*60}")
        print(f"[2D {T('Mode', 'Mode')}] {T('Processing', 'Processing')}: {source_folder}")
        print(f"{T('Output', 'Output')}: {output_folder}")
        print(f"{T('Timestamp format', 'Timestamp format')}: YYYY-MM-DD_HH:MM:SS.mmm")
        print(f"{'='*60}\n")
        
        source_path = Path(source_folder)
        output_path = Path(output_folder)
        
        images_output = output_path / "images"
        annotations_output = output_path / "annotations"
        images_output.mkdir(parents=True, exist_ok=True)
        annotations_output.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        all_image_files = self.get_image_files_recursive(source_path)
        print(f"{T('Found', 'Found')} {len(all_image_files)} {T('images total', 'images total')}")
        
        # Filter check images
        check_images = [f for f in all_image_files if f.name.lower().startswith('check')]
        
        if not check_images:
            raise ValueError(T("ERROR: No images starting with 'check' found!", "ERROR: No images starting with 'check' found!"))
        
        print(f"{T('Processing', 'Processing')} {len(check_images)} {T('check images', 'check images')}\n")
        
        # Process each image
        images_info = []
        filename_mapping = {}
        date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for idx, img_path in enumerate(check_images, start=1):
            original_name = img_path.name
            
            # Extract and validate timestamp
            timestamp_tuple = self.extract_timestamp(original_name)
            formatted_timestamp = self.format_timestamp(timestamp_tuple)
            
            # Generate new filename with STRICT format
            new_filename = f"check_color_img_{formatted_timestamp}.jpg"
            filename_mapping[original_name] = new_filename
            
            images_info.append({
                "id": idx,
                "width": 1280,
                "height": 1024,
                "file_name": new_filename,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })
            
            # Copy file
            shutil.copy2(img_path, images_output / new_filename)
            print(f"  [{idx}] {original_name}")
            print(f"      -> {new_filename}")
        
        # Process JSON files
        self._process_json_files(source_path, annotations_output, filename_mapping, images_info, "2d")
        
        print(f"\n{'='*60}")
        print(f"[2D {T('Complete', 'Complete')}]")
        print(f"  {T('Images', 'Images')}: {images_output} ({len(check_images)} {T('files', 'files')})")
        print(f"  {T('Annotations', 'Annotations')}: {annotations_output / 'instances_default.json'}")
        print(f"{'='*60}\n")
        
        return True
    
    # ==================== 2. 3D Processing ====================
    
    def process_3d(self, source_folder, output_folder):
        """
        Process 3D data - handle color* and depth* images
        Output format: 
          - color_img_YYYY-MM-DD_HH:MM:SS.mmm.jpg
          - depth_img_YYYY-MM-DD_HH:MM:SS.mmm.png
        """
        print(f"\n{'='*60}")
        print(f"[3D {T('Mode', 'Mode')}] {T('Processing', 'Processing')}: {source_folder}")
        print(f"{T('Output', 'Output')}: {output_folder}")
        print(f"{T('Timestamp format', 'Timestamp format')}: YYYY-MM-DD_HH:MM:SS.mmm")
        print(f"{'='*60}\n")
        
        source_path = Path(source_folder)
        output_path = Path(output_folder)
        
        images_output = output_path / "images"
        depths_output = output_path / "depths"
        annotations_output = output_path / "annotations"
        images_output.mkdir(parents=True, exist_ok=True)
        depths_output.mkdir(parents=True, exist_ok=True)
        annotations_output.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        all_image_files = self.get_image_files_recursive(source_path)
        print(f"{T('Found', 'Found')} {len(all_image_files)} {T('images total', 'images total')}")
        
        # Classify by prefix
        color_images = [f for f in all_image_files if f.name.lower().startswith('color')]
        depth_images = [f for f in all_image_files if f.name.lower().startswith('depth')]
        
        print(f"  Color: {len(color_images)} {T('files', 'files')}")
        print(f"  Depth: {len(depth_images)} {T('files', 'files')}")
        
        if not color_images:
            raise ValueError(T("ERROR: No color images found!", "ERROR: No color images found!"))
        
        print()
        
        # Process color images
        images_info = []
        filename_mapping = {}
        date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[{T('Processing Color images', 'Processing Color images')}]")
        for idx, img_path in enumerate(color_images, start=1):
            original_name = img_path.name
            
            timestamp_tuple = self.extract_timestamp(original_name)
            formatted_timestamp = self.format_timestamp(timestamp_tuple)
            
            new_filename = f"color_img_{formatted_timestamp}.jpg"
            filename_mapping[original_name] = new_filename
            
            images_info.append({
                "id": idx,
                "width": 1536,
                "height": 1280,
                "file_name": new_filename,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })
            
            shutil.copy2(img_path, images_output / new_filename)
            print(f"  [{idx}] {original_name} -> {new_filename}")
        
        # Process depth images
        print(f"\n[{T('Processing Depth images', 'Processing Depth images')}]")
        for idx, img_path in enumerate(depth_images, start=1):
            original_name = img_path.name
            
            timestamp_tuple = self.extract_timestamp(original_name)
            formatted_timestamp = self.format_timestamp(timestamp_tuple)
            
            new_filename = f"depth_img_{formatted_timestamp}.png"
            filename_mapping[original_name] = new_filename
            
            shutil.copy2(img_path, depths_output / new_filename)
            print(f"  [{idx}] {original_name} -> {new_filename}")
        
        # Process JSON files
        self._process_json_files(source_path, annotations_output, filename_mapping, images_info, "3d")
        
        print(f"\n{'='*60}")
        print(f"[3D {T('Complete', 'Complete')}]")
        print(f"  Color: {images_output} ({len(color_images)} {T('files', 'files')})")
        print(f"  Depth: {depths_output} ({len(depth_images)} {T('files', 'files')})")
        print(f"  JSON: {annotations_output / 'instances_default.json'}")
        print(f"{'='*60}\n")
        
        return True
    
    # ==================== 3. Refined Processing (Selective Prefix) ====================
    
    def classify_images_by_prefix(self, image_files):
        """Classify images by filename prefix (before timestamp)"""
        classified = {}
        other_images = []
        
        for img_path in image_files:
            name_lower = img_path.stem.lower()
            
            # Find timestamp start (4 consecutive digits = year)
            timestamp_match = re.search(r'\d{4}', name_lower)
            if timestamp_match:
                timestamp_start = timestamp_match.start()
                if timestamp_start > 0:
                    prefix = name_lower[:timestamp_start].rstrip('_-')
                    if prefix:
                        if prefix not in classified:
                            classified[prefix] = []
                        classified[prefix].append(img_path)
                        continue
            
            other_images.append(img_path)
        
        # Sort each category
        for prefix in classified:
            classified[prefix] = sorted(classified[prefix])
        
        return classified, sorted(other_images)
    
    def process_selective(self, source_folder, output_folder, target_prefixes=None):
        """
        Process images with selected prefixes only
        Only normalizes timestamp format, preserves original prefix
        
        Args:
            source_folder: Source folder path
            output_folder: Output folder path
            target_prefixes: List of prefixes to process (e.g., ['depth', 'color', 'check'])
        """
        print(f"\n{'='*60}")
        print(f"[{T('Selective Mode', 'Selective Mode')}] {T('Processing', 'Processing')}: {source_folder}")
        print(f"{T('Output', 'Output')}: {output_folder}")
        print(f"{T('Timestamp format', 'Timestamp format')}: YYYY-MM-DD_HH:MM:SS.mmm")
        if target_prefixes:
            print(f"{T('Selected prefixes', 'Selected prefixes')}: {', '.join(target_prefixes)}")
        else:
            print(f"{T('Selected prefixes', 'Selected prefixes')}: {T('All', 'All')}")
        print(f"{'='*60}\n")
        
        source_path = Path(source_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        all_image_files = self.get_image_files_recursive(source_path)
        print(f"{T('Found', 'Found')} {len(all_image_files)} {T('images total', 'images total')}")
        
        if not all_image_files:
            raise ValueError(T("ERROR: No image files found!", "ERROR: No image files found!"))
        
        # Classify by prefix
        classified, other_images = self.classify_images_by_prefix(all_image_files)
        
        if not classified:
            raise ValueError(T("ERROR: Could not classify images by prefix!", "ERROR: Could not classify images by prefix!"))
        
        # Filter by target prefixes
        if target_prefixes:
            target_prefixes_lower = [p.lower() for p in target_prefixes]
            filtered_classified = {}
            for prefix, images in classified.items():
                if prefix.lower() in target_prefixes_lower:
                    filtered_classified[prefix] = images
            classified = filtered_classified
        
        print(f"\n{T('Prefix classification', 'Prefix classification')}:")
        for prefix, images in sorted(classified.items()):
            print(f"  - {prefix}: {len(images)} {T('files', 'files')}")
        print()
        
        if not classified:
            raise ValueError(T("ERROR: No images match the selected prefixes!", "ERROR: No images match the selected prefixes!"))
        
        # Process each prefix
        stats = {}
        total_processed = 0
        total_failed = 0
        
        for prefix, image_files in sorted(classified.items()):
            print(f"[{T('Processing', 'Processing')}: {prefix}]")
            
            prefix_output = output_path / prefix
            prefix_output.mkdir(parents=True, exist_ok=True)
            
            processed = 0
            failed = 0
            
            for idx, img_path in enumerate(image_files, 1):
                original_name = img_path.name
                original_stem = img_path.stem
                original_ext = img_path.suffix.lower()
                
                try:
                    # Extract timestamp
                    timestamp_tuple = self.extract_timestamp(original_stem)
                    formatted_timestamp = self.format_timestamp(timestamp_tuple)
                    
                    # Determine extension (depth -> png, others -> jpg)
                    if prefix.lower() == 'depth':
                        new_ext = '.png'
                    else:
                        new_ext = '.jpg'
                    
                    # Generate new filename: {prefix}_{timestamp}.{ext}
                    new_filename = f"{prefix.lower()}_{formatted_timestamp}{new_ext}"
                    output_file = prefix_output / new_filename
                    
                    # Copy and convert if needed
                    if PIL_AVAILABLE and original_ext != new_ext:
                        with Image.open(img_path) as img:
                            if new_ext == '.jpg':
                                if img.mode in ('RGBA', 'LA', 'P'):
                                    background = Image.new('RGB', img.size, (255, 255, 255))
                                    if img.mode == 'P':
                                        img = img.convert('RGBA')
                                    if img.mode in ('RGBA', 'LA'):
                                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                                        img = background
                                    else:
                                        img = img.convert('RGB')
                                elif img.mode != 'RGB':
                                    img = img.convert('RGB')
                                img.save(output_file, 'JPEG', quality=95, optimize=True)
                            else:  # PNG
                                if img.mode == 'P':
                                    img = img.convert('RGBA')
                                img.save(output_file, 'PNG', optimize=True)
                    else:
                        shutil.copy2(img_path, output_file)
                    
                    # Preserve modification time
                    stat = img_path.stat()
                    os.utime(output_file, (stat.st_atime, stat.st_mtime))
                    
                    processed += 1
                    if idx <= 3 or idx % 100 == 0:
                        print(f"  [{idx}/{len(image_files)}] {original_name} -> {new_filename}")
                        
                except TimestampError:
                    raise  # Re-raise timestamp errors
                except Exception as e:
                    failed += 1
                    print(f"  [{T('FAILED', 'FAILED')}] {original_name}: {e}")
            
            stats[prefix] = {'processed': processed, 'failed': failed}
            total_processed += processed
            total_failed += failed
            
            print(f"  [{prefix}] {T('Done', 'Done')}: {processed} OK, {failed} failed\n")
        
        print(f"{'='*60}")
        print(f"[{T('Selective Mode Complete', 'Selective Mode Complete')}]")
        print(f"  {T('Total processed', 'Total processed')}: {total_processed}")
        print(f"  {T('Total failed', 'Total failed')}: {total_failed}")
        print(f"{'='*60}\n")
        
        return stats
    
    # ==================== 4. JSON Fix ====================
    
    def fix_json_format(self, json_path, output_path=None, backup=True):
        """
        Fix JSON format errors and normalize timestamps in file_name fields
        
        Args:
            json_path: JSON file path
            output_path: Output path (default: overwrite original)
            backup: Whether to create backup
        Returns:
            (success, issues_list)
        """
        issues = []
        
        try:
            # Read file
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix trailing commas
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            if content != original_content:
                issues.append(T("Fixed trailing commas", "Fixed trailing commas"))
            
            # Fix single quotes
            if "'" in content:
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    new_content = re.sub(r"(?<!\\)'", '"', content)
                    if new_content != content:
                        content = new_content
                        issues.append(T("Fixed quotes", "Fixed quotes"))
            
            # Parse JSON
            data = json.loads(content)
            
            # Fix images and normalize timestamps
            if isinstance(data, dict) and 'images' in data:
                for img in data['images']:
                    if isinstance(img, dict) and 'file_name' in img:
                        old_file_name = img['file_name']
                        try:
                            # Extract timestamp
                            timestamp_tuple = self.extract_timestamp(old_file_name)
                            formatted_timestamp = self.format_timestamp(timestamp_tuple)
                            
                            # Replace timestamp in filename
                            name_without_ext = Path(old_file_name).stem
                            ext = Path(old_file_name).suffix
                            
                            # Find and replace timestamp pattern
                            timestamp_pattern = re.compile(
                                r'(\d{4})[^\d]?(\d{2})[^\d]?(\d{2})[^\d]?(\d{2})[^\d]?(\d{2})[^\d]?(\d{2})[^\d]?(\d{3})'
                            )
                            match = timestamp_pattern.search(name_without_ext)
                            if match:
                                new_name = timestamp_pattern.sub(formatted_timestamp, name_without_ext)
                                # Determine extension
                                if new_name.lower().startswith('depth'):
                                    new_ext = '.png'
                                else:
                                    new_ext = ext if ext else '.jpg'
                                new_file_name = new_name + new_ext
                                
                                if new_file_name != old_file_name:
                                    img['file_name'] = new_file_name
                                    issues.append(f"{T('Timestamp', 'Timestamp')}: {old_file_name} -> {new_file_name}")
                        except TimestampError:
                            pass  # Keep original if no valid timestamp
            
            # Determine output path
            if output_path is None:
                output_path = json_path
            output_path = Path(output_path)
            
            # Create backup
            if backup and str(output_path) == str(json_path) and output_path.exists():
                backup_path = output_path.with_suffix('.json.backup')
                shutil.copy2(json_path, backup_path)
                issues.append(T("Backup created", "Backup created"))
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Preserve modification time
            if Path(json_path).exists():
                stat = Path(json_path).stat()
                os.utime(output_path, (stat.st_atime, stat.st_mtime))
            
            return True, issues
            
        except Exception as e:
            return False, [f"{T('Error', 'Error')}: {str(e)}"]
    
    def batch_fix_json(self, source_folder, output_folder=None, backup=True):
        """
        Batch fix all JSON files
        """
        print(f"\n{'='*60}")
        print(f"[{T('JSON Fix Mode', 'JSON Fix Mode')}]")
        print(f"{T('Source', 'Source')}: {source_folder}")
        print(f"{T('Timestamp format', 'Timestamp format')}: YYYY-MM-DD_HH:MM:SS.mmm")
        print(f"{'='*60}\n")
        
        source_path = Path(source_folder)
        json_files = self.find_json_files_recursive(source_path)
        
        if not json_files:
            raise ValueError(T("ERROR: No JSON files found!", "ERROR: No JSON files found!"))
        
        print(f"{T('Found', 'Found')} {len(json_files)} JSON {T('files', 'files')}\n")
        
        success_count = 0
        fail_count = 0
        
        for idx, json_file in enumerate(json_files, 1):
            if output_folder:
                rel_path = json_file.relative_to(source_path)
                output_file = Path(output_folder) / rel_path
            else:
                output_file = None
            
            success, issues = self.fix_json_format(json_file, output_file, backup)
            
            if success:
                success_count += 1
                status = "OK"
                issue_str = f" ({', '.join(issues[:2])})" if issues else ""
            else:
                fail_count += 1
                status = "FAIL"
                issue_str = f" ({', '.join(issues)})"
            
            print(f"  [{idx}/{len(json_files)}] {status}: {json_file.name}{issue_str}")
        
        print(f"\n{'='*60}")
        print(f"[{T('JSON Fix Complete', 'JSON Fix Complete')}]")
        print(f"  {T('Success', 'Success')}: {success_count}, {T('Failed', 'Failed')}: {fail_count}")
        print(f"{'='*60}\n")
        
        return success_count, fail_count
    
    # ==================== Helper: Process JSON files ====================
    
    def _process_json_files(self, source_path, annotations_output, filename_mapping, images_info, mode):
        """
        Process JSON annotation files
        - Build file_name mapping from old to new
        - Build image_id mapping from old to new
        - Convert annotations with new IDs
        - Create standard output JSON structure
        """
        json_files = self.find_json_files_recursive(source_path)
        print(f"\n{len(json_files)} JSON {T('files found', 'files found')}")
        
        all_annotations = []
        annotation_id = 1
        
        for json_file in json_files:
            try:
                data = self.load_json(json_file)
                print(f"  {T('Processing', 'Processing')}: {json_file.name}")
                
                # Step 1: Build file_name mapping (old path -> new filename)
                file_name_mapping = {}
                if "images" in data:
                    for img in data["images"]:
                        old_file_name = img.get("file_name", "")
                        old_basename = Path(old_file_name).name
                        
                        # Skip depth images in 3D mode
                        if mode == "3d" and old_basename.lower().startswith('depth'):
                            continue
                        
                        # Direct match by basename
                        if old_basename in filename_mapping:
                            file_name_mapping[old_file_name] = filename_mapping[old_basename]
                        else:
                            # Try fuzzy match by timestamp
                            try:
                                old_timestamp = self.extract_timestamp(old_basename)
                            except TimestampError:
                                old_timestamp = None
                            
                            if old_timestamp:
                                for orig_name, new_name in filename_mapping.items():
                                    try:
                                        orig_timestamp = self.extract_timestamp(orig_name)
                                    except TimestampError:
                                        continue
                                    if old_timestamp == orig_timestamp:
                                        file_name_mapping[old_file_name] = new_name
                                        break
                
                # Step 2: Build image_id mapping (old_id -> new_id)
                id_mapping = {}
                if "images" in data:
                    for i, img in enumerate(data["images"]):
                        old_id = img.get("id", i + 1)
                        old_file_name = img.get("file_name", "")
                        old_basename = Path(old_file_name).name
                        
                        # Skip depth images in 3D mode
                        if mode == "3d" and old_basename.lower().startswith('depth'):
                            continue
                        
                        # Direct match
                        if old_basename in filename_mapping:
                            new_file_name = filename_mapping[old_basename]
                            for info in images_info:
                                if info["file_name"] == new_file_name:
                                    id_mapping[old_id] = info["id"]
                                    break
                        else:
                            # Try timestamp matching
                            try:
                                old_timestamp = self.extract_timestamp(old_basename)
                            except TimestampError:
                                old_timestamp = None
                            
                            if old_timestamp:
                                for idx, (orig_name, new_name) in enumerate(filename_mapping.items(), 1):
                                    try:
                                        orig_timestamp = self.extract_timestamp(orig_name)
                                    except TimestampError:
                                        continue
                                    if orig_timestamp == old_timestamp:
                                        id_mapping[old_id] = idx
                                        break
                
                # Step 3: Convert annotations
                if "annotations" in data:
                    for ann in data["annotations"]:
                        old_image_id = ann.get("image_id", 0)
                        
                        # In 3D mode, check if this annotation is for a color image
                        if mode == "3d":
                            is_color = False
                            if "images" in data:
                                for img in data["images"]:
                                    if img.get("id") == old_image_id:
                                        check_file_name = img.get("file_name", "")
                                        if not Path(check_file_name).name.lower().startswith('depth'):
                                            is_color = True
                                        break
                            if not is_color:
                                continue
                        
                        # Only include if we have a mapping for this image_id
                        if old_image_id in id_mapping:
                            new_ann = ann.copy()
                            new_ann["id"] = annotation_id
                            new_ann["image_id"] = id_mapping[old_image_id]
                            all_annotations.append(new_ann)
                            annotation_id += 1
                            
            except TimestampError:
                raise  # Re-raise timestamp errors
            except Exception as e:
                print(f"  {T('Warning', 'Warning')}: {json_file.name} - {e}")
        
        # Step 4: Create and save standard JSON
        date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if mode == "2d":
            standard_json = self._create_json_2d(images_info, date_created)
        else:
            standard_json = self._create_json_3d(images_info, date_created)
        
        standard_json["annotations"] = all_annotations
        
        output_json_path = annotations_output / "instances_default.json"
        self.save_json(standard_json, output_json_path)
        print(f"\nJSON {T('saved', 'saved')}: {output_json_path}")
    
    def _create_json_2d(self, images_info, date_created):
        """Create 2D format JSON structure"""
        return {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "",
                "date_created": date_created,
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            "categories": self.CATEGORIES_2D.copy(),
            "images": images_info,
            "annotations": []
        }
    
    def _create_json_3d(self, images_info, date_created):
        """Create 3D format JSON structure"""
        return {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "",
                "date_created": date_created,
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            "categories": self.CATEGORIES_3D.copy(),
            "images": images_info,
            "annotations": []
        }


class ImageSelector:
    """Image preview and selection dialog - like mainstream photo software"""
    
    THUMB_SIZE = (150, 150)  # Thumbnail size
    GRID_COLS = 5  # Number of columns in grid
    
    def __init__(self, parent, image_files, converter):
        self.parent = parent
        self.image_files = image_files  # List of Path objects
        self.converter = converter
        self.selected = {}  # {path: BooleanVar}
        self.thumbnails = {}  # {path: PhotoImage}
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title(T("Select Images to Process", "Select Images to Process"))
        self.window.geometry("1200x800")
        self.window.minsize(1000, 600)
        
        # Detect OS for font
        system = platform.system()
        if system == "Linux":
            self.font_family = "Arial"
        elif system == "Darwin":
            self.font_family = "Helvetica"
        else:
            self.font_family = "Microsoft YaHei"
        
        self.create_ui()
        self.load_images()
        
        # Make modal
        self.window.transient(parent)
        self.window.grab_set()
        parent.wait_window(self.window)
    
    def create_ui(self):
        """Create the selection UI"""
        # Top control panel
        control_frame = tk.Frame(self.window, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        # Title and count
        self.title_label = tk.Label(
            control_frame,
            text=f"{T('Total', 'Total')}: {len(self.image_files)} {T('images', 'images')}",
            font=(self.font_family, 12, "bold")
        )
        self.title_label.pack(side=tk.LEFT)
        
        self.count_label = tk.Label(
            control_frame,
            text=f"{T('Selected', 'Selected')}: 0",
            font=(self.font_family, 11),
            fg="#4CAF50"
        )
        self.count_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Control buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        tk.Button(
            btn_frame,
            text=T("Select All", "Select All"),
            command=self.select_all,
            width=12,
            font=(self.font_family, 9)
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_frame,
            text=T("Deselect All", "Deselect All"),
            command=self.deselect_all,
            width=12,
            font=(self.font_family, 9)
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_frame,
            text=T("Invert", "Invert"),
            command=self.invert_selection,
            width=12,
            font=(self.font_family, 9)
        ).pack(side=tk.LEFT, padx=2)
        
        # Filter frame
        filter_frame = tk.Frame(self.window, padx=10, pady=5)
        filter_frame.pack(fill=tk.X)
        
        tk.Label(
            filter_frame,
            text=T("Filter by prefix:", "Filter by prefix:"),
            font=(self.font_family, 9)
        ).pack(side=tk.LEFT)
        
        self.filter_var = tk.StringVar(value="")
        self.filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            values=[""],
            width=15,
            state="readonly"
        )
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.apply_filter)
        
        tk.Button(
            filter_frame,
            text=T("Clear Filter", "Clear Filter"),
            command=self.clear_filter,
            font=(self.font_family, 9)
        ).pack(side=tk.LEFT, padx=5)
        
        # Canvas with scrollbar for thumbnails
        canvas_frame = tk.Frame(self.window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
        # Bottom buttons
        bottom_frame = tk.Frame(self.window, padx=10, pady=10)
        bottom_frame.pack(fill=tk.X)
        
        self.process_btn = tk.Button(
            bottom_frame,
            text=f"{T('Process Selected', 'Process Selected')} (0)",
            command=self.confirm_selection,
            bg="#4CAF50",
            fg="white",
            font=(self.font_family, 11, "bold"),
            width=20,
            height=2
        )
        self.process_btn.pack(side=tk.RIGHT, padx=5)
        
        tk.Button(
            bottom_frame,
            text=T("Cancel", "Cancel"),
            command=self.cancel,
            font=(self.font_family, 11),
            width=12,
            height=2
        ).pack(side=tk.RIGHT, padx=5)
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def load_images(self):
        """Load and display thumbnails"""
        # Get unique prefixes for filter
        prefixes = set()
        
        for img_path in self.image_files:
            # Get prefix
            name_lower = img_path.stem.lower()
            match = re.search(r'\d{4}', name_lower)
            if match:
                prefix = name_lower[:match.start()].rstrip('_-')
                if prefix:
                    prefixes.add(prefix)
        
        # Update filter combo
        self.filter_combo['values'] = [""] + sorted(prefixes)
        
        # Create thumbnail grid
        self.display_thumbnails(self.image_files)
    
    def display_thumbnails(self, image_list):
        """Display thumbnails in grid"""
        # Clear existing
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.selected.clear()
        self.thumbnails.clear()
        
        row, col = 0, 0
        
        for idx, img_path in enumerate(image_list):
            # Create frame for each image
            frame = tk.Frame(
                self.scrollable_frame,
                bg="white",
                highlightthickness=2,
                highlightbackground="#ddd"
            )
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Load and create thumbnail
            try:
                with Image.open(img_path) as img:
                    # Calculate resize to fit while maintaining aspect ratio
                    img.thumbnail(self.THUMB_SIZE)
                    
                    # Convert to PhotoImage
                    if img.mode == 'RGBA':
                        # Create white background for transparent images
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    photo = ImageTk.PhotoImage(img)
                    self.thumbnails[img_path] = photo
            except Exception:
                # Create placeholder for failed loads
                photo = None
            
            # Image label (clickable)
            if photo:
                img_label = tk.Label(frame, image=photo, bg="white", cursor="hand2")
            else:
                img_label = tk.Label(
                    frame,
                    text=T("[No Preview]", "[No Preview]"),
                    bg="white",
                    width=15,
                    height=8
                )
            img_label.pack(padx=5, pady=5)
            
            # Filename label (truncated)
            display_name = img_path.name[:25] + "..." if len(img_path.name) > 28 else img_path.name
            name_label = tk.Label(
                frame,
                text=display_name,
                bg="white",
                font=(self.font_family, 8),
                wraplength=140
            )
            name_label.pack(padx=5)
            
            # Extract and display timestamp
            try:
                ts = self.converter.extract_timestamp(img_path.name)
                formatted = self.converter.format_timestamp(ts)
                ts_text = formatted
            except:
                ts_text = T("[Invalid Timestamp]", "[Invalid Timestamp]")
            
            ts_label = tk.Label(
                frame,
                text=ts_text,
                bg="white",
                font=(self.font_family, 7),
                fg="#666",
                wraplength=140
            )
            ts_label.pack(padx=5)
            
            # Checkbox
            var = tk.BooleanVar(value=True)
            self.selected[img_path] = var
            
            cb = tk.Checkbutton(
                frame,
                text=T("Select", "Select"),
                variable=var,
                bg="white",
                font=(self.font_family, 8),
                command=self.update_count
            )
            cb.pack(pady=(0, 5))
            
            # Click on image to toggle selection
            for widget in [img_label, frame]:
                widget.bind("<Button-1>", lambda e, p=img_path: self.toggle_image(p))
            
            # Grid layout
            col += 1
            if col >= self.GRID_COLS:
                col = 0
                row += 1
        
        # Configure grid weights
        for c in range(self.GRID_COLS):
            self.scrollable_frame.grid_columnconfigure(c, weight=1)
        
        self.update_count()
    
    def toggle_image(self, img_path):
        """Toggle selection of an image"""
        if img_path in self.selected:
            self.selected[img_path].set(not self.selected[img_path].get())
            self.update_count()
    
    def update_count(self):
        """Update selected count display"""
        count = sum(1 for v in self.selected.values() if v.get())
        self.count_label.config(text=f"{T('Selected', 'Selected')}: {count}")
        self.process_btn.config(text=f"{T('Process Selected', 'Process Selected')} ({count})")
    
    def select_all(self):
        """Select all visible images"""
        for var in self.selected.values():
            var.set(True)
        self.update_count()
    
    def deselect_all(self):
        """Deselect all images"""
        for var in self.selected.values():
            var.set(False)
        self.update_count()
    
    def invert_selection(self):
        """Invert selection"""
        for var in self.selected.values():
            var.set(not var.get())
        self.update_count()
    
    def apply_filter(self, event=None):
        """Apply prefix filter"""
        filter_prefix = self.filter_var.get().lower()
        
        if not filter_prefix:
            self.display_thumbnails(self.image_files)
            return
        
        filtered = [
            p for p in self.image_files
            if p.stem.lower().startswith(filter_prefix)
        ]
        self.display_thumbnails(filtered)
    
    def clear_filter(self):
        """Clear filter"""
        self.filter_var.set("")
        self.display_thumbnails(self.image_files)
    
    def confirm_selection(self):
        """Confirm and close with selected images"""
        self.result = [p for p, v in self.selected.items() if v.get()]
        self.window.destroy()
    
    def cancel(self):
        """Cancel and close"""
        self.result = []
        self.window.destroy()
    
    def get_selected(self):
        """Get the selected image paths"""
        return getattr(self, 'result', [])


class ConverterGUI:
    """GUI class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("2D/3D Data Converter")
        
        # Detect OS and set font
        self.system = platform.system()
        if self.system == "Linux":
            self.root.geometry("800x600")
            self.font_family = "Arial"
            self.font_mono = "Courier"
        elif self.system == "Darwin":
            self.root.geometry("800x600")
            self.font_family = "Helvetica"
            self.font_mono = "Courier"
        else:  # Windows
            self.root.geometry("800x600")
            self.font_family = "Microsoft YaHei"
            self.font_mono = "Courier New"
        
        self.root.minsize(750, 550)
        
        self.converter = DataConverter()
        self.source_path = tk.StringVar()
        self.output_path = tk.StringVar()
        
        self.create_ui()
        
    def create_ui(self):
        """Create user interface"""
        main_container = tk.Frame(self.root, padx=20, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_container, 
            text="2D/3D Data Format Converter", 
            font=(self.font_family, 16, "bold"),
            fg="#2196F3"
        )
        title_label.pack(pady=(0, 10))
        
        # Timestamp format reminder
        format_frame = tk.Frame(main_container, bg="#E3F2FD", padx=10, pady=5)
        format_frame.pack(fill=tk.X, pady=(0, 15))
        format_label = tk.Label(
            format_frame,
            text="Output Timestamp Format: YYYY-MM-DD_HH:MM:SS.mmm",
            font=(self.font_family, 10, "bold"),
            fg="#1565C0",
            bg="#E3F2FD"
        )
        format_label.pack()
        
        # Source folder
        source_frame = tk.LabelFrame(
            main_container, 
            text=" Source Folder ", 
            font=(self.font_family, 10, "bold"),
            padx=10, pady=10
        )
        source_frame.pack(fill=tk.X, pady=8)
        
        source_entry = tk.Entry(
            source_frame, 
            textvariable=self.source_path, 
            font=(self.font_mono, 10)
        )
        source_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        source_btn = tk.Button(
            source_frame, 
            text="Browse", 
            command=self.browse_source,
            width=10,
            font=(self.font_family, 9)
        )
        source_btn.pack(side=tk.RIGHT)
        
        # Output folder
        output_frame = tk.LabelFrame(
            main_container, 
            text=" Output Folder ", 
            font=(self.font_family, 10, "bold"),
            padx=10, pady=10
        )
        output_frame.pack(fill=tk.X, pady=8)
        
        output_entry = tk.Entry(
            output_frame, 
            textvariable=self.output_path, 
            font=(self.font_mono, 10)
        )
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        output_btn = tk.Button(
            output_frame, 
            text="Browse", 
            command=self.browse_output,
            width=10,
            font=(self.font_family, 9)
        )
        output_btn.pack(side=tk.RIGHT)
        
        # Mode selection
        mode_frame = tk.LabelFrame(
            main_container, 
            text=" Conversion Mode ", 
            font=(self.font_family, 10, "bold"),
            padx=10, pady=10
        )
        mode_frame.pack(fill=tk.X, pady=10)
        
        self.mode_var = tk.StringVar(value="2d")
        
        mode_2d = tk.Radiobutton(
            mode_frame, 
            text="2D Mode: Process 'check' images -> check_color_img_YYYY-MM-DD_HH:MM:SS.mmm.jpg", 
            variable=self.mode_var, 
            value="2d",
            font=(self.font_family, 10),
            anchor=tk.W,
            command=self.update_ui_visibility
        )
        mode_2d.pack(fill=tk.X, pady=3)
        
        mode_3d = tk.Radiobutton(
            mode_frame, 
            text="3D Mode: color->images/, depth->depths/ folder", 
            variable=self.mode_var, 
            value="3d",
            font=(self.font_family, 10),
            anchor=tk.W,
            command=self.update_ui_visibility
        )
        mode_3d.pack(fill=tk.X, pady=3)
        
        mode_selective = tk.Radiobutton(
            mode_frame,
            text="Selective Mode: Preview and select images to process (all prefixes)",
            variable=self.mode_var,
            value="selective",
            font=(self.font_family, 10),
            anchor=tk.W,
            command=self.update_ui_visibility
        )
        mode_selective.pack(fill=tk.X, pady=3)
        
        mode_json = tk.Radiobutton(
            mode_frame, 
            text="JSON Fix: Fix JSON format and normalize timestamps", 
            variable=self.mode_var, 
            value="json",
            font=(self.font_family, 10),
            anchor=tk.W,
            command=self.update_ui_visibility
        )
        mode_json.pack(fill=tk.X, pady=3)
        
        # Buttons
        btn_frame = tk.Frame(main_container)
        btn_frame.pack(pady=20)
        
        self.convert_btn = tk.Button(
            btn_frame, 
            text="Start Conversion", 
            command=self.start_conversion,
            width=18,
            height=2,
            bg="#4CAF50",
            fg="white",
            font=(self.font_family, 11, "bold"),
            relief=tk.RAISED,
            bd=3
        )
        self.convert_btn.pack(side=tk.LEFT, padx=10)
        
        exit_btn = tk.Button(
            btn_frame, 
            text="Exit", 
            command=self.root.quit,
            width=12,
            height=2,
            font=(self.font_family, 11),
            relief=tk.RAISED,
            bd=3
        )
        exit_btn.pack(side=tk.LEFT, padx=10)
        
        # Instructions
        info_frame = tk.LabelFrame(
            main_container, 
            text=" Instructions ", 
            font=(self.font_family, 9),
            fg="#666",
            padx=10,
            pady=10
        )
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = (
            "- Output timestamp format is STRICT: YYYY-MM-DD_HH:MM:SS.mmm\n"
            "- 2D Mode: Only processes images starting with 'check'\n"
            "- 3D Mode: Separates color and depth into different folders\n"
            "- Selective Mode: Preview images and select which to process\n"
            "- JSON Mode: Fixes format errors and normalizes timestamps\n"
            "- Timestamp errors will be reported with detailed information"
        )
        
        info_label = tk.Label(
            info_frame, 
            text=info_text,
            justify=tk.LEFT,
            font=(self.font_family, 9),
            fg="#333"
        )
        info_label.pack(anchor=tk.W)
    
    def update_ui_visibility(self):
        """Update UI elements visibility based on selected mode"""
        pass  # No special UI to toggle anymore
    
    def browse_source(self):
        """Browse source folder"""
        folder = filedialog.askdirectory(title="Select Source Folder")
        if folder:
            self.source_path.set(folder)
            mode = self.mode_var.get()
            suffix = f"_{mode}output"
            output_folder = Path(folder).parent / f"{Path(folder).name}{suffix}"
            self.output_path.set(str(output_folder))
    
    def browse_output(self):
        """Browse output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)
    
    def _handle_selective_mode(self, source, output):
        """Handle selective mode with image preview and selection"""
        # Scan images first
        self.convert_btn.config(state=tk.DISABLED, text="Scanning images...")
        self.root.update()
        
        try:
            source_path = Path(source)
            image_files = self.converter.get_image_files_recursive(source_path)
            
            if not image_files:
                messagebox.showerror("Error", T("No image files found!", "No image files found!"))
                self.convert_btn.config(state=tk.NORMAL, text="Start Conversion")
                return
            
            print(f"\n{'='*60}")
            print(f"[{T('Selective Mode', 'Selective Mode')}] Found {len(image_files)} images")
            print(f"{'='*60}\n")
            
            # Show image selector (with all images)
            self.convert_btn.config(state=tk.NORMAL, text="Start Conversion")
            self.root.update()
            
            selector = ImageSelector(self.root, image_files, self.converter)
            selected_images = selector.get_selected()
            
            if not selected_images:
                messagebox.showinfo("Cancelled", T("No images selected. Operation cancelled.", "No images selected. Operation cancelled."))
                return
            
            # Confirm processing
            confirm = messagebox.askyesno(
                "Confirm Processing",
                f"{T('Selected', 'Selected')}: {len(selected_images)} {T('images', 'images')}\n"
                f"{T('Output', 'Output')}: {output}\n\n"
                f"{T('Start processing?', 'Start processing?')}"
            )
            
            if not confirm:
                return
            
            # Process selected images
            self.convert_btn.config(state=tk.DISABLED, text="Processing...")
            self.root.update()
            
            stats = self._process_selected_images(selected_images, output)
            
            messagebox.showinfo("Complete",
                f"{T('Processing complete!', 'Processing complete!')}\n\n"
                f"{T('Processed', 'Processed')}: {stats['processed']}\n"
                f"{T('Failed', 'Failed')}: {stats['failed']}\n\n"
                f"{T('Output', 'Output')}:\n{output}")
            
        except Exception as e:
            messagebox.showerror("Error", f"{T('Processing failed', 'Processing failed')}: {str(e)}")
        finally:
            self.convert_btn.config(state=tk.NORMAL, text="Start Conversion")
    
    def _process_selected_images(self, selected_images, output_folder):
        """Process only the selected images"""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        failed = 0
        
        print(f"\n{'='*60}")
        print(f"[{T('Processing Selected Images', 'Processing Selected Images')}]")
        print(f"{'='*60}\n")
        
        for idx, img_path in enumerate(selected_images, 1):
            original_name = img_path.name
            original_stem = img_path.stem
            original_ext = img_path.suffix.lower()
            
            try:
                # Extract timestamp
                timestamp_tuple = self.converter.extract_timestamp(original_stem)
                formatted_timestamp = self.converter.format_timestamp(timestamp_tuple)
                
                # Get prefix
                name_lower = original_stem.lower()
                match = re.search(r'\d{4}', name_lower)
                if match:
                    prefix = name_lower[:match.start()].rstrip('_-')
                else:
                    prefix = "img"
                
                # Determine extension
                if prefix == 'depth':
                    new_ext = '.png'
                else:
                    new_ext = '.jpg'
                
                # Create prefix folder
                prefix_output = output_path / prefix
                prefix_output.mkdir(parents=True, exist_ok=True)
                
                # Generate new filename
                new_filename = f"{prefix}_{formatted_timestamp}{new_ext}"
                output_file = prefix_output / new_filename
                
                # Copy/convert file
                if PIL_AVAILABLE and original_ext != new_ext:
                    with Image.open(img_path) as img:
                        if new_ext == '.jpg':
                            if img.mode in ('RGBA', 'LA', 'P'):
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                if img.mode == 'P':
                                    img = img.convert('RGBA')
                                if img.mode in ('RGBA', 'LA'):
                                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                                    img = background
                                else:
                                    img = img.convert('RGB')
                            elif img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(output_file, 'JPEG', quality=95, optimize=True)
                        else:  # PNG
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            img.save(output_file, 'PNG', optimize=True)
                else:
                    shutil.copy2(img_path, output_file)
                
                # Preserve modification time
                stat = img_path.stat()
                os.utime(output_file, (stat.st_atime, stat.st_mtime))
                
                processed += 1
                print(f"  [{idx}/{len(selected_images)}] {original_name} -> {prefix}/{new_filename}")
                
            except TimestampError as e:
                failed += 1
                print(f"  [{T('FAILED', 'FAILED')}] {original_name}: {T('Invalid timestamp', 'Invalid timestamp')}")
            except Exception as e:
                failed += 1
                print(f"  [{T('FAILED', 'FAILED')}] {original_name}: {e}")
        
        print(f"\n{'='*60}")
        print(f"{T('Done', 'Done')}: {processed} OK, {failed} failed")
        print(f"{'='*60}\n")
        
        return {'processed': processed, 'failed': failed}
    
    def start_conversion(self):
        """Start conversion"""
        source = self.source_path.get().strip()
        output = self.output_path.get().strip()
        mode = self.mode_var.get()
        
        if not source:
            messagebox.showerror("Error", "Please select source folder")
            return
        
        if not os.path.exists(source):
            messagebox.showerror("Error", "Source folder does not exist")
            return
        
        if not output:
            suffix = f"_{mode}output"
            output = str(Path(source).parent / f"{Path(source).name}{suffix}")
        
        # Handle Selective Mode with image preview
        if mode == "selective":
            self._handle_selective_mode(source, output)
            return
        
        # Set confirmation message for other modes
        if mode == "json":
            confirm_msg = "Mode: JSON Fix\n\n"
        else:
            confirm_msg = f"Mode: {mode.upper()} Data Conversion\n\n"
        
        result = messagebox.askyesno(
            "Confirm Conversion", 
            f"{confirm_msg}"
            f"Source:\n{source}\n\n"
            f"Output:\n{output}\n\n"
            f"Timestamp format: YYYY-MM-DD_HH:MM:SS.mmm\n\n"
            f"Start conversion?"
        )
        
        if result:
            self.convert_btn.config(state=tk.DISABLED, text="Processing...")
            self.root.update()
            
            try:
                if mode == "2d":
                    self.converter.process_2d(source, output)
                    messagebox.showinfo("Complete", f"2D Conversion successful!\n\nOutput:\n{output}")
                    
                elif mode == "3d":
                    self.converter.process_3d(source, output)
                    messagebox.showinfo("Complete", f"3D Conversion successful!\n\nOutput:\n{output}")
                    
                elif mode == "json":
                    success_count, fail_count = self.converter.batch_fix_json(source, output, backup=True)
                    messagebox.showinfo("Complete", 
                        f"JSON fix complete!\n\n"
                        f"Success: {success_count}\n"
                        f"Failed: {fail_count}\n\n"
                        f"Output:\n{output}")
                    
            except (TimestampError, ValueError) as e:
                # Show detailed timestamp error
                messagebox.showerror("Timestamp Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Conversion failed:\n{str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                self.convert_btn.config(state=tk.NORMAL, text="Start Conversion")
    
    def run(self):
        """Run GUI"""
        self.root.mainloop()


def console_mode():
    """Command line mode"""
    print("=" * 60)
    print("  2D/3D Data Format Converter")
    print("  Output Format: YYYY-MM-DD_HH:MM:SS.mmm")
    print("=" * 60)
    print()
    
    converter = DataConverter()
    
    # Get source folder
    while True:
        source = input("Enter source folder path: ").strip().strip('"')
        if os.path.exists(source):
            break
        print("Path does not exist, please try again")
    
    # Select mode
    while True:
        print("\nSelect conversion mode:")
        print("  1. 2D mode - Process 'check' images")
        print("  2. 3D mode - Process color and depth images")
        print("  3. Selective mode - Process selected prefixes")
        print("  4. JSON mode - Fix JSON format and normalize timestamps")
        mode_input = input("\nEnter (1-4): ").strip()
        
        if mode_input in ("1", "2d", "2D", "2"):
            mode = "2d"
            break
        elif mode_input in ("2", "3d", "3D", "3"):
            mode = "3d"
            break
        elif mode_input in ("3", "selective", "SELECTIVE", "s"):
            mode = "selective"
            break
        elif mode_input in ("4", "json", "JSON"):
            mode = "json"
            break
        else:
            print("Invalid input, please try again")
    
    # Get prefix list for selective mode
    target_prefixes = None
    if mode == "selective":
        print("\nSelective mode: Process images with specific prefixes")
        print("Available: depth, color, check, ir, rgb, etc.")
        prefix_input = input("Enter prefixes (comma-separated) [default: all]: ").strip()
        if prefix_input:
            target_prefixes = [p.strip() for p in prefix_input.split(',') if p.strip()]
    
    # Get output folder
    default_output = os.path.join(os.path.dirname(source), f"{Path(source).name}_{mode}output")
    output = input(f"\nEnter output folder [default: {default_output}]: ").strip().strip('"')
    if not output:
        output = default_output
    
    print(f"\n{'='*60}")
    print(f"Starting conversion...")
    print(f"  Mode: {mode.upper()}")
    if mode == "selective" and target_prefixes:
        print(f"  Prefixes: {', '.join(target_prefixes)}")
    print(f"  Source: {source}")
    print(f"  Output: {output}")
    print(f"  Timestamp: YYYY-MM-DD_HH:MM:SS.mmm (STRICT)")
    print(f"{'='*60}\n")
    
    confirm = input("Confirm? (y/n): ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("Cancelled")
        return
    
    try:
        if mode == "2d":
            converter.process_2d(source, output)
        elif mode == "3d":
            converter.process_3d(source, output)
        elif mode == "selective":
            converter.process_selective(source, output, target_prefixes)
        elif mode == "json":
            converter.batch_fix_json(source, output, backup=True)
        print("\nConversion complete!")
    except (TimestampError, ValueError) as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nConversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='2D/3D Data Format Converter')
    parser.add_argument('--cli', action='store_true', help='Use command line mode')
    parser.add_argument('--source', '-s', help='Source folder path')
    parser.add_argument('--output', '-o', help='Output folder path')
    parser.add_argument('--mode', '-m', choices=['2d', '3d', 'selective', 'json'], help='Conversion mode')
    parser.add_argument('--prefix', '-p', help='Comma-separated prefixes for selective mode (e.g., depth,color,check)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup (for json mode)')
    
    args = parser.parse_args()
    
    if args.source and args.mode:
        converter = DataConverter()
        if not args.output:
            args.output = os.path.join(os.path.dirname(args.source), f"{Path(args.source).name}_{args.mode}output")
        
        # Parse prefixes for selective mode
        target_prefixes = None
        if args.prefix:
            target_prefixes = [p.strip() for p in args.prefix.split(',') if p.strip()]
        
        try:
            if args.mode == "2d":
                converter.process_2d(args.source, args.output)
            elif args.mode == "3d":
                converter.process_3d(args.source, args.output)
            elif args.mode == "selective":
                converter.process_selective(args.source, args.output, target_prefixes)
            elif args.mode == "json":
                converter.batch_fix_json(args.source, args.output, backup=not args.no_backup)
        except (TimestampError, ValueError) as e:
            print(f"\n{e}")
            sys.exit(1)
        except Exception as e:
            print(f"\nConversion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    elif args.cli:
        console_mode()
    else:
        app = ConverterGUI()
        app.run()


if __name__ == "__main__":
    main()