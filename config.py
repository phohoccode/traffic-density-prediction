#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
from collections import OrderedDict
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"

DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt"]


OBJECT_COUNTER = None
OBJECT_COUNTER1 = None

# Traffic Thresholds (Ngưỡng mật độ giao thông)
THRESHOLD_CLEAR = 10      # Dưới 10 xe là thông thoáng
THRESHOLD_CONGESTED = 20  # Trên 20 xe là tắc nghẽn

# Database Update Frequency (Số lượng frame xử lý trước khi lưu log vào DB)
# Ví dụ: 30 frame lưu 1 lần để tránh làm chậm hệ thống và tràn dữ liệu
DB_UPDATE_INTERVAL = 30

# Object Tracking Configuration
# Tracker type: bytetrack hoặc botsort
TRACKER_TYPE = "bytetrack.yaml"  # ByteTrack - fast and accurate
# Set để lưu ID các xe đã tracking (reset khi chạy mới)
TRACKED_VEHICLE_IDS = set()

# ==================== Device Configuration ====================
import torch

# Auto detect: cuda nếu có GPU, cpu nếu không
# Có thể thay bằng 'cpu' để force CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kiểm tra thông tin GPU
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
else:
    GPU_NAME = "None"
    GPU_VRAM = 0

print(f"Device: {DEVICE.upper()}")
if torch.cuda.is_available():
    print(f"GPU: {GPU_NAME}")
    print(f"VRAM: {GPU_VRAM:.2f} GB")