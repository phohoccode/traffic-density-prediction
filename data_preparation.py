"""
Data Preparation Module - Chuẩn bị dữ liệu huấn luyện cho mô hình dự báo
"""

import pandas as pd
from datetime import datetime, timedelta
import pymongo
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_training_data(lookback_minutes=30, forecast_horizon=60):
    """
    Chuẩn bị dữ liệu huấn luyện cho mô hình dự báo
    
    Args:
        lookback_minutes: Số phút nhìn lại (tính năng) - Default: 30
        forecast_horizon: Dự báo bao nhiêu phút trước - Default: 60
    
    Returns:
        X: Ma trận đầu vào (samples, features)
        y: Vector đầu ra (mục tiêu)
        scaler: MinMaxScaler để chuẩn hóa
    
    Ví dụ:
        X: [[8, 10, 12, 15, ...], [10, 12, 15, 18, ...], ...]  (30 phút quá khứ)
        y: [22, 25, 28, ...]  (dự báo sau 60 phút)
    """
    
    logger.info("Bắt đầu chuẩn bị dữ liệu huấn luyện...")
    
    # Kết nối MongoDB
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client["traffic_system"]
        collection = db["density_logs"]
        logger.info("Kết nối MongoDB thành công")
    except Exception as e:
        logger.error(f"Không thể kết nối MongoDB: {e}")
        return None, None, None
    
    # Lấy dữ liệu 14 ngày qua (training data)
    two_weeks_ago = datetime.now() - timedelta(days=14)
    
    try:
        cursor = collection.find(
            {"timestamp": {"$gte": two_weeks_ago}},
            {"timestamp": 1, "total_vehicles": 1}
        ).sort("timestamp", 1)
        
        data = list(cursor)
        logger.info(f"Lấy được {len(data)} bản ghi từ MongoDB")
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu: {e}")
        return None, None, None
    
    # Kiểm tra đủ dữ liệu
    if len(data) < lookback_minutes + forecast_horizon:
        logger.error(f"Không đủ dữ liệu: {len(data)} < {lookback_minutes + forecast_horizon}")
        logger.warning(f"   Cần ít nhất {lookback_minutes + forecast_horizon} bản ghi")
        return None, None, None
    
    # Chuyển thành DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Kiểm tra outliers
    mean = df['total_vehicles'].mean()
    std = df['total_vehicles'].std()
    logger.info(f"Thống kê dữ liệu:")
    logger.info(f"   Min: {df['total_vehicles'].min()}, Max: {df['total_vehicles'].max()}")
    logger.info(f"   Mean: {mean:.2f}, Std: {std:.2f}")
    
    # Chuẩn hóa dữ liệu [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1)) # Chuẩn hóa tổng số xe trong khoảng [0, 1]
    scaled_data = scaler.fit_transform(df[['total_vehicles']].values)
    
    logger.info(f"Dữ liệu đã chuẩn hóa [0, 1]")
    
    # Tạo sequences
    X = []
    y = []
    
    for i in range(len(scaled_data) - lookback_minutes - forecast_horizon):
        # Lấy lookback_minutes dữ liệu quá khứ
        X.append(scaled_data[i:i+lookback_minutes].flatten())
        # Mục tiêu: dữ liệu sau forecast_horizon phút
        y.append(scaled_data[i + lookback_minutes + forecast_horizon][0])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Dữ liệu huấn luyện:")
    logger.info(f"   X shape: {X.shape} (samples, features)")
    logger.info(f"   y shape: {y.shape} (samples,)")
    logger.info(f"   Lookback: {lookback_minutes} phút")
    logger.info(f"   Forecast horizon: {forecast_horizon} phút")
    
    return X, y, scaler

def get_recent_data(minutes=30):
    """
    Lấy N bản ghi gần nhất để dự báo
    (Không phụ thuộc timezone)
    """
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client["traffic_system"]
        collection = db["density_logs"]

        cursor = collection.find(
            {},
            {"timestamp": 1, "total_vehicles": 1}
        ).sort("timestamp", -1).limit(minutes)

        data = list(cursor)

        if len(data) < minutes:
            logger.warning(f"Không đủ dữ liệu: {len(data)}/{minutes}")
            return None

        # Đảo ngược lại theo thứ tự thời gian
        data.reverse()

        vehicles = np.array([doc["total_vehicles"] for doc in data])
        return vehicles

    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu gần đây: {e}")
        return None


if __name__ == "__main__":
    # Test
    X, y, scaler = prepare_training_data(lookback_minutes=30, forecast_horizon=60)
    
    if X is not None:
        logger.info("Dữ liệu đã sẵn sàng cho huấn luyện!")
    else:
        logger.error("Chuẩn bị dữ liệu thất bại")
