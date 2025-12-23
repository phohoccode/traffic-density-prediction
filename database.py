import pymongo
from datetime import datetime
import pandas as pd
import streamlit as st

# --- CẤU HÌNH KẾT NỐI MONGODB ---
# Nếu bạn dùng MongoDB Atlas (Cloud), hãy thay chuỗi kết nối vào đây.
# Ví dụ: "mongodb+srv://<user>:<password>@cluster0.mongodb.net/..."
MONGO_URI = "mongodb://localhost:27017/" 
DB_NAME = "traffic_system"
COLLECTION_NAME = "density_logs"

# Cache connection globally (not using @st.cache_resource to avoid Streamlit element conflicts)
_db_connection = None
_connection_attempted = False

def get_db_connection():
    """
    Tạo và cache kết nối đến MongoDB.
    """
    global _db_connection, _connection_attempted
    
    if _db_connection is not None:
        return _db_connection
    
    if _connection_attempted:
        return None
    
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Kiểm tra kết nối
        client.server_info()
        _db_connection = client[DB_NAME]
        _connection_attempted = True
        return _db_connection
    except Exception as e:
        _connection_attempted = True
        st.error(f"❌ Không thể kết nối MongoDB: {e}")
        st.warning("Ứng dụng vẫn hoạt động nhưng dữ liệu sẽ không được lưu vào database.")
        return None

def init_db():
    """
    Với MongoDB không cần tạo bảng trước (schema-less).
    Hàm này dùng để tạo Index giúp truy vấn nhanh hơn.
    """
    db = get_db_connection()
    if db is not None:
        # Tạo index giảm dần cho timestamp để lấy dữ liệu mới nhất nhanh hơn
        db[COLLECTION_NAME].create_index([("timestamp", -1)])

def insert_density(total_vehicles, status):
    """Lưu thông tin mật độ vào Collection"""
    db = get_db_connection()
    if db is not None:
        try:
            record = {
                "timestamp": datetime.now(),
                "total_vehicles": int(total_vehicles),
                "status": status,
            }
            db[COLLECTION_NAME].insert_one(record)
        except Exception as e:
            st.warning(f"Lỗi khi lưu vào DB: {e}")

def get_density_history(limit=100):
    """Lấy dữ liệu lịch sử để vẽ biểu đồ"""
    db = get_db_connection()
    if db is None:
        return pd.DataFrame()

    collection = db[COLLECTION_NAME]
    
    # Query: Lấy 'limit' bản ghi mới nhất (sort -1)
    # Projection: Chỉ lấy timestamp và total_vehicles, bỏ _id
    cursor = collection.find(
        {}, 
        {"_id": 0, "timestamp": 1, "total_vehicles": 1}
    ).sort("timestamp", -1).limit(limit)
    
    data = list(cursor)
    
    if data:
        df = pd.DataFrame(data)
        # Streamlit cần dữ liệu theo thời gian tăng dần để vẽ line chart đúng chiều
        df = df.sort_values(by="timestamp")
        return df
    else:
        # Trả về DataFrame rỗng có cấu trúc nếu chưa có dữ liệu
        return pd.DataFrame(columns=["timestamp", "total_vehicles"])

def get_density_history_filtered(start_date=None, end_date=None, limit=500):
    """Lấy dữ liệu lịch sử theo khoảng thời gian"""
    db = get_db_connection()
    if db is None:
        return pd.DataFrame()

    collection = db[COLLECTION_NAME]
    
    # Xây dựng query filter theo thời gian
    query = {}
    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            query["timestamp"]["$gte"] = start_date
        if end_date:
            query["timestamp"]["$lte"] = end_date
    
    cursor = collection.find(
        query, 
        {"_id": 0, "timestamp": 1, "total_vehicles": 1, "status": 1}
    ).sort("timestamp", -1).limit(limit)
    
    data = list(cursor)
    
    if data:
        df = pd.DataFrame(data)
        df = df.sort_values(by="timestamp")
        return df
    else:
        return pd.DataFrame(columns=["timestamp", "total_vehicles", "status"])

def insert_vehicle_detail(vehicle_class, vehicle_type, direction, confidence, track_id=None):
    """Lưu thông tin chi tiết từng xe phát hiện được"""
    db = get_db_connection()
    if db is not None:
        try:
            record = {
                "timestamp": datetime.now(),
                "vehicle_class": int(vehicle_class),  # COCO class ID
                "vehicle_type": vehicle_type,  # Tên loại xe (car, motorcycle, bus, truck)
                "direction": direction,  # "in" hoặc "out" hoặc "unknown"
                "confidence": float(confidence),
                "track_id": int(track_id) if track_id is not None else None  # ID tracking của xe
            }
            db["vehicle_details"].insert_one(record)
        except Exception as e:
            # Silent fail để không làm chậm video processing
            pass

def insert_tracked_vehicle(track_id, vehicle_class, vehicle_type, confidence, first_seen_time=None):
    """Lưu thông tin xe mới được tracking (chỉ lưu 1 lần khi xe xuất hiện lần đầu)"""
    db = get_db_connection()
    if db is not None:
        try:
            record = {
                "track_id": int(track_id),
                "vehicle_class": int(vehicle_class),
                "vehicle_type": vehicle_type,
                "confidence": float(confidence),
                "first_seen": first_seen_time or datetime.now(),
                "last_seen": datetime.now(),
                "frame_count": 1  # Số frame xuất hiện
            }
            # Insert với upsert=True để update last_seen nếu đã tồn tại
            db["tracked_vehicles"].update_one(
                {"track_id": int(track_id)},
                {"$set": record},
                upsert=True
            )
        except Exception as e:
            pass

def update_tracked_vehicle_last_seen(track_id):
    """Cập nhật thời gian xuất hiện cuối cùng của xe"""
    db = get_db_connection()
    if db is not None:
        try:
            db["tracked_vehicles"].update_one(
                {"track_id": int(track_id)},
                {
                    "$set": {"last_seen": datetime.now()},
                    "$inc": {"frame_count": 1}
                }
            )
        except Exception as e:
            pass

def get_vehicle_details(limit=100, start_date=None, end_date=None):
    """Lấy danh sách chi tiết các xe đã phát hiện"""
    db = get_db_connection()
    if db is None:
        return pd.DataFrame()

    collection = db["vehicle_details"]
    
    # Xây dựng query filter
    query = {}
    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            query["timestamp"]["$gte"] = start_date
        if end_date:
            query["timestamp"]["$lte"] = end_date
    
    cursor = collection.find(
        query,
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit)
    
    data = list(cursor)
    
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame(columns=["timestamp", "vehicle_class", "vehicle_type", "direction", "confidence"])

def get_vehicle_stats_by_type(start_date=None, end_date=None):
    """Thống kê số lượng xe theo loại"""
    db = get_db_connection()
    if db is None:
        return pd.DataFrame()

    collection = db["vehicle_details"]
    
    # Xây dựng match filter
    match_filter = {}
    if start_date or end_date:
        match_filter["timestamp"] = {}
        if start_date:
            match_filter["timestamp"]["$gte"] = start_date
        if end_date:
            match_filter["timestamp"]["$lte"] = end_date
    
    # Aggregation pipeline
    pipeline = [
        {"$match": match_filter} if match_filter else {"$match": {}},
        {
            "$group": {
                "_id": "$vehicle_type",
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"count": -1}}
    ]
    
    result = list(collection.aggregate(pipeline))
    
    if result:
        df = pd.DataFrame(result)
        df.columns = ["vehicle_type", "count"]
        return df
    else:
        return pd.DataFrame(columns=["vehicle_type", "count"])