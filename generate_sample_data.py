"""
Generate Sample Data Script - Tạo dữ liệu mẫu cho density_logs
Chạy lệnh: python generate_sample_data.py

Tạo ngẫu nhiên 2000 bản ghi trong 14 ngày cho collection density_logs
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(num_records=2000, days=14):
    """
    Tạo dữ liệu mẫu ngẫu nhiên cho density_logs
    
    Args:
        num_records: Số lượng bản ghi cần tạo (default: 2000)
        days: Số ngày dữ liệu (default: 14)
    """
    
    logger.info("=" * 70)
    logger.info("TAO DU LIEU MAU CHO DENSITY_LOGS")
    logger.info("=" * 70)
    
    try:
        # Kết nối MongoDB
        logger.info("\nKet noi MongoDB...")
        client = MongoClient("mongodb://localhost:27017/")
        db = client["traffic_system"]
        collection = db["density_logs"]
        
        logger.info("Ket noi thanh cong!")
        
        # Tính toán thời gian bắt đầu
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"\nTao du lieu tu {start_time} den {end_time}")
        logger.info(f"Tong so ban ghi: {num_records}")
        
        # Khởi tạo dữ liệu
        records = []
        current_time = start_time
        time_interval = timedelta(minutes=int((days * 24 * 60) / num_records))
        
        # Danh sách loại xe
        vehicle_types = ["motorcycle", "car", "truck", "bus", "bicycle"]
        
        logger.info("\nTao du lieu...")
        
        for i in range(num_records):
            # Tính toán mật độ dựa trên giờ trong ngày (cao điểm buổi sáng/chiều)
            hour = current_time.hour
            
            # Mật độ cao vào giờ cao điểm (7-9h, 16-18h)
            if (7 <= hour <= 9) or (16 <= hour <= 18):
                base_density = random.randint(15, 25)  # Tắc nghẽn
                status = "Tac"
            elif (10 <= hour <= 16):
                base_density = random.randint(8, 15)   # Bình thường
                status = "Binh thuong"
            elif (19 <= hour <= 23):
                base_density = random.randint(5, 12)   # Nhẹ
                status = "Nhe"
            else:
                base_density = random.randint(2, 8)    # Thông thoáng (đêm/sáng sớm)
                status = "Thong thoang"
            
            # Thêm nhiễu ngẫu nhiên
            total_vehicles = base_density + random.randint(-2, 3)
            total_vehicles = max(1, min(30, total_vehicles))  # Giới hạn 1-30
            
            # Cập nhật status dựa vào total_vehicles
            if total_vehicles > 15:
                status = "Tac"
            elif total_vehicles > 10:
                status = "Binh thuong"
            elif total_vehicles > 5:
                status = "Nhe"
            else:
                status = "Thong thoang"
            
            # Tạo bản ghi (chỉ 3 trường: timestamp, total_vehicles, status)
            record = {
                "timestamp": current_time,
                "total_vehicles": total_vehicles,
                "status": status
            }
            
            records.append(record)
            current_time += time_interval
            
            # Hiển thị tiến độ
            if (i + 1) % 200 == 0:
                logger.info(f"  Tao {i + 1}/{num_records} ban ghi...")
        
        # Chèn vào MongoDB
        logger.info(f"\nChen {len(records)} ban ghi vao MongoDB...")
        result = collection.insert_many(records)
        logger.info(f"Chen thanh cong! {len(result.inserted_ids)} ban ghi da duoc them")
        
        total_count = collection.count_documents({})
        logger.info(f"\nTong so ban ghi trong density_logs: {total_count}")
        
        # Thống kê trạng thái
        pipeline = [
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        logger.info("\nPhan bo theo trang thai:")
        for item in collection.aggregate(pipeline):
            logger.info(f"  - {item['_id']}: {item['count']} ban ghi")
        
        # Thống kê mật độ
        logger.info("\nThong ke mat do:")
        total_sum = collection.aggregate([{"$group": {"_id": None, "total": {"$sum": "$total_vehicles"}}}])
        for result in total_sum:
            avg_density = result['total'] / total_count if total_count > 0 else 0
            logger.info(f"  - Trung binh mat do: {avg_density:.2f} xe/gio")
        
        min_vehicles = collection.find().sort({"total_vehicles": 1}).limit(1)
        max_vehicles = collection.find().sort({"total_vehicles": -1}).limit(1)
        
        logger.info("\n" + "=" * 70)
        logger.info("TAO DU LIEU THANH CONG!")
        logger.info("=" * 70)
        logger.info("\nBan co the su dung du lieu nay de huan luyen mo hinh")
        logger.info("Chay lenh: python train_model.py")
        
        client.close()
        
    except Exception as e:
        logger.error(f"\nLoi: {e}")
        logger.error("Kiem tra xem MongoDB da chay chua? (mongosh)")


def main():
    """Entry point"""
    try:
        # Tạo 2000 bản ghi trong 14 ngày
        generate_sample_data(num_records=2000, days=14)
    except KeyboardInterrupt:
        logger.info("\nQua trinh bi dung boi nguoi dung")
    except Exception as e:
        logger.error(f"\nLoi: {e}")


if __name__ == "__main__":
    main()
