"""
Prediction Engine - Engine dự báo tích hợp cho Streamlit
"""

import joblib
import numpy as np
from datetime import datetime, timedelta
from data_preparation import get_recent_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Engine dự báo mật độ giao thông
    Tích hợp model, scaler, và logic dự báo
    """
    
    def __init__(self, model_path='weights/traffic_predictor.pkl', 
                 scaler_path='weights/traffic_scaler.pkl'):
        """
        Khởi tạo prediction engine
        
        Args:
            model_path: Đường dẫn model đã huấn luyện
            scaler_path: Đường dẫn scaler
        """
        
        logger.info("Khởi tạo Prediction Engine...")
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.lookback_minutes = 30
            logger.info("Model và Scaler đã tải thành công")
        
        except FileNotFoundError as e:
            logger.error(f"Không tìm thấy file model hoặc scaler: {e}")
            self.model = None
            self.scaler = None
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo engine: {e}")
            self.model = None
            self.scaler = None
    
    
    def is_ready(self):
        """Kiểm tra engine đã sẵn sàng"""
        return self.model is not None and self.scaler is not None
    
    
    def predict_next_hour(self):
        """
        Dự báo mật độ trong 1 giờ tới
        
        Returns:
            tuple: (predicted_vehicles, (status, color))
                - predicted_vehicles: Số xe dự báo (int)
                - status: Trạng thái ("Thong thoang", "Dong duc", "Tac nghen")
                - color: Icon ("🟢", "🟡", "🔴")
        """
        
        if not self.is_ready():
            logger.error("Engine chưa sẵn sàng")
            return None, None
        
        # Lấy dữ liệu gần nhất
        history = get_recent_data(self.lookback_minutes)
        
        if history is None:
            logger.warning("Không có đủ dữ liệu để dự báo")
            return None, None
        
        try:
            # Chuẩn hóa
            history_normalized = self.scaler.transform(
                history.reshape(-1, 1)
            ).flatten()
            
            # Dự báo
            prediction_normalized = self.model.predict(
                history_normalized.reshape(1, -1)
            )[0]
            
            # Denormalize
            prediction_actual = self.scaler.inverse_transform(
                [[prediction_normalized]]
            )[0][0]
            
            # Làm tròn
            prediction_actual = max(0, int(prediction_actual))
            
            # Xác định trạng thái dự báo
            if prediction_actual < 10:
                status = "Thong thoang"
                color = "🟢"
            elif prediction_actual > 20:
                status = "Tac nghen"
                color = "🔴"
            else:
                status = "Dong duc"
                color = "🟡"
            
            logger.info(f"Dự báo 1 giờ tới: {prediction_actual} xe ({status})")
            
            return prediction_actual, (status, color)
        
        except Exception as e:
            logger.error(f"Lỗi khi dự báo: {e}")
            return None, None
    
    
    def predict_next_hours(self, num_hours=3):
        """
        Dự báo cho nhiều giờ tiếp theo
        
        Args:
            num_hours: Số giờ cần dự báo (1-3)
        
        Returns:
            list: Danh sách dự báo
                [{
                    'hour': 1,
                    'timestamp': datetime,
                    'vehicles': 22,
                    'status': 'Dong duc',
                    'color': '🟡'
                }, ...]
        """
        
        if not self.is_ready():
            logger.error("Engine chưa sẵn sàng")
            return []
        
        predictions = []
        num_hours = min(num_hours, 3)  # Giới hạn tối đa 3 giờ
        
        try:
            for hour in range(1, num_hours + 1):
                # Lấy dữ liệu gần nhất
                history = get_recent_data(self.lookback_minutes)
                
                if history is None:
                    logger.warning(f"Không có dữ liệu cho dự báo giờ {hour}")
                    break
                
                # Chuẩn hóa
                history_normalized = self.scaler.transform(
                    history.reshape(-1, 1)
                ).flatten()
                
                # Dự báo
                pred_normalized = self.model.predict(
                    history_normalized.reshape(1, -1)
                )[0]
                
                # Denormalize
                pred_actual = self.scaler.inverse_transform(
                    [[pred_normalized]]
                )[0][0]
                
                pred_actual = max(0, int(pred_actual))
                
                # Xác định trạng thái
                if pred_actual < 10:
                    status = "Thong thoang"
                    color = "🟢"
                elif pred_actual > 20:
                    status = "Tac nghen"
                    color = "🔴"
                else:
                    status = "Dong duc"
                    color = "🟡"
                
                predictions.append({
                    'hour': hour,
                    'timestamp': datetime.now() + timedelta(hours=hour),
                    'vehicles': pred_actual,
                    'status': status,
                    'color': color
                })
        
        except Exception as e:
            logger.error(f"Lỗi khi dự báo nhiều giờ: {e}")
        
        logger.info(f"Dự báo {len(predictions)} giờ tiếp theo")
        return predictions
    
    
    def get_confidence_level(self, prediction):
        """
        Tính mức độ tin cậy của dự báo
        (Có thể mở rộng với thông tin từ model)
        
        Args:
            prediction: Giá trị dự báo
        
        Returns:
            float: Mức độ tin cậy (0-1)
        """
        
        # Quy tắc đơn giản: dự báo giữa 10-20 có tin cậy cao
        if 10 <= prediction <= 20:
            confidence = 0.90
        elif 5 <= prediction < 10 or 20 < prediction <= 30:
            confidence = 0.80
        else:
            confidence = 0.70
        
        return confidence
    
    
    def get_recommendation(self, prediction):
        """
        Đưa khuyến nghị dựa trên dự báo
        
        Args:
            prediction: Giá trị dự báo
        
        Returns:
            str: Khuyến nghị cho người dùng
        """
        
        if prediction < 10:
            return "Giao thông thông thoáng. Có thể di chuyển bình thường."
        elif prediction < 15:
            return "Giao thông bình thường. Lưu ý tăng tốc độ mong muốn."
        elif prediction < 20:
            return "Giao thông đông. Chuẩn bị đợi hoặc chọn tuyến khác."
        else:
            return "Giao thông tắc nghẽn. Khuyến khích chọn tuyến đi khác."


# Singleton pattern - Một instance duy nhất
_engine_instance = None


def get_prediction_engine(model_path='weights/traffic_predictor.pkl',
                         scaler_path='weights/traffic_scaler.pkl'):
    """
    Lấy instance của PredictionEngine (Singleton)
    
    Args:
        model_path: Đường dẫn model
        scaler_path: Đường dẫn scaler
    
    Returns:
        PredictionEngine: Instance engine dự báo
    """
    
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = PredictionEngine(model_path, scaler_path)
    
    return _engine_instance


if __name__ == "__main__":
    # Test
    engine = get_prediction_engine()
    
    if engine.is_ready():
        logger.info("\nTesting Prediction Engine...")
        
        # Dự báo 1 giờ
        pred_1h, (status, color) = engine.predict_next_hour()
        if pred_1h is not None:
            logger.info(f"Dự báo 1 giờ tới: {pred_1h} xe ({status})")
        
        # Dự báo 3 giờ
        predictions = engine.predict_next_hours(3)
        if predictions:
            logger.info("\nDự báo 3 giờ tới:")
            for pred in predictions:
                logger.info(f"Giờ {pred['hour']}: {pred['vehicles']} xe")
    else:
        logger.error("Engine chưa sẵn sàng. Vui lòng kiểm tra model và scaler.")
