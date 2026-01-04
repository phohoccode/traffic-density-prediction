"""
Traffic Predictor Module - Lớp mô hình dự báo mật độ giao thông
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficPredictor:
    """
    Mô hình dự báo mật độ giao thông sử dụng Machine Learning
    
    Hỗ trợ 3 loại mô hình:
    - Random Forest: Nhanh, ổn định
    - Gradient Boosting: Chính xác cao
    - Neural Network: Linh hoạt nhất
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Khởi tạo mô hình
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', hoặc 'neural_network'
        """
        
        logger.info(f"Khởi tạo mô hình: {model_type}")
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, # số cây trong rừng
                max_depth=20,       # độ sâu tối đa của mỗi cây
                min_samples_split=5, # số mẫu tối thiểu để chia nút
                min_samples_leaf=2, # số mẫu tối thiểu tại lá
                random_state=42, # để tái lập kết quả
                n_jobs=-1, # sử dụng tất cả CPU cores
                verbose=0 # không in log chi tiết
            )
        
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100, # số cây
                learning_rate=0.1, # tốc độ học
                max_depth=5,      # độ sâu tối đa
                min_samples_split=5, # số mẫu tối thiểu để chia nút
                min_samples_leaf=2, # số mẫu tối thiểu tại lá
                random_state=42, # để tái lập kết quả
                verbose=0 # không in log chi tiết
            )
        
        elif model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16), # 3 lớp ẩn với 64, 32, 16 neurons
                activation='relu', # hàm kích hoạt ReLU
                solver='adam', # thuật toán tối ưu Adam
                alpha=0.0001, # hệ số điều chuẩn
                batch_size=32, # kích thước batch
                learning_rate='adaptive', # tốc độ học thích nghi
                max_iter=500, # số epoch tối đa
                random_state=42, # để tái lập kết quả
                early_stopping=True, # dừng sớm nếu không cải thiện
                validation_fraction=0.1 # dùng 10% dữ liệu để validation
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        self.scaler = None
        self.lookback_minutes = 30
        self.is_trained = False
    
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Huấn luyện mô hình
        
        Args:
            X_train: Dữ liệu huấn luyện
            y_train: Nhãn huấn luyện
            X_val: Dữ liệu validation (tùy chọn)
            y_val: Nhãn validation (tùy chọn)
        """
        
        logger.info(f"\nBắt đầu huấn luyện mô hình {self.model_type.upper()}...")
        logger.info(f"   Training samples: {X_train.shape[0]}")
        
        if X_val is not None:
            logger.info(f"   Validation samples: {X_val.shape[0]}")
        
        # Huấn luyện
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info("Huấn luyện hoàn tất")
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện: {e}")
            return
        
        # Đánh giá training
        y_pred_train = self.model.predict(X_train)  # Dự báo trên tập training
        train_mse = mean_squared_error(y_train, y_pred_train) # MSE training
        train_rmse = np.sqrt(train_mse) # RMSE training
        train_mae = mean_absolute_error(y_train, y_pred_train) # MAE training
        train_r2 = r2_score(y_train, y_pred_train) # R² training
        
        logger.info(f"\nKết Quả Training:")
        logger.info(f"   MSE:  {train_mse:.6f}") # MSE: là số bình phương trung bình
        logger.info(f"   RMSE: {train_rmse:.4f}") # RMSE: là căn bậc hai của MSE
        logger.info(f"   MAE:  {train_mae:.4f}") # MAE: là sai số tuyệt đối trung bình
        logger.info(f"   R²:   {train_r2:.4f}") # R²: hệ số xác định
        
        # Đánh giá validation
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            val_mape = mean_absolute_percentage_error(y_val, y_pred_val)
            
            logger.info(f"\nKết Quả Validation:")
            logger.info(f"   MSE:  {val_mse:.6f}")
            logger.info(f"   RMSE: {val_rmse:.4f}")
            logger.info(f"   MAE:  {val_mae:.4f}")
            logger.info(f"   R²:   {val_r2:.4f}")
            logger.info(f"   MAPE: {val_mape:.4f}")
            
            # Cảnh báo nếu overfitting
            if train_r2 - val_r2 > 0.1:
                logger.warning("Cảnh báo: Có dấu hiệu Overfitting (Train R² >> Val R²)")
    
    
    def predict(self, X, scaler):
        """
        Dự báo mật độ giao thông
        
        Args:
            X: Dữ liệu đầu vào (history lookback phút)
            scaler: MinMaxScaler để denormalize kết quả
        
        Returns:
            Dự báo mật độ xe (số lượng)
        """
        
        if not self.is_trained:
            logger.error("Mô hình chưa được huấn luyện")
            return None
        
        # Reshape nếu cần
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        try:
            # Dự báo (normalized)
            y_pred_normalized = self.model.predict(X)
            
            # Denormalize về giá trị thực
            y_pred_actual = scaler.inverse_transform(
                y_pred_normalized.reshape(-1, 1)
            ).flatten()
            
            return y_pred_actual[0]
        
        except Exception as e:
            logger.error(f"Lỗi khi dự báo: {e}")
            return None
    
    
    def predict_batch(self, X, scaler):
        """
        Dự báo nhiều mẫu
        
        Args:
            X: Ma trận đầu vào (samples, features)
            scaler: MinMaxScaler
        
        Returns:
            np.array: Dự báo cho tất cả mẫu
        """
        
        if not self.is_trained:
            logger.error("Mô hình chưa được huấn luyện")
            return None
        
        try:
            y_pred_normalized = self.model.predict(X)
            y_pred_actual = scaler.inverse_transform(
                y_pred_normalized.reshape(-1, 1)
            ).flatten()
            
            return y_pred_actual
        
        except Exception as e:
            logger.error(f"Lỗi khi dự báo batch: {e}")
            return None
    
    
    def get_feature_importance(self):
        """
        Lấy độ quan trọng của các feature
        (Chỉ hoạt động với Random Forest & Gradient Boosting)
        
        Returns:
            np.array: Độ quan trọng của mỗi feature
        """
        
        if self.model_type == 'neural_network':
            logger.warning("Neural Network không hỗ trợ feature importance")
            return None
        
        try:
            importance = self.model.feature_importances_
            
            logger.info("Độ quan trọng các feature (Top 10):")
            top_indices = np.argsort(importance)[-10:][::-1]
            for idx in top_indices:
                logger.info(f"   Feature {idx}: {importance[idx]:.4f}")
            
            return importance
        
        except Exception as e:
            logger.error(f"Lỗi khi lấy feature importance: {e}")
            return None
    
    
    def save(self, filepath):
        """
        Lưu mô hình vào file
        
        Args:
            filepath: Đường dẫn file lưu
        """
        
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Mô hình đã lưu: {filepath}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")
    
    
    def load(self, filepath):
        """
        Tải mô hình từ file
        
        Args:
            filepath: Đường dẫn file mô hình
        """
        
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Mô hình đã tải: {filepath}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")


if __name__ == "__main__":
    logger.info("Module traffic_predictor sẵn sàng sử dụng")

