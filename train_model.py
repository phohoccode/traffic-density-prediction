"""
Train Model Script - Huấn luyện mô hình dự báo
Chạy lệnh: python train_model.py
"""

from data_preparation import prepare_training_data
from traffic_predictor import TrafficPredictor
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save_models():
    """
    Quy trình huấn luyện và lưu mô hình
    1. Chuẩn bị dữ liệu
    2. So sánh 3 loại mô hình
    3. Lưu mô hình tốt nhất
    """
    
    logger.info("=" * 70)
    logger.info("BAT DAU QUY TRINH HUAN LUYEN MO HINH")
    logger.info("=" * 70)
    
    # ===== BƯỚC 1: CHUẨN BỊ DỮ LIỆU =====
    logger.info("\nBUOC 1: Chuan bi du lieu...")
    logger.info("-" * 70)
    
    X, y, scaler = prepare_training_data(
        lookback_minutes=30, # sử dụng 30 phút lịch sử
        forecast_horizon=60 # dự báo 60 phút tới
    )
    
    if X is None:
        logger.error("Chuan bi du lieu that bai. Kiem tra MongoDB!")
        return
    
    # Chia train/validation
    logger.info("\nChia du lieu...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training: {X_train.shape[0]} samples")
    logger.info(f"Validation: {X_val.shape[0]} samples")
    
    # ===== BƯỚC 2: HUẤN LUYỆN VÀ SO SÁNH 3 MÔ HÌNH =====
    logger.info("\n" + "=" * 70)
    logger.info("BUOC 2: Huan luyen va so sanh 3 mo hinh")
    logger.info("=" * 70)
    
    models_to_test = [
        'random_forest',     # là mô hình rừng ngẫu nhiên
        'gradient_boosting', # là mô hình tăng cường gradient
        'neural_network'     # là mô hình mạng nơ-ron nhân tạo
    ]
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_model_type = None
    
    for model_type in models_to_test:
        logger.info(f"\n{'='*70}")
        logger.info(f"Mo hinh: {model_type.upper()}")
        logger.info(f"{'='*70}")
        
        # Tạo và huấn luyện
        predictor = TrafficPredictor(model_type=model_type)
        predictor.train(X_train, y_train, X_val, y_val)
        
        # Lưu kết quả
        y_val_pred = predictor.model.predict(X_val)
        r2 = predictor.model.score(X_val, y_val)
        mae = np.mean(np.abs(y_val - y_val_pred))
        
        results[model_type] = {
            'model': predictor,
            'r2': r2,
            'mae': mae
        }
        
        # Chọn mô hình tốt nhất
        if r2 > best_r2:
            best_r2 = r2
            best_model = predictor
            best_model_type = model_type
    
    # ===== BƯỚC 3: HIỂN THỊ KẾT QUẢ SO SÁNH =====
    logger.info("\n" + "=" * 70)
    logger.info("KET QUA SO SANH")
    logger.info("=" * 70)
    
    logger.info("\n{:<25} {:<15} {:<15}".format("Mô hình", "R² Score", "MAE"))
    logger.info("-" * 55)
    
    for model_type, metrics in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        logger.info("{:<25} {:<15.4f} {:<15.4f}".format(
            model_type.upper(),
            metrics['r2'],
            metrics['mae']
        ))
    
    # ===== BƯỚC 4: LƯU MÔ HÌNH TỐT NHẤT =====
    logger.info("\n" + "=" * 70)
    logger.info("BUOC 3: Luu mo hinh tot nhat")
    logger.info("=" * 70)
    
    logger.info(f"\nMo hinh tot nhat: {best_model_type.upper()}")
    logger.info(f"   R² Score: {best_r2:.4f}") # R² là 
    logger.info(f"   MAE:      {results[best_model_type]['mae']:.4f}")
    
    # Tạo thư mục weights nếu chưa có
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    # Lưu model
    model_path = weights_dir / 'traffic_predictor.pkl'
    best_model.save(str(model_path))
    
    # Lưu scaler
    scaler_path = weights_dir / 'traffic_scaler.pkl'
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"Scaler da luu: {scaler_path}")
    
    # ===== BƯỚC 5: THỐNG KÊ CUỐI CÙNG =====
    logger.info("\n" + "=" * 70)
    logger.info("THONG KE MO HINH CUOI CUNG")
    logger.info("=" * 70)
    
    y_train_pred = best_model.predict_batch(X_train, scaler)
    y_val_pred = best_model.model.predict(X_val)
    
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    val_mae = np.mean(np.abs(y_val - y_val_pred))
    
    logger.info(f"\nTraining:")
    logger.info(f"   MAE: {train_mae:.4f} xe")
    
    logger.info(f"\nValidation:")
    logger.info(f"   MAE: {val_mae:.4f} xe")
    logger.info(f"   R²:  {best_r2:.4f}")
    
    # Kiểm tra feature importance
    if best_model_type != 'neural_network':
        logger.info(f"\nFeature Importance:")
        best_model.get_feature_importance()
    
    logger.info("\n" + "=" * 70)
    logger.info("HUAN LUYEN THANH CONG!")
    logger.info("=" * 70)
    logger.info(f"\nFiles da luu:")
    logger.info(f"   - Model: weights/traffic_predictor.pkl")
    logger.info(f"   - Scaler: weights/traffic_scaler.pkl")
    logger.info(f"\nSu dung mo hinh nay trong Streamlit App!")
    
    return best_model, scaler


def main():
    """Entry point"""
    try:
        train_and_save_models()
    except KeyboardInterrupt:
        logger.info("\nQua trinh huan luyen bi dung boi nguoi dung")
    except Exception as e:
        logger.error(f"\nLoi: {e}")


if __name__ == "__main__":
    main()

