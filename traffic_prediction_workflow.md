# Quy trình hoạt động của mô hình dự đoán mật độ giao thông

## 1. Chuẩn bị dữ liệu

- **Dữ liệu lịch sử giao thông**: Dữ liệu được thu thập từ các cảm biến hoặc nguồn khác, bao gồm số lượng xe trong các khoảng thời gian nhất định.
- **Tiền xử lý dữ liệu**:
  - Dữ liệu được chuẩn hóa (normalization) bằng cách sử dụng `MinMaxScaler` để đưa các giá trị về cùng một khoảng, giúp mô hình học hiệu quả hơn.
  - Dữ liệu được định dạng thành các chuỗi thời gian (time series) với `lookback_minutes` (ví dụ: 30 phút lịch sử) để dự đoán `forecast_horizon` (ví dụ: 60 phút tiếp theo).

## 2. Chọn mô hình dự đoán

Ba loại mô hình được hỗ trợ:

1. **Random Forest**:

   - Là một tập hợp các cây quyết định (decision trees) hoạt động theo nguyên tắc "bagging" (Bootstrap Aggregating).
   - Mỗi cây được huấn luyện trên một tập con dữ liệu và đưa ra dự đoán. Kết quả cuối cùng là trung bình của các dự đoán từ các cây.
   - **Ưu điểm**: Nhanh, ổn định, ít bị overfitting.
   - **Nhược điểm**: Hiệu suất có thể kém hơn trên dữ liệu phức tạp.

2. **Gradient Boosting**:

   - Là một tập hợp các cây quyết định hoạt động theo nguyên tắc "boosting".
   - Các cây được xây dựng tuần tự, mỗi cây mới cố gắng sửa lỗi của cây trước đó.
   - **Ưu điểm**: Chính xác cao, hiệu quả trên dữ liệu phức tạp.
   - **Nhược điểm**: Chậm hơn Random Forest, dễ bị overfitting nếu không điều chỉnh tốt.

3. **Neural Network**:
   - Là một mạng nơ-ron nhân tạo với các lớp ẩn (hidden layers) và các hàm kích hoạt (activation functions) như `ReLU`.
   - Mạng học các mẫu phức tạp trong dữ liệu thông qua quá trình lan truyền ngược (backpropagation).
   - **Ưu điểm**: Linh hoạt, có thể học các mẫu phi tuyến tính phức tạp.
   - **Nhược điểm**: Cần nhiều dữ liệu và tài nguyên tính toán, dễ bị overfitting.

## 3. Huấn luyện mô hình

- **Chia dữ liệu**: Dữ liệu được chia thành tập huấn luyện (training) và tập kiểm tra (validation).
- **Huấn luyện**:
  - Mô hình được huấn luyện trên tập huấn luyện bằng cách tối ưu hóa hàm mất mát (loss function).
  - Các siêu tham số (hyperparameters) như số lượng cây, độ sâu cây, số lượng lớp ẩn, v.v., được điều chỉnh để đạt hiệu suất tốt nhất.
- **Đánh giá**:
  - Mô hình được đánh giá trên tập kiểm tra bằng các chỉ số như:
    - **R² Score**: Đo lường mức độ phù hợp của mô hình.
    - **MAE (Mean Absolute Error)**: Sai số trung bình tuyệt đối.
    - **MSE (Mean Squared Error)**: Sai số bình phương trung bình.

## 4. Lựa chọn mô hình tốt nhất

- Kết quả của các mô hình được so sánh dựa trên các chỉ số đánh giá.
- Mô hình có hiệu suất tốt nhất (cao nhất về R², thấp nhất về MAE) được chọn làm mô hình cuối cùng.

## 5. Lưu mô hình và scaler

- Mô hình tốt nhất và scaler được lưu vào file (`traffic_predictor.pkl` và `traffic_scaler.pkl`) để sử dụng trong ứng dụng dự báo.

## 6. Dự báo mật độ giao thông

- **Dữ liệu đầu vào**: Lấy dữ liệu lịch sử gần nhất (ví dụ: 30 phút).
- **Chuẩn hóa**: Dữ liệu được chuẩn hóa bằng scaler đã lưu.
- **Dự báo**:
  - Mô hình dự đoán số lượng xe trong khoảng thời gian tiếp theo (ví dụ: 1 giờ).
  - Kết quả được "denormalize" để đưa về giá trị thực tế.
- **Xác định trạng thái giao thông**:
  - Dựa trên số lượng xe dự báo, trạng thái giao thông được phân loại:
    - **Thông thoáng**: 🟢
    - **Đông đúc**: 🟡
    - **Tắc nghẽn**: 🔴

## 7. Ứng dụng trong hệ thống

- Mô hình được tích hợp vào ứng dụng Streamlit để hiển thị dự báo và khuyến nghị cho người dùng.
- Người dùng có thể xem dự báo giao thông trong 1-3 giờ tới và nhận khuyến nghị về tuyến đường.
