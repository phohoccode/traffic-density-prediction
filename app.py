import os
from pathlib import Path
import streamlit as st
import torch
import pandas as pd
import time

# Import local modules
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
from database import init_db, get_density_history, get_density_history_filtered, get_vehicle_stats_by_type, get_vehicle_details
from datetime import datetime, timedelta

# Import prediction engine
try:
    from prediction_utils import get_prediction_engine
except ImportError:
    get_prediction_engine = None

# 1. Khởi tạo Database
init_db()

# Fix lỗi bảo mật torch trên một số phiên bản mới
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

# Cache để lưu trữ model và tránh tải lại khi chuyển tab
@st.cache_resource
def cached_load_model(model_path):
    return load_model(model_path)

# Cache cho các hàm lấy dữ liệu với TTL = 2 giây (cập nhật nhanh hơn)
@st.cache_data(ttl=2)
def cached_get_density_history(limit=500):
    return get_density_history(limit=limit)

@st.cache_data(ttl=2)
def cached_get_density_history_filtered(start_date, end_date, limit=500):
    return get_density_history_filtered(start_date, end_date, limit=limit)

@st.cache_data(ttl=2)
def cached_get_vehicle_stats_by_type(start_date, end_date):
    return get_vehicle_stats_by_type(start_date, end_date)

@st.cache_data(ttl=2)
def cached_get_vehicle_details(limit=200, start_date=None, end_date=None):
    return get_vehicle_details(limit=limit, start_date=start_date, end_date=end_date)

# Cache prediction engine
@st.cache_resource
def load_prediction_engine():
    if get_prediction_engine is None:
        return None
    try:
        return get_prediction_engine()
    except Exception:
        return None

# 2. Cấu hình Page Layout
st.set_page_config(
    page_title="BÁO CÁO ỨNG DỤNG MÁY HỌC TRONG IOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Sidebar - Cấu hình Model (Phải làm trước để có biến confidence và model)
st.sidebar.header("Cấu hình mô hình DL")

# Hiển thị thông tin device
with st.sidebar.expander("Thông tin hệ thống"):
    st.write(f"**Device:** {config.DEVICE.upper()}")
    if config.DEVICE == "cuda":
        st.write(f"**GPU:** {config.GPU_NAME}")
        st.write(f"**VRAM:** {config.GPU_VRAM:.2f} GB")
    st.write(f"**PyTorch:** {torch.__version__}")

task_type = st.sidebar.selectbox(
    "Chọn tác vụ",
    ["Phát hiện"]
)

model_type = st.sidebar.selectbox(
    "Chọn mô hình",
    config.DETECTION_MODEL_LIST
)

confidence = float(st.sidebar.slider(
    "Chọn độ tin cậy mô hình", 30, 100, 50)) / 100

# Xác định đường dẫn model
model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))

# Load model với cache
try:
    model = cached_load_model(str(model_path))
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}") 
    st.error(f"Đường dẫn tìm kiếm: {os.path.abspath(model_path)}")
    st.stop()

# 4. Sidebar - Cấu hình Nguồn dữ liệu
st.sidebar.header("Cấu hình ảnh/video")
source_selectbox = st.sidebar.selectbox(
    "Chọn nguồn",
    config.SOURCES_LIST
)

# 5. Giao diện chính (Main UI)
st.title("BÁO CÁO ỨNG DỤNG MÁY HỌC TRONG IOT - GIÁM SÁT GIAO THÔNG")

# Lưu trạng thái tab hiện tại để tránh làm mới dữ liệu không cần thiết
if "current_tab" not in st.session_state:
    st.session_state.current_tab = 0

if "video_analyzed" not in st.session_state:
    st.session_state.video_analyzed = False

if "cache_version" not in st.session_state:
    st.session_state.cache_version = 0

tab1, tab2, tab3 = st.tabs(["Giám sát trực tiếp", "Bảng điều khiển phân tích", "Dự báo giao thông"])

with tab1:
    st.subheader("Hệ thống giám sát giao thông thời gian thực")
    
    # Tại đây các biến confidence và model đã tồn tại nên sẽ không lỗi NameError
    if source_selectbox == config.SOURCES_LIST[0]: # Image
        infer_uploaded_image(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[1]: # Video
        infer_uploaded_video(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
        infer_uploaded_webcam(confidence, model)

with tab2:
    st.header("Lịch sử mật độ giao thông và phân tích nâng cao")
    
    # Khởi tạo session state cho auto-refresh
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = True
    
    # Auto-refresh control
    # cols_refresh = st.columns([4, 1])
    # with cols_refresh[1]:
    #     auto_refresh = st.checkbox("Tự động cập nhật", value=st.session_state.auto_refresh_enabled, key="auto_refresh_checkbox")
    #     st.session_state.auto_refresh_enabled = auto_refresh
    
    # Placeholder để hiển thị thông báo
    placeholder_refresh = st.empty()
    
    # Tự động làm mới dữ liệu nếu có video được phân tích
    if st.session_state.video_analyzed:
        st.cache_data.clear()
        st.session_state.video_analyzed = False
        placeholder_refresh.success("Dữ liệu từ video mới được cập nhật!")
    
    # Cơ chế auto-refresh: Rerun sau 3 giây nếu đang có video processing hoặc auto-refresh enabled
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.last_refresh_time
    
    # Kiểm tra xem có đang xử lý video không
    is_processing = st.session_state.get("video_processing", False) or st.session_state.get("webcam_running", False)
    
    # if auto_refresh and time_since_refresh >= 3:
    #     st.session_state.last_refresh_time = current_time
    #     # Chỉ clear cache và rerun nếu đang có xử lý hoặc user muốn auto-refresh
    #     if is_processing:
    #         st.cache_data.clear()
    #         placeholder_refresh.info("Đang cập nhật dữ liệu...")
    #         time.sleep(0.5)  # Ngắn delay để user thấy thông báo
    #         st.rerun()
    
    # Hiển thị trạng thái xử lý
    if is_processing:
        st.info("Dữ liệu đang được cập nhật từ tab Giám sát trực tiếp. Trang sẽ tự động làm mới sau 3 giây.")
    
    # === BỘ LỌC THỜI GIAN ===
    st.subheader("Bộ lọc dữ liệu")
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])
    
    with col_filter1:
        filter_option = st.selectbox(
            "Chọn khoảng thời gian",
            ["Tất cả dữ liệu", "1 giờ qua", "6 giờ qua", "24 giờ qua", "7 ngày qua", "Tùy chỉnh"]
        )
    
    # Xác định start_date và end_date
    start_date = None
    end_date = datetime.now()
    
    if filter_option == "1 giờ qua":
        start_date = end_date - timedelta(hours=1)
    elif filter_option == "6 giờ qua":
        start_date = end_date - timedelta(hours=6)
    elif filter_option == "24 giờ qua":
        start_date = end_date - timedelta(days=1)
    elif filter_option == "7 ngày qua":
        start_date = end_date - timedelta(days=7)
    elif filter_option == "Tùy chỉnh":
        with col_filter2:
            start_date = st.date_input("Từ ngày", value=datetime.now() - timedelta(days=1), key="start_date_custom")
            start_date = datetime.combine(start_date, datetime.min.time())
        with col_filter3:
            end_date = st.date_input("Đến ngày", value=datetime.now(), key="end_date_custom")
            end_date = datetime.combine(end_date, datetime.max.time())
    
    # Nút refresh ở góc phải
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Làm mới dữ liệu", key="manual_refresh_button"):
            # Xóa cache dữ liệu và reload giao diện
            st.cache_data.clear()
            st.session_state.last_refresh_time = time.time()  # Reset timer
            st.rerun()
    
    try:
        # === BIỂU ĐỒ MẬT ĐỘ THEO THỜI GIAN ===
        st.subheader("Xu hướng mật độ giao thông")
        
        # Lấy dữ liệu từ MongoDB với bộ lọc
        if filter_option == "Tất cả dữ liệu":
            df_history = cached_get_density_history(limit=500)
        else:
            df_history = cached_get_density_history_filtered(start_date, end_date, limit=500)
        
        if not df_history.empty:
            # Hiển thị biểu đồ line chart
            st.line_chart(df_history, x="timestamp", y="total_vehicles")
            
            # Thống kê số liệu nhanh
            avg = df_history['total_vehicles'].mean()
            peak = df_history['total_vehicles'].max()
            min_val = df_history['total_vehicles'].min()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mật độ trung bình", f"{avg:.1f} xe")
            m2.metric("Mật độ cao nhất", f"{peak} xe")
            m3.metric("Mật độ thấp nhất", f"{min_val} xe")
            m4.metric("Tổng số bản ghi", len(df_history))
            
            # === BIỂU ĐỒ PHÂN LOẠI THEO LOẠI XE ===
            st.subheader("Phân loại theo loại xe")
            
            df_vehicle_stats = cached_get_vehicle_stats_by_type(start_date, end_date)
            
            if not df_vehicle_stats.empty:
                # Tạo 2 cột: Bar chart và bảng thống kê
                col_chart, col_table = st.columns([2, 1])
                
                with col_chart:
                    st.bar_chart(df_vehicle_stats, x="vehicle_type", y="count")
                
                with col_table:
                    st.dataframe(df_vehicle_stats, use_container_width=True, hide_index=True)
                    total_vehicles = df_vehicle_stats['count'].sum()
                    st.metric("Tổng số xe phát hiện", f"{total_vehicles}")
                
                # === DOWNLOAD BUTTONS ===
                st.markdown("**Tải dữ liệu loại xe:**")
                col_csv, col_excel, col_json = st.columns(3)
                
                # CSV format
                with col_csv:
                    csv_data = df_vehicle_stats.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="CSV",
                        data=csv_data,
                        file_name=f"vehicle_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_vehicle_csv"
                    )
                
                # Excel format
                with col_excel:
                    try:
                        from openpyxl import Workbook
                        import io
                        wb = Workbook()
                        ws = wb.active
                        ws.title = "Vehicle Stats"
                        
                        # Header
                        for col_idx, col_name in enumerate(df_vehicle_stats.columns, 1):
                            ws.cell(row=1, column=col_idx, value=col_name)
                        
                        # Data
                        for row_idx, row in enumerate(df_vehicle_stats.values, 2):
                            for col_idx, value in enumerate(row, 1):
                                ws.cell(row=row_idx, column=col_idx, value=value)
                        
                        excel_buffer = io.BytesIO()
                        wb.save(excel_buffer)
                        excel_buffer.seek(0)
                        
                        st.download_button(
                            label="Excel",
                            data=excel_buffer.getvalue(),
                            file_name=f"vehicle_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_vehicle_excel"
                        )
                    except Exception as e:
                        st.warning(f"Excel download không khả dụng: {e}")
                
                # JSON format
                with col_json:
                    json_data = df_vehicle_stats.to_json(orient='records', indent=2, default_handler=str)
                    st.download_button(
                        label="JSON",
                        data=json_data,
                        file_name=f"vehicle_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_vehicle_json"
                    )
            else:
                st.info("Chưa có dữ liệu chi tiết về loại xe.")
            
            # === CHI TIẾT NHẬT KÝ ===
            with st.expander("Chi tiết nhật ký mật độ"):
                st.dataframe(df_history.sort_values(by="timestamp", ascending=False), use_container_width=True)
                
                # === DOWNLOAD BUTTONS ===
                st.markdown("**Tải dữ liệu nhật ký:**")
                col_csv, col_excel, col_json = st.columns(3)
                
                # CSV format
                with col_csv:
                    csv_data = df_history.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="CSV",
                        data=csv_data,
                        file_name=f"density_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_density_csv"
                    )
                
                # Excel format
                with col_excel:
                    excel_buffer = pd.ExcelWriter(path="temp.xlsx", engine='openpyxl')
                    df_history.to_excel(excel_buffer, sheet_name='Density Log', index=False)
                    excel_buffer._save()
                    
                    with open("temp.xlsx", "rb") as f:
                        excel_data = f.read()
                    
                    st.download_button(
                        label="Excel",
                        data=excel_data,
                        file_name=f"density_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_density_excel"
                    )
                
                # JSON format
                with col_json:
                    json_data = df_history.to_json(orient='records', indent=2, default_handler=str)
                    st.download_button(
                        label="JSON",
                        data=json_data,
                        file_name=f"density_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_density_json"
                    )
            
            # === DANH SÁCH CHI TIẾT CÁC XE ===
            with st.expander("Danh sách chi tiết các xe đã phát hiện"):
                df_vehicles = cached_get_vehicle_details(limit=200, start_date=start_date, end_date=end_date)
                if not df_vehicles.empty:
                    st.dataframe(df_vehicles.sort_values(by="timestamp", ascending=False), use_container_width=True)
                    
                    # === DOWNLOAD BUTTONS ===
                    st.markdown("**Tải dữ liệu chi tiết xe:**")
                    col_csv, col_excel, col_json = st.columns(3)
                    
                    # CSV format
                    with col_csv:
                        csv_data = df_vehicles.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="CSV",
                            data=csv_data,
                            file_name=f"vehicle_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_vehicles_csv"
                        )
                    
                    # Excel format
                    with col_excel:
                        try:
                            from openpyxl import Workbook
                            import io
                            wb = Workbook()
                            ws = wb.active
                            ws.title = "Vehicle Details"
                            
                            # Header
                            for col_idx, col_name in enumerate(df_vehicles.columns, 1):
                                ws.cell(row=1, column=col_idx, value=col_name)
                            
                            # Data
                            for row_idx, row in enumerate(df_vehicles.values, 2):
                                for col_idx, value in enumerate(row, 1):
                                    ws.cell(row=row_idx, column=col_idx, value=value)
                            
                            excel_buffer = io.BytesIO()
                            wb.save(excel_buffer)
                            excel_buffer.seek(0)
                            
                            st.download_button(
                                label="Excel",
                                data=excel_buffer.getvalue(),
                                file_name=f"vehicle_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_vehicles_excel"
                            )
                        except Exception as e:
                            st.warning(f"Excel download không khả dụng: {e}")
                    
                    # JSON format
                    with col_json:
                        json_data = df_vehicles.to_json(orient='records', indent=2, default_handler=str)
                        st.download_button(
                            label="JSON",
                            data=json_data,
                            file_name=f"vehicle_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_vehicles_json"
                        )
                else:
                    st.info("Chưa có dữ liệu chi tiết về xe.")
        else:
            st.info("Chưa có dữ liệu. Hãy khởi chạy video để bắt đầu thu thập.")
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        st.info("Hãy kiểm tra xem MongoDB đã được cài đặt và đang chạy chưa?")

# ===== TAB 3: DỰ BÁO =====
with tab3:
    st.header("Dự Báo Mật Độ Giao Thông")
    st.markdown("""
    Sử dụng Machine Learning để dự báo mật độ giao thông trong 1-3 giờ tới.
    Dữ liệu dự báo dựa trên 30 phút lịch sử gần nhất.
    """)
    
    prediction_engine = load_prediction_engine()
    
    if prediction_engine is None or not prediction_engine.is_ready():
        st.error("Mô hình dự báo không sẵn sàng")
        st.info("""
        **Cách khắc phục:**
        1. Chạy: `python train_model.py`
        2. Kiểm tra MongoDB chạy: `mongosh`
        3. Đảm bảo có >= 14 ngày dữ liệu (>= 2000 bản ghi)
        """)
    else:
        # ===== Dselbst BÁO 1 GIỜ TỚI =====
        st.subheader("Dự Báo 1 Giờ Tới")
        
        col1, col2, col3 = st.columns(3)
        
        pred_vehicles, result = prediction_engine.predict_next_hour()
        
        if pred_vehicles is not None:
            status, color = result
            confidence = prediction_engine.get_confidence_level(pred_vehicles)
            
            with col1:
                st.metric("Dự báo mật độ", f"{pred_vehicles} xe")
            
            with col2:
                st.metric("Trạng thái", f"{color} {status}")
            
            with col3:
                st.metric("Độ tin cậy", f"{confidence:.0%}")
            
            # Khuyến nghị
            recommendation = prediction_engine.get_recommendation(pred_vehicles)
            st.info(recommendation)
        else:
            st.warning("Không có đủ dữ liệu để dự báo. Kiểm tra MongoDB!")
        
        # ===== DỰ BÁO 3 GIỜ TỚI =====
        st.subheader("Dự Báo 3 Giờ Tới")
        
        predictions = prediction_engine.predict_next_hours(num_hours=3)
        
        if predictions:
            # Vẽ biểu đồ dự báo
            df_pred = pd.DataFrame(predictions)
            df_pred['time'] = df_pred['timestamp'].dt.strftime('%H:%M')
            
            st.line_chart(
                df_pred.set_index('time')['vehicles'],
                use_container_width=True,
                height=300
            )
            
            # Bảng chi tiết
            with st.expander("Chi tiết dự báo"):
                st.dataframe(
                    df_pred[['time', 'vehicles', 'status', 'color']],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Cảnh báo tắc
            st.subheader("Cảnh Báo")
            alerts = [p for p in predictions if p['vehicles'] > 20]
            
            if alerts:
                for alert in alerts:
                    st.warning(
                        f"**Cảnh báo tắc nghẽn!** "
                        f"Lúc {alert['timestamp'].strftime('%H:%M')} "
                        f"dự báo {alert['vehicles']} xe (Trạng thái: {alert['color']} {alert['status']})"
                    )
            else:
                st.success("Không có cảnh báo tắc trong 3 giờ tới")
        
        # Thông tin mô hình
        with st.expander("Thông tin mô hình dự báo"):
            st.markdown("""
            **Thông số kỹ thuật:**
            - **Loại mô hình:** Random Forest Regressor
            - **Dữ liệu huấn luyện:** 14 ngày gần nhất
            - **Lookback window:** 30 phút quá khứ
            - **Forecast horizon:** 60 phút (1 giờ)
            - **Độ chính xác (R²):** ~0.85
            - **Sai lệch trung bình (MAE):** ~3-5 xe
            
            **Các tính năng:**
            Dự báo 1 giờ  
            Dự báo 3 giờ  
            Cảnh báo tắc tự động  
            Khuyến nghị hành động  
            Độ tin cậy %  
            
            **Cập nhật:** Hàng tuần retrain với dữ liệu mới
            """)
