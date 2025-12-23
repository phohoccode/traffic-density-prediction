
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import config
from database import insert_density, insert_tracked_vehicle, update_tracked_vehicle_last_seen, insert_vehicle_detail
from datetime import datetime
import threading
import queue
import requests
import json

# Biến đếm frame toàn cục để kiểm soát tốc độ ghi DB
frame_counter = 0
# Set để lưu ID các xe đã tracking (để chỉ lưu xe mới)
tracked_vehicle_ids = set()
# WebSocket Server URL
WEBSOCKET_SERVER_URL = "http://localhost:5000"

# COCO dataset class names mapping (các class ID liên quan đến xe)
COCO_VEHICLE_CLASSES = {
    2: "car",           # Xe ô tô
    3: "motorcycle",    # Xe máy
    5: "bus",           # Xe buýt
    7: "truck"          # Xe tải
}

# ==================== WebSocket Communication Functions ====================
def send_to_websocket(endpoint, data):
    """
    Gửi dữ liệu đến WebSocket server qua REST API
    :param endpoint: API endpoint (e.g., 'density_update', 'vehicle_types_update')
    :param data: Dictionary chứa dữ liệu cần gửi
    """
    try:
        url = f"{WEBSOCKET_SERVER_URL}/emit"
        payload = {
            "event": endpoint,
            "data": data
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers, timeout=2)
        if response.status_code == 200:
            print(f"[WebSocket] Sent {endpoint} successfully")
        else:
            print(f"[WebSocket] Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[WebSocket] Connection refused - server may not be running at {WEBSOCKET_SERVER_URL}")
    except Exception as e:
        print(f"[WebSocket Error] {str(e)}")

def emit_websocket_event(event_name, data):
    """
    Emit event tới WebSocket server
    Wrapper function cho send_to_websocket
    """
    send_to_websocket(event_name, data)

def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    #image = cv2.resize(image, (720, int(720 * (9 / 16))))

    global frame_counter, tracked_vehicle_ids

    # Sử dụng track() thay vì predict() để có tracking ID
    # persist=True giữ tracking ID giữa các frame
    res = model.track(image, conf=conf, persist=True, tracker=config.TRACKER_TYPE)

    # 2. Lấy số lượng xe và đếm theo loại với tracking
    boxes = res[0].boxes
    vehicle_count = 0
    vehicle_types_in_frame = {}  # Đếm số xe từng loại trong frame này
    
    # Đếm xe theo loại trong frame hiện tại và tracking từng xe
    for box in boxes:
        try:
            cls_id = int(box.cls[0])
            
            # Chỉ đếm nếu là xe (class ID trong COCO_VEHICLE_CLASSES)
            if cls_id in COCO_VEHICLE_CLASSES:
                vehicle_count += 1
                vehicle_type = COCO_VEHICLE_CLASSES[cls_id]
                vehicle_types_in_frame[vehicle_type] = vehicle_types_in_frame.get(vehicle_type, 0) + 1
                
                # Lấy tracking ID nếu có
                if box.id is not None:
                    track_id = int(box.id[0])
                    confidence = float(box.conf[0])
                    
                    # Nếu là xe mới (chưa tracking), lưu vào database
                    if track_id not in tracked_vehicle_ids:
                        tracked_vehicle_ids.add(track_id)
                        insert_tracked_vehicle(
                            track_id=track_id,
                            vehicle_class=cls_id,
                            vehicle_type=vehicle_type,
                            confidence=confidence,
                            first_seen_time=datetime.now()
                        )
                        # Thêm vào vehicle_details để có dữ liệu phân loại
                        insert_vehicle_detail(
                            vehicle_class=cls_id,
                            vehicle_type=vehicle_type,
                            direction="unknown",
                            confidence=confidence,
                            track_id=track_id
                        )
                        st.toast(f"Phát hiện {vehicle_type} mới (ID: {track_id})")
                    else:
                        # Update last_seen time cho xe đã tracking
                        update_tracked_vehicle_last_seen(track_id)
        except Exception:
            pass
    
    current_count = vehicle_count
    
    # 3. Logic Cảnh báo (Thresholds)
    status = "Normal"
    color = (0, 255, 0) # Green
    
    if current_count < config.THRESHOLD_CLEAR:
        status = "Thong thoang"
        color = (0, 255, 0)
    elif current_count > config.THRESHOLD_CONGESTED:
        status = "Tac nghen"
        color = (0, 0, 255) # Red
        st.toast(f"Mat do cao: {current_count} xe!")
    else:
        status = "Dong duc"
        color = (0, 255, 255) # Yellow

    # 4. Vẽ thông tin lên Frame
    res_plotted = res[0].plot()
    
    # Vẽ overlay nền đen cho chữ dễ đọc
    cv2.rectangle(res_plotted, (0, 0), (450, 60), (0, 0, 0), -1)
    cv2.putText(res_plotted, f"Count: {current_count} | {status}", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 5. Lưu mật độ vào MongoDB (định kỳ mỗi 30 frame)
    frame_counter += 1
    if frame_counter % config.DB_UPDATE_INTERVAL == 0:
        # Chỉ lưu mật độ tổng thể, không lưu chi tiết vì đã lưu khi tracking xe mới
        insert_density(current_count, status)
        
        # === GỬI DỮ LIỆU QUA WEBSOCKET ===
        # 1. Gửi update mật độ
        emit_websocket_event('density_update', {
            'count': current_count,
            'status': status
        })
        
        # 2. Gửi update loại xe
        if vehicle_types_in_frame:
            emit_websocket_event('vehicle_types_update', {
                'vehicle_types': vehicle_types_in_frame
            })

    inText = 'Xe vào'
    outText = 'Xe ra'
    if config.OBJECT_COUNTER1 != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER1.items()):
            inText += ' - ' + str(key) + ": " +str(value)
    if config.OBJECT_COUNTER != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER.items()):
            outText += ' - ' + str(key) + ": " +str(value)
    
    print(inText)
    print(outText)

    # Display counter info and detected video
    st_count.write(inText + '\n\n' + outText)
    st_count.write(f"### Trạng thái: {status}\nSố xe hiện tại: {current_count}")
    st_frame.image(res_plotted, caption='Real-time Analysis', channels="BGR", width='stretch')


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model loaded on the configured device.
    """
    model = YOLO(model_path)
    # Move model to configured device (GPU or CPU)
    model.to(config.DEVICE)
    st.write(f"Model loaded on {config.DEVICE.upper()}")
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Chọn ảnh...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=uploaded_image,
                caption="Ảnh đã tải lên",
                width='stretch'
            )

    if source_img:
        if st.button("Thực thi"):
            with st.spinner("Đang chạy..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Ảnh đã phát hiện",
                             width='stretch')
                    try:
                        with st.expander("Kết quả phát hiện"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("Chưa có ảnh được tải lên!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    global frame_counter, tracked_vehicle_ids
    
    # Khởi tạo session states
    if "video_processing" not in st.session_state:
        st.session_state.video_processing = False
    if "video_stop_requested" not in st.session_state:
        st.session_state.video_stop_requested = False
    
    source_video = st.sidebar.file_uploader(
        label="Chọn video...",
    )

    if source_video:
        st.video(source_video)

    if source_video:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            start_button = st.button("Bắt đầu phân tích", disabled=st.session_state.video_processing)
        with col2:
            stop_button = st.button("Dừng lại", disabled=not st.session_state.video_processing)
        
        if stop_button:
            st.session_state.video_stop_requested = True
            st.session_state.video_processing = False
            st.info("Đang dừng xử lý video...")
            return
        
        if start_button and not st.session_state.video_processing:
            st.session_state.video_processing = True
            st.session_state.video_stop_requested = False
            
            with st.spinner("Đang xử lý video..."):
                try:
                    # === GỬI SIGNAL BẮT ĐẦU PHÂN TÍCH ===
                    emit_websocket_event('analysis_start', {
                        'message': 'Video analysis started'
                    })
                    
                    frame_counter = 0
                    tracked_vehicle_ids.clear()  # Reset tracking IDs
                    config.OBJECT_COUNTER1 = None
                    config.OBJECT_COUNTER = None
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(source_video.read())
                    tfile.close()
                    
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st_count = st.empty()
                    st_frame = st.empty()
                    st_progress = st.progress(0)
                    
                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    current_frame = 0
                    
                    while vid_cap.isOpened() and not st.session_state.video_stop_requested:
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_count,
                                                     st_frame,
                                                     image
                                                     )
                            current_frame += 1
                            if total_frames > 0:
                                st_progress.progress(current_frame / total_frames)
                        else:
                            vid_cap.release()
                            break
                    
                    vid_cap.release()
                    st_progress.empty()
                    
                    # === GỬI SIGNAL HOÀN THÀNH PHÂN TÍCH ===
                    emit_websocket_event('analysis_complete', {
                        'message': 'Video analysis completed successfully',
                        'total_frames': current_frame
                    })
                    
                    # Báo hiệu cho tab2 biết rằng có dữ liệu mới
                    st.session_state.video_analyzed = True
                    st.session_state.video_processing = False
                    st.success("Xử lý video hoàn tất!")
                    
                except Exception as e:
                    st.error(f"Lỗi khi tải video: {e}")
                    st.session_state.video_processing = False
        
        # Hiển thị trạng thái
        if st.session_state.video_processing:
            st.warning("⚠️ Video đang được xử lý. Bạn có thể chuyển sang tab Bảng điều khiển để xem dữ liệu cập nhật theo thời gian thực.")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    global frame_counter, tracked_vehicle_ids
    
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "webcam_initialized" not in st.session_state:
        st.session_state.webcam_initialized = False
    
    col1, col2 = st.columns([4, 1])
    with col2:
        webcam_toggle = st.checkbox("Chạy Webcam", value=st.session_state.webcam_running)
        
        if webcam_toggle != st.session_state.webcam_running:
            st.session_state.webcam_running = webcam_toggle
            if not webcam_toggle:
                st.session_state.webcam_initialized = False
    
    if st.session_state.webcam_running:
        st.info("Webcam đang chạy. Bạn có thể chuyển sang tab Bảng điều khiển để xem dữ liệu cập nhật.")
        try:
            if not st.session_state.webcam_initialized:
                frame_counter = 0
                tracked_vehicle_ids.clear()  # Reset tracking IDs
                st.session_state.webcam_initialized = True
            
            vid_cap = cv2.VideoCapture(0)
            st_count = st.empty()
            st_frame = st.empty()
            
            # Chỉ chạy một số frame giới hạn mỗi lần rerun để không block quá lâu
            frame_limit = 30  # Xử lý tối đa 30 frames mỗi rerun (~1 giây)
            frame_processed = 0
            
            while st.session_state.webcam_running and frame_processed < frame_limit:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_count,
                        st_frame,
                        image
                    )
                    frame_processed += 1
                else:
                    vid_cap.release()
                    break
            
            vid_cap.release()
            
            # Báo hiệu cho tab2 biết rằng có dữ liệu mới
            st.session_state.video_analyzed = True
            
            # Tự động rerun để tiếp tục xử lý webcam
            if st.session_state.webcam_running:
                st.rerun()
                
        except Exception as e:
            st.error(f"Lỗi khi tải video: {str(e)}")
            st.session_state.webcam_running = False
            st.session_state.webcam_initialized = False
