#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebSocket Server using Flask and Socket.IO
Receives real-time traffic analysis data and broadcasts to connected clients
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import json
from datetime import datetime
from collections import deque, defaultdict

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Data storage for broadcasting to clients
class AnalysisDataStore:
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.density_history = deque(maxlen=max_history)  # Lịch sử mật độ
        self.vehicle_types_count = defaultdict(int)  # Đếm loại xe
        self.current_density = 0
        self.current_status = "Normal"
        self.total_vehicles_detected = 0
        self.connected_clients = set()
        self.lock = threading.Lock()
    
    def add_density_record(self, count, status="Normal"):
        """Thêm bản ghi mật độ mới"""
        with self.lock:
            timestamp = datetime.now().isoformat()
            record = {
                "timestamp": timestamp,
                "density": count,
                "status": status
            }
            self.density_history.append(record)
            self.current_density = count
            self.current_status = status
    
    def add_vehicle_detection(self, vehicle_type, count=1):
        """Thêm số lượng xe theo loại"""
        with self.lock:
            self.vehicle_types_count[vehicle_type] += count
            self.total_vehicles_detected += count
    
    def update_vehicle_types(self, vehicle_types_dict):
        """Cập nhật số lượng loại xe từ frame hiện tại"""
        with self.lock:
            for vehicle_type, count in vehicle_types_dict.items():
                self.vehicle_types_count[vehicle_type] += count
    
    def get_current_state(self):
        """Lấy trạng thái hiện tại"""
        with self.lock:
            return {
                "current_density": self.current_density,
                "current_status": self.current_status,
                "density_history": list(self.density_history),
                "vehicle_types": dict(self.vehicle_types_count),
                "total_vehicles_detected": self.total_vehicles_detected,
                "connected_clients": len(self.connected_clients)
            }
    
    def reset(self):
        """Đặt lại dữ liệu"""
        with self.lock:
            self.density_history.clear()
            self.vehicle_types_count.clear()
            self.current_density = 0
            self.current_status = "Normal"
            self.total_vehicles_detected = 0

data_store = AnalysisDataStore()

@app.route('/')
def index():
    """Serve index.html"""
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    """REST API endpoint để lấy dữ liệu"""
    return jsonify(data_store.get_current_state())

@app.route('/emit', methods=['POST'])
def emit_event():
    """
    HTTP endpoint để emit events tới clients
    POST body format: {"event": "density_update", "data": {...}}
    """
    try:
        payload = request.get_json()
        event_name = payload.get('event')
        event_data = payload.get('data', {})
        
        if event_name == 'density_update':
            count = event_data.get('count', 0)
            status = event_data.get('status', 'Normal')
            
            # Lưu vào data store
            data_store.add_density_record(count, status)
            
            # Broadcast tới tất cả clients
            socketio.emit('density_update', {
                'timestamp': datetime.now().isoformat(),
                'density': count,
                'status': status
            }, to=None, namespace='/')
            
            print(f"[Emit] Density update: {count} vehicles, status: {status}")
        
        elif event_name == 'vehicle_types_update':
            vehicle_types = event_data.get('vehicle_types', {})
            
            # Cập nhật vào data store
            data_store.update_vehicle_types(vehicle_types)
            
            # Broadcast tới tất cả clients
            socketio.emit('vehicle_types_update', {
                'timestamp': datetime.now().isoformat(),
                'vehicle_types': dict(data_store.vehicle_types_count),
                'total_detected': data_store.total_vehicles_detected
            }, to=None, namespace='/')
            
            print(f"[Emit] Vehicle types update: {vehicle_types}")
        
        elif event_name == 'analysis_start':
            data_store.reset()
            socketio.emit('analysis_start', event_data, to=None, namespace='/')
            socketio.emit('current_state', data_store.get_current_state(), to=None, namespace='/')
            print(f"[Emit] Analysis started")
        
        elif event_name == 'analysis_complete':
            socketio.emit('analysis_complete', event_data, to=None, namespace='/')
            print(f"[Emit] Analysis completed: {event_data}")
        
        return jsonify({"status": "success", "event": event_name})
    
    except Exception as e:
        print(f"[Error] Failed to emit event: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

# ==================== Socket.IO Events ====================

@socketio.on('connect')
def handle_connect():
    """Khi client kết nối"""
    from flask import request as flask_request
    sid = flask_request.sid if hasattr(flask_request, 'sid') else 'unknown'
    data_store.connected_clients.add(sid)
    print(f"[Socket] Client connected: {sid}")
    
    # Gửi trạng thái hiện tại cho client mới kết nối
    emit('current_state', data_store.get_current_state())

@socketio.on('disconnect')
def handle_disconnect():
    """Khi client ngắt kết nối"""
    from flask import request as flask_request
    sid = flask_request.sid if hasattr(flask_request, 'sid') else 'unknown'
    data_store.connected_clients.discard(sid)
    print(f"[Socket] Client disconnected: {sid}")

@socketio.on('request_state')
def handle_request_state():
    """Khi client yêu cầu trạng thái hiện tại"""
    emit('current_state', data_store.get_current_state())

if __name__ == '__main__':
    # Chạy server trên port 5000
    print("=" * 60)
    print("Starting WebSocket Server")
    print("=" * 60)
    print("Server URL: http://localhost:5000")
    print("API Endpoint: http://localhost:5000/api/data")
    print("Emit Endpoint: POST http://localhost:5000/emit")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
