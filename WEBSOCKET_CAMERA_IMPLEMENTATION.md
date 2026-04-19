# Triển Khai Phương Án 1: Browser Camera + WebSocket Stream

## 🎯 Mục Tiêu
Chuyển từ Server Camera (không hoạt động trên Cloud) → **Browser Camera + WebSocket Stream**
- Client (Browser): Capture camera qua `getUserMedia()` → Encode frames → Send via WebSocket
- Server (Backend): Receive frames → Run MediaPipe + TensorFlow predict → Send results back via WebSocket

---

## 📊 Kiến Trúc

```
Browser (Client)
├─ getUserMedia() → Canvas capture
├─ Frame buffer (30 FPS)
├─ Send frames via WebSocket every 33ms
└─ Receive predictions → Display on canvas

Flask Backend
├─ WebSocket listener (@socketio.on('frame'))
├─ Frame buffer queue (FIFO)
├─ Predict worker thread
│  ├─ Read frame từ queue
│  ├─ Run MediaPipe holistic
│  ├─ Run TensorFlow model
│  └─ Emit result via WebSocket
├─ /upload_video endpoint (keep unchanged)
└─ /status endpoint (return current action)
```

---

## 📦 Dependencies Cần Thêm

### Backend (`requirements.txt`)
```
Flask==2.3.2
Flask-SocketIO==5.3.4
python-socketio==5.9.0
python-engineio==4.7.1
opencv-python-headless==4.8.0.74
tensorflow==2.13.0
mediapipe==0.10.7
numpy==1.24.3
Werkzeug==2.3.6
```

### Frontend
- No additional library needed - Use native HTML5 `Canvas` API + SocketIO JavaScript client

---

## 🔧 Backend Implementation Changes

### 1. Cập Nhật imports và khởi tạo SocketIO

**File**: `03_web_app/web_app.py`

```python
# Thêm vào đầu file sau Flask import
from flask_socketio import SocketIO, emit
from base64 import b64decode
import base64

# Sau app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables cho WebSocket
websocket_frame_queue = Queue(maxsize=2)  # Keep recent frame only
websocket_connected_clients = set()
```

### 2. Tạo WebSocket Event Handlers

**Thêm sau class definitions, trước @app.route**

```python
# ============ WEBSOCKET EVENTS ============

@socketio.on('connect')
def handle_connect():
    """Client vừa kết nối WebSocket"""
    client_id = request.sid
    websocket_connected_clients.add(client_id)
    logger.info(f"[WebSocket] Client {client_id} connected. Total: {len(websocket_connected_clients)}")
    emit('connection_response', {'status': 'connected', 'message': 'Server ready'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client ngắt kết nối"""
    client_id = request.sid
    websocket_connected_clients.discard(client_id)
    logger.info(f"[WebSocket] Client {client_id} disconnected. Total: {len(websocket_connected_clients)}")

@socketio.on('frame')
def handle_frame(data):
    """
    Nhận frame từ client (base64 encoded JPEG)
    
    Expected data format:
    {
        'image': 'data:image/jpeg;base64,...',
        'timestamp': 1234567890
    }
    """
    try:
        if not data or 'image' not in data:
            emit('error', {'message': 'Invalid frame data'})
            return
        
        # Decode base64 frame
        base64_string = data['image']
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        frame_data = b64decode(base64_string)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('error', {'message': 'Failed to decode frame'})
            return
        
        # Enqueue frame (drop oldest if full)
        try:
            websocket_frame_queue.put_nowait(frame)
        except Full:
            try:
                websocket_frame_queue.get_nowait()
                websocket_frame_queue.put_nowait(frame)
            except Empty:
                websocket_frame_queue.put_nowait(frame)
                
    except Exception as e:
        logger.error(f"[WebSocket] Frame processing error: {str(e)}")
        emit('error', {'message': f'Frame error: {str(e)}'})

@socketio.on('start_stream')
def handle_start_stream():
    """Client yêu cầu bắt đầu stream"""
    global camera_active
    with camera_lock:
        camera_active = True
    logger.info("[WebSocket] Stream started by client")
    emit('stream_status', {'status': 'started'})

@socketio.on('stop_stream')
def handle_stop_stream():
    """Client yêu cầu dừng stream"""
    global camera_active
    with camera_lock:
        camera_active = False
    logger.info("[WebSocket] Stream stopped by client")
    emit('stream_status', {'status': 'stopped'})
```

### 3. Tạo Predict Worker cho WebSocket Frames

**Thêm hàm này trước các @app.route**

```python
# ============ WEBSOCKET PREDICT WORKER ============

websocket_predict_worker_thread = None
websocket_worker_running = False

def _websocket_predict_worker_loop():
    """
    Background worker: Lấy frame từ websocket_frame_queue
    → Run prediction → Broadcast results to all clients
    """
    global websocket_worker_running, camera_active
    
    predict_interval = 5  # seconds - predict every 5s
    last_predict_time = time.time()
    frame_buffer = []
    
    while websocket_worker_running:
        try:
            # Get frame từ queue (non-blocking)
            frame = websocket_frame_queue.get(timeout=0.1)
            frame_buffer.append(frame)
            
            # Predict every 5 seconds
            current_time = time.time()
            if current_time - last_predict_time >= predict_interval and len(frame_buffer) > 0:
                # Use latest frame
                predict_frame = frame_buffer[-1]
                frame_buffer.clear()
                
                with model_lock:
                    try:
                        # Run MediaPipe
                        with mp_holistic.Holistic(
                            static_image_mode=False,
                            model_complexity=1,
                            smooth_landmarks=True,
                            refine_face_landmarks=False
                        ) as holistic:
                            results = holistic.process(cv2.cvtColor(predict_frame, cv2.COLOR_BGR2RGB))
                        
                        if results.pose_landmarks is None:
                            logger.warning("[WebSocket] No pose detected")
                            socketio.emit('prediction', {
                                'action': 'No pose detected',
                                'confidence': 0.0,
                                'sentence': latest_sentence
                            }, to=list(websocket_connected_clients))
                            last_predict_time = current_time
                            continue
                        
                        # Extract landmarks
                        landmarks_list = []
                        if results.pose_landmarks:
                            for lm in results.pose_landmarks.landmark:
                                landmarks_list.extend([lm.x, lm.y, lm.z])
                        if results.left_hand_landmarks:
                            for lm in results.left_hand_landmarks.landmark:
                                landmarks_list.extend([lm.x, lm.y, lm.z])
                        if results.right_hand_landmarks:
                            for lm in results.right_hand_landmarks.landmark:
                                landmarks_list.extend([lm.x, lm.y, lm.z])
                        
                        # Pad to 225 features
                        while len(landmarks_list) < 225:
                            landmarks_list.append(0.0)
                        landmarks_list = landmarks_list[:225]
                        
                        # Run TensorFlow model
                        input_data = np.array([landmarks_list], dtype=np.float32)
                        prediction = model.predict(input_data, verbose=0)
                        action_idx = np.argmax(prediction[0])
                        confidence = float(prediction[0][action_idx])
                        
                        if confidence < 0.5:
                            action = "Idle"
                        else:
                            action = label_map.get(int(action_idx), f"Unknown({action_idx})")
                        
                        # Store latest for API
                        with camera_lock:
                            globals()['latest_action'] = action
                            globals()['latest_confidence'] = confidence
                        
                        # Send to all connected WebSocket clients
                        socketio.emit('prediction', {
                            'action': action,
                            'confidence': float(confidence),
                            'sentence': latest_sentence,
                            'timestamp': time.time()
                        }, to=list(websocket_connected_clients))
                        
                        logger.info(f"[WebSocket] Prediction: {action} ({confidence:.2%})")
                        
                    except Exception as e:
                        logger.error(f"[WebSocket] Prediction error: {str(e)}")
                        socketio.emit('error', {
                            'message': f'Prediction failed: {str(e)}'
                        }, to=list(websocket_connected_clients))
                
                last_predict_time = current_time
        
        except Empty:
            pass
        except Exception as e:
            logger.error(f"[WebSocket] Worker error: {str(e)}")
            time.sleep(0.1)

def start_websocket_predict_worker():
    """Khởi động worker thread"""
    global websocket_predict_worker_thread, websocket_worker_running
    if websocket_predict_worker_thread is None or not websocket_predict_worker_thread.is_alive():
        websocket_worker_running = True
        websocket_predict_worker_thread = threading.Thread(
            target=_websocket_predict_worker_loop,
            daemon=True
        )
        websocket_predict_worker_thread.start()
        logger.info("[WebSocket] Predict worker started")

def stop_websocket_predict_worker():
    """Dừng worker thread"""
    global websocket_worker_running
    websocket_worker_running = False
    if websocket_predict_worker_thread and websocket_predict_worker_thread.is_alive():
        websocket_predict_worker_thread.join(timeout=2)
    logger.info("[WebSocket] Predict worker stopped")
```

### 4. Khởi động Worker khi app start

**Tìm và sửa `if __name__ == '__main__':`**

```python
if __name__ == '__main__':
    # Start WebSocket predict worker
    start_websocket_predict_worker()
    
    # Run Flask + SocketIO
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )
```

---

## 🎨 Frontend Implementation Changes

### 1. Thêm Camera Canvas và Video Elements

**File**: `03_web_app/templates/index.html`

**Tìm section `<div id="videoCard" class="card">` và thay thế:**

```html
<!-- HIDDEN: Video input element -->
<input type="hidden" id="cameraInput" accept="video/*">

<!-- Video element từ camera (hidden) -->
<video id="cameraVideo" autoplay playsinline muted style="display:none;"></video>

<!-- Canvas để vẽ video stream từ camera -->
<canvas id="videoCanvas" style="display:none;"></canvas>

<!-- Section hiển thị realtime -->
<div id="videoCard" class="card">
    <div class="card-header">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span class="text-subtle">Video Stream</span>
            <div id="liveDot" class="live-indicator"></div>
        </div>
    </div>
    
    <div class="video-container">
        <!-- Canvas hiển thị realtime camera preview -->
        <canvas 
            id="displayCanvas" 
            class="stream-canvas"
            style="width: 100%; height: 100%; border-radius: var(--radius-lg);"
        ></canvas>
        
        <!-- Placeholder khi chưa start -->
        <div id="videoPlaceholder" class="video-placeholder">
            <div style="text-align: center;">
                <p style="font-size: 14px; margin: 0 0 12px;">Nhấp Start để bắt đầu stream từ camera</p>
                <button id="startBtn" class="btn btn-primary">🎥 Start Camera</button>
            </div>
        </div>
        
        <!-- Prediction overlay -->
        <div id="predictionOverlay" class="prediction-overlay">
            <div style="font-size: 12px; color: var(--cyan); margin-bottom: 4px;">Action:</div>
            <div id="predictionText" style="font-size: 16px; font-weight: 700; color: var(--violet);">
                Waiting...
            </div>
        </div>
    </div>
</div>

<!-- Control buttons -->
<div style="display: flex; gap: 8px; margin-top: 12px;">
    <button id="stopBtn" class="btn btn-secondary" style="flex: 1;">⏹ Stop Stream</button>
    <button id="permissionBtn" class="btn btn-tertiary" style="flex: 1;">🔓 Request Camera</button>
</div>

<!-- Status indicator -->
<div id="statusBadge" class="status-badge">
    <span id="statusText">● Offline</span>
</div>
```

### 2. Thêm JavaScript Camera & WebSocket Handler

**Thêm vào section `<script>` trong `index.html`, sau `const SERVER_URL = '...';`:**

```javascript
// ============ WEBSOCKET & CAMERA SETUP ============

// Initialize SocketIO
const socket = io();

// Camera & Canvas variables
let cameraVideo = null;
let displayCanvas = null;
let displayCtx = null;
let cameraStream = null;
let isStreamActive = false;
let frameInterval = null;
let cameraReady = false;

// Socket event handlers
socket.on('connect', function() {
    console.log('[WebSocket] Connected to server');
    updateStatusBadge('connected');
});

socket.on('disconnect', function() {
    console.log('[WebSocket] Disconnected from server');
    updateStatusBadge('disconnected');
});

socket.on('connection_response', function(data) {
    console.log('[WebSocket] Server response:', data);
});

socket.on('prediction', function(data) {
    console.log('[WebSocket] Prediction received:', data);
    document.getElementById('predictionText').textContent = 
        `${data.action} (${(data.confidence * 100).toFixed(1)}%)`;
});

socket.on('stream_status', function(data) {
    console.log('[WebSocket] Stream status:', data.status);
});

socket.on('error', function(data) {
    console.error('[WebSocket] Server error:', data.message);
    showNotification(`Server: ${data.message}`, 'error');
});

// Status update function
function updateStatusBadge(status) {
    const badge = document.getElementById('statusBadge');
    const text = document.getElementById('statusText');
    if (status === 'connected') {
        badge.style.backgroundColor = 'rgba(16, 185, 129, 0.2)';
        text.textContent = '● Online';
        text.style.color = 'var(--emerald)';
    } else {
        badge.style.backgroundColor = 'rgba(244, 63, 94, 0.2)';
        text.textContent = '● Offline';
        text.style.color = 'var(--rose)';
    }
}

// Initialize camera elements
function initCameraElements() {
    cameraVideo = document.getElementById('cameraVideo');
    displayCanvas = document.getElementById('displayCanvas');
    displayCtx = displayCanvas.getContext('2d');
}

// Request browser camera permission
async function requestCameraPermission() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        cameraVideo.srcObject = cameraStream;
        cameraReady = true;
        console.log('[Camera] Permission granted, stream initialized');
        showNotification('Camera permission granted', 'success');
        updateStatusBadge('connected');
        return true;
    } catch (err) {
        console.error('[Camera] Permission denied or error:', err);
        showNotification(`Camera error: ${err.message}`, 'error');
        return false;
    }
}

// Send frame to server via WebSocket
function sendFrameToServer() {
    if (!cameraReady || !isStreamActive) return;
    
    try {
        // Set canvas size to match video
        if (cameraVideo.readyState === cameraVideo.HAVE_ENOUGH_DATA) {
            displayCanvas.width = cameraVideo.videoWidth;
            displayCanvas.height = cameraVideo.videoHeight;
            
            // Draw video frame to canvas
            displayCtx.drawImage(cameraVideo, 0, 0);
            
            // Convert canvas to JPEG base64
            const base64Frame = displayCanvas.toDataURL('image/jpeg', 0.7);
            
            // Send to server
            socket.emit('frame', {
                image: base64Frame,
                timestamp: Date.now()
            });
            
            // Mirror display for user (optional)
            displayCtx.scale(-1, 1);
            displayCtx.drawImage(cameraVideo, -displayCanvas.width, 0);
            displayCtx.scale(-1, 1);
        }
    } catch (err) {
        console.error('[Camera] Frame send error:', err);
    }
}

// Start camera stream
async function startCameraStream() {
    if (!cameraReady) {
        const permitted = await requestCameraPermission();
        if (!permitted) return;
    }
    
    isStreamActive = true;
    document.getElementById('videoPlaceholder').style.display = 'none';
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('liveDot').classList.add('pulse');
    
    // Send start signal to server
    socket.emit('start_stream');
    
    // Send frames every 33ms (30 FPS)
    if (frameInterval) clearInterval(frameInterval);
    frameInterval = setInterval(sendFrameToServer, 33);
    
    console.log('[Camera] Stream started');
    showNotification('Camera stream started', 'success');
}

// Stop camera stream
function stopCameraStream() {
    isStreamActive = false;
    if (frameInterval) clearInterval(frameInterval);
    
    document.getElementById('videoPlaceholder').style.display = 'grid';
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('liveDot').classList.remove('pulse');
    
    // Send stop signal to server
    socket.emit('stop_stream');
    
    // Clear canvas
    if (displayCtx) {
        displayCtx.fillStyle = '#0a0a0c';
        displayCtx.fillRect(0, 0, displayCanvas.width, displayCanvas.height);
    }
    
    console.log('[Camera] Stream stopped');
    showNotification('Camera stream stopped', 'success');
}

// Stop all and release camera
function releaseCameraStream() {
    stopCameraStream();
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraReady = false;
    console.log('[Camera] Camera released');
}

// Show notification helper
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // Optional: Implement toast notification here
}

// ============ EVENT LISTENERS ============

// Wait for DOM ready
document.addEventListener('DOMContentLoaded', function() {
    initCameraElements();
    
    // Button listeners
    document.getElementById('startBtn').addEventListener('click', startCameraStream);
    document.getElementById('stopBtn').addEventListener('click', stopCameraStream);
    document.getElementById('permissionBtn').addEventListener('click', requestCameraPermission);
    
    // Handle page unload
    window.addEventListener('beforeunload', releaseCameraStream);
});
```

### 3. Thêm CSS cho Camera Canvas

**Thêm vào section `<style>` trong `index.html`:**

```css
/* Camera Canvas */
.stream-canvas {
    display: block;
    background: #111;
    border-radius: var(--radius-lg);
}

/* Video Placeholder */
.video-placeholder {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(6, 182, 212, 0.05));
    border-radius: var(--radius-lg);
    backdrop-filter: blur(8px);
    z-index: 1;
}

/* Camera Status */
#statusBadge {
    padding: 8px 12px;
    border-radius: var(--radius-md);
    font-size: 13px;
    font-weight: 600;
    transition: all 0.3s ease;
    background: rgba(244, 63, 94, 0.2);
}

#statusText {
    color: var(--rose);
}

/* Live indicator pulse */
.live-indicator.pulse {
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
```

### 4. Sửa existing button IDs (nếu cần)

**Xác nhận các button IDs tồn tại trong form upload:**
```html
<input type="file" id="videoFileInput" accept="video/*" style="display:none;">
<button id="uploadBtn" class="btn btn-secondary">📤 Upload Video</button>
<div id="uploadResult"></div>
```

---

## ✅ Checklist Triển Khai

- [ ] 1. Thêm imports `flask_socketio, base64` vào `web_app.py`
- [ ] 2. Khởi tạo SocketIO: `socketio = SocketIO(app, ...)`
- [ ] 3. Tạo WebSocket event handlers: `@socketio.on('frame')`, `@socketio.on('connect')`, etc.
- [ ] 4. Tạo WebSocket predict worker function
- [ ] 5. Update `if __name__ == '__main__':` để dùng `socketio.run()`
- [ ] 6. Thêm `<video>`, `<canvas>` elements vào HTML
- [ ] 7. Thêm JavaScript camera capture code
- [ ] 8. Thêm JavaScript WebSocket emit/receive handlers
- [ ] 9. Thêm CSS styles cho canvas
- [ ] 10. Test locally: `python 03_web_app/web_app.py`
- [ ] 11. Test camera permission request
- [ ] 12. Test frame sending & predictions
- [ ] 13. Deploy to AWS EC2 (uncomment/remove server camera code)

---

## 🚀 Testing Locally

```bash
# Terminal 1: Activate venv
.\.venv310\Scripts\Activate.ps1

# Terminal 2: Install dependencies
pip install flask-socketio python-socketio python-engineio

# Terminal 3: Run Flask
python 03_web_app/web_app.py
```

Visit: `http://localhost:5000`

1. Nhấp "🔓 Request Camera" → Cho phép trình duyệt truy cập camera
2. Nhấp "🎥 Start Camera" → Bắt đầu stream
3. Quan sát predictions trong overlay

---

## 📡 Triển Khai AWS EC2

Sau khi code hoạt động locally:

1. Push code lên GitHub
2. SSH vào EC2: `ssh -i key.pem ubuntu@<ip>`
3. Chạy AWS_EC2_DEPLOYMENT.md instructions
4. Access: `http://<ec2-public-ip>:5000`

---

## ⚠️ Ghi Chú Quan Trọng

- **Network Bandwidth**: 30 FPS × 640×480 JPEG (0.7 quality) ≈ 1-2 Mbps
- **Latency**: Network round-trip + predict time (~500ms-1s)
- **No Server Camera Code**: Xóa/comment code camera server cũ
- **/video_feed endpoint**: Sẽ deprecated, nhưng keep `/upload_video`
- **Browser Compatibility**: Chrome/Firefox/Edge (need HTTPS for production)

---

## 📝 Code Diff Summary

| File | Changes | Lines |
|------|---------|-------|
| `web_app.py` | Add SocketIO, frame handler, predict worker | +150 lines |
| `index.html` | Add canvas, camera JS, WebSocket handler | +200 lines |
| `requirements.txt` | Add flask-socketio | +3 lines |

Total effort: ~350 lines new code, minimal breaking changes

