from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import json
import time
import threading
import logging
import importlib.util
from queue import Queue, Empty, Full
import tensorflow as tf
from collections import Counter, defaultdict
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

import mediapipe as mp
try:
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    mp_holistic = mp.solutions.holistic

try:
    import torch
except ImportError:
    torch = None

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mutemotion-web")

camera_active = False
latest_action = "Waiting..."  # Trạng thái dự đoán mới nhất cho giao diện web.
latest_confidence = 0.0
latest_sentence = ""
sentence_words = []
last_error = ""
camera_lock = threading.Lock()
model_lock = threading.Lock()
predict_worker_lock = threading.Lock()
landmark_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "custom_videos")
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

SIGN_WEB_DIR = os.path.join(PROJECT_ROOT, "sign_language_web")
SIGN_MODEL_DIR = os.path.join(SIGN_WEB_DIR, "models", "wlasl", "asl100")
SIGN_LABELS_PATH = os.path.join(SIGN_WEB_DIR, "models", "wlasl", "wlasl_class_list.txt")
SIGN_MODEL_WEIGHTS = "FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt"
SIGN_MODEL_PATH = os.path.join(SIGN_MODEL_DIR, SIGN_MODEL_WEIGHTS)
SIGN_NUM_CLASSES = 100
SIGN_TARGET_FRAMES = 64

sign_model = None
sign_labels = []
sign_device = None
sign_model_error = ""

os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Đang nạp mô hình AI lên Web Server...")
from tensorflow.keras.layers import BatchNormalization

def resolve_model_path(base_dir):
    preferred = [
        os.path.join(base_dir, "best_action_model.keras"),
        os.path.join(base_dir, "best_action_model.h5"),
    ]
    for p in preferred:
        if os.path.exists(p):
            return p

    candidates = []
    for ext in ("*.keras", "*.h5"):
        for name in os.listdir(base_dir):
            if name.lower().endswith(ext[1:]):
                candidates.append(os.path.join(base_dir, name))

    if not candidates:
        raise FileNotFoundError("Không tìm thấy file model (.keras/.h5) trong thư mục project")

    # Chọn model có thời gian sửa gần nhất.
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def resolve_label_path(base_dir):
    preferred = [
        os.path.join(base_dir, "label_map.json"),
        os.path.join(base_dir, "data", "processed", "label_map.json"),
    ]
    for p in preferred:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Không tìm thấy label_map.json")

def parse_label_map(raw_map):
    # Hỗ trợ cả định dạng {word: id} và {id: word}.
    if all(isinstance(k, str) for k in raw_map.keys()) and all(isinstance(v, int) for v in raw_map.values()):
        label_to_id = raw_map
        id_to_label = {v: k for k, v in raw_map.items()}
        return label_to_id, id_to_label

    converted = {}
    for k, v in raw_map.items():
        if isinstance(k, str) and k.isdigit() and isinstance(v, str):
            converted[v] = int(k)
        elif isinstance(k, int) and isinstance(v, str):
            converted[v] = k

    if converted:
        return converted, {v: k for k, v in converted.items()}

    raise ValueError("label_map.json không đúng định dạng hỗ trợ")

def load_sign_labels(txt_path, num_classes):
    labels = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                labels.append(" ".join(parts[1:]))
            else:
                labels.append(line.strip())
    return labels[:num_classes]

def load_sign_language_model():
    global sign_model, sign_labels, sign_device, sign_model_error

    if sign_model is not None:
        return True

    if torch is None:
        sign_model_error = "Thiếu thư viện torch cho pipeline upload video"
        logger.error(sign_model_error)
        return False

    if not os.path.exists(SIGN_MODEL_PATH):
        sign_model_error = f"Không tìm thấy model upload tại: {SIGN_MODEL_PATH}"
        logger.error(sign_model_error)
        return False

    if not os.path.exists(SIGN_LABELS_PATH):
        sign_model_error = f"Không tìm thấy labels upload tại: {SIGN_LABELS_PATH}"
        logger.error(sign_model_error)
        return False

    pytorch_i3d_path = os.path.join(SIGN_WEB_DIR, "translator", "pytorch_i3d.py")
    if not os.path.exists(pytorch_i3d_path):
        sign_model_error = f"Không tìm thấy kiến trúc I3D tại: {pytorch_i3d_path}"
        logger.error(sign_model_error)
        return False

    try:
        spec = importlib.util.spec_from_file_location("sign_pytorch_i3d", pytorch_i3d_path)
        sign_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sign_module)
        InceptionI3d = sign_module.InceptionI3d

        sign_labels = load_sign_labels(SIGN_LABELS_PATH, SIGN_NUM_CLASSES)
        sign_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_obj = InceptionI3d(400, in_channels=3)
        model_obj.replace_logits(SIGN_NUM_CLASSES)
        model_obj.load_state_dict(torch.load(SIGN_MODEL_PATH, map_location=sign_device))
        model_obj.to(sign_device)
        model_obj.eval()
        sign_model = model_obj
        logger.info("Upload model loaded: %s", SIGN_MODEL_PATH)
        return True
    except Exception as ex:
        sign_model_error = f"Không thể nạp model upload: {ex}"
        logger.exception("Upload model initialization failed")
        return False

def extract_frames_for_sign_model(video_path, target_frames=64):
    cap = cv2.VideoCapture(video_path)
    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w // 2) - (min_dim // 2)
        start_y = (h // 2) - (min_dim // 2)
        frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
        frame = cv2.resize(frame, (224, 224))
        frame = (frame / 255.0) * 2.0 - 1.0
        frames.append(frame)

    total_frames = len(frames)
    cap.release()

    if total_frames == 0:
        frames = [np.zeros((224, 224, 3), dtype=np.float32)] * target_frames
    elif total_frames < target_frames:
        frames.extend([frames[-1]] * (target_frames - total_frames))
    elif total_frames > target_frames:
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]

    frames_np = np.array(frames, dtype=np.float32).transpose(3, 0, 1, 2)
    video_tensor = torch.from_numpy(frames_np).unsqueeze(0).float().to(sign_device)
    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0
    return video_tensor, duration_sec

def analyze_uploaded_video_with_sign_model(video_path):
    if not load_sign_language_model():
        raise RuntimeError(sign_model_error or "Không thể nạp pipeline upload video")

    with torch.no_grad():
        video_tensor, duration_sec = extract_frames_for_sign_model(video_path, target_frames=SIGN_TARGET_FRAMES)
        predictions = sign_model(video_tensor)
        logits = predictions[0] if isinstance(predictions, tuple) else predictions
        pooled_logits = torch.max(logits, dim=2)[0]
        probs = torch.softmax(pooled_logits[0], dim=0)

        topk = min(5, probs.shape[0])
        top_values, top_indices = torch.topk(probs, k=topk)

        top_actions = []
        for value, idx in zip(top_values.tolist(), top_indices.tolist()):
            label = sign_labels[idx] if idx < len(sign_labels) else f"class_{idx}"
            top_actions.append({
                "action": label,
                "score": round(float(value), 4),
                "count": 1,
            })

        best_idx = int(top_indices[0].item())
        final_action = sign_labels[best_idx] if best_idx < len(sign_labels) else f"class_{best_idx}"
        final_conf = float(top_values[0].item())

    return {
        "final_action": final_action,
        "final_confidence": round(final_conf, 4),
        "duration_sec": duration_sec,
        "samples": [],
        "top_actions": top_actions,
        "model_source": "sign_language_web",
    }

class CompatInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Tương thích model lưu batch_shape thay cho batch_input_shape.
        if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        super().__init__(*args, **kwargs)

MODEL_PATH = resolve_model_path(PROJECT_ROOT)
LABEL_PATH = resolve_label_path(PROJECT_ROOT)
logger.info("Using model: %s", MODEL_PATH)
logger.info("Using label map: %s", LABEL_PATH)

model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "InputLayer": CompatInputLayer,
        "DTypePolicy": tf.keras.mixed_precision.Policy,
    },
) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
with open(LABEL_PATH, 'r', encoding='utf-8') as f:
    raw_label_map = json.load(f)
label_map, actions = parse_label_map(raw_label_map)

model_class_count = int(model.output_shape[-1])
if model_class_count != len(actions):
    logger.warning(
        "Model output classes (%s) != label map classes (%s). Sẽ tự căn theo số lớp model.",
        model_class_count,
        len(actions),
    )
    for cls_id in range(model_class_count):
        if cls_id not in actions:
            actions[cls_id] = f"class_{cls_id}"
    label_map = {label: cls_id for cls_id, label in actions.items()}
IDLE_LABEL = next((k for k in label_map.keys() if k.lower() == "idle"), "idle")
IDLE_ID = label_map.get(IDLE_LABEL)
WORD_APPEND_INTERVAL_SECONDS = 5.0
INFER_QUEUE_SIZE = 1

latest_infer_ms = 0.0
predict_worker_thread = None
predict_queue = None
latest_pose_landmarks = None
latest_left_hand_landmarks = None
latest_right_hand_landmarks = None

def get_update_interval_seconds(confidence):
    # Giữ tương quan confidence cao thì thời gian cập nhật ngắn hơn.
    if confidence >= 0.85:
        return 1.0
    if confidence >= 0.70:
        return 2.0
    return 3.0

def allowed_video_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def select_action_with_idle_control(probabilities, previous_action):
    ranked_ids = np.argsort(probabilities)[::-1]
    top1_id = int(ranked_ids[0])
    top2_id = int(ranked_ids[1]) if len(ranked_ids) > 1 else top1_id

    top1_conf = float(probabilities[top1_id])
    top2_conf = float(probabilities[top2_id])
    margin = top1_conf - top2_conf

    chosen_id = top1_id

    # Chỉ ưu tiên idle khi vượt trội rõ ràng so với lớp kế tiếp.
    if IDLE_ID is not None and top1_id == IDLE_ID:
        if top1_conf < 0.92 or margin < 0.30:
            if top2_conf >= 0.25:
                chosen_id = top2_id

    chosen_conf = float(probabilities[chosen_id])
    chosen_action = actions.get(chosen_id, "---")

    # Giữ action trước đó khi confidence thấp để giảm nhiễu dự đoán.
    if chosen_conf < 0.32:
        if previous_action not in ("Waiting...", "Đang nhận diện...", "ĐÃ DỪNG", "---", "LỖI CAMERA", "LỖI STREAM"):
            return previous_action, chosen_conf, top1_id, top1_conf, top2_id, top2_conf, margin
        return "---", chosen_conf, top1_id, top1_conf, top2_id, top2_conf, margin

    return chosen_action, chosen_conf, top1_id, top1_conf, top2_id, top2_conf, margin

def estimate_hand_motion(sequence):
    # Ước lượng chuyển động tay từ 126 chiều cuối (2 tay x 21 điểm x 3 tọa độ).
    if len(sequence) < 2:
        return 0.0
    prev_hands = sequence[-2][99:225]
    curr_hands = sequence[-1][99:225]
    return float(np.mean(np.abs(curr_hands - prev_hands)))

def build_adjusted_probs(raw_probs, sequence, results, ema_probs):
    if ema_probs is None:
        ema_probs = raw_probs
    else:
        ema_probs = 0.65 * ema_probs + 0.35 * raw_probs

    adjusted_probs = np.copy(ema_probs)

    has_hand_landmarks = bool(results.left_hand_landmarks) or bool(results.right_hand_landmarks)
    hand_motion = estimate_hand_motion(sequence)
    if IDLE_ID is not None and has_hand_landmarks and hand_motion > 0.002:
        adjusted_probs[IDLE_ID] *= 0.10

    debug = {
        "has_hand_landmarks": has_hand_landmarks,
        "hand_motion": hand_motion,
    }
    return adjusted_probs, ema_probs, debug

def predict_action_from_probs(raw_probs, previous_action, sequence, results, ema_probs):
    adjusted_probs, ema_probs, debug = build_adjusted_probs(raw_probs, sequence, results, ema_probs)

    (
        predicted_action,
        confidence,
        top1_id,
        top1_conf,
        top2_id,
        top2_conf,
        margin,
    ) = select_action_with_idle_control(adjusted_probs, previous_action)

    debug.update({
        "top1_label": actions.get(top1_id, str(top1_id)),
        "top1_conf": float(top1_conf),
        "top2_label": actions.get(top2_id, str(top2_id)),
        "top2_conf": float(top2_conf),
        "margin": float(margin),
    })
    return predicted_action, float(confidence), ema_probs, debug

def select_sentence_word(probabilities, sentence_words):
    ranked_ids = np.argsort(probabilities)[::-1]
    used_words = set(sentence_words)

    # Không lặp lại từ đã xuất hiện trong chuỗi.
    for cand_id in ranked_ids:
        cand_id = int(cand_id)
        cand_word = actions.get(cand_id, "---")
        cand_conf = float(probabilities[cand_id])
        if cand_word not in used_words:
            return cand_id, cand_word, cand_conf

    # Không thêm từ mới nếu toàn bộ từ vựng đã xuất hiện.
    return None, None, None

def analyze_uploaded_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở video đã tải lên")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frame_index = 0
    sequence = []
    ema_probs = None
    previous_action = "---"
    sampled_predictions = []
    action_scores = defaultdict(float)
    action_counts = Counter()
    sample_stride = max(1, int(round(fps)))

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=False,
    ) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = holistic.process(img_rgb)

            keypoints = extract_1659_landmarks(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                with model_lock:
                    raw_probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_action, confidence, ema_probs, _ = predict_action_from_probs(
                    raw_probs,
                    previous_action,
                    sequence,
                    results,
                    ema_probs,
                )

                previous_action = predicted_action
                if predicted_action not in ("---", "Waiting...", "Đang nhận diện..."):
                    action_scores[predicted_action] += confidence
                    action_counts[predicted_action] += 1

                if frame_index % sample_stride == 0:
                    sampled_predictions.append({
                        "time_sec": round(frame_index / fps, 2),
                        "action": predicted_action,
                        "confidence": round(confidence, 4),
                    })

            frame_index += 1

    cap.release()

    if not sampled_predictions:
        return {
            "final_action": "Không đủ dữ liệu (video quá ngắn)",
            "final_confidence": 0.0,
            "duration_sec": round(frame_index / fps, 2),
            "samples": [],
            "top_actions": [],
        }

    if action_scores:
        ranked = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        final_action, final_score = ranked[0]
        final_confidence = final_score / max(1, action_counts[final_action])
        top_actions = [
            {
                "action": action,
                "score": round(score, 4),
                "count": int(action_counts[action]),
            }
            for action, score in ranked[:5]
        ]
    else:
        final_action = sampled_predictions[-1]["action"]
        final_confidence = sampled_predictions[-1]["confidence"]
        top_actions = []

    return {
        "final_action": final_action,
        "final_confidence": round(float(final_confidence), 4),
        "duration_sec": round(frame_index / fps, 2),
        "samples": sampled_predictions[:30],
        "top_actions": top_actions,
    }

def open_camera_capture():
    # Ưu tiên backend ổn định trên Windows trước khi fallback mặc định.
    candidates = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("DEFAULT", None),
    ]
    for backend_name, backend in candidates:
        cap = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
        if cap.isOpened():
            logger.info("Camera opened with backend: %s", backend_name)
            return cap, backend_name
        cap.release()
    return None, None

def extract_1659_landmarks(results):
    landmarks = np.zeros((553, 3))
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
    
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[33 + 478 + i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[33 + 478 + 21 + i] = [lm.x, lm.y, lm.z]
            
    flat_landmarks = landmarks.flatten()
    return np.concatenate([flat_landmarks[0:99], flat_landmarks[1533:1659]])

def _enqueue_latest_frame(frame_bgr):
    global predict_queue
    if predict_queue is None:
        return

    frame_copy = frame_bgr.copy()
    try:
        predict_queue.put_nowait(frame_copy)
    except Full:
        try:
            predict_queue.get_nowait()
        except Empty:
            pass
        try:
            predict_queue.put_nowait(frame_copy)
        except Full:
            pass

def _predict_worker_loop():
    global latest_action, latest_confidence, latest_sentence, sentence_words, last_error, latest_infer_ms
    global latest_pose_landmarks, latest_left_hand_landmarks, latest_right_hand_landmarks

    sequence = []
    ema_probs = None
    last_word_append_ts = time.time()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while camera_active:
            try:
                frame = predict_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False

                t0 = time.time()
                results = holistic.process(img_rgb)

                with landmark_lock:
                    latest_pose_landmarks = results.pose_landmarks
                    latest_left_hand_landmarks = results.left_hand_landmarks
                    latest_right_hand_landmarks = results.right_hand_landmarks

                keypoints = extract_1659_landmarks(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    with model_lock:
                        raw_probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                    adjusted_probs, ema_probs, debug = build_adjusted_probs(
                        raw_probs,
                        sequence,
                        results,
                        ema_probs,
                    )

                    ranked_ids = np.argsort(adjusted_probs)[::-1]
                    top1_id = int(ranked_ids[0])
                    top2_id = int(ranked_ids[1]) if len(ranked_ids) > 1 else top1_id
                    top1_conf = float(adjusted_probs[top1_id])
                    top2_conf = float(adjusted_probs[top2_id])
                    margin = top1_conf - top2_conf

                    latest_confidence = top1_conf

                    now = time.time()
                    if (now - last_word_append_ts) >= WORD_APPEND_INTERVAL_SECONDS:
                        probs_for_sentence = np.copy(adjusted_probs)
                        if sentence_words and IDLE_ID is not None:
                            probs_for_sentence[IDLE_ID] = -1.0

                        _, sentence_word, sentence_word_conf = select_sentence_word(
                            probs_for_sentence,
                            sentence_words,
                        )

                        if sentence_word is not None:
                            sentence_words.append(sentence_word)
                            latest_sentence = " ".join(sentence_words)
                            latest_action = sentence_word
                            latest_confidence = sentence_word_conf
                        else:
                            latest_action = "Đã dùng hết từ vựng"
                            latest_confidence = 0.0

                        last_word_append_ts = now
                        logger.info(
                            "Sentence tick: %s | cand_word=%s conf=%.2f | interval=%.1fs | hand=%s motion=%.4f | top1=%s(%.2f) top2=%s(%.2f) margin=%.2f",
                            latest_sentence,
                            sentence_word,
                            latest_confidence,
                            WORD_APPEND_INTERVAL_SECONDS,
                            debug["has_hand_landmarks"],
                            debug["hand_motion"],
                            actions.get(top1_id, str(top1_id)),
                            top1_conf,
                            actions.get(top2_id, str(top2_id)),
                            top2_conf,
                            margin,
                        )

                latest_infer_ms = (time.time() - t0) * 1000.0
            except Exception as ex:
                last_error = f"Lỗi stream: {ex}"
                latest_action = "LỖI STREAM"
                logger.exception("Unhandled exception in predict worker")

def start_predict_worker():
    global predict_worker_thread, predict_queue
    global latest_pose_landmarks, latest_left_hand_landmarks, latest_right_hand_landmarks

    with predict_worker_lock:
        if predict_worker_thread is not None and predict_worker_thread.is_alive():
            return

        with landmark_lock:
            latest_pose_landmarks = None
            latest_left_hand_landmarks = None
            latest_right_hand_landmarks = None

        predict_queue = Queue(maxsize=INFER_QUEUE_SIZE)
        predict_worker_thread = threading.Thread(target=_predict_worker_loop, daemon=True)
        predict_worker_thread.start()

def stop_predict_worker():
    global predict_worker_thread, predict_queue
    global latest_pose_landmarks, latest_left_hand_landmarks, latest_right_hand_landmarks

    with predict_worker_lock:
        if predict_worker_thread is not None:
            predict_worker_thread.join(timeout=1.0)
        predict_worker_thread = None
        predict_queue = None
        with landmark_lock:
            latest_pose_landmarks = None
            latest_left_hand_landmarks = None
            latest_right_hand_landmarks = None

def gen_frames():
    global camera_active, latest_action, latest_confidence, latest_sentence, sentence_words, last_error

    with camera_lock:
        first_frame_logged = False

        cap, backend_name = open_camera_capture()
        if cap is None:
            last_error = "Không thể mở camera. Hãy kiểm tra xem camera có đang bị ứng dụng khác sử dụng không."
            latest_action = "LỖI CAMERA"
            logger.error(last_error)
            return

        logger.info("Starting stream loop. backend=%s", backend_name)
        start_predict_worker()

        try:
            while camera_active:
                success, frame = cap.read()
                if not success:
                    last_error = "Không đọc được khung hình từ camera."
                    latest_action = "LỖI CAMERA"
                    logger.error(last_error)
                    break

                if not first_frame_logged:
                    logger.info("First camera frame received")
                    first_frame_logged = True

                frame = cv2.flip(frame, 1)
                image = frame.copy()

                _enqueue_latest_frame(frame)

                with landmark_lock:
                    pose_landmarks = latest_pose_landmarks
                    left_hand_landmarks = latest_left_hand_landmarks
                    right_hand_landmarks = latest_right_hand_landmarks

                if pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                    )
                if left_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                    )
                if right_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                    )

                # Vẽ thanh trạng thái trên khung hình video.
                cv2.rectangle(image, (0, 0), (640, 40), (40, 40, 40), -1)
                cv2.putText(
                    image,
                    f"Conf: {latest_confidence*100:.1f}% | Infer: {latest_infer_ms:.0f}ms",
                    (300, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    logger.warning("Frame encode failed, skipping frame")
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as ex:
            last_error = f"Lỗi stream: {ex}"
            latest_action = "LỖI STREAM"
            logger.exception("Unhandled exception in gen_frames")
        finally:
            stop_predict_worker()
            cap.release()
            logger.info("Camera released")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame', headers=headers)

@app.route('/toggle', methods=['POST'])
def toggle():
    global camera_active, latest_action, latest_confidence, latest_sentence, sentence_words, last_error
    data = request.get_json(silent=True) or {}
    action = data.get('action')

    if action == 'start':
        camera_active = True
        last_error = ""
        latest_action = "Đang nhận diện..."
        latest_confidence = 0.0
        latest_sentence = ""
        sentence_words = []
    elif action == 'stop':
        camera_active = False
        latest_action = "ĐÃ DỪNG"
    else:
        return jsonify(status="error", message="Action không hợp lệ"), 400

    return jsonify(status="success", camera_active=camera_active)

# API trả kết quả dự đoán liên tục cho giao diện web.
@app.route('/get_action')
def get_action():
    return jsonify(action=latest_action, confidence=round(latest_confidence, 4), sentence=latest_sentence, camera_active=camera_active, error=last_error)

@app.route('/status')
def status():
    return jsonify(camera_active=camera_active, action=latest_action, confidence=round(latest_confidence, 4), sentence=latest_sentence, error=last_error)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify(status="error", message="Thiếu file video"), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify(status="error", message="Bạn chưa chọn file video"), 400

    if not allowed_video_file(file.filename):
        return jsonify(status="error", message="Định dạng file không hỗ trợ"), 400

    safe_name = secure_filename(file.filename)
    unique_name = f"{int(time.time())}_{safe_name}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        file.save(save_path)
        logger.info("Uploaded video: %s", save_path)
        # Prefer the I3D upload pipeline when assets exist; otherwise fallback to TensorFlow.
        can_use_i3d = (
            torch is not None
            and os.path.isdir(SIGN_WEB_DIR)
            and os.path.exists(SIGN_MODEL_PATH)
            and os.path.exists(SIGN_LABELS_PATH)
        )

        if can_use_i3d:
            result = analyze_uploaded_video_with_sign_model(save_path)
        else:
            result = analyze_uploaded_video(save_path)
            result["model_source"] = "tensorflow_fallback"

        return jsonify(status="success", **result)
    except Exception as ex:
        logger.exception("Analyze uploaded video failed")
        return jsonify(status="error", message=f"Phân tích video thất bại: {ex}"), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)