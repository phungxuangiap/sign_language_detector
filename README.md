# WLASL Project - Tong Quan Va Tat Ca Luong Xu Ly

Du an nay xay dung he thong nhan dien ngon ngu ky hieu voi 2 huong ung dung chinh:
- Huong 1: Pipeline du lieu + model TensorFlow cho suy luan realtime webcam (Flask).
- Huong 2: Pipeline video upload dung model PyTorch I3D (co trong `sign_language_web`).

README nay mo ta day du cac luong xu ly de ban co the hieu, van hanh, va mo rong du an.

## 1. Kien truc tong the

Du an duoc chia thanh 4 khoi:
- `01_data_pipeline/`: Xu ly va hop nhat du lieu.
- `02_inference/`: Suy luan realtime tren webcam bang model TensorFlow.
- `03_web_app/`: Web Flask cho realtime va upload video test.
- `sign_language_web/`: Web Django rieng, suy luan upload/camera bang PyTorch I3D.

Tai nguyen mo hinh va nhan:
- `best_action_model.h5`: Model TensorFlow/Keras cho pipeline realtime.
- `label_map.json` (hoac `data/processed/label_map.json`): Anh xa nhan <-> id.
- `dataset_final.json`: Metadata video tu crawl/thu thap.
- `dataset_landmarks/`: Landmark `.npy` theo tu vung.

## 2. Luong xu ly du lieu (Data Pipeline)

### 2.1. Luong A - Loc du lieu goc tu Kaggle/WLASL
Muc tieu: giu cac tu co du so video, tao bo du lieu gon va on dinh.

Script: `01_data_pipeline/filter_kaggle_data.py`

Cac buoc:
1. Doc `data/raw/WLASL_v0.3.json`.
2. Loc cac gloss co so video >= `MIN_VIDEOS` (mac dinh 15).
3. Gom cac `video_id` hop le.
4. Quet 3 file:
   - `data/raw/landmarks_v1.npz`
   - `data/raw/landmarks_v2.npz`
   - `data/raw/landmarks_v3.npz`
5. Chi giu landmark cua cac `video_id` hop le.
6. Luu output:
   - `data/processed/WLASL_filtered_15plus.json`
   - `data/processed/WLASL_filtered_15plus.npz`

Ket qua: bo du lieu Kaggle da duoc loc va nen lai, san sang hop nhat.

### 2.2. Luong B - Trich xuat landmark tu video tu thu thap
Muc tieu: dong bo du lieu tu thu thap ve dung cau truc dac trung.

Script lien quan:
- `01_data_pipeline/extract_scraped_data_chuankaggle.py` (vector 1662 chieu)
- `01_data_pipeline/extract_all_1659.py` (dang (553, 3), phuc vu nhanh 1659)

`extract_scraped_data_chuankaggle.py`:
1. Doc danh sach video trong `dataset_final.json` (truong `local_path`).
2. Dung MediaPipe Holistic trich landmark moi frame.
3. Tao vector 1662 chieu/frame:
   - Pose: 132 (co `visibility`)
   - Face: 1404
   - Left hand: 63
   - Right hand: 63
4. Luu file `.npy` vao `dataset_landmarks/<word>/<video_id>.npy`.
5. Cap nhat lai `landmark_path` trong `dataset_final.json`.

`extract_all_1659.py`:
1. Xu ly idle video va video tu thu thap.
2. Trich landmarks dang mang (553,3) => tuong ung 1659 gia tri neu flatten.
3. Ghi `.npy` cho idle vao `data/idle_npy/`.
4. Co the cap nhat `dataset_final.json` voi `landmark_path` moi.

Luu y quan trong:
- Mot so script dung 1662 (co `visibility` cua pose),
- Mot so script huan luyen/suy luan dung 1659 (chi x,y,z).
- Khi train/infer can dam bao dong nhat shape theo model dang dung.

### 2.3. Luong C - Tao lop idle
Muc tieu: bo sung lop "khong hanh dong" de model giam du doan sai.

Script: `01_data_pipeline/extract_idle.py`

Cac buoc:
1. Lay video `idle_*.mp4/.avi` tu `data/custom_videos/`.
2. Trich landmark MediaPipe tung frame.
3. Xuat `.npy` vao `data/idle_npy/`.

### 2.4. Luong D - Hop nhat tat ca nguon du lieu
Muc tieu: tao bo du lieu MASTER duy nhat cho huan luyen.

Script: `01_data_pipeline/merge_all_data.py`

Cac nguon du lieu dau vao:
1. Kaggle da loc:
   - `data/processed/WLASL_filtered_15plus.json`
   - `data/processed/WLASL_filtered_15plus.npz`
2. Du lieu tu thu thap:
   - `dataset_final.json`
   - `dataset_landmarks/...`
3. Lop idle:
   - `data/idle_npy/*.npy`

Output:
- `data/processed/MASTER_DATASET.json`
- `data/processed/MASTER_DATASET.npz`

### 2.5. Luong E - Chuan hoa chuoi frame de train
Muc tieu: tao tensor train co kich thuoc dong nhat.

Script: `01_data_pipeline/prepare_X_y.py`

Cach xu ly:
1. Doc `MASTER_DATASET.json` + `MASTER_DATASET.npz`.
2. Tao `label_map` tu danh sach gloss.
3. Chuyen moi sample ve dang `(frames, features)`.
4. Kiem tra shape feature, script hien tai yeu cau `1659`.
5. Chuan hoa do dai chuoi ve `MAX_FRAMES = 30`:
   - Ngan hon 30: padding 0.
   - Dai hon 30: cat giua.
6. Luu output:
   - `data/processed/X.npy`
   - `data/processed/y.npy`
   - `data/processed/label_map.json`

Ghi chu:
- Co script `02_inference/check_shape.py` de kiem tra shape landmark goc trong file NPZ.

## 3. Luong suy luan realtime (TensorFlow)

### 3.1. Luong F - Suy luan webcam truc tiep bang script
Script: `02_inference/realtime_predict.py`

Luong xu ly:
1. Nap model `best_action_model.h5` + `label_map.json`.
2. Mo camera.
3. Moi frame:
   - MediaPipe trich landmarks.
   - Tao vector 225 chieu tu 1659:
     - Pose: 99
     - Hands: 126
     - Bo phan face de nhe va nhanh.
4. Giu cua so 30 frame gan nhat => input `(1,30,225)`.
5. Predict model, lay top-1 confidence.
6. Neu confidence > 0.50 hien thi nhan, nguoc lai gan `idle`.
7. Hien thi ket qua va confidence tren OpenCV window.

## 4. Luong web Flask (Realtime + Upload)

Script server: `03_web_app/web_app.py`
Template: `03_web_app/templates/index.html`

### 4.1. Luong G - Realtime stream qua web
1. Client goi `/toggle` de `start`.
2. Server mo camera, stream MJPEG qua `/video_feed`.
3. Backend:
   - Trich keypoint moi frame,
   - Tao sequence 30 frame,
   - Predict model TensorFlow,
   - Lam muot xac suat bang EMA,
   - Co logic uu tien/chong nhiu cho `idle`.
4. Frontend poll `/get_action` de cap nhat tu dang nhan dien + cau dang ghep.
5. Sau moi khoang thoi gian (`WORD_APPEND_INTERVAL_SECONDS`), backend append tu moi vao cau.

### 4.2. Luong H - Upload video de test
1. Client upload qua `/upload_video`.
2. Server luu tam file vao `data/custom_videos/`.
3. Chay pipeline `analyze_uploaded_video_with_sign_model`:
   - Nap model PyTorch I3D trong `sign_language_web/translator/pytorch_i3d.py`.
   - Chuan hoa video ve 64 frame, crop center, resize 224x224, normalize [-1,1].
   - Predict va tra top action.
4. Tra JSON gom:
   - `final_action`, `final_confidence`, `duration_sec`, `top_actions`.
5. Xoa file tam sau khi xu ly.

## 5. Luong web Django (nhanh gon, doc lap)

Thu muc: `sign_language_web/`

Thanh phan chinh:
- `sign_language_web/translator/views.py`: Xu ly upload/camera va predict.
- `sign_language_web/translator/templates/index.html`: Giao dien tab Camera/Upload.
- `sign_language_web/models/wlasl/...`: trong so va tu dien nhan I3D.

Luong xu ly:
1. Nguoi dung quay 3 giay hoac upload video.
2. Video gui POST ve route `/`.
3. Backend trich/chuan hoa frame ve 64 frame.
4. Predict bang InceptionI3D, max-pool theo chieu thoi gian.
5. Lay class score cao nhat, map sang tu WLASL va tra JSON.

## 6. Huong dan chay nhanh

## 6.1. Chuan bi moi truong
Tai goc du an `E:\WLASL_Project`:

```powershell
# kich hoat venv (neu da co)
.\.venv\Scripts\Activate.ps1

# cai thu vien co ban cho pipeline Flask + inference
pip install tensorflow opencv-python mediapipe numpy flask

# neu can chay nhanh nhanh trong sign_language_web
pip install -r .\sign_language_web\requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Neu may khong co CUDA, cai torch ban CPU tu trang huong dan chinh thuc PyTorch.

## 6.2. Chay pipeline du lieu (theo thu tu)

```powershell
python .\01_data_pipeline\filter_kaggle_data.py
python .\01_data_pipeline\extract_scraped_data_chuankaggle.py
python .\01_data_pipeline\extract_idle.py
python .\01_data_pipeline\merge_all_data.py
python .\01_data_pipeline\prepare_X_y.py
```

## 6.3. Chay suy luan realtime bang script

```powershell
python .\02_inference\realtime_predict.py
```

## 6.4. Chay web Flask

```powershell
python .\03_web_app\web_app.py
```

Mo trinh duyet tai:
- `http://127.0.0.1:5000`

## 6.5. Chay web Django (nhanh doc lap)

```powershell
cd .\sign_language_web
python manage.py runserver
```

Mo:
- `http://127.0.0.1:8000`

## 7. Thu muc output quan trong

- `data/processed/WLASL_filtered_15plus.json`: metadata Kaggle sau loc.
- `data/processed/WLASL_filtered_15plus.npz`: landmark Kaggle sau loc.
- `data/processed/MASTER_DATASET.json`: metadata tong hop tat ca nguon.
- `data/processed/MASTER_DATASET.npz`: landmark tong hop tat ca nguon.
- `data/processed/X.npy`, `data/processed/y.npy`: du lieu train.
- `data/processed/label_map.json`: map nhan dung cho train/infer.

## 8. Ghi chu ky thuat quan trong

- Du an dang co 3 muc feature xuat hien:
  - 1662 (co pose visibility),
  - 1659 (pose/face/hands chi x,y,z),
  - 225 (pose + hands da rut gon cho infer nhanh).
- Truoc khi train lai model, can xac nhan shape input model ban muon dung.
- `03_web_app/web_app.py` co co che fallback tim model `.keras/.h5` va tu can so lop theo label map.
- Neu camera khong mo duoc tren Windows, app Flask da co fallback backend `CAP_DSHOW`, `CAP_MSMF`, `DEFAULT`.

## 9. Goi y mo rong

- Dong bo hoan toan mot cau hinh duy nhat cho feature (uu tien 1659 hoac 225) de tranh lech train/infer.
- Them script train ro rang (vi du `04_training/train.py`) va luu lich su metric.
- Them danh gia offline tren tap validation/test rieng cho ca hai model (TensorFlow va I3D).

## 10. Danh sach file lon khong the up len GitHub

GitHub gioi han 100MB/file (voi git thong thuong). Cac file duoi day hien tai vuot nguong nay.

### 10.1. File du lieu/model nen up Google Drive

| STT | File | Kich thuoc (MB) | Link Drive |
|---|---|---:|---|
| 1 | `data/raw/landmarks_V3.npz` | 8508.65 | TODO: dan link |
| 2 | `data/raw/landmarks_V1.npz` | 2596.33 | TODO: dan link |
| 3 | `data/raw/landmarks_V2.npz` | 2574.93 | TODO: dan link |
| 4 | `data/processed/MASTER_DATASET.npz` | 366.12 | TODO: dan link |
| 5 | `data/processed/X.npy` | 270.36 | TODO: dan link |
| 6 | `data/processed/WLASL_filtered_15plus.npz` | 266.18 | TODO: dan link |

### 10.2. File lon thuoc moi truong ao (khong can up Drive)

Nhung file nay thuoc thu muc `venv/.venv`, co the tai lai bang `pip install`, khong nen dua len GitHub hoac Drive.

| STT | File | Kich thuoc (MB) | Ghi chu |
|---|---|---:|---|
| 1 | `sign_language_web/venv/Lib/site-packages/tensorflow/python/_pywrap_tensorflow_common.dll` | 1007.68 | Tu dong sinh khi cai TensorFlow |
| 2 | `.venv/Lib/site-packages/tensorflow/python/_pywrap_tensorflow_internal.pyd` | 697.64 | Tu dong sinh khi cai TensorFlow |
| 3 | `sign_language_web/venv/Lib/site-packages/torch/lib/torch_cpu.dll` | 252.67 | Tu dong sinh khi cai PyTorch |

### 10.3. Mau cach dan link Drive vao README

Ban thay cot `TODO: dan link` bang markdown link, vi du:

`[Tai file](https://drive.google.com/...)`

