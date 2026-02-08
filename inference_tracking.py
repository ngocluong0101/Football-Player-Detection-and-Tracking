import cv2
import torch
import os
import warnings
import numpy as np
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort


try:
    from team_classifier import TeamClassifier
except ImportError:
    print("[ERROR] Không tìm thấy file 'team_classifier.py'. Hãy tạo file này trước!")
    exit()

if __name__ == "__main__":
    # --- 1. CẤU HÌNH (SETTINGS) ---
    warnings.filterwarnings("ignore")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ĐƯỜNG DẪN VIDEO VÀ MODEL
    VIDEO_PATH = "/home/ngocluong/code/projects/dataset/football_test/Match_1864_1_0_subclip/Match_1864_1_0_subclip.mp4"
    WEIGHTS_PATH = "weights/best.pt" 
    
    OUTPUT_DIR = "outputs"
    IMG_SIZE = 640          
    CONF_THRES = 0.45       
    IOU_THRES = 0.40

    # --- 2. KHỞI TẠO (INITIALIZATION) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using Device: {device.upper()}")

    # Khởi tạo DeepSort Tracker
    tracker = DeepSort(
        max_age=50,
        n_init=3,
        max_iou_distance=0.7,
        nms_max_overlap=1.0, 
        max_cosine_distance=0.2,
        embedder="mobilenet", 
        embedder_gpu=True if device == 'cuda' else False
    )

    # Load Model YOLOv5
    print(f"[INFO] Loading YOLOv5 model from {WEIGHTS_PATH}...")
    try:
        model = torch.hub.load('yolov5', 'custom', path=WEIGHTS_PATH, source='local') 
    except:
        print("[WARN] Local YOLOv5 not found, trying GitHub...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=WEIGHTS_PATH)

    model.conf = CONF_THRES
    model.iou = IOU_THRES
    model.to(device)

    # --- 3. CHUẨN BỊ VIDEO ---
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video path not found: {VIDEO_PATH}")
        exit()

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{video_name}_{timestamp}_tracked.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # --- 4. CẤU HÌNH PHÂN LOẠI ĐỘI BÓNG (TEAM CLASSIFIER) ---
    classifier = TeamClassifier()
    frames_for_calibration = 30  # 30 frame đầu để học màu áo
    calibration_crops = []       
    is_calibrated = False        
    
    # Bộ nhớ đệm (Cache) để tăng tốc độ: {track_id: team_id}
    team_cache = {} 

    # Màu hiển thị: Team 0 (Xanh), Team 1 (Đỏ) - Hệ màu BGR
    TEAM_COLORS = {0: (255, 0, 0), 1: (0, 0, 255)} 

    # --- 5. VÒNG LẶP CHÍNH (MAIN LOOP) ---
    print("[INFO] Starting tracking loop...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # A. DETECTION (YOLO)
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()

        player_dets = [] 

        for x1, y1, x2, y2, conf, cls in detections:
            # Giả định class 0 là Player
            if int(cls) == 0: 
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                player_dets.append(([x1, y1, w, h], conf, 'player'))

        # B. TRACKING (DeepSort)
        tracks = tracker.update_tracks(player_dets, frame=frame)

        # C. LOGIC PHÂN LOẠI ĐỘI BÓNG
        
        # Giai đoạn 1: Thu thập dữ liệu (Calibration)
        if not is_calibrated and frame_count <= frames_for_calibration:
            for track in tracks:
                if not track.is_confirmed(): continue
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
                # Check biên an toàn để không crash
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                player_crop = frame[y1:y2, x1:x2]
                if player_crop.size > 0:
                    calibration_crops.append(player_crop)

            cv2.putText(frame, f"Dang hoc mau ao... {frame_count}/{frames_for_calibration}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Giai đoạn 2: Training K-Means (Chạy 1 lần duy nhất)
        elif not is_calibrated and frame_count > frames_for_calibration:
            print(f"[INFO] Đã thu thập {len(calibration_crops)} mẫu áo. Bắt đầu phân loại...")
            if len(calibration_crops) > 0:
                classifier.fit_teams(calibration_crops)
                is_calibrated = True
            else:
                print("[WARN] Không đủ dữ liệu để học màu áo!")

        # Giai đoạn 3: Vẽ và Dự đoán (Prediction)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Màu mặc định (Xám) khi chưa biết đội
            color = (128, 128, 128) 
            team_text = ""

            if is_calibrated:
                # --- TỐI ƯU TỐC ĐỘ (SPEED OPTIMIZATION) ---
                # Nếu ID này đã biết đội rồi -> Lấy luôn từ Cache
                if track_id in team_cache:
                    team_id = team_cache[track_id]
                else:
                    # Nếu chưa biết -> Mới phải tính toán (chậm)
                    bx1, by1 = max(0, x1), max(0, y1)
                    bx2, by2 = min(width, x2), min(height, y2)
                    
                    # Gọi hàm phân loại
                    team_id = classifier.predict(frame, [bx1, by1, bx2, by2])
                    
                    # Lưu ngay vào Cache
                    team_cache[track_id] = team_id
                # ------------------------------------------

                color = TEAM_COLORS.get(team_id, (255, 255, 255))
                team_text = f"Doi {team_id+1}" # Đội 1 hoặc Đội 2

            # Vẽ Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ Nhãn (Label)
            label = f"ID:{track_id} {team_text}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # In thông báo tiến độ mỗi 20 frames
        if frame_count % 20 == 0:
            print(f"[Processing] Frame {frame_count} - Calibrated: {is_calibrated}")

        out.write(frame)

    # --- 6. KẾT THÚC ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Video saved to: {OUTPUT_PATH}")