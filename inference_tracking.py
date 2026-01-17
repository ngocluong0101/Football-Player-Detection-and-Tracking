import cv2
import torch
import os
import warnings
import numpy as np
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

if __name__ == "__main__":
    # 1. SETUP
    warnings.filterwarnings("ignore", category=FutureWarning)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    VIDEO_PATH = "/home/ngocluong/code/projects/dataset/football_test/Match_1864_1_0_subclip/Match_1864_1_0_subclip.mp4" 
    WEIGHTS_PATH = "weights/best.pt"
    OUTPUT_DIR = "outputs"
    IMG_SIZE = 640
    CONF_THRES = 0.45  # Tăng lên để giảm nhiễu
    IOU_THRES = 0.40

    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using Device: {device.upper()}")

    # 2. INIT TRACKER
    # max_age: Số frame giữ ID khi bị che khuất. Bóng đá nên để cao (30-50)
    tracker = DeepSort(
        max_age=50,
        n_init=3,
        max_iou_distance=0.7,
        nms_max_overlap=1.0, 
        max_cosine_distance=0.2,
        embedder="mobilenet", # Hoặc 'torchreid' nếu muốn chính xác hơn (nhưng nặng hơn)
        embedder_gpu=True if device == 'cuda' else False
    )

    # 3. LOAD MODEL
    print(f"[INFO] Loading YOLOv5 from {WEIGHTS_PATH}...")

    model = torch.hub.load('yolov5', 'custom', path=WEIGHTS_PATH, source='local') 
    model.conf = CONF_THRES
    model.iou = IOU_THRES
    model.to(device)

    # 4. VIDEO SETUP
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video path not found: {VIDEO_PATH}")
        exit()

    cap = cv2.VideoCapture(VIDEO_PATH)
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{video_name}_{timestamp}_tracked.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Codec mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # 5. HELPER: COLOR GENERATOR
    id_colors = {}
    def get_color(track_id):
        if track_id not in id_colors:
            np.random.seed(int(track_id))
            id_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return id_colors[track_id]

    # 6. MAIN LOOP
    print("[INFO] Starting tracking loop...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Detection
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy() # Chuyển về CPU để xử lý logic

        player_dets = []
        ball_dets = []

        # Format của deep_sort_realtime: [[left, top, w, h], confidence, detection_class]
        for x1, y1, x2, y2, conf, cls in detections:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = int(cls)

            if cls == 0:  # Player (Giả định class 0 là player)
                player_dets.append(([x1, y1, w, h], conf, 'player'))
            else:         # Ball (Giả định các class khác là bóng)
                ball_dets.append((x1, y1, x2, y2, conf))

        # --- TRACKING LOGIC (SỬA LỖI THỤT DÒNG TẠI ĐÂY) ---
        # Tracks phải được cập nhật 1 lần mỗi frame, KHÔNG phải trong vòng lặp detections
        tracks = tracker.update_tracks(player_dets, frame=frame)
        # --------------------------------------------------

        # Draw Tracks (Players)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            
            # Vẽ Box
            l, t, r, b = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            color = get_color(int(track_id))
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            # Vẽ ID Background cho dễ nhìn
            label = f"ID: {track_id}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (l, t - 20), (l + w_text, t), color, -1)
            cv2.putText(frame, label, (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Detections (Ball - No Tracking)
        for x1, y1, x2, y2, conf in ball_dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Màu vàng cho bóng
            cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Status update
        if frame_count % 20 == 0:
            print(f"[Run] Processing frame {frame_count}...")

        out.write(frame)

    cap.release()
    out.release()
    print(f"[DONE] Video saved to: {OUTPUT_PATH}")