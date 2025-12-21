
import cv2
import torch
import os
import warnings
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort


# SUPPRESS WARNING
warnings.filterwarnings("ignore", category=FutureWarning)


# GET CURRENT TIMESTAMP
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# CONFIG
VIDEO_PATH = "/home/ngocluong/code/project/dataset/football_train/Match_2031_5_0_test/Match_2031_5_0_test.mp4"
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH = f"outputs/{video_name}_{timestamp}_tracked.mp4"
WEIGHTS_PATH = "weights/best.pt"

IMG_SIZE = 640
CONF_THRES = 0.3

CLASS_NAMES = ['player', 'ball']


# INIT TRACKER (PLAYER ONLY)
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)


# LOAD YOLOv5
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load(
    'yolov5',
    'custom',
    path=WEIGHTS_PATH,
    source='local'
)

model.conf = CONF_THRES
model.iou = 0.45
model.classes = None


# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("[INFO] Processing video with tracking...")


# COLOR MAP FOR PLAYER ID
id_colors = {}

def get_color(track_id):
    if track_id not in id_colors:
        id_colors[track_id] = (
            int(track_id * 37 % 255),
            int(track_id * 17 % 255),
            int(track_id * 29 % 255),
        )
    return id_colors[track_id]

frame_count = 0

# PROCESS FRAME
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame, size=IMG_SIZE)
    detections = results.xyxy[0]

    player_dets = []
    ball_dets = []

    for *bbox, conf, cls in detections:
        x1, y1, x2, y2 = map(int, bbox)
        cls = int(cls)

        if cls == 0:  # player
            player_dets.append(
                ([x1, y1, x2 - x1, y2 - y1], conf, 'player')
            )
        else:  # ball
            ball_dets.append((x1, y1, x2, y2, conf))

    
    # TRACK PLAYER
        tracks = tracker.update_tracks(player_dets, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        l, t, w, h = map(int, track.to_ltrb())

        color = get_color(track_id)
        label = f"player_{track_id}"

        cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)
        cv2.putText(
            frame,
            label,
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    
    # DRAW BALL (NO TRACKING)
    for x1, y1, x2, y2, conf in ball_dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"ball {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    out.write(frame)

    print(f"[INFO] Processed frame {frame_count}")


# RELEASE
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[DONE] Tracking output saved to {OUTPUT_PATH}")
