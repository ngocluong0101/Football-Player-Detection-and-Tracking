import cv2
import torch
import os
import warnings
from datetime import datetime


# SUPPRESS WARNING
warnings.filterwarnings("ignore", category=FutureWarning)

# GET CURRENT TIMESTAMP
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# CONFIG
VIDEO_PATH = "/home/ngocluong/code/project/dataset/football_train/Match_2031_5_0_test/Match_2031_5_0_test.mp4"
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH = f"outputs/{video_name}_{timestamp}.mp4"
WEIGHTS_PATH = "weights/best.pt"

IMG_SIZE = 640
CONF_THRES = 0.4

# CLASS NAMES
CLASS_NAMES = ['player', 'ball']


# LOAD YOLOv5 MODEL
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
model.max_det = 1000


# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output folder if not exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("[INFO] Processing video...")


# PROCESS FRAME BY FRAME
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # YOLO inference
    results = model(frame, size=IMG_SIZE)

    # Get detections
    detections = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

    for *bbox, conf, cls in detections:
        x1, y1, x2, y2 = map(int, bbox)
        cls = int(cls)

        class_name = CLASS_NAMES[cls]
        label = f"{class_name} {conf:.2f}"

        # Color by class
        if cls == 0:       # player
            color = (255, 0, 0)
        else:              # ball
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    out.write(frame)

    if frame_count % 50 == 0:
        print(f"[INFO] Processed frame {frame_count}")


# RELEASE
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[DONE] Output saved to {OUTPUT_PATH}")
