import cv2
import torch
import os

# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = "/home/ngocluong/code/project/dataset/football_train/Match_2031_5_0_test/Match_2031_5_0_test.mp4"      # video đầu vào
OUTPUT_PATH = "output_video.mp4"    # video kết quả
WEIGHTS_PATH = "weights/best.pt"    # model đã train
IMG_SIZE = 640
CONF_THRES = 0.4

# ----------------------------
# LOAD YOLOv5 MODEL
# ----------------------------
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load(
    'yolov5',
    'custom',
    path=WEIGHTS_PATH,
    source='local'
)

model.conf = CONF_THRES
model.iou = 0.45
model.classes = None  # detect tất cả class
model.max_det = 1000

# ----------------------------
# OPEN VIDEO
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("[INFO] Processing video...")

# ----------------------------
# PROCESS FRAME BY FRAME
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, size=IMG_SIZE)

    # Get detections
    detections = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

    for *bbox, conf, cls in detections:
        x1, y1, x2, y2 = map(int, bbox)

        label = f"player {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    out.write(frame)

# ----------------------------
# RELEASE
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[DONE] Output saved to {OUTPUT_PATH}")