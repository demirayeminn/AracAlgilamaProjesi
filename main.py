from ultralytics import YOLO
import cv2
import time
import numpy as np
from sort import Sort

model = YOLO("yolov8s.pt")
video_kaynak = 'video1.mp4'
cap = cv2.VideoCapture(video_kaynak)
pixel_to_meter = 3 / 60
tracker = Sort()
previous_time = None

def calculate_elapsed_time():
    global previous_time
    current_time = time.time()
    if previous_time is None:
        previous_time = current_time
        return 0
    elapsed_time = current_time - previous_time
    previous_time = current_time
    return elapsed_time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    elapsed_time = calculate_elapsed_time()
    if elapsed_time == 0:
        continue
    results = model(frame, conf=0.25, imgsz=640)
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in [2, 5, 7]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            detections.append([x1, y1, x2, y2, confidence])
    tracked_objects = tracker.update(np.array(detections))
    for obj in tracked_objects:
        x1, y1, x2, y2, object_id = map(int, obj[:5])
        w, h = x2 - x1, y2 - y1
        current_position = ((x1 + x2) // 2, (y1 + y2) // 2)
        if elapsed_time > 0:
            speed = (w * pixel_to_meter) / elapsed_time * 3.6
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {int(object_id)} HIZ: {int(speed)} km/h",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Araç Algılama ve Hız Ölçümü", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

