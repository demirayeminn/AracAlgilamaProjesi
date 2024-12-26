from ultralytics import YOLO
import cv2
import time
import numpy as np
from sort import Sort  # Araçları takip etmek için SORT algoritmasını kullanıyoruz

# YOLO modeli yükleniyor. Bu model, araçları algılamak için kullanılıyor.
model = YOLO("yolov8s.pt")  # Daha hassas sonuçlar için 'yolov8m.pt' gibi daha büyük bir model seçebilirsiniz

# Video kaynağı seçiliyor. Webcam kullanmak için 0 yazabilir, dosya için dosya adını belirtebilirsiniz.
video_kaynak = 'video1.mp4'
cap = cv2.VideoCapture(video_kaynak)

# Piksel/metre oranı belirleniyor. Gerçek dünyadaki mesafeyi piksel cinsine çevirmek için bir ölçek gerekiyor.
# Örnekte, 3 metre = 60 piksel olarak kabul ediliyor.
pixel_to_meter = 3 / 60

# SORT algoritması araçların kimliklerini çerçeveler arasında takip etmek için kullanılıyor.
tracker = Sort()

# Zaman hesaplaması için bir önceki çerçeve zamanı saklanıyor.
previous_time = None

def calculate_elapsed_time():
    """
    Çerçeve arası geçen süreyi hesaplar.
    Bu süre, hız hesaplamasında kullanılır.
    """
    global previous_time
    current_time = time.time()
    if previous_time is None:
        previous_time = current_time
        return 0  # İlk çerçevede zaman farkı hesaplanamaz
    elapsed_time = current_time - previous_time
    previous_time = current_time
    return elapsed_time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Video akışı bittiğinde döngü sonlandırılır.

    # Çerçeve arası geçen süreyi hesaplıyoruz. İlk çerçevede süre hesaplanamaz.
    elapsed_time = calculate_elapsed_time()
    if elapsed_time == 0:
        continue

    # YOLO modeliyle araçları algılıyoruz.
    results = model(frame, conf=0.25, imgsz=640)

    # Algılanan araçların kutularını topluyoruz.
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])  # Algılanan nesnenin sınıfı
            if cls not in [2, 5, 7]:  # Yalnızca araçlarla ilgileniyoruz (örneğin: otomobil, kamyonet, kamyon)
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Algılama kutusunun koordinatları
            confidence = float(box.conf[0])  # Algılamanın doğruluk oranı
            detections.append([x1, y1, x2, y2, confidence])

    # SORT algoritmasıyla araçları takip ediyoruz.
    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, object_id = map(int, obj[:5])  # Algılanan aracın kutusu ve kimliği
        w, h = x2 - x1, y2 - y1

        # Araç merkez konumunu hesaplıyoruz.
        current_position = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Hız hesaplama
        if elapsed_time > 0:
            speed = (w * pixel_to_meter) / elapsed_time * 3.6  # m/s biriminden km/h birimine dönüştürülüyor

            # Ekrana hız ve araç kimliği yazdırıyoruz.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {int(object_id)} HIZ: {int(speed)} km/h",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # İşlenen görüntüyü ekranda gösteriyoruz.
    cv2.imshow("Araç Algılama ve Hız Ölçümü", frame)

    # 'q' tuşuna basıldığında döngü sonlanır ve program kapanır.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
