
# import torch
#
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


from ultralytics import YOLO
import cv2
import cvzone
import time
import torch

# ---------------- GPU CHECK ----------------
device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", "CUDA" if device == 0 else "CPU")

# ---------------- WEBCAM SETUP ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

# ---------------- YOLO MODEL ----------------
model = YOLO("../Yolo-Weights/yolov8s.pt")  # s = speed + accuracy
model.to(device)

# ---------------- CLASS NAMES ----------------
classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
              "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
              "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
              "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
              "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
              "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
              "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
              "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
              "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
              "teddy bear","hair drier","toothbrush"]

# ---------------- FPS ----------------
prev_time = 0

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        print("❌ Frame grab failed")
        break

    # YOLO inference (GPU)
    results = model(
        img,
        device=device,
        imgsz=640,
        conf=0.4,
        verbose=False
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h))
            cls = int(box.cls[0])
            conf = round(float(box.conf[0]), 2)

            cvzone.putTextRect(
                img,
                f'{classNames[cls]} {conf}',
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1
            )

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 40), scale=1.5)

    cv2.imshow("YOLOv8 GPU Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
