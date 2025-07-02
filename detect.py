import cv2
import time
from ultralytics import YOLO

def yolov10_inference(frame, model, image_size, conf_threshold):
    results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
    frame = results[0].plot()
    return frame

def main():
    image_size = 640  # Sesuaikan sesuai kebutuhan
    conf_threshold = 0.25  # Sesuaikan sesuai kebutuhan
    model = YOLO("best.pt")  # Menggunakan model best.pt
    cap = cv2.VideoCapture(0)  # Menggunakan webcam (0 untuk kamera default)
    
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return
    
    while True:
        ret, frame = cap.read()
        start_time = time.time()
        
        if not ret:
            break
        
        frame = yolov10_inference(frame, model, image_size, conf_threshold)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        framefps = "FPS: {:.2f}".format(fps)
        cv2.rectangle(frame, (10, 10), (120, 20), (0, 0, 0), -1)
        cv2.putText(frame, framefps, (15, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Deteksi Objek YOLOv8", frame)  # Tampilkan frame yang sudah diberi anotasi
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Keluar jika menekan tombol 'q'
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
