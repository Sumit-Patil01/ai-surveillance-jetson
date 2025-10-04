import cv2
from ultralytics import YOLO

# Example URL for testing Camera 1's sub stream
VIDEO_SOURCE = "rtsp://admin:123%40admin@192.168.29.24:554/cam/realmonitor?channel=2&subtype=1"

# --- Load the model (will auto-download on first run) ---
model = YOLO('yolov8n.pt')

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source at {VIDEO_SOURCE}")
        return

    print("Successfully connected. Running detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()