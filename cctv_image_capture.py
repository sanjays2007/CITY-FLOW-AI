import cv2
import time
import os


def capture_image(camera_index, output_path):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        return

    cv2.imwrite(output_path, frame)
    cap.release()
    print("Image captured and saved as", output_path)


def main():
    camera_index = 0          # change if needed
    interval_seconds = 5      # capture every 5 seconds

    os.makedirs("static", exist_ok=True)
    output_path = "static/latest_frame.jpg"   # <<< must match app.py

    while True:
        capture_image(camera_index, output_path)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
