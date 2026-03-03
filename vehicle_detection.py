"""
Vehicle Detection Module with Advanced YOLO Tracking
Uses YOLOTracker for ByteTrack-based object tracking and counting.
Integrated from: https://github.com/Pushtogithub23/Real-Time-YOLO-Car-Counter
"""

import os
import time
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

# Import the advanced tracker module
try:
    from yolo_tracker import YOLOTracker, VideoProcessor, SUPERVISION_AVAILABLE
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    SUPERVISION_AVAILABLE = False

IMAGE_PATH = "static/latest_frame.jpg"   # frame from capture script
CONF_THRESHOLD = 0.3                     # detection confidence threshold
NMS_IOU_THRESHOLD = 0.45                 # NMS IoU threshold

# COCO class ids for vehicles in YOLOv8
VEHICLE_CLASS_IDS = {2, 3, 5, 7}         # car, motorcycle, bus, truck

# Initialize tracker or fallback to basic model
tracker = None
model = None

if TRACKER_AVAILABLE:
    print("[YOLOv8] Initializing advanced tracker with ByteTrack...")
    tracker = YOLOTracker(
        model_path="yolov8n.pt",
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=NMS_IOU_THRESHOLD,
        use_smoothing=True
    )
    print("[YOLOv8] Advanced tracker ready (supervision available: {})".format(SUPERVISION_AVAILABLE))
else:
    print("[YOLOv8] loading model yolov8n.pt (basic mode)...")
    model = YOLO("yolov8n.pt")               # will auto-download first run
    print("[YOLOv8] model ready")


def count_vehicles(image):
    """
    Run YOLOv8 on the image and return NS/EW counts and annotated frame.
    Uses advanced tracking if available, otherwise falls back to basic detection.
    """
    # Use advanced tracker if available
    if TRACKER_AVAILABLE and tracker is not None:
        return tracker.count_vehicles_simple(image)
    
    # Fallback to basic detection (original implementation)
    height, width, _ = image.shape
    mid_y = height // 2  # top = NS, bottom = EW

    # Run inference (results list, one item per image)
    results = model(
        image,
        verbose=False,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
    )

    count_ns = 0
    count_ew = 0

    annotated = image.copy()

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)   # [x1, y1, x2, y2]
        cls = r.boxes.cls.cpu().numpy().astype(int)      # class ids
        confs = r.boxes.conf.cpu().numpy()               # confidences

        for (x1, y1, x2, y2, c, conf) in zip(
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2],
            boxes[:, 3],
            cls,
            confs,
        ):
            if c not in VEHICLE_CLASS_IDS:
                continue

            center_y = (y1 + y2) // 2

            if center_y < mid_y:
                count_ns += 1
                color = (0, 255, 0)      # NS = green
            else:
                count_ew += 1
                color = (0, 191, 255)    # EW = yellow

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"veh {conf:.2f}",
                (x1, max(10, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # draw split line
    cv2.line(annotated, (0, mid_y), (width, mid_y), (255, 255, 255), 1)

    return count_ns, count_ew, annotated


def process_video_stream(source=0, output_path=None):
    """
    Process live video stream with vehicle counting.
    Uses advanced VideoProcessor if available.
    
    Args:
        source: Video source (camera index, file path, or URL)
        output_path: Optional path to save processed video
    """
    if TRACKER_AVAILABLE and tracker is not None:
        processor = VideoProcessor(
            source=str(source),
            tracker=tracker,
            output_path=output_path
        )
        
        def on_frame(count_ns, count_ew, frame):
            # Write counts for controller/Flask
            with open("vehicle_count_ns.txt", "w") as f:
                f.write(str(count_ns))
            with open("vehicle_count_ew.txt", "w") as f:
                f.write(str(count_ew))
        
        print(f"Processing video from: {source}")
        print("Press 'q' to quit")
        
        stats = processor.process(show_preview=True, callback=on_frame)
        
        print("\n--- Final Statistics ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        # Fallback to basic frame-by-frame processing
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            count_ns, count_ew, annotated = count_vehicles(frame)
            
            print(f"NS vehicles: {count_ns} | EW vehicles: {count_ew}")
            
            with open("vehicle_count_ns.txt", "w") as f:
                f.write(str(count_ns))
            with open("vehicle_count_ew.txt", "w") as f:
                f.write(str(count_ew))
            
            cv2.imshow("Vehicle Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection with YOLO")
    parser.add_argument("--mode", type=str, default="image", 
                       choices=["image", "video", "stream"],
                       help="Detection mode: image (from file), video (from file), stream (live camera)")
    parser.add_argument("--source", type=str, default="0",
                       help="Video source (camera index, file path, or URL)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video path (for video/stream modes)")
    
    args = parser.parse_args()
    
    if args.mode == "video" or args.mode == "stream":
        # Video/stream processing mode
        source = int(args.source) if args.source.isdigit() else args.source
        process_video_stream(source, args.output)
    else:
        # Image processing mode (original behavior)
        while True:
            if not os.path.exists(IMAGE_PATH):
                print("No captured images yet. Waiting...")
                time.sleep(2)
                continue

            image = cv2.imread(IMAGE_PATH)
            if image is None:
                print("Failed to read image, skipping...")
                time.sleep(2)
                continue

            count_ns, count_ew, annotated = count_vehicles(image)

            print(f"NS vehicles: {count_ns} | EW vehicles: {count_ew}")

            # write counts for controller/Flask
            with open("vehicle_count_ns.txt", "w") as f:
                f.write(str(count_ns))
            with open("vehicle_count_ew.txt", "w") as f:
                f.write(str(count_ew))

            cv2.imshow("YOLOv8 Vehicle Detection (NS upper, EW lower)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(1)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
