"""
YOLO Tracker Module - Integrated from Real-Time-YOLO-Car-Counter
https://github.com/Pushtogithub23/Real-Time-YOLO-Car-Counter

Features:
- ByteTrack object tracking for consistent vehicle IDs across frames
- Detection smoothing to reduce jitter
- Line-crossing counting with zone-based detection
- Advanced annotations with bounding boxes, labels, and traces
"""

import cv2 as cv
import numpy as np
from ultralytics import YOLO

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("[WARNING] supervision library not installed. Install with: pip install supervision")

from typing import Tuple, List, Dict, Set, Optional
from dataclasses import dataclass, field


@dataclass
class CountingZone:
    """Defines a polygon zone for counting vehicles"""
    points: np.ndarray
    name: str = "Zone"
    color: Tuple[int, int, int] = (255, 0, 0)
    
    
@dataclass
class CountingLine:
    """Defines a line for counting vehicles crossing"""
    start: Tuple[int, int]
    end: Tuple[int, int]
    name: str = "Line"
    tolerance: int = 15  # pixels above/below line to trigger count


@dataclass
class TrackingStats:
    """Statistics for vehicle tracking"""
    total_count: int = 0
    count_ns: int = 0  # North-South direction
    count_ew: int = 0  # East-West direction
    tracked_ids: Set[int] = field(default_factory=set)
    ids_ns: Set[int] = field(default_factory=set)
    ids_ew: Set[int] = field(default_factory=set)


class YOLOTracker:
    """
    Advanced YOLO-based vehicle tracker with ByteTrack integration.
    Provides real-time vehicle detection, tracking, and counting.
    """
    
    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        track_buffer: int = 30,
        use_smoothing: bool = True,
    ):
        """
        Initialize the YOLO Tracker.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            track_buffer: Number of frames to keep track of lost objects
            use_smoothing: Whether to use detection smoothing
        """
        print(f"[YOLOTracker] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Get vehicle class IDs from model
        self.vehicle_class_ids = [
            cls_id for cls_id, name in self.model.names.items()
            if name in ['car', 'motorcycle', 'motorbike', 'bus', 'truck']
        ]
        
        # Initialize tracking components if supervision is available
        self.tracker = None
        self.smoother = None
        self.use_smoothing = use_smoothing
        
        if SUPERVISION_AVAILABLE:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=conf_threshold,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=30
            )
            if use_smoothing:
                self.smoother = sv.DetectionsSmoother()
            print("[YOLOTracker] ByteTrack tracker initialized")
        else:
            print("[YOLOTracker] Running without tracking (supervision not available)")
        
        # Annotators (will be set based on video resolution)
        self._annotators_initialized = False
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None
        self.ellipse_annotator = None
        
        # Counting statistics
        self.stats = TrackingStats()
        
        # Counting zones and lines
        self.zones: List[CountingZone] = []
        self.counting_lines: List[CountingLine] = []
        
        print("[YOLOTracker] Ready")
    
    def _init_annotators(self, frame_shape: Tuple[int, int]):
        """Initialize annotators based on frame resolution"""
        if not SUPERVISION_AVAILABLE or self._annotators_initialized:
            return
            
        h, w = frame_shape[:2]
        resolution = (w, h)
        
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution)
        
        self.box_annotator = sv.RoundBoxAnnotator(
            thickness=thickness,
            color_lookup=sv.ColorLookup.TRACK
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.TOP_CENTER,
            color_lookup=sv.ColorLookup.TRACK
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=30,
            position=sv.Position.CENTER,
            color_lookup=sv.ColorLookup.TRACK
        )
        
        self.ellipse_annotator = sv.EllipseAnnotator(
            thickness=thickness,
            color_lookup=sv.ColorLookup.TRACK
        )
        
        self._annotators_initialized = True
    
    def set_counting_line(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        name: str = "Line",
        tolerance: int = 15
    ):
        """Add a counting line for vehicle crossing detection"""
        self.counting_lines.append(CountingLine(
            start=start,
            end=end,
            name=name,
            tolerance=tolerance
        ))
    
    def set_ns_ew_zones(self, mid_y: int, frame_height: int, frame_width: int):
        """
        Set up North-South and East-West counting zones.
        NS zone: top half of frame
        EW zone: bottom half of frame
        """
        # NS zone (top half)
        ns_points = np.array([
            [0, 0],
            [frame_width, 0],
            [frame_width, mid_y],
            [0, mid_y]
        ], dtype=np.int32)
        
        # EW zone (bottom half)
        ew_points = np.array([
            [0, mid_y],
            [frame_width, mid_y],
            [frame_width, frame_height],
            [0, frame_height]
        ], dtype=np.int32)
        
        self.zones = [
            CountingZone(ns_points, "NS", (0, 255, 0)),
            CountingZone(ew_points, "EW", (0, 191, 255))
        ]
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Run YOLO detection on a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of (detections array, list of detection dicts)
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            
            for box, cls_id, conf in zip(boxes, classes, confs):
                if cls_id in self.vehicle_class_ids:
                    detections.append({
                        'box': box,  # [x1, y1, x2, y2]
                        'class_id': cls_id,
                        'class_name': self.model.names[cls_id],
                        'confidence': conf
                    })
        
        return results, detections
    
    def track(self, frame: np.ndarray) -> Tuple[Optional[object], List[dict]]:
        """
        Run detection and tracking on a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of (supervision detections object, list of tracked detection dicts)
        """
        if not SUPERVISION_AVAILABLE:
            _, detections = self.detect(frame)
            return None, detections
        
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )[0]
        
        # Convert to supervision detections
        sv_detections = sv.Detections.from_ultralytics(results)
        
        # Filter to vehicle classes only
        vehicle_mask = np.isin(sv_detections.class_id, self.vehicle_class_ids)
        sv_detections = sv_detections[vehicle_mask]
        
        # Update tracker
        if self.tracker is not None:
            sv_detections = self.tracker.update_with_detections(sv_detections)
        
        # Apply smoothing
        if self.smoother is not None and self.use_smoothing:
            sv_detections = self.smoother.update_with_detections(sv_detections)
        
        # Convert to list of dicts for easier handling
        tracked_detections = []
        if sv_detections.tracker_id is not None:
            for i, (xyxy, conf, cls_id, tracker_id) in enumerate(zip(
                sv_detections.xyxy,
                sv_detections.confidence,
                sv_detections.class_id,
                sv_detections.tracker_id
            )):
                tracked_detections.append({
                    'box': xyxy.astype(int),
                    'class_id': cls_id,
                    'class_name': self.model.names[cls_id],
                    'confidence': conf,
                    'tracker_id': tracker_id
                })
        
        return sv_detections, tracked_detections
    
    def count_in_zones(
        self,
        sv_detections: object,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[int, int]:
        """
        Count vehicles in NS and EW zones.
        
        Returns:
            Tuple of (count_ns, count_ew)
        """
        if not SUPERVISION_AVAILABLE or sv_detections is None:
            return 0, 0
            
        if len(self.zones) < 2:
            h, w = frame_shape[:2]
            self.set_ns_ew_zones(h // 2, h, w)
        
        count_ns = 0
        count_ew = 0
        
        if sv_detections.tracker_id is not None:
            for tracker_id, center in zip(
                sv_detections.tracker_id,
                sv_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            ):
                cx, cy = map(int, center)
                
                # Check NS zone (index 0)
                if cv.pointPolygonTest(self.zones[0].points, (cx, cy), False) >= 0:
                    if tracker_id not in self.stats.ids_ns:
                        self.stats.ids_ns.add(tracker_id)
                        count_ns += 1
                
                # Check EW zone (index 1)
                elif cv.pointPolygonTest(self.zones[1].points, (cx, cy), False) >= 0:
                    if tracker_id not in self.stats.ids_ew:
                        self.stats.ids_ew.add(tracker_id)
                        count_ew += 1
        
        self.stats.count_ns = len(self.stats.ids_ns)
        self.stats.count_ew = len(self.stats.ids_ew)
        
        return count_ns, count_ew
    
    def count_vehicles_simple(self, frame: np.ndarray) -> Tuple[int, int, np.ndarray]:
        """
        Simple vehicle counting for traffic management - counts by zone.
        Compatible with existing vehicle_detection.py interface.
        
        Args:
            frame: BGR image
            
        Returns:
            Tuple of (count_ns, count_ew, annotated_frame)
        """
        height, width = frame.shape[:2]
        mid_y = height // 2
        
        # Set up zones if not already done
        if len(self.zones) < 2:
            self.set_ns_ew_zones(mid_y, height, width)
        
        # Initialize annotators
        self._init_annotators(frame.shape)
        
        # Run detection and tracking
        sv_detections, tracked_dets = self.track(frame)
        
        annotated = frame.copy()
        count_ns = 0
        count_ew = 0
        
        if SUPERVISION_AVAILABLE and sv_detections is not None and sv_detections.tracker_id is not None:
            # Count by position
            for tracker_id, center in zip(
                sv_detections.tracker_id,
                sv_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            ):
                cx, cy = map(int, center)
                
                if cy < mid_y:
                    count_ns += 1
                else:
                    count_ew += 1
            
            # Annotate with supervision
            if self.box_annotator:
                self.box_annotator.annotate(annotated, sv_detections)
            
            if self.label_annotator:
                labels = [
                    f"#{tid} {self.model.names[cid]}"
                    for tid, cid in zip(sv_detections.tracker_id, sv_detections.class_id)
                ]
                self.label_annotator.annotate(annotated, sv_detections, labels=labels)
            
            if self.trace_annotator:
                self.trace_annotator.annotate(annotated, sv_detections)
        else:
            # Fallback without supervision
            for det in tracked_dets:
                box = det['box']
                x1, y1, x2, y2 = box
                center_y = (y1 + y2) // 2
                
                if center_y < mid_y:
                    count_ns += 1
                    color = (0, 255, 0)  # Green for NS
                else:
                    count_ew += 1
                    color = (0, 191, 255)  # Yellow for EW
                
                cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det.get('tracker_id', '')} {det['class_name']} {det['confidence']:.2f}"
                cv.putText(annotated, label, (x1, max(10, y1 - 5)),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw zone divider line
        cv.line(annotated, (0, mid_y), (width, mid_y), (255, 255, 255), 2)
        
        # Draw overlay for zones
        self._draw_overlay(annotated, self.zones[0].points, (0, 255, 0), 0.1)
        self._draw_overlay(annotated, self.zones[1].points, (0, 191, 255), 0.1)
        
        # Draw count display
        self._draw_counts(annotated, count_ns, count_ew)
        
        return count_ns, count_ew, annotated
    
    def _draw_overlay(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.25
    ):
        """Draw semi-transparent polygon overlay"""
        overlay = frame.copy()
        cv.fillPoly(overlay, [points], color)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def _draw_counts(
        self,
        frame: np.ndarray,
        count_ns: int,
        count_ew: int
    ):
        """Draw count display on frame"""
        # Background rectangle
        cv.rectangle(frame, (10, 10), (200, 100), (255, 255, 255), cv.FILLED)
        cv.rectangle(frame, (10, 10), (200, 100), (0, 0, 0), 2)
        
        # NS count (green)
        cv.putText(frame, f"NS: {count_ns}", (20, 40),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)
        
        # EW count (orange)
        cv.putText(frame, f"EW: {count_ew}", (20, 70),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
        
        # Total
        total = count_ns + count_ew
        cv.putText(frame, f"Total: {total}", (20, 95),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def reset_counts(self):
        """Reset all counting statistics"""
        self.stats = TrackingStats()
    
    def get_stats(self) -> Dict:
        """Get current tracking statistics"""
        return {
            'total_count': self.stats.count_ns + self.stats.count_ew,
            'count_ns': self.stats.count_ns,
            'count_ew': self.stats.count_ew,
            'unique_vehicles': len(self.stats.tracked_ids),
            'unique_ns': len(self.stats.ids_ns),
            'unique_ew': len(self.stats.ids_ew)
        }


class VideoProcessor:
    """
    Process video files or streams with YOLO tracking and counting.
    Adapted from Real-Time-YOLO-Car-Counter.
    """
    
    def __init__(
        self,
        source: str,  # video path, camera index, or RTSP URL
        tracker: Optional[YOLOTracker] = None,
        output_path: Optional[str] = None
    ):
        """
        Initialize video processor.
        
        Args:
            source: Video source (file path, camera index as string, or URL)
            tracker: YOLOTracker instance (creates one if not provided)
            output_path: Optional path to save processed video
        """
        self.source = source
        self.tracker = tracker or YOLOTracker()
        self.output_path = output_path
        
        # Try to open as integer (camera index) or string (file/URL)
        try:
            self.source_int = int(source)
        except (ValueError, TypeError):
            self.source_int = source
        
        self.cap = None
        self.writer = None
        self.fps = 30
        self.width = 0
        self.height = 0
    
    def _init_capture(self):
        """Initialize video capture"""
        self.cap = cv.VideoCapture(self.source_int)
        
        if not self.cap.isOpened():
            raise Exception(f"Error: Could not open video source: {self.source}")
        
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        if self.output_path:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.writer = cv.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
    
    def process(
        self,
        show_preview: bool = True,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        Process the video source.
        
        Args:
            show_preview: Whether to display preview window
            callback: Optional function called with (count_ns, count_ew, frame) for each frame
            
        Returns:
            Final statistics dictionary
        """
        self._init_capture()
        self.tracker.reset_counts()
        
        frame_count = 0
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                count_ns, count_ew, annotated = self.tracker.count_vehicles_simple(frame)
                
                frame_count += 1
                
                # Call callback if provided
                if callback:
                    callback(count_ns, count_ew, annotated)
                
                # Write output
                if self.writer:
                    self.writer.write(annotated)
                
                # Show preview
                if show_preview:
                    cv.imshow("YOLO Vehicle Counter", annotated)
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('p'):
                        break
        
        finally:
            self._cleanup()
        
        stats = self.tracker.get_stats()
        stats['frames_processed'] = frame_count
        return stats
    
    def _cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv.destroyAllWindows()


# Standalone execution for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Vehicle Tracker")
    parser.add_argument("--source", type=str, default="0", help="Video source (file, camera index, or URL)")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="YOLO model path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    
    args = parser.parse_args()
    
    tracker = YOLOTracker(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    processor = VideoProcessor(
        source=args.source,
        tracker=tracker,
        output_path=args.output
    )
    
    print(f"Processing video from: {args.source}")
    print("Press 'q' to quit")
    
    stats = processor.process(show_preview=True)
    
    print("\n--- Final Statistics ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")
