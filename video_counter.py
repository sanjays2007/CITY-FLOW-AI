"""
Real-Time Video Vehicle Counter
Standalone script for advanced vehicle counting with YOLO + ByteTrack.

Features:
- Multi-zone counting (NS/EW directions)
- Line-crossing detection
- Real-time tracking with unique IDs
- Trace visualization for vehicle paths
- Detection smoothing for stable counts
- Video file or live camera support

Integrated from: https://github.com/Pushtogithub23/Real-Time-YOLO-Car-Counter
"""

import cv2 as cv
import numpy as np
import argparse
import os
from datetime import datetime

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("[WARNING] supervision library not installed. Install with: pip install supervision")

from ultralytics import YOLO


class AdvancedVideoCounter:
    """
    Advanced video-based vehicle counter with polygon zones and line crossing.
    Provides full features from Real-Time-YOLO-Car-Counter.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
    ):
        print(f"[VideoCounter] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Vehicle classes
        self.vehicle_classes = {'car', 'motorcycle', 'motorbike', 'bus', 'truck'}
        self.selected_classes = [
            cls_id for cls_id, name in self.model.names.items()
            if name in self.vehicle_classes
        ]
        
        # Tracking components
        self.tracker = None
        self.smoother = None
        
        if SUPERVISION_AVAILABLE:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=conf_threshold,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30
            )
            self.smoother = sv.DetectionsSmoother()
        
        # Counting storage
        self.total_counts = []
        self.crossed_ids = set()
        self.counts_ns = []
        self.ids_ns = set()
        self.counts_ew = []
        self.ids_ew = set()
        
        # Zones
        self.zone_points = []
        self.zones = []
        
        print("[VideoCounter] Ready")
    
    def setup_zones_from_frame(self, frame_height: int, frame_width: int, mid_y: int = None):
        """Set up NS/EW counting zones based on frame dimensions"""
        if mid_y is None:
            mid_y = frame_height // 2
        
        # NS zone (top half)
        zone_ns = np.array([
            [0, 0],
            [frame_width, 0],
            [frame_width, mid_y],
            [0, mid_y]
        ], dtype=np.int32)
        
        # EW zone (bottom half)
        zone_ew = np.array([
            [0, mid_y],
            [frame_width, mid_y],
            [frame_width, frame_height],
            [0, frame_height]
        ], dtype=np.int32)
        
        self.zone_points = [zone_ns, zone_ew]
        
        if SUPERVISION_AVAILABLE:
            self.zones = [sv.PolygonZone(points) for points in self.zone_points]
    
    def setup_custom_zones(self, zone_points_list: list):
        """Set up custom polygon zones"""
        self.zone_points = [np.array(pts, dtype=np.int32) for pts in zone_points_list]
        
        if SUPERVISION_AVAILABLE:
            self.zones = [sv.PolygonZone(points) for points in self.zone_points]
    
    def draw_overlay(self, frame, points, color, alpha=0.25):
        """Draw semi-transparent polygon overlay"""
        overlay = frame.copy()
        cv.fillPoly(overlay, [points], color)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def count_vehicle_in_zone(self, track_id, cx, cy, zone_idx):
        """Count vehicle if it's in the specified zone"""
        if cv.pointPolygonTest(self.zone_points[zone_idx], (cx, cy), False) >= 0:
            if track_id not in self.crossed_ids:
                self.total_counts.append(track_id)
                self.crossed_ids.add(track_id)
            
            if zone_idx == 0 and track_id not in self.ids_ns:
                self.counts_ns.append(track_id)
                self.ids_ns.add(track_id)
            elif zone_idx == 1 and track_id not in self.ids_ew:
                self.counts_ew.append(track_id)
                self.ids_ew.add(track_id)
    
    def process_frame(self, frame):
        """Process a single frame with detection and counting"""
        h, w = frame.shape[:2]
        
        # Setup zones if not done
        if len(self.zone_points) == 0:
            self.setup_zones_from_frame(h, w)
        
        # Run YOLO
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )[0]
        
        annotated = frame.copy()
        count_ns = 0
        count_ew = 0
        
        if SUPERVISION_AVAILABLE:
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter to vehicle classes
            vehicle_mask = np.isin(detections.class_id, self.selected_classes)
            detections = detections[vehicle_mask]
            
            # Track
            if self.tracker:
                detections = self.tracker.update_with_detections(detections)
            
            # Smooth
            if self.smoother:
                detections = self.smoother.update_with_detections(detections)
            
            # Draw zone overlays
            self.draw_overlay(annotated, self.zone_points[0], (0, 255, 0), 0.15)  # NS - green
            self.draw_overlay(annotated, self.zone_points[1], (0, 191, 255), 0.15)  # EW - yellow
            
            # Draw zone boundaries
            cv.polylines(annotated, [self.zone_points[0]], True, (0, 255, 0), 2)
            cv.polylines(annotated, [self.zone_points[1]], True, (0, 191, 255), 2)
            
            if detections.tracker_id is not None:
                # Calculate optimal sizes
                thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
                text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
                
                # Annotators
                box_annotator = sv.RoundBoxAnnotator(
                    thickness=thickness,
                    color_lookup=sv.ColorLookup.TRACK
                )
                label_annotator = sv.LabelAnnotator(
                    text_scale=text_scale,
                    text_thickness=thickness,
                    text_position=sv.Position.TOP_CENTER,
                    color_lookup=sv.ColorLookup.TRACK
                )
                trace_annotator = sv.TraceAnnotator(
                    thickness=thickness,
                    trace_length=30,
                    position=sv.Position.CENTER,
                    color_lookup=sv.ColorLookup.TRACK
                )
                
                # Draw annotations
                box_annotator.annotate(annotated, detections)
                
                labels = [
                    f"#{tid} {self.model.names[cid]}"
                    for tid, cid in zip(detections.tracker_id, detections.class_id)
                ]
                label_annotator.annotate(annotated, detections, labels=labels)
                trace_annotator.annotate(annotated, detections)
                
                # Count and draw centers
                for track_id, center in zip(
                    detections.tracker_id,
                    detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
                ):
                    cx, cy = map(int, center)
                    cv.circle(annotated, (cx, cy), 4, (0, 255, 255), cv.FILLED)
                    
                    # Count in zones
                    self.count_vehicle_in_zone(track_id, cx, cy, 0)  # NS
                    self.count_vehicle_in_zone(track_id, cx, cy, 1)  # EW
                
                count_ns = len(self.counts_ns)
                count_ew = len(self.counts_ew)
        else:
            # Fallback without supervision
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                mid_y = h // 2
                
                for box, cls_id, conf in zip(boxes, classes, confs):
                    if cls_id not in self.selected_classes:
                        continue
                    
                    x1, y1, x2, y2 = box
                    center_y = (y1 + y2) // 2
                    
                    if center_y < mid_y:
                        count_ns += 1
                        color = (0, 255, 0)
                    else:
                        count_ew += 1
                        color = (0, 191, 255)
                    
                    cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv.putText(annotated, f"{self.model.names[cls_id]} {conf:.2f}",
                              (x1, max(10, y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv.line(annotated, (0, mid_y), (w, mid_y), (255, 255, 255), 2)
        
        # Draw count display
        self._draw_counter_display(annotated, count_ns, count_ew)
        
        return count_ns, count_ew, annotated
    
    def _draw_counter_display(self, frame, count_ns, count_ew):
        """Draw count overlay on frame"""
        total = count_ns + count_ew if not SUPERVISION_AVAILABLE else len(self.total_counts)
        
        # Background
        cv.rectangle(frame, (10, 10), (220, 130), (255, 255, 255), cv.FILLED)
        cv.rectangle(frame, (10, 10), (220, 130), (0, 0, 0), 2)
        
        # Counts
        cv.putText(frame, f"TOTAL: {total}", (20, 40),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv.putText(frame, f"NS: {count_ns}", (20, 75),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)
        cv.putText(frame, f"EW: {count_ew}", (20, 110),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    
    def reset(self):
        """Reset all counts"""
        self.total_counts = []
        self.crossed_ids = set()
        self.counts_ns = []
        self.ids_ns = set()
        self.counts_ew = []
        self.ids_ew = set()
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'total': len(self.total_counts),
            'ns': len(self.counts_ns),
            'ew': len(self.counts_ew),
            'unique_ids': len(self.crossed_ids)
        }


def process_video(
    source,
    output_path=None,
    model_path="yolov8n.pt",
    show_preview=True,
    write_counts=True,
    skip_frames=2,  # Process every Nth frame for speed
    target_fps=15   # Target playback speed
):
    """
    Process video file or live camera stream.
    
    Args:
        source: Video source (file path, camera index, or URL)
        output_path: Optional path to save output video
        model_path: Path to YOLO model
        show_preview: Whether to show preview window
        write_counts: Whether to write counts to files
        skip_frames: Process every Nth frame (higher = faster)
        target_fps: Target FPS for playback speed
    """
    # Parse source
    try:
        source_int = int(source)
        is_camera = True
    except (ValueError, TypeError):
        source_int = source
        is_camera = False
    
    cap = cv.VideoCapture(source_int)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return
    
    # Get video properties
    original_fps = int(cap.get(cv.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # For video files, use frame skipping for speed
    if not is_camera:
        # Adjust skip_frames based on video FPS for smooth playback
        effective_fps = original_fps / skip_frames
        print(f"Video: {width}x{height} @ {original_fps}fps (processing at ~{effective_fps:.1f}fps)")
    else:
        skip_frames = 1  # Don't skip frames for live camera
        print(f"Camera: {width}x{height} @ {original_fps}fps")
    
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    
    # Setup video writer
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_path, fourcc, target_fps, (width, height))
        print(f"Output: {output_path}")
    
    # Initialize counter
    counter = AdvancedVideoCounter(model_path=model_path)
    
    # Update tracker frame rate
    if SUPERVISION_AVAILABLE and counter.tracker:
        counter.tracker = sv.ByteTrack(
            track_activation_threshold=counter.conf_threshold,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=original_fps // skip_frames
        )
    
    frame_count = 0
    processed_count = 0
    start_time = datetime.now()
    last_annotated = None
    
    print("\nProcessing started. Press 'q' to quit.\n")
    
    # Ensure static folder exists for frame saving
    os.makedirs("static", exist_ok=True)
    
    # Calculate delay for target playback speed
    # For video files, we want faster playback
    frame_delay = 1  # 1ms for fastest processing
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # If video file ended, optionally loop
                if not is_camera and total_frames > 0:
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                break
            
            frame_count += 1
            
            # Skip frames for speed (only for video files)
            if not is_camera and frame_count % skip_frames != 0:
                # Still show the last processed frame for smoother display
                if show_preview and last_annotated is not None:
                    cv.imshow("YOLO Vehicle Counter", last_annotated)
                    if cv.waitKey(frame_delay) & 0xFF in [ord('q'), ord('p')]:
                        break
                continue
            
            processed_count += 1
            
            # Process frame
            count_ns, count_ew, annotated = counter.process_frame(frame)
            last_annotated = annotated
            
            # Write counts to files
            if write_counts:
                with open("vehicle_count_ns.txt", "w") as f:
                    f.write(str(count_ns))
                with open("vehicle_count_ew.txt", "w") as f:
                    f.write(str(count_ew))
            
            # Save annotated frame for web dashboard
            cv.imwrite("static/latest_frame.jpg", annotated)
            
            # Write to output video
            if writer:
                writer.write(annotated)
            
            # Show preview
            if show_preview:
                cv.imshow("YOLO Vehicle Counter", annotated)
                key = cv.waitKey(frame_delay) & 0xFF
                if key == ord('q') or key == ord('p'):
                    break
            
            # Print periodic stats
            if processed_count % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                stats = counter.get_stats()
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"Frame {frame_count}/{total_frames if total_frames > 0 else '?'} ({progress:.1f}%) | "
                      f"FPS: {fps_actual:.1f} | Total: {stats['total']} | NS: {stats['ns']} | EW: {stats['ew']}")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv.destroyAllWindows()
    
    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()
    stats = counter.get_stats()
    
    print("\n" + "="*50)
    print("FINAL STATISTICS")
    print("="*50)
    print(f"Frames processed: {frame_count}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Average FPS: {frame_count/elapsed:.1f}" if elapsed > 0 else "N/A")
    print(f"Total vehicles: {stats['total']}")
    print(f"  NS direction: {stats['ns']}")
    print(f"  EW direction: {stats['ew']}")
    print(f"Unique IDs tracked: {stats['unique_ids']}")
    print("="*50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time YOLO Vehicle Counter with ByteTrack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_counter.py --source 0                          # Webcam
  python video_counter.py --source video.mp4                  # Video file
  python video_counter.py --source video.mp4 --output out.mp4 # Save output
  python video_counter.py --source video.mp4 --skip 3         # Faster processing
  python video_counter.py --source rtsp://... --no-preview    # RTSP stream
        """
    )
    
    parser.add_argument("--source", type=str, default="0",
                       help="Video source: camera index, file path, or URL")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video path (optional)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--no-preview", action="store_true",
                       help="Disable preview window")
    parser.add_argument("--no-counts", action="store_true",
                       help="Don't write counts to files")
    parser.add_argument("--skip", type=int, default=2,
                       help="Skip frames: process every Nth frame (default: 2, higher=faster)")
    
    args = parser.parse_args()
    
    process_video(
        source=args.source,
        output_path=args.output,
        model_path=args.model,
        show_preview=not args.no_preview,
        write_counts=not args.no_counts,
        skip_frames=args.skip
    )


if __name__ == "__main__":
    main()
