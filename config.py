# config.py

# =============================================================================
# TRAFFIC SIGNAL TIMING CONFIGURATION
# =============================================================================

# Base green times (seconds)
BASE_GREEN_NS = 25      # little lower base
BASE_GREEN_EW = 25

# Per-vehicle increment
PER_VEHICLE = 3         # stronger effect: +3s per vehicle

# Bounds
MIN_GREEN = 15          # never less than 15s
MAX_GREEN = 120         # allow long greens for heavy queues

# Yellow and all-red times (seconds)
YELLOW_TIME = 3
ALL_RED_TIME = 2

# Time-of-day profiles (24-hr clock)
PEAK_MORNING = (8, 11)    # 08:00–11:59
PEAK_EVENING = (17, 20)   # 17:00–20:59

PEAK_MULTIPLIER_NS = 1.5  # NS boosts more in peak
PEAK_MULTIPLIER_EW = 1.2
NIGHT_MULTIPLIER = 0.7    # shorter greens at night

# =============================================================================
# YOLO DETECTION CONFIGURATION
# =============================================================================

# Model settings
YOLO_MODEL_PATH = "yolov8x.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
CONF_THRESHOLD = 0.3            # Detection confidence threshold (0.0 - 1.0)
IOU_THRESHOLD = 0.45            # NMS IoU threshold (0.0 - 1.0)

# =============================================================================
# TRACKING CONFIGURATION (ByteTrack)
# Integrated from: https://github.com/Pushtogithub23/Real-Time-YOLO-Car-Counter
# =============================================================================

# ByteTrack parameters
TRACK_ACTIVATION_THRESHOLD = 0.3   # Confidence threshold to activate a track
LOST_TRACK_BUFFER = 30             # Frames to keep lost tracks before removing
MINIMUM_MATCHING_THRESHOLD = 0.8   # Minimum IoU for track matching

# Detection smoothing
USE_DETECTION_SMOOTHING = True     # Enable detection smoothing for stable counts

# Trace visualization
TRACE_LENGTH = 30                  # Number of frames to show vehicle trace/path

# =============================================================================
# VIDEO SOURCE CONFIGURATION
# =============================================================================

# Default video source (0 for webcam, or path/URL)
DEFAULT_VIDEO_SOURCE = "0"

# Frame capture interval for image mode (seconds)
CAPTURE_INTERVAL = 5

# Image path for frame capture mode
IMAGE_PATH = "static/latest_frame.jpg"

# =============================================================================
# COUNTING ZONES CONFIGURATION
# =============================================================================

# Zone split mode: "horizontal" (NS/EW by top/bottom) or "custom"
ZONE_SPLIT_MODE = "horizontal"

# For horizontal mode: fraction of frame height for the split line (0.5 = middle)
ZONE_SPLIT_RATIO = 0.5

# Custom zone definitions (used when ZONE_SPLIT_MODE = "custom")
# Each zone is a list of [x, y] points forming a polygon
# Example for custom zones:
# CUSTOM_ZONES = [
#     [[0, 0], [640, 0], [640, 360], [0, 360]],      # NS zone
#     [[0, 360], [640, 360], [640, 720], [0, 720]]   # EW zone
# ]
CUSTOM_ZONES = None

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output directories
LOG_DIR = "data/logs"
OUTPUT_VIDEO_DIR = "data/outputs"

# Write vehicle counts to files (for Flask integration)
WRITE_COUNT_FILES = True
COUNT_FILE_NS = "vehicle_count_ns.txt"
COUNT_FILE_EW = "vehicle_count_ew.txt"

