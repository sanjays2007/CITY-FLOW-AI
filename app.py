import os
import sys
import time
import csv
import subprocess
import signal
import io
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, send_file, request, Response
from werkzeug.utils import secure_filename

from controller import TrafficController   # NEW: use your smart controller
import database as db  # Database module
from camera_manager import get_camera_manager  # Multi-camera support
from emergency_detector import get_emergency_detector  # Emergency vehicle detection
from signal_cycle import get_signal_controller  # Signal cycle controller

app = Flask(__name__)

# Initialize camera manager and emergency detector
camera_manager = get_camera_manager()
emergency_detector = get_emergency_detector()
signal_controller = get_signal_controller()

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
db.init_database()

# -------------------------------------------------------
# DETECTION PROCESS MANAGEMENT
# -------------------------------------------------------
detection_process = None
current_source = None
current_source_type = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------
# SMART CONTROLLER INSTANCE
# -------------------------------------------------------
controller = TrafficController()  # uses config.py + internal CSV logging


# -------------------------------------------------------
# EXTRA CSV LOGGING (OPTIONAL, can be removed if you only want controller.log)
# -------------------------------------------------------
def log_cycle_simple(phase, ns_count, ew_count, green_time):
    """
    Kept only if you still want app.py-side logging; otherwise not used.
    History widget already reads the controller's CSV.
    """
    os.makedirs("data/logs", exist_ok=True)
    csv_path = "data/logs/cycles_app.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "phase",
                "vehicle_count_ns",
                "vehicle_count_ew",
                "green_time",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "phase": phase,
                "vehicle_count_ns": ns_count,
                "vehicle_count_ew": ew_count,
                "green_time": green_time,
            }
        )


# -------------------------------------------------------
# INDEX
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------
# STATUS (used by dashboard JS)
# -------------------------------------------------------
@app.route("/status")
def status():
    """
    Frontend expects:
      phase           -> "NS", "EW", "YELLOW", "ALL_RED"
      remaining_time  -> number
      green_time      -> number
      vehicle_ns      -> number
      vehicle_ew      -> number
      has_image       -> bool
    """

    # Read current counts from YOLO output text files
    count_ns = 0
    count_ew = 0

    if os.path.exists("vehicle_count_ns.txt"):
        try:
            with open("vehicle_count_ns.txt") as f:
                count_ns = int(f.read().strip() or 0)
        except ValueError:
            count_ns = 0

    if os.path.exists("vehicle_count_ew.txt"):
        try:
            with open("vehicle_count_ew.txt") as f:
                count_ew = int(f.read().strip() or 0)
        except ValueError:
            count_ew = 0

    # Ask smart controller for phase + timings + load
    ui_phase, remaining, green_time, load = controller.update_phase(count_ns, count_ew)

    has_image = os.path.exists("static/latest_frame.jpg")
    
    # Record to database every time status is checked (throttled by frontend polling)
    if has_image and (count_ns > 0 or count_ew > 0):
        try:
            db.insert_detection_record(
                vehicle_ns=count_ns,
                vehicle_ew=count_ew,
                phase=ui_phase,
                green_time=green_time,
                source_type=current_source_type or 'unknown'
            )
        except Exception as e:
            pass  # Don't fail status if DB insert fails

    return jsonify(
        {
            "phase": ui_phase,
            "remaining_time": remaining,
            "green_time": green_time,
            "vehicle_ns": count_ns,
            "vehicle_ew": count_ew,
            "has_image": has_image,
            # Emergency detection status
            "emergency_detected": emergency_detector.emergency_detected,
            "emergency_type": emergency_detector.emergency_type,
            "emergency_confidence": emergency_detector.emergency_confidence,
            # Multi-camera info
            "active_camera": camera_manager.active_camera_id,
            "camera_count": len(camera_manager.cameras)
        }
    )


# -------------------------------------------------------
# HISTORY (last 10 rows of controller CSV)
# -------------------------------------------------------
@app.route("/history")
def history():
    # controller already writes to data/logs/cycles.csv
    csv_path = "data/logs/cycles.csv"
    if not os.path.exists(csv_path):
        return jsonify([])

    cycles = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows[-10:]:
            cycles.append(
                {
                    "timestamp": row["timestamp"],
                    "phase": row["phase"],
                    "vehicle_ns": row["vehicle_count_ns"],
                    "vehicle_ew": row["vehicle_count_ew"],
                    "green_time": row["green_time"],
                }
            )

    return jsonify(cycles)


# -------------------------------------------------------
# IMAGE (served to <img id="trafficImage"> when has_image = true)
# -------------------------------------------------------
@app.route("/image")
def image():
    """
    Expects your capture script to keep writing a frame to:
      static/latest_frame.jpg
    """
    img_path = "static/latest_frame.jpg"
    if os.path.exists(img_path):
        return send_file(img_path, mimetype="image/jpeg")
    return ("", 404)


# -------------------------------------------------------
# VIDEO SOURCE CONTROL ENDPOINTS
# -------------------------------------------------------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Handle video file upload"""
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided"})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            "success": True,
            "filename": filename,
            "path": filepath
        })
    
    return jsonify({"success": False, "error": "Invalid file type"})


@app.route("/start_detection", methods=["POST"])
def start_detection():
    """Start vehicle detection with specified source"""
    global detection_process, current_source, current_source_type
    
    data = request.get_json()
    source = data.get('source', '0')
    source_type = data.get('type', 'webcam')
    skip_frames = data.get('skip_frames', 2)  # Skip frames for faster video processing
    
    # Stop existing process if running
    if detection_process is not None:
        try:
            detection_process.terminate()
            detection_process.wait(timeout=5)
        except:
            try:
                detection_process.kill()
            except:
                pass
        detection_process = None
    
    try:
        # Start the video counter process (use same Python as current process)
        cmd = [
            sys.executable, 'video_counter.py',
            '--source', str(source),
            '--no-preview',
            '--skip', str(skip_frames if source_type == 'video' else 1)  # Only skip for video files
        ]
        
        detection_process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        current_source = source
        current_source_type = source_type
        
        # Give it a moment to start
        time.sleep(1)
        
        # Check if process is still running
        if detection_process.poll() is None:
            return jsonify({
                "success": True,
                "message": f"Detection started with source: {source}",
                "pid": detection_process.pid
            })
        else:
            stderr = detection_process.stderr.read().decode()
            return jsonify({
                "success": False,
                "error": f"Process exited immediately: {stderr[:200]}"
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    """Stop vehicle detection"""
    global detection_process, current_source, current_source_type
    
    if detection_process is None:
        return jsonify({"success": True, "message": "No detection running"})
    
    try:
        if os.name == 'nt':
            # Windows
            detection_process.terminate()
        else:
            # Unix
            os.killpg(os.getpgid(detection_process.pid), signal.SIGTERM)
        
        detection_process.wait(timeout=5)
        detection_process = None
        current_source = None
        current_source_type = None
        
        return jsonify({"success": True, "message": "Detection stopped"})
    except Exception as e:
        # Force kill if terminate fails
        try:
            detection_process.kill()
            detection_process = None
        except:
            pass
        return jsonify({"success": True, "message": f"Detection force stopped: {e}"})


@app.route("/detection_status")
def detection_status():
    """Get current detection status"""
    global detection_process, current_source, current_source_type
    
    running = False
    if detection_process is not None:
        running = detection_process.poll() is None
        if not running:
            detection_process = None
            current_source = None
            current_source_type = None
    
    return jsonify({
        "running": running,
        "source": current_source,
        "type": current_source_type
    })


@app.route("/list_videos")
def list_videos():
    """List uploaded videos"""
    videos = []
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if allowed_file(f):
                videos.append({
                    "name": f,
                    "path": os.path.join(UPLOAD_FOLDER, f)
                })
    return jsonify(videos)


# -------------------------------------------------------
# DATABASE & ANALYTICS ENDPOINTS
# -------------------------------------------------------
@app.route("/api/record_detection", methods=["POST"])
def record_detection():
    """Record a detection entry to database"""
    data = request.get_json()
    try:
        record_id = db.insert_detection_record(
            vehicle_ns=data.get('vehicle_ns', 0),
            vehicle_ew=data.get('vehicle_ew', 0),
            phase=data.get('phase', 'NS'),
            green_time=data.get('green_time', 0),
            source_type=data.get('source_type', 'webcam'),
            frame_number=data.get('frame_number', 0)
        )
        return jsonify({"success": True, "id": record_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/stats/today")
def get_today_stats():
    """Get today's statistics"""
    stats = db.get_today_stats()
    return jsonify(stats if stats else {})


@app.route("/api/stats/hourly")
def get_hourly_stats():
    """Get hourly breakdown for a date"""
    date = request.args.get('date')
    hourly = db.get_hourly_breakdown(date)
    return jsonify(hourly)


@app.route("/api/stats/weekly")
def get_weekly_stats():
    """Get weekly summary"""
    weekly = db.get_weekly_summary()
    return jsonify(weekly)


@app.route("/api/stats/daily")
def get_daily_stats():
    """Get detailed daily summary"""
    date = request.args.get('date')
    summary = db.get_daily_summary(date)
    return jsonify(summary)


@app.route("/api/history/detection")
def get_detection_history():
    """Get detection history records"""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    records = db.get_detection_history(limit, offset)
    return jsonify(records)


@app.route("/api/history/signals")
def get_signal_history():
    """Get signal phase history"""
    limit = request.args.get('limit', 50, type=int)
    signals = db.get_signal_history(limit)
    return jsonify(signals)


# -------------------------------------------------------
# EXPORT ENDPOINTS
# -------------------------------------------------------
@app.route("/export/csv")
def export_csv():
    """Export detection data as CSV"""
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    csv_data = db.export_to_csv(start_date, end_date)
    
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=traffic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        }
    )


@app.route("/export/pdf")
def export_pdf():
    """Export traffic report as PDF"""
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Get report data
        report_data = db.generate_report_data(start_date, end_date)
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=20)
        elements.append(Paragraph("Smart Traffic Management Report", title_style))
        
        # Period
        period = report_data.get('period', {})
        elements.append(Paragraph(f"Period: {period.get('start', 'N/A')} to {period.get('end', 'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Generated: {report_data.get('generated_at', 'N/A')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Overall Statistics
        elements.append(Paragraph("Overall Statistics", styles['Heading2']))
        overall = report_data.get('overall', {})
        stats_data = [
            ['Metric', 'Value'],
            ['Total Records', str(overall.get('total_records', 0))],
            ['Total Vehicles (NS)', str(overall.get('total_ns', 0))],
            ['Total Vehicles (EW)', str(overall.get('total_ew', 0))],
            ['Total Vehicles', str(overall.get('total_vehicles', 0))],
            ['Avg NS Count', f"{overall.get('avg_ns', 0):.1f}"],
            ['Avg EW Count', f"{overall.get('avg_ew', 0):.1f}"],
            ['Avg Green Time', f"{overall.get('avg_green_time', 0):.1f}s"],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 20))
        
        # Daily Breakdown
        daily = report_data.get('daily_breakdown', [])
        if daily:
            elements.append(Paragraph("Daily Breakdown", styles['Heading2']))
            daily_data = [['Date', 'Vehicles', 'Avg NS', 'Avg EW']]
            for d in daily:
                daily_data.append([
                    str(d.get('date', '')),
                    str(d.get('vehicles', 0)),
                    f"{d.get('avg_ns', 0):.1f}",
                    f"{d.get('avg_ew', 0):.1f}"
                ])
            
            daily_table = Table(daily_data, colWidths=[2*inch, 1.5*inch, 1.25*inch, 1.25*inch])
            daily_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0ea5e9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(daily_table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'traffic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
        
    except ImportError:
        # If reportlab is not installed, return JSON instead
        report_data = db.generate_report_data(start_date, end_date)
        return jsonify({
            "error": "PDF export requires reportlab. Install with: pip install reportlab",
            "data": report_data
        }), 500


@app.route("/api/summaries/generate", methods=["POST"])
def generate_summaries():
    """Manually trigger summary generation"""
    try:
        db.update_hourly_stats()
        return jsonify({"success": True, "message": "Summaries updated"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# -------------------------------------------------------
# MULTI-CAMERA MANAGEMENT ENDPOINTS
# -------------------------------------------------------
@app.route("/api/cameras")
def get_cameras():
    """Get list of all cameras"""
    cameras = camera_manager.get_all_cameras()
    return jsonify({
        "cameras": cameras,
        "active_camera": camera_manager.active_camera_id
    })


@app.route("/api/cameras/add", methods=["POST"])
def add_camera():
    """Add a new camera source"""
    data = request.get_json()
    source_type = data.get('type', 'webcam')  # webcam, video, rtsp
    name = data.get('name', 'New Camera')
    source = data.get('source', 0)
    intersection = data.get('intersection', '')
    
    try:
        if source_type == 'webcam':
            camera = camera_manager.add_webcam(name, int(source), intersection)
        elif source_type == 'video':
            camera = camera_manager.add_video_source(name, source, intersection)
        elif source_type == 'rtsp':
            camera = camera_manager.add_rtsp_source(name, source, intersection)
        else:
            return jsonify({"success": False, "error": "Invalid source type"})
        
        return jsonify({
            "success": True,
            "camera_id": camera.camera_id,
            "message": f"Camera '{name}' added successfully"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/cameras/remove", methods=["POST"])
def remove_camera():
    """Remove a camera"""
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_manager.remove_camera(camera_id):
        return jsonify({"success": True, "message": "Camera removed"})
    return jsonify({"success": False, "error": "Camera not found"})


@app.route("/api/cameras/select", methods=["POST"])
def select_camera():
    """Set active camera"""
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_manager.set_active_camera(camera_id):
        return jsonify({"success": True, "active_camera": camera_id})
    return jsonify({"success": False, "error": "Camera not found"})


@app.route("/api/cameras/connect", methods=["POST"])
def connect_camera():
    """Connect to a camera"""
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_manager.connect_camera(camera_id):
        return jsonify({"success": True, "message": "Camera connected"})
    return jsonify({"success": False, "error": "Failed to connect camera"})


# -------------------------------------------------------
# EMERGENCY DETECTION ENDPOINTS
# -------------------------------------------------------
@app.route("/api/emergency/status")
def emergency_status():
    """Get current emergency detection status"""
    return jsonify({
        "emergency_detected": emergency_detector.emergency_detected,
        "emergency_type": emergency_detector.emergency_type,
        "confidence": emergency_detector.emergency_confidence,
        "priority_active": emergency_detector.priority_active,
        "priority_direction": emergency_detector.priority_direction
    })


@app.route("/api/emergency/history")
def emergency_history():
    """Get emergency detection history from all cameras"""
    emergencies = camera_manager.get_emergency_status()
    return jsonify(emergencies)


@app.route("/api/emergency/acknowledge", methods=["POST"])
def acknowledge_emergency():
    """Acknowledge and clear emergency alert"""
    emergency_detector.emergency_detected = False
    emergency_detector.emergency_type = None
    emergency_detector.emergency_confidence = 0.0
    emergency_detector.priority_active = False
    
    return jsonify({"success": True, "message": "Emergency acknowledged"})


@app.route("/api/emergency/test", methods=["POST"])
def test_emergency():
    """Trigger a test emergency alert (for testing UI)"""
    data = request.get_json()
    emergency_type = data.get('type', 'Ambulance')
    
    emergency_detector.emergency_detected = True
    emergency_detector.emergency_type = emergency_type
    emergency_detector.emergency_confidence = 0.95
    emergency_detector.priority_active = True
    emergency_detector.priority_direction = 'NS'
    
    return jsonify({
        "success": True,
        "message": f"Test emergency triggered: {emergency_type}"
    })


# -------------------------------------------------------
# SIGNAL CYCLE API ENDPOINTS
# -------------------------------------------------------

@app.route("/api/signal/start", methods=["POST"])
def start_signal_cycle():
    """Start the automated signal cycle"""
    global detection_process
    
    data = request.get_json() or {}
    
    # Update parameters if provided
    if 'vehicle_threshold' in data:
        signal_controller.vehicle_threshold = int(data['vehicle_threshold'])
    if 'max_detection_time' in data:
        signal_controller.max_detection_time = int(data['max_detection_time'])
    if 'base_green_time' in data:
        signal_controller.base_green_time = int(data['base_green_time'])
    
    # Use video source from request, or current source, or default to webcam
    video_source = data.get('source', current_source or '0')
    source_type = data.get('type', current_source_type or 'webcam')
    
    # Set the video source in signal controller
    signal_controller.set_video_source(video_source, source_type)
    
    # Stop any existing detection process from dashboard
    if detection_process is not None:
        try:
            detection_process.terminate()
            detection_process.wait(timeout=3)
        except:
            pass
        detection_process = None
    
    signal_controller.start()
    
    return jsonify({
        "success": True,
        "message": "Signal cycle started",
        "parameters": {
            "vehicle_threshold": signal_controller.vehicle_threshold,
            "max_detection_time": signal_controller.max_detection_time,
            "base_green_time": signal_controller.base_green_time,
            "video_source": video_source
        }
    })


@app.route("/api/signal/stop", methods=["POST"])
def stop_signal_cycle():
    """Stop the automated signal cycle"""
    signal_controller.stop()
    
    return jsonify({
        "success": True,
        "message": "Signal cycle stopped"
    })


@app.route("/api/signal/state")
def get_signal_state():
    """Get current signal cycle state"""
    from dataclasses import asdict
    state = signal_controller.get_state()
    return jsonify(asdict(state))


@app.route("/api/signal/parameters", methods=["GET", "POST"])
def signal_parameters():
    """Get or update signal cycle parameters"""
    if request.method == "POST":
        data = request.get_json()
        signal_controller.set_parameters(
            vehicle_threshold=data.get('vehicle_threshold'),
            max_detection_time=data.get('max_detection_time'),
            base_green_time=data.get('base_green_time'),
            per_vehicle_time=data.get('per_vehicle_time')
        )
        return jsonify({"success": True, "message": "Parameters updated"})
    
    return jsonify({
        "vehicle_threshold": signal_controller.vehicle_threshold,
        "max_detection_time": signal_controller.max_detection_time,
        "base_green_time": signal_controller.base_green_time,
        "per_vehicle_time": signal_controller.per_vehicle_time,
        "min_green_time": signal_controller.min_green_time,
        "max_green_time": signal_controller.max_green_time,
        "yellow_time": signal_controller.yellow_time
    })


# -------------------------------------------------------
if __name__ == "__main__":
    # debug=True for development
    app.run(host="0.0.0.0", port=5000, debug=True)
