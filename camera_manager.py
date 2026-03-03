"""
Multi-Camera Manager Module
Handles multiple camera sources for monitoring multiple intersections
"""

import cv2
import threading
import time
from collections import defaultdict
import numpy as np

class CameraSource:
    """Represents a single camera source"""
    
    def __init__(self, camera_id, name, source, intersection_name=""):
        self.camera_id = camera_id
        self.name = name
        self.source = source  # Can be int (webcam), string (RTSP/file), or URL
        self.intersection_name = intersection_name
        self.cap = None
        self.is_active = False
        self.last_frame = None
        self.last_frame_time = 0
        self.fps = 0
        self.frame_count = 0
        self.error = None
        self.lock = threading.Lock()
        
        # Detection stats
        self.vehicle_count = {'NS': 0, 'EW': 0}
        self.emergency_detected = False
        self.emergency_type = None
        
    def connect(self):
        """Connect to camera source"""
        try:
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if self.cap.isOpened():
                self.is_active = True
                self.error = None
                return True
            else:
                self.error = "Failed to open camera"
                return False
        except Exception as e:
            self.error = str(e)
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_active = False
    
    def read_frame(self):
        """Read a frame from the camera"""
        if not self.is_active or self.cap is None:
            return None
        
        with self.lock:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.last_frame_time = time.time()
                self.frame_count += 1
                return frame
            else:
                self.error = "Failed to read frame"
                return None
    
    def get_status(self):
        """Get camera status"""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'intersection': self.intersection_name,
            'is_active': self.is_active,
            'error': self.error,
            'fps': self.fps,
            'vehicle_count': self.vehicle_count,
            'emergency_detected': self.emergency_detected,
            'emergency_type': self.emergency_type
        }


class CameraManager:
    """Manages multiple camera sources"""
    
    def __init__(self):
        self.cameras = {}  # camera_id -> CameraSource
        self.active_camera_id = None
        self.lock = threading.Lock()
        
        # Default cameras
        self._add_default_cameras()
        
    def _add_default_cameras(self):
        """Add default camera configurations"""
        # Main webcam
        self.add_camera(
            camera_id="cam_main",
            name="Main Camera",
            source=0,
            intersection_name="Main Intersection"
        )
        
    def add_camera(self, camera_id, name, source, intersection_name=""):
        """Add a new camera source"""
        with self.lock:
            camera = CameraSource(camera_id, name, source, intersection_name)
            self.cameras[camera_id] = camera
            
            # Set as active if first camera
            if self.active_camera_id is None:
                self.active_camera_id = camera_id
            
            return camera
    
    def remove_camera(self, camera_id):
        """Remove a camera source"""
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].disconnect()
                del self.cameras[camera_id]
                
                # Update active camera if needed
                if self.active_camera_id == camera_id:
                    if self.cameras:
                        self.active_camera_id = list(self.cameras.keys())[0]
                    else:
                        self.active_camera_id = None
                return True
            return False
    
    def get_camera(self, camera_id):
        """Get a camera by ID"""
        return self.cameras.get(camera_id)
    
    def get_active_camera(self):
        """Get the currently active camera"""
        if self.active_camera_id:
            return self.cameras.get(self.active_camera_id)
        return None
    
    def set_active_camera(self, camera_id):
        """Set the active camera"""
        with self.lock:
            if camera_id in self.cameras:
                self.active_camera_id = camera_id
                return True
            return False
    
    def connect_camera(self, camera_id):
        """Connect a specific camera"""
        camera = self.cameras.get(camera_id)
        if camera:
            return camera.connect()
        return False
    
    def disconnect_camera(self, camera_id):
        """Disconnect a specific camera"""
        camera = self.cameras.get(camera_id)
        if camera:
            camera.disconnect()
            return True
        return False
    
    def get_all_cameras(self):
        """Get list of all cameras"""
        return [cam.get_status() for cam in self.cameras.values()]
    
    def get_active_cameras(self):
        """Get list of active (connected) cameras"""
        return [cam.get_status() for cam in self.cameras.values() if cam.is_active]
    
    def update_camera_stats(self, camera_id, vehicle_count=None, emergency_detected=None, emergency_type=None):
        """Update camera detection stats"""
        camera = self.cameras.get(camera_id)
        if camera:
            if vehicle_count is not None:
                camera.vehicle_count = vehicle_count
            if emergency_detected is not None:
                camera.emergency_detected = emergency_detected
            if emergency_type is not None:
                camera.emergency_type = emergency_type
    
    def get_emergency_status(self):
        """Get emergency status across all cameras"""
        emergencies = []
        for camera in self.cameras.values():
            if camera.emergency_detected:
                emergencies.append({
                    'camera_id': camera.camera_id,
                    'camera_name': camera.name,
                    'intersection': camera.intersection_name,
                    'emergency_type': camera.emergency_type
                })
        return emergencies
    
    def add_video_source(self, name, video_path, intersection_name=""):
        """Add a video file as a camera source"""
        camera_id = f"video_{int(time.time())}"
        return self.add_camera(
            camera_id=camera_id,
            name=name,
            source=video_path,
            intersection_name=intersection_name
        )
    
    def add_rtsp_source(self, name, rtsp_url, intersection_name=""):
        """Add an RTSP stream as a camera source"""
        camera_id = f"rtsp_{int(time.time())}"
        return self.add_camera(
            camera_id=camera_id,
            name=name,
            source=rtsp_url,
            intersection_name=intersection_name
        )
    
    def add_webcam(self, name, webcam_index, intersection_name=""):
        """Add a webcam as a camera source"""
        camera_id = f"webcam_{webcam_index}"
        return self.add_camera(
            camera_id=camera_id,
            name=name,
            source=webcam_index,
            intersection_name=intersection_name
        )
    
    def get_grid_layout(self, max_cameras=4):
        """Get camera frames for grid display"""
        frames = {}
        count = 0
        for camera_id, camera in self.cameras.items():
            if camera.is_active and camera.last_frame is not None:
                frames[camera_id] = {
                    'frame': camera.last_frame,
                    'name': camera.name,
                    'intersection': camera.intersection_name,
                    'emergency': camera.emergency_detected
                }
                count += 1
                if count >= max_cameras:
                    break
        return frames
    
    def create_mosaic(self, width=1280, height=720):
        """Create a mosaic view of all active cameras"""
        active_cams = [c for c in self.cameras.values() if c.is_active and c.last_frame is not None]
        
        if not active_cams:
            # Return blank frame
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        n = len(active_cams)
        
        if n == 1:
            # Single camera - full size
            return cv2.resize(active_cams[0].last_frame, (width, height))
        elif n == 2:
            # Two cameras - side by side
            h, w = height, width // 2
            frames = [cv2.resize(c.last_frame, (w, h)) for c in active_cams]
            return np.hstack(frames)
        elif n <= 4:
            # 2x2 grid
            h, w = height // 2, width // 2
            frames = [cv2.resize(c.last_frame, (w, h)) for c in active_cams[:4]]
            while len(frames) < 4:
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            top = np.hstack(frames[:2])
            bottom = np.hstack(frames[2:4])
            return np.vstack([top, bottom])
        else:
            # 3x3 grid for more cameras
            h, w = height // 3, width // 3
            frames = [cv2.resize(c.last_frame, (w, h)) for c in active_cams[:9]]
            while len(frames) < 9:
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            rows = [np.hstack(frames[i:i+3]) for i in range(0, 9, 3)]
            return np.vstack(rows)


# Singleton instance
_camera_manager = None

def get_camera_manager():
    """Get or create camera manager instance"""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager
