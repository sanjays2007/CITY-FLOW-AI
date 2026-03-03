"""
Emergency Vehicle Detection Module
Detects ambulances, fire trucks, and police vehicles using:
1. Color detection (red/blue flashing lights)
2. YOLOv8 vehicle classification
3. Audio detection placeholder for sirens
"""

import cv2
import numpy as np
from collections import deque
import time

class EmergencyDetector:
    def __init__(self):
        # Emergency vehicle types
        self.emergency_types = ['ambulance', 'fire truck', 'police', 'emergency']
        
        # Color ranges for emergency lights (HSV)
        # Red color range (fire trucks, ambulances)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Blue color range (police, some ambulances)
        self.blue_lower = np.array([100, 100, 100])
        self.blue_upper = np.array([130, 255, 255])
        
        # Detection history for flashing light detection
        self.light_history = deque(maxlen=30)  # Track last 30 frames
        self.flash_threshold = 5  # Minimum flashes to confirm emergency
        
        # Emergency state
        self.emergency_detected = False
        self.emergency_type = None
        self.emergency_confidence = 0.0
        self.emergency_location = None
        self.last_emergency_time = 0
        self.emergency_cooldown = 5.0  # seconds before resetting
        
        # Priority override state
        self.priority_active = False
        self.priority_direction = None  # 'NS' or 'EW'
        
    def detect_emergency_colors(self, frame):
        """Detect red and blue emergency light colors"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for red colors
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Create mask for blue colors
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        # Count colored pixels
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = red_pixels / total_pixels
        blue_ratio = blue_pixels / total_pixels
        
        return {
            'red_detected': red_ratio > 0.001,  # More than 0.1% red
            'blue_detected': blue_ratio > 0.001,
            'red_ratio': red_ratio,
            'blue_ratio': blue_ratio,
            'red_mask': red_mask,
            'blue_mask': blue_mask
        }
    
    def detect_flashing_lights(self, frame):
        """Detect flashing emergency lights by analyzing color changes"""
        color_result = self.detect_emergency_colors(frame)
        
        # Record current state
        current_state = {
            'red': color_result['red_detected'],
            'blue': color_result['blue_detected'],
            'timestamp': time.time()
        }
        self.light_history.append(current_state)
        
        if len(self.light_history) < 10:
            return False, 0
        
        # Count state changes (flashing pattern)
        red_changes = 0
        blue_changes = 0
        
        for i in range(1, len(self.light_history)):
            if self.light_history[i]['red'] != self.light_history[i-1]['red']:
                red_changes += 1
            if self.light_history[i]['blue'] != self.light_history[i-1]['blue']:
                blue_changes += 1
        
        # Flashing lights change rapidly
        total_changes = red_changes + blue_changes
        is_flashing = total_changes >= self.flash_threshold
        
        return is_flashing, total_changes
    
    def check_vehicle_type(self, detections, class_names):
        """Check if any detected vehicles are emergency types"""
        emergency_vehicles = []
        
        for detection in detections:
            if hasattr(detection, 'class_id'):
                class_id = detection.class_id
            elif isinstance(detection, dict):
                class_id = detection.get('class_id', -1)
            else:
                continue
                
            if class_id < len(class_names):
                vehicle_type = class_names[class_id].lower()
                
                # Check for emergency vehicle keywords
                for emergency_type in self.emergency_types:
                    if emergency_type in vehicle_type:
                        emergency_vehicles.append({
                            'type': vehicle_type,
                            'detection': detection
                        })
                        break
                
                # Also check for trucks (could be fire trucks)
                if 'truck' in vehicle_type:
                    emergency_vehicles.append({
                        'type': 'potential_fire_truck',
                        'detection': detection
                    })
        
        return emergency_vehicles
    
    def analyze_frame(self, frame, detections=None, class_names=None):
        """
        Main analysis function - combines all detection methods
        
        Args:
            frame: Current video frame
            detections: Optional YOLO detections
            class_names: Optional class names for YOLO
            
        Returns:
            dict with emergency detection results
        """
        result = {
            'emergency_detected': False,
            'emergency_type': None,
            'confidence': 0.0,
            'priority_override': False,
            'direction': None,
            'alert_message': None
        }
        
        # Method 1: Color detection
        color_result = self.detect_emergency_colors(frame)
        
        # Method 2: Flashing light detection
        is_flashing, flash_count = self.detect_flashing_lights(frame)
        
        # Method 3: Vehicle type detection (if YOLO detections provided)
        emergency_vehicles = []
        if detections is not None and class_names is not None:
            emergency_vehicles = self.check_vehicle_type(detections, class_names)
        
        # Combine evidence
        confidence = 0.0
        emergency_type = None
        
        # Flashing lights are strong indicator
        if is_flashing:
            confidence += 0.5
            if color_result['red_ratio'] > color_result['blue_ratio']:
                emergency_type = 'Fire Truck / Ambulance'
            else:
                emergency_type = 'Police Vehicle'
        
        # Strong color presence
        if color_result['red_ratio'] > 0.005:  # Significant red
            confidence += 0.2
            emergency_type = emergency_type or 'Ambulance / Fire Truck'
        if color_result['blue_ratio'] > 0.005:  # Significant blue
            confidence += 0.2
            emergency_type = emergency_type or 'Police Vehicle'
        
        # Emergency vehicle detected by YOLO
        if emergency_vehicles:
            confidence += 0.4
            emergency_type = emergency_vehicles[0]['type'].replace('_', ' ').title()
        
        # Determine if emergency is confirmed
        confidence = min(confidence, 1.0)
        
        if confidence >= 0.5:
            self.emergency_detected = True
            self.emergency_type = emergency_type
            self.emergency_confidence = confidence
            self.last_emergency_time = time.time()
            
            result['emergency_detected'] = True
            result['emergency_type'] = emergency_type
            result['confidence'] = confidence
            result['priority_override'] = True
            result['alert_message'] = f"🚨 EMERGENCY: {emergency_type} detected! Activating priority signal."
            
            # Determine direction based on position in frame
            # Assuming camera faces intersection, left/right = EW, top/bottom = NS
            result['direction'] = 'NS'  # Default, can be enhanced with position tracking
            
        else:
            # Check cooldown
            if time.time() - self.last_emergency_time > self.emergency_cooldown:
                self.emergency_detected = False
                self.emergency_type = None
                self.emergency_confidence = 0.0
        
        return result
    
    def draw_emergency_overlay(self, frame, result):
        """Draw emergency detection visualization on frame"""
        if result['emergency_detected']:
            # Red border for emergency
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)
            
            # Emergency banner
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
            
            # Alert text
            text = f"EMERGENCY: {result['emergency_type']}"
            cv2.putText(frame, text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Confidence bar
            conf_width = int(200 * result['confidence'])
            cv2.rectangle(frame, (w-220, 15), (w-20, 45), (100, 100, 100), -1)
            cv2.rectangle(frame, (w-220, 15), (w-220+conf_width, 45), (0, 255, 0), -1)
            cv2.putText(frame, f"{result['confidence']*100:.0f}%", (w-210, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_priority_signal(self, current_phase):
        """
        Get recommended signal change for emergency priority
        
        Args:
            current_phase: Current signal phase ('NS_GREEN' or 'EW_GREEN')
            
        Returns:
            dict with signal recommendation
        """
        if not self.emergency_detected:
            return {'override': False, 'phase': current_phase}
        
        # Priority: Give green to emergency vehicle direction
        # For now, always prioritize NS (can be enhanced with position tracking)
        return {
            'override': True,
            'phase': 'NS_GREEN',
            'min_duration': 30,  # Minimum 30 seconds for emergency
            'reason': f'Emergency vehicle: {self.emergency_type}'
        }


# Singleton instance
_emergency_detector = None

def get_emergency_detector():
    """Get or create emergency detector instance"""
    global _emergency_detector
    if _emergency_detector is None:
        _emergency_detector = EmergencyDetector()
    return _emergency_detector
