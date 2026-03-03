"""
Signal Cycle Controller
-----------------------
Implements a batch detection → signal phase → repeat cycle:
1. DETECTION PHASE: Run vehicle detection until threshold reached (e.g., 30 vehicles or 60 seconds)
2. SIGNAL PHASE: Calculate and display signal times based on detected vehicles
3. Repeat the cycle

Author: Smart Traffic Management System
"""

import time
import threading
import subprocess
import sys
import os
import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Callable


class CyclePhase(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    RED_SIGNAL = "red_signal"
    GREEN_SIGNAL = "green_signal"
    YELLOW_SIGNAL = "yellow_signal"


@dataclass
class SignalState:
    phase: str
    direction: str  # "NS" or "EW"
    remaining_time: int
    total_time: int
    vehicle_count: int
    cycle_count: int
    message: str


class SignalCycleController:
    """
    Controls the traffic signal cycle with batch detection.
    """
    
    def __init__(self):
        # Detection thresholds
        self.vehicle_threshold = 30  # Stop detection after this many vehicles
        self.max_detection_time = 60  # Or stop after this many seconds
        
        # Signal timing parameters
        self.base_green_time = 20  # Base green signal time (seconds)
        self.per_vehicle_time = 2  # Additional seconds per vehicle
        self.min_green_time = 15   # Minimum green time
        self.max_green_time = 90   # Maximum green time
        self.yellow_time = 3       # Yellow signal duration
        self.min_red_time = 10     # Minimum red time for other direction
        
        # Current state
        self.current_phase = CyclePhase.IDLE
        self.current_direction = "NS"  # Start with North-South
        self.vehicle_count = 0
        self.cycle_count = 0
        self.phase_start_time = 0
        self.phase_duration = 0
        
        # Threading
        self._running = False
        self._cycle_thread: Optional[threading.Thread] = None
        self._detection_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self.on_state_change: Optional[Callable[[SignalState], None]] = None
        
        # Vehicle count file paths (NS and EW)
        self.count_file_ns = "vehicle_count_ns.txt"
        self.count_file_ew = "vehicle_count_ew.txt"
        
        # Video source (can be set before starting)
        self.video_source = "0"  # Default to webcam
        self.video_source_type = "webcam"
        
    def get_state(self) -> SignalState:
        """Get current signal state."""
        with self._lock:
            remaining = 0
            if self.phase_start_time > 0 and self.phase_duration > 0:
                elapsed = time.time() - self.phase_start_time
                remaining = max(0, int(self.phase_duration - elapsed))
            
            messages = {
                CyclePhase.IDLE: "System idle - Click Start to begin",
                CyclePhase.DETECTING: f"Detecting vehicles... ({self.vehicle_count}/{self.vehicle_threshold})",
                CyclePhase.RED_SIGNAL: f"🔴 RED - {self.current_direction} traffic STOP",
                CyclePhase.GREEN_SIGNAL: f"🟢 GREEN - {self.current_direction} traffic GO",
                CyclePhase.YELLOW_SIGNAL: f"🟡 YELLOW - Prepare to stop",
            }
            
            return SignalState(
                phase=self.current_phase.value,
                direction=self.current_direction,
                remaining_time=remaining,
                total_time=int(self.phase_duration),
                vehicle_count=self.vehicle_count,
                cycle_count=self.cycle_count,
                message=messages.get(self.current_phase, "")
            )
    
    def calculate_green_time(self, vehicle_count: int) -> int:
        """Calculate green signal time based on vehicle count."""
        green_time = self.base_green_time + (vehicle_count * self.per_vehicle_time)
        return int(max(self.min_green_time, min(green_time, self.max_green_time)))
    
    def calculate_red_time(self, vehicle_count: int) -> int:
        """Calculate red signal time (for opposite direction)."""
        # Red time is based on assumed waiting vehicles on other side
        # Use a proportion of detected vehicles
        other_side_estimate = max(5, vehicle_count // 2)
        red_time = self.base_green_time + (other_side_estimate * self.per_vehicle_time)
        return int(max(self.min_red_time, min(red_time, self.max_green_time // 2)))
    
    def read_vehicle_count(self) -> int:
        """Read current vehicle count from NS and EW files."""
        total = 0
        try:
            if os.path.exists(self.count_file_ns):
                with open(self.count_file_ns, "r") as f:
                    data = f.read().strip()
                    if data:
                        total += int(data)
        except Exception as e:
            print(f"Error reading NS count: {e}")
        
        try:
            if os.path.exists(self.count_file_ew):
                with open(self.count_file_ew, "r") as f:
                    data = f.read().strip()
                    if data:
                        total += int(data)
        except Exception as e:
            print(f"Error reading EW count: {e}")
        
        return total
    
    def start_detection(self):
        """Start the vehicle detection process."""
        try:
            # Stop any existing process
            self.stop_detection()
            
            # Build command with configurable source (use same Python as current process)
            cmd = [
                sys.executable, 'video_counter.py',
                '--source', str(self.video_source),
                '--no-preview'
            ]
            
            # Add skip frames for video files
            if self.video_source_type == 'video':
                cmd.extend(['--skip', '2'])
            
            print(f"Starting detection with source: {self.video_source}")
            
            self._detection_process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            print("Detection process started")
        except Exception as e:
            print(f"Error starting detection: {e}")
    
    def set_video_source(self, source: str, source_type: str = "webcam"):
        """Set the video source for detection."""
        self.video_source = source
        self.video_source_type = source_type
        print(f"Video source set to: {source} (type: {source_type})")
    
    def stop_detection(self):
        """Stop the vehicle detection process."""
        if self._detection_process is not None:
            try:
                self._detection_process.terminate()
                self._detection_process.wait(timeout=5)
            except:
                try:
                    self._detection_process.kill()
                except:
                    pass
            self._detection_process = None
            print("Detection process stopped")
    
    def _notify_state_change(self):
        """Notify listeners of state change."""
        if self.on_state_change:
            try:
                self.on_state_change(self.get_state())
            except:
                pass
    
    def _run_cycle(self):
        """Main cycle loop - runs in a separate thread."""
        while self._running:
            try:
                # =====================================
                # PHASE 1: DETECTION
                # =====================================
                with self._lock:
                    self.current_phase = CyclePhase.DETECTING
                    self.vehicle_count = 0
                    self.phase_start_time = time.time()
                    self.phase_duration = self.max_detection_time
                
                self._notify_state_change()
                
                # Reset vehicle count files
                with open(self.count_file_ns, "w") as f:
                    f.write("0")
                with open(self.count_file_ew, "w") as f:
                    f.write("0")
                
                # Start detection
                self.start_detection()
                
                # Monitor detection until threshold or timeout
                detection_start = time.time()
                while self._running:
                    current_count = self.read_vehicle_count()
                    
                    with self._lock:
                        self.vehicle_count = current_count
                    
                    self._notify_state_change()
                    
                    # Check if threshold reached
                    if current_count >= self.vehicle_threshold:
                        print(f"Vehicle threshold reached: {current_count}")
                        break
                    
                    # Check if timeout reached
                    if time.time() - detection_start >= self.max_detection_time:
                        print(f"Detection timeout reached with {current_count} vehicles")
                        break
                    
                    time.sleep(1)
                
                if not self._running:
                    break
                
                # Stop detection
                self.stop_detection()
                final_count = self.vehicle_count
                
                # =====================================
                # PHASE 2: RED SIGNAL (for current direction)
                # =====================================
                red_time = self.calculate_red_time(final_count)
                
                with self._lock:
                    self.current_phase = CyclePhase.RED_SIGNAL
                    self.phase_start_time = time.time()
                    self.phase_duration = red_time
                
                self._notify_state_change()
                print(f"RED signal for {self.current_direction}: {red_time}s (vehicles: {final_count})")
                
                # Wait for red signal duration
                red_start = time.time()
                while self._running and (time.time() - red_start) < red_time:
                    self._notify_state_change()
                    time.sleep(0.5)
                
                if not self._running:
                    break
                
                # =====================================
                # PHASE 3: GREEN SIGNAL
                # =====================================
                green_time = self.calculate_green_time(final_count)
                
                with self._lock:
                    self.current_phase = CyclePhase.GREEN_SIGNAL
                    self.phase_start_time = time.time()
                    self.phase_duration = green_time
                
                self._notify_state_change()
                print(f"GREEN signal for {self.current_direction}: {green_time}s")
                
                # Wait for green signal duration
                green_start = time.time()
                while self._running and (time.time() - green_start) < green_time:
                    self._notify_state_change()
                    time.sleep(0.5)
                
                if not self._running:
                    break
                
                # =====================================
                # PHASE 4: YELLOW SIGNAL
                # =====================================
                with self._lock:
                    self.current_phase = CyclePhase.YELLOW_SIGNAL
                    self.phase_start_time = time.time()
                    self.phase_duration = self.yellow_time
                
                self._notify_state_change()
                print(f"YELLOW signal: {self.yellow_time}s")
                
                time.sleep(self.yellow_time)
                
                if not self._running:
                    break
                
                # =====================================
                # Switch direction and increment cycle
                # =====================================
                with self._lock:
                    self.current_direction = "EW" if self.current_direction == "NS" else "NS"
                    self.cycle_count += 1
                
                print(f"Cycle {self.cycle_count} complete. Switching to {self.current_direction}")
                
            except Exception as e:
                print(f"Error in cycle: {e}")
                time.sleep(1)
        
        # Cleanup
        self.stop_detection()
        with self._lock:
            self.current_phase = CyclePhase.IDLE
        self._notify_state_change()
    
    def start(self):
        """Start the signal cycle."""
        if self._running:
            return
        
        self._running = True
        self._cycle_thread = threading.Thread(target=self._run_cycle, daemon=True)
        self._cycle_thread.start()
        print("Signal cycle started")
    
    def stop(self):
        """Stop the signal cycle."""
        self._running = False
        self.stop_detection()
        
        if self._cycle_thread:
            self._cycle_thread.join(timeout=5)
            self._cycle_thread = None
        
        with self._lock:
            self.current_phase = CyclePhase.IDLE
        
        self._notify_state_change()
        print("Signal cycle stopped")
    
    def is_running(self) -> bool:
        """Check if cycle is running."""
        return self._running
    
    def set_parameters(self, vehicle_threshold: int = None, max_detection_time: int = None,
                       base_green_time: int = None, per_vehicle_time: int = None):
        """Update cycle parameters."""
        with self._lock:
            if vehicle_threshold is not None:
                self.vehicle_threshold = vehicle_threshold
            if max_detection_time is not None:
                self.max_detection_time = max_detection_time
            if base_green_time is not None:
                self.base_green_time = base_green_time
            if per_vehicle_time is not None:
                self.per_vehicle_time = per_vehicle_time


# Singleton instance
_signal_controller: Optional[SignalCycleController] = None


def get_signal_controller() -> SignalCycleController:
    """Get the singleton signal controller instance."""
    global _signal_controller
    if _signal_controller is None:
        _signal_controller = SignalCycleController()
    return _signal_controller


if __name__ == "__main__":
    # Test the controller
    controller = get_signal_controller()
    
    def on_state(state: SignalState):
        print(f"[{state.phase}] {state.message} - {state.remaining_time}s remaining")
    
    controller.on_state_change = on_state
    controller.vehicle_threshold = 10  # Lower for testing
    controller.max_detection_time = 30  # Shorter for testing
    
    print("Starting signal cycle controller...")
    controller.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        controller.stop()
