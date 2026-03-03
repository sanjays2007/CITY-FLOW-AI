import time
import csv
import os
from dataclasses import dataclass
from enum import Enum
import datetime as dt

from config import (
    BASE_GREEN_NS, BASE_GREEN_EW, PER_VEHICLE,
    MIN_GREEN, MAX_GREEN,
    YELLOW_TIME, ALL_RED_TIME,
    PEAK_MORNING, PEAK_EVENING,
    PEAK_MULTIPLIER_NS, PEAK_MULTIPLIER_EW,
    NIGHT_MULTIPLIER
)

class Phase(Enum):
    NS_GREEN = "NS_GREEN"
    NS_YELLOW = "NS_YELLOW"
    ALL_RED = "ALL_RED"
    EW_GREEN = "EW_GREEN"
    EW_YELLOW = "EW_YELLOW"

@dataclass
class PhaseState:
    name: Phase
    start_time: float
    duration: int

class TrafficController:
    def __init__(self, log_path="data/logs/cycles.csv"):
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "phase",
                    "vehicle_count_ns",
                    "vehicle_count_ew",
                    "green_time",
                    "load_category"
                ])

        now = time.time()
        self.current_phase = PhaseState(
            name=Phase.NS_GREEN,
            start_time=now,
            duration=BASE_GREEN_NS
        )
        self.last_green_duration = BASE_GREEN_NS

    def _time_of_day_multiplier(self, phase: Phase):
        hour = dt.datetime.now().hour

        # Night profile
        if hour >= 22 or hour < 6:
            return NIGHT_MULTIPLIER

        # Peak profiles
        in_morning_peak = PEAK_MORNING[0] <= hour < PEAK_MORNING[1]
        in_evening_peak = PEAK_EVENING[0] <= hour < PEAK_EVENING[1]

        if in_morning_peak or in_evening_peak:
            if phase == Phase.NS_GREEN:
                return PEAK_MULTIPLIER_NS
            elif phase == Phase.EW_GREEN:
                return PEAK_MULTIPLIER_EW

        return 1.0

    def compute_green_time(self, phase: Phase, count_ns: int, count_ew: int):
        if phase == Phase.NS_GREEN:
            base = BASE_GREEN_NS
            vehicles = count_ns
        elif phase == Phase.EW_GREEN:
            base = BASE_GREEN_EW
            vehicles = count_ew
        else:
            return 0

        green = base + PER_VEHICLE * vehicles
        green *= self._time_of_day_multiplier(phase)
        green = int(max(MIN_GREEN, min(green, MAX_GREEN)))
        return green

    def categorize_load(self, total_count):
        if total_count == 0:
            return "FREE"
        elif total_count <= 5:
            return "LIGHT"
        elif total_count <= 15:
            return "MODERATE"
        else:
            return "HEAVY"

    def log_cycle(self, phase: Phase, count_ns: int, count_ew: int, green_time: int):
        if phase not in (Phase.NS_GREEN, Phase.EW_GREEN):
            return  # log only full green phases

        total = count_ns + count_ew
        load = self.categorize_load(total)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, phase.value, count_ns, count_ew, green_time, load])

    def update_phase(self, count_ns: int, count_ew: int):
        """
        Call regularly. Returns (phase_name_for_ui, remaining_time, green_time, load_category).
        phase_name_for_ui: "NS", "EW", or "YELLOW/ALL_RED".
        """
        now = time.time()
        elapsed = now - self.current_phase.start_time

        # compute remaining first
        if elapsed < self.current_phase.duration:
            remaining = int(self.current_phase.duration - elapsed)
        else:
            # phase ended -> transition
            phase = self.current_phase.name

            if phase == Phase.NS_GREEN:
                self.log_cycle(phase, count_ns, count_ew, self.current_phase.duration)
                self.current_phase = PhaseState(
                    name=Phase.NS_YELLOW,
                    start_time=now,
                    duration=YELLOW_TIME
                )
            elif phase == Phase.NS_YELLOW:
                self.current_phase = PhaseState(
                    name=Phase.ALL_RED,
                    start_time=now,
                    duration=ALL_RED_TIME
                )
            elif phase == Phase.ALL_RED:
                # choose next green based on queues
                if count_ns >= count_ew:
                    next_phase = Phase.NS_GREEN
                else:
                    next_phase = Phase.EW_GREEN
                green = self.compute_green_time(next_phase, count_ns, count_ew)
                self.current_phase = PhaseState(
                    name=next_phase,
                    start_time=now,
                    duration=green
                )
            elif phase == Phase.EW_GREEN:
                self.log_cycle(phase, count_ns, count_ew, self.current_phase.duration)
                self.current_phase = PhaseState(
                    name=Phase.EW_YELLOW,
                    start_time=now,
                    duration=YELLOW_TIME
                )
            elif phase == Phase.EW_YELLOW:
                self.current_phase = PhaseState(
                    name=Phase.ALL_RED,
                    start_time=now,
                    duration=ALL_RED_TIME
                )

            now = time.time()
            elapsed = now - self.current_phase.start_time
            remaining = int(self.current_phase.duration - elapsed)

        phase_name = self.current_phase.name
        if phase_name == Phase.NS_GREEN:
            ui_phase = "NS"
        elif phase_name == Phase.EW_GREEN:
            ui_phase = "EW"
        elif phase_name in (Phase.NS_YELLOW, Phase.EW_YELLOW):
            ui_phase = "YELLOW"
        else:
            ui_phase = "ALL_RED"

        total = count_ns + count_ew
        load = self.categorize_load(total)

        return ui_phase, remaining, self.current_phase.duration, load
