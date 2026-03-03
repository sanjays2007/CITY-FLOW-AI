import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

IMAGE_UPDATE_INTERVAL = 1000   # ms
COUNT_UPDATE_INTERVAL = 1000   # ms

class TrafficUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Traffic Management - Live Dashboard")
        self.geometry("900x600")

        # Labels
        self.image_label = tk.Label(self, text="Waiting for image...", bg="black")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_frame = tk.Frame(self)
        self.info_frame.pack(fill="x", padx=10, pady=10)

        self.vehicle_count_var = tk.StringVar(value="Vehicles detected: -")
        self.green_time_var = tk.StringVar(value="Green time: - sec")

        tk.Label(self.info_frame, textvariable=self.vehicle_count_var,
                 font=("Segoe UI", 14)).pack(anchor="w")
        tk.Label(self.info_frame, textvariable=self.green_time_var,
                 font=("Segoe UI", 14)).pack(anchor="w")

        # Start periodic updates
        self.after(IMAGE_UPDATE_INTERVAL, self.update_image)
        self.after(COUNT_UPDATE_INTERVAL, self.update_stats)

    def get_latest_image_path(self):
        files = [f for f in os.listdir(".")
                 if f.startswith("captured_image_") and f.endswith(".jpg")]
        if not files:
            return None
        files.sort()
        return files[-1]

    def update_image(self):
        path = self.get_latest_image_path()
        if path:
            try:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (800, 400))
                    im_pil = Image.fromarray(img)
                    im_tk = ImageTk.PhotoImage(im_pil)
                    self.image_label.configure(image=im_tk, text="")
                    self.image_label.image = im_tk
            except Exception:
                pass
        else:
            self.image_label.configure(text="No captured images yet...")

        self.after(IMAGE_UPDATE_INTERVAL, self.update_image)

    def update_stats(self):
        # read vehicle_count.txt
        if os.path.exists("vehicle_count.txt"):
            try:
                with open("vehicle_count.txt", "r") as f:
                    data = f.read().strip()
                    vehicle_count = int(data) if data else 0
            except Exception:
                vehicle_count = 0
        else:
            vehicle_count = 0

        # same formula as your green_time_signal.py
        base_green_time = 30
        vehicle_multiplier = 2
        green_time = base_green_time + vehicle_count * vehicle_multiplier

        self.vehicle_count_var.set(f"Vehicles detected: {vehicle_count}")
        self.green_time_var.set(f"Green time: {green_time} sec")

        self.after(COUNT_UPDATE_INTERVAL, self.update_stats)

def start_ui():
    app = TrafficUI()
    app.mainloop()

if __name__ == "__main__":
    start_ui()
