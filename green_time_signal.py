import time
import os

BASE_GREEN_TIME = 30      # seconds
VEHICLE_MULTIPLIER = 2    # +2s per vehicle
READ_INTERVAL = 3         # check every 3 seconds

def adjust_green_signal_time(vehicle_count):
    green_time = BASE_GREEN_TIME + (vehicle_count * VEHICLE_MULTIPLIER)
    return green_time

def main():
    last_count = None

    while True:
        if not os.path.exists("vehicle_count.txt"):
            print("vehicle_count.txt not found. Waiting for detection...")
            time.sleep(READ_INTERVAL)
            continue

        try:
            with open("vehicle_count.txt", "r") as file:
                data = file.read().strip()
                if not data:
                    print("Empty vehicle_count.txt. Waiting...")
                    time.sleep(READ_INTERVAL)
                    continue
                vehicle_count = int(data)

            if vehicle_count < 0:
                print("Invalid vehicle count in the file. Count must be non‑negative.")
                time.sleep(READ_INTERVAL)
                continue
        except Exception as e:
            print(f"Error reading vehicle count from file: {e}")
            time.sleep(READ_INTERVAL)
            continue

        # Only print when value changes
        if vehicle_count != last_count:
            last_count = vehicle_count
            green_time = adjust_green_signal_time(vehicle_count)
            print("\n======================")
            print(f"🚗 Vehicles detected      : {vehicle_count}")
            print(f"✅ Green signal time (sec): {green_time}")
            print("======================")

            # place to connect to real signal / UI

        time.sleep(READ_INTERVAL)

if __name__ == "__main__":
    main()
