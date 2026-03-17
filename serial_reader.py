import serial
import time
import json
import argparse

PORT = "COM3"
BAUD = 9600

def main(filename):
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    print(f"Connected. Saving data to {filename}\n")

    start = time.time()
    duration = 120
    data = []

    try:
        while time.time() - start < duration:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors="ignore").strip()
                line = int(line)
                if line < 300:
                    pressure_feel = "very light"
                elif line >= 300 and line < 600:
                    pressure_feel = "light"
                elif line >= 600 and line < 800:
                    pressure_feel = "medium"
                elif line >= 800 and line < 900:
                    pressure_feel = "high"
                else:
                    pressure_feel = "very high"
                
                data.append({
                    "timestamp": time.time() - start,
                    "value": line,
                    "category": pressure_feel
                })

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        ser.close()

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Data saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serial data logger")
    parser.add_argument("filename", help="Output JSON file name")

    args = parser.parse_args()
    main(args.filename)