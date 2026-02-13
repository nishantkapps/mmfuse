import serial
import time

PORT = "COM3"        # change this (e.g. "/dev/ttyUSB0" on Linux)
BAUD = 9600        # MUST match Arduino Serial.begin()

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # give Arduino time to reset

    print("Connected. Reading serial...\n")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors="ignore").strip()
                print(line)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
