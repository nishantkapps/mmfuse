"""
Arduino communication interface for robotic arm control
Handles serial communication with Arduino Uno microcontroller
"""

import serial
import serial.tools.list_ports
import time
import threading
from typing import Optional, List, Tuple
import logging


logger = logging.getLogger(__name__)


class ArduinoController:
    """
    Interface for Arduino-based robotic arm control
    
    Protocol:
    - Send command: "MOVE:angle1,angle2,angle3,gripper_force\n"
    - Receive feedback: "READY" or sensor readings
    """
    
    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: int = 9600,
        timeout: float = 0.1,
        write_delay: float = 0.01
    ):
        """
        Initialize Arduino Controller
        
        Args:
            port: Serial port (e.g., "/dev/ttyUSB0", "COM3")
                  If None, auto-detects first available port
            baud_rate: Serial communication speed
            timeout: Read/write timeout in seconds
            write_delay: Delay between consecutive writes
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.write_delay = write_delay
        self.ser = None
        self.connected = False
        self._lock = threading.Lock()
        self.last_write_time = 0
    
    def auto_detect_port(self) -> Optional[str]:
        """
        Auto-detect Arduino COM port
        
        Returns:
            Port name if found, else None
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "Arduino" in port.description or "USB" in port.description:
                logger.info(f"Auto-detected Arduino on {port.device}")
                return port.device
        
        logger.warning("No Arduino detected. Manual port configuration required.")
        return None
    
    def connect(self) -> bool:
        """
        Establish connection to Arduino
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Auto-detect if port not specified
            port_to_use = self.port or self.auto_detect_port()
            
            if not port_to_use:
                logger.error("Cannot connect: no port available")
                return False
            
            self.ser = serial.Serial(
                port=port_to_use,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            
            time.sleep(2)  # Wait for Arduino to initialize
            self.connected = True
            logger.info(f"Connected to Arduino on {port_to_use} @ {self.baud_rate} baud")
            
            return True
        
        except serial.SerialException as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close connection to Arduino"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            logger.info("Disconnected from Arduino")
    
    def send_command(
        self,
        angle1: float,
        angle2: float,
        angle3: float,
        gripper_force: float
    ) -> bool:
        """
        Send motor command to Arduino
        
        Args:
            angle1: Joint 1 angle (degrees)
            angle2: Joint 2 angle (degrees)
            angle3: Joint 3 angle (degrees)
            gripper_force: Gripper force (0-100%)
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.connected or not self.ser:
            logger.warning("Arduino not connected. Command dropped.")
            return False
        
        try:
            with self._lock:
                # Enforce write delay to avoid overwhelming Arduino
                elapsed = time.time() - self.last_write_time
                if elapsed < self.write_delay:
                    time.sleep(self.write_delay - elapsed)
                
                # Format command: "MOVE:angle1,angle2,angle3,gripper\n"
                command = f"MOVE:{angle1:.1f},{angle2:.1f},{angle3:.1f},{gripper_force:.1f}\n"
                self.ser.write(command.encode())
                self.last_write_time = time.time()
                
                return True
        
        except serial.SerialException as e:
            logger.error(f"Failed to send command: {e}")
            self.connected = False
            return False
    
    def read_sensors(self):
        if not self.connected or not self.ser:
            return None

        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="ignore").strip()

                # Log raw serial (optional but useful)
                logger.info(f"[RAW SERIAL] {line}")

                if not line.startswith("SENSORS:"):
                    return None

                # Strip prefix and split
                payload = line.replace("SENSORS:", "")
                parts = payload.split(",")

                if len(parts) != 4:
                    logger.warning(f"Malformed sensor packet: {line}")
                    return None

                # Convert to floats
                pressure = float(parts[0])
                emg_1 = float(parts[1])
                emg_2 = float(parts[2])
                emg_3 = float(parts[3])

                return {
                    "pressure": pressure,
                    "emg_1": emg_1,
                    "emg_2": emg_2,
                    "emg_3": emg_3
                }

        except Exception as e:
            logger.error(f"Serial read error: {e}", exc_info=True)

        return None

    
    def get_status(self) -> str:
        """Get connection status"""
        if self.connected and self.ser and self.ser.is_open:
            return f"CONNECTED on {self.ser.port} @ {self.baud_rate} baud"
        else:
            return "DISCONNECTED"


class SensorBuffer:
    """
    Thread-safe circular buffer for sensor data
    """
    
    def __init__(self, size: int = 1000):
        """
        Initialize Sensor Buffer
        
        Args:
            size: Buffer size (number of samples)
        """
        self.size = size
        self.pressure = []
        self.emg = [[], [], []]
        self._lock = threading.Lock()
    
    def append(self, sensor_data: dict):
        """Add sensor reading to buffer"""
        with self._lock:
            if "pressure" in sensor_data:
                self.pressure.append(sensor_data["pressure"])
                if len(self.pressure) > self.size:
                    self.pressure.pop(0)
            
            for i in range(3):
                emg_key = f"emg_{i+1}"
                if emg_key in sensor_data:
                    self.emg[i].append(sensor_data[emg_key])
                    if len(self.emg[i]) > self.size:
                        self.emg[i].pop(0)
    
    def get_snapshot(self) -> Optional[Tuple]:
        """Get current buffer snapshot as numpy arrays"""
        import numpy as np
        
        with self._lock:
            if len(self.pressure) > 0 and all(len(e) > 0 for e in self.emg):
                return (
                    np.array(self.pressure),
                    np.array(self.emg[0]),
                    np.array(self.emg[1]),
                    np.array(self.emg[2])
                )
        
        return None
