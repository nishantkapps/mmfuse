"""
Mock/Stub Arduino Controller for testing without hardware
Simulates sensor readings and motor outputs for pipeline validation
"""

import torch
import numpy as np
import threading
import os
import time
import logging
from typing import Optional, Dict, Tuple
from collections import deque


logger = logging.getLogger(__name__)


class MockArduinoController:
    """
    Simulates Arduino Uno with realistic sensor data
    
    Generates synthetic pressure and EMG signals that respond to
    simulated robot movements for end-to-end testing
    """
    
    def __init__(
        self,
        baud_rate: int = 115200,
        timeout: float = 0.1,
        noise_level: float = 0.1
    ):
        """
        Initialize Mock Arduino Controller
        
        Args:
            baud_rate: Simulated baud rate (unused, for API compatibility)
            timeout: Read/write timeout in seconds
            noise_level: Gaussian noise std dev for sensor simulation (0-1)
        """
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.noise_level = noise_level
        self.connected = True
        
        # Simulated motor state
        self.motor_angles = np.array([0.0, 0.0, 0.0])
        self.gripper_force = 0.0
        
        # Sensor state (influenced by motor position)
        self.pressure_value = 0.0
        self.emg_values = np.array([0.0, 0.0, 0.0])
        
        # Command history for logging
        self.command_history = deque(maxlen=100)
        self.command_count = 0
        
        # Sensor simulation parameters
        self.pressure_offset = np.random.uniform(40, 60)  # Base pressure (ADC 0-1023)
        self.emg_offsets = np.random.uniform(100, 150, 3)  # Base EMG readings
        self.emg_noise = np.random.uniform(5, 15, 3)  # EMG noise amplitude
        
        logger.info("Mock Arduino initialized (simulated mode)")
    
    def connect(self) -> bool:
        """Simulate connection (always succeeds)"""
        self.connected = True
        logger.info("Mock Arduino connected")
        return True
    
    def disconnect(self):
        """Simulate disconnection"""
        self.connected = False
        logger.info("Mock Arduino disconnected")
    
    def send_command(
        self,
        angle1: float,
        angle2: float,
        angle3: float,
        gripper_force: float
    ) -> bool:
        """
        Simulate motor command
        
        Args:
            angle1: Joint 1 angle (degrees)
            angle2: Joint 2 angle (degrees)
            angle3: Joint 3 angle (degrees)
            gripper_force: Gripper force (0-100%)
        
        Returns:
            True if successful
        """
        if not self.connected:
            logger.warning("Mock Arduino not connected")
            return False
        
        try:
            # Update internal motor state
            self.motor_angles = np.array([angle1, angle2, angle3])
            self.gripper_force = gripper_force
            
            # Update sensor values based on motor state
            self._update_sensor_state()
            
            # Log command
            self.command_count += 1
            self.command_history.append({
                'timestamp': time.time(),
                'angles': (angle1, angle2, angle3),
                'gripper': gripper_force
            })
            
            return True
        
        except Exception as e:
            logger.error(f"Mock command error: {e}")
            return False
    
    def _update_sensor_state(self):
        """
        Simulate sensor readings based on current motor state
        Pressure and EMG respond to arm configuration
        """
        # Pressure increases with gripper force
        pressure_from_gripper = self.gripper_force * 0.5
        
        # Pressure also responds to vertical position (motor 2)
        pressure_from_position = abs(self.motor_angles[1]) * 0.3
        
        self.pressure_value = (
            self.pressure_offset +
            pressure_from_gripper +
            pressure_from_position +
            np.random.normal(0, self.noise_level * 5)
        )
        self.pressure_value = np.clip(self.pressure_value, 0, 1023)
        
        # EMG responses to each motor (activity simulated)
        for i in range(3):
            motor_activity = abs(self.motor_angles[i]) / 90.0  # Normalized 0-1
            emg_value = (
                self.emg_offsets[i] +
                motor_activity * 100 +  # Activity component
                np.sin(time.time() * 5 + i) * self.emg_noise[i] +  # Oscillation
                np.random.normal(0, self.noise_level * 10)  # Noise
            )
            self.emg_values[i] = np.clip(emg_value, 0, 1023)
    
    def read_sensors(self) -> Optional[dict]:
        """
        Read simulated sensor data
        
        Returns:
            Dict with pressure and EMG values, or None on error
        """
        if not self.connected:
            return None
        
        try:
            return {
                "pressure": float(self.pressure_value),
                "emg_1": float(self.emg_values[0]),
                "emg_2": float(self.emg_values[1]),
                "emg_3": float(self.emg_values[2])
            }
        except Exception as e:
            logger.debug(f"Mock sensor read error: {e}")
            return None
    
    def get_status(self) -> str:
        """Get connection status"""
        if self.connected:
            return f"MOCK Arduino (simulated) | Commands: {self.command_count}"
        else:
            return "MOCK Arduino (DISCONNECTED)"
    
    def get_simulation_state(self) -> Dict:
        """Get full simulation state for debugging"""
        return {
            'motor_angles': self.motor_angles.tolist(),
            'gripper_force': float(self.gripper_force),
            'pressure': float(self.pressure_value),
            'emg': self.emg_values.tolist(),
            'commands_sent': self.command_count
        }


def create_arduino_controller(mock_mode: bool = None, **kwargs) -> any:
    """
    Factory function to create either real or mock Arduino controller.

    Behavior:
    - If `mock_mode` is explicitly provided (True/False), respects it.
    - If `mock_mode` is None, attempts to read `config/streaming_config.yaml` and
      determine mode from `arduino.mode` ("mock" or "real"). Falls back to mock.

    **kwargs are forwarded to the chosen controller constructor.
    """
    # Lazy import of config loader to avoid circular imports
    if mock_mode is None:
        try:
            # Load YAML directly to avoid import name collisions with module 'config.py'
            import yaml
            repo_root = os.path.dirname(os.path.dirname(__file__))
            cfg_path = os.path.join(repo_root, 'config', 'streaming_config.yaml')
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f)
            else:
                cfg = {}

            mode = cfg.get('arduino', {}).get('mode', 'mock')
            mock_mode = True if str(mode).lower() == 'mock' else False
            # Merge default args from config if not provided explicitly
            ar_cfg = cfg.get('arduino', {})
            if 'port' in ar_cfg and kwargs.get('port', None) is None:
                kwargs['port'] = ar_cfg.get('port')
            if 'baud_rate' in ar_cfg and kwargs.get('baud_rate', None) is None:
                kwargs['baud_rate'] = ar_cfg.get('baud_rate')
            if 'timeout' in ar_cfg and kwargs.get('timeout', None) is None:
                kwargs['timeout'] = ar_cfg.get('timeout')
            if 'write_delay' in ar_cfg and kwargs.get('write_delay', None) is None:
                kwargs['write_delay'] = ar_cfg.get('write_delay')
        except Exception:
            logger.warning("Could not load streaming config; defaulting to mock Arduino")
            mock_mode = True

    if mock_mode:
        logger.info("Using MOCK Arduino controller (no hardware required)")
        # Only pass parameters that MockArduinoController accepts
        allowed_keys = ['baud_rate', 'timeout', 'noise_level']
        mock_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        return MockArduinoController(**mock_kwargs)
    else:
        from io.arduino_controller import ArduinoController
        logger.info("Using REAL Arduino controller (hardware required)")
        # Only pass parameters that ArduinoController accepts
        allowed_keys = ['port', 'baud_rate', 'timeout', 'write_delay']
        real_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        return ArduinoController(**real_kwargs)
