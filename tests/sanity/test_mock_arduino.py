"""
Unit tests for Mock Arduino Controller
Validates sensor simulation and command handling
"""

import sys
import os

# Ensure proper path resolution to avoid 'io' module conflict
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_root)

# Import from the io subdirectory directly
io_path = os.path.join(repo_root, 'io')
sys.path.insert(0, io_path)

import logging
import time
import numpy as np
from mock_arduino_controller import MockArduinoController, create_arduino_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mock_arduino_basic():
    """Test basic mock Arduino functionality"""
    
    logger.info("=" * 70)
    logger.info("TEST 1: Mock Arduino Basic Operations")
    logger.info("=" * 70)
    
    # Create mock Arduino
    arduino = MockArduinoController()
    
    # Test connection
    assert arduino.connect(), "Failed to connect"
    assert arduino.connected, "Not marked as connected"
    logger.info("✓ Connection successful\n")
    
    # Test command sending
    success = arduino.send_command(
        angle1=45.0,
        angle2=30.0,
        angle3=60.0,
        gripper_force=50.0
    )
    assert success, "Failed to send command"
    logger.info("✓ Command sent successfully\n")
    
    # Test sensor reading
    sensors = arduino.read_sensors()
    assert sensors is not None, "Failed to read sensors"
    assert 'pressure' in sensors, "Missing pressure data"
    assert 'emg_1' in sensors, "Missing EMG_1 data"
    assert 'emg_2' in sensors, "Missing EMG_2 data"
    assert 'emg_3' in sensors, "Missing EMG_3 data"
    logger.info(f"✓ Sensors read successfully")
    logger.info(f"  Pressure: {sensors['pressure']:.1f}")
    logger.info(f"  EMG: [{sensors['emg_1']:.1f}, {sensors['emg_2']:.1f}, {sensors['emg_3']:.1f}]\n")
    
    # Test status
    status = arduino.get_status()
    assert "MOCK Arduino" in status, "Invalid status string"
    logger.info(f"✓ Status: {status}\n")
    
    # Test disconnection
    arduino.disconnect()
    assert not arduino.connected, "Still marked as connected"
    logger.info("✓ Disconnection successful\n")


def test_mock_arduino_sensor_response():
    """Test that sensors respond to motor commands"""
    
    logger.info("=" * 70)
    logger.info("TEST 2: Sensor Response to Motor Commands")
    logger.info("=" * 70)
    
    arduino = MockArduinoController(noise_level=0.01)  # Low noise for testing
    arduino.connect()
    
    # Record sensors with no motor movement
    arduino.send_command(0, 0, 0, 0)
    baseline = arduino.read_sensors()
    baseline_pressure = baseline['pressure']
    logger.info(f"Baseline pressure: {baseline_pressure:.1f}\n")
    
    # Move gripper, pressure should increase
    arduino.send_command(0, 0, 0, 100)  # Full gripper force
    time.sleep(0.01)
    with_gripper = arduino.read_sensors()
    with_gripper_pressure = with_gripper['pressure']
    logger.info(f"Pressure with gripper: {with_gripper_pressure:.1f}")
    assert with_gripper_pressure > baseline_pressure, "Pressure didn't respond to gripper"
    logger.info("✓ Pressure responds to gripper force\n")
    
    # Move arm vertically
    arduino.send_command(0, 45, 0, 0)
    time.sleep(0.01)
    with_position = arduino.read_sensors()
    with_position_pressure = with_position['pressure']
    logger.info(f"Pressure with arm position: {with_position_pressure:.1f}")
    logger.info("✓ Pressure responds to arm position\n")
    
    # Check EMG responses to motor activity
    baseline_emg = np.array([baseline['emg_1'], baseline['emg_2'], baseline['emg_3']])
    with_move_emg = np.array([with_position['emg_1'], with_position['emg_2'], with_position['emg_3']])
    
    logger.info(f"Baseline EMG: {baseline_emg}")
    logger.info(f"With movement EMG: {with_move_emg}")
    
    # Motor 2 moved, so EMG_2 should show more activity
    assert with_move_emg[1] > baseline_emg[1], "EMG didn't respond to motor movement"
    logger.info("✓ EMG responds to motor activity\n")


def test_mock_arduino_command_history():
    """Test command history tracking"""
    
    logger.info("=" * 70)
    logger.info("TEST 3: Command History")
    logger.info("=" * 70)
    
    arduino = MockArduinoController()
    arduino.connect()
    
    # Send multiple commands
    for i in range(5):
        arduino.send_command(i*10, i*5, i*15, i*20)
    
    assert arduino.command_count == 5, f"Expected 5 commands, got {arduino.command_count}"
    logger.info(f"✓ Tracked {arduino.command_count} commands\n")
    
    # Check history
    history = list(arduino.command_history)
    assert len(history) == 5, "History size mismatch"
    logger.info("Command history:")
    for idx, cmd in enumerate(history):
        logger.info(f"  {idx+1}: angles={cmd['angles']}, gripper={cmd['gripper']:.1f}")
    logger.info("")


def test_factory_function():
    """Test factory function for creating controllers"""
    
    logger.info("=" * 70)
    logger.info("TEST 4: Factory Function")
    logger.info("=" * 70)
    
    # Test mock mode
    mock_arduino = create_arduino_controller(mock_mode=True)
    assert isinstance(mock_arduino, MockArduinoController), "Wrong type returned"
    logger.info("✓ Factory creates mock controller\n")
    
    # Test that it works
    mock_arduino.connect()
    mock_arduino.send_command(10, 20, 30, 50)
    sensors = mock_arduino.read_sensors()
    assert sensors is not None, "Mock controller not working"
    logger.info("✓ Mock controller functional\n")


def test_simulation_state():
    """Test getting full simulation state"""
    
    logger.info("=" * 70)
    logger.info("TEST 5: Simulation State Export")
    logger.info("=" * 70)
    
    arduino = MockArduinoController()
    arduino.connect()
    
    arduino.send_command(45, 30, 60, 75)
    state = arduino.get_simulation_state()
    
    assert 'motor_angles' in state, "Missing motor angles"
    assert 'gripper_force' in state, "Missing gripper force"
    assert 'pressure' in state, "Missing pressure"
    assert 'emg' in state, "Missing EMG"
    assert 'commands_sent' in state, "Missing command count"
    
    logger.info("Simulation state:")
    logger.info(f"  Motor angles: {state['motor_angles']}")
    logger.info(f"  Gripper force: {state['gripper_force']:.1f}%")
    logger.info(f"  Pressure: {state['pressure']:.1f}")
    logger.info(f"  EMG: {[f'{e:.1f}' for e in state['emg']]}")
    logger.info(f"  Commands sent: {state['commands_sent']}\n")


def main():
    """Run all tests"""
    
    logger.info("\n")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 15 + "MOCK ARDUINO CONTROLLER UNIT TESTS" + " " * 19 + "║")
    logger.info("╚" + "═" * 68 + "╝\n")
    
    tests = [
        test_mock_arduino_basic,
        test_mock_arduino_sensor_response,
        test_mock_arduino_command_history,
        test_factory_function,
        test_simulation_state
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            logger.error(f"✗ Test failed: {e}\n")
            failed += 1
        except Exception as e:
            logger.error(f"✗ Test error: {e}\n")
            failed += 1
    
    logger.info("=" * 70)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
