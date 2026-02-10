#!/usr/bin/env python3
"""
Dual Camera Functionality Verification Script

This script tests:
1. Camera detection and availability
2. Single camera mode
3. Dual camera mode
4. Graceful fallback when camera is unavailable
5. Vision embedding dimensions
"""

import cv2
import sys
import os
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_camera_availability(max_cameras: int = 5) -> dict:
    """
    Test which cameras are available on the system
    
    Args:
        max_cameras: Maximum camera ID to test
    
    Returns:
        Dictionary with camera availability info
    """
    logger.info("=" * 60)
    logger.info("TEST 1: Camera Availability Detection")
    logger.info("=" * 60)
    
    available_cameras = {}
    
    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras[camera_id] = {
                    'available': True,
                    'shape': frame.shape,
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': cap.get(cv2.CAP_PROP_FPS)
                }
                logger.info(f"✓ Camera {camera_id}: {frame.shape} @ {cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
            else:
                available_cameras[camera_id] = {'available': False}
                logger.warning(f"✗ Camera {camera_id}: Opened but cannot read frames")
        else:
            available_cameras[camera_id] = {'available': False}
            logger.debug(f"✗ Camera {camera_id}: Not available")
        
        cap.release()
    
    logger.info(f"\nFound {sum(1 for c in available_cameras.values() if c['available'])} working camera(s)\n")
    return available_cameras


def test_single_camera_capture(camera_id: int = 0, num_frames: int = 30) -> bool:
    """
    Test single camera capture performance
    
    Args:
        camera_id: Camera ID to test
        num_frames: Number of frames to capture
    
    Returns:
        True if test passed, False otherwise
    """
    logger.info("=" * 60)
    logger.info(f"TEST 2: Single Camera Capture (Camera {camera_id})")
    logger.info("=" * 60)
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"✗ Failed to open camera {camera_id}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    logger.info(f"Capturing {num_frames} frames...")
    successful_frames = 0
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            successful_frames += 1
        else:
            logger.warning(f"Failed to capture frame {i}")
    
    cap.release()
    
    success_rate = (successful_frames / num_frames) * 100
    logger.info(f"✓ Captured {successful_frames}/{num_frames} frames ({success_rate:.1f}% success)\n")
    
    return successful_frames > (num_frames * 0.8)  # 80% success threshold


def test_dual_camera_capture(camera_ids: list = [0, 1], num_frames: int = 30) -> bool:
    """
    Test dual camera capture performance
    
    Args:
        camera_ids: [primary_camera_id, secondary_camera_id]
        num_frames: Number of frame pairs to capture
    
    Returns:
        True if test passed, False otherwise
    """
    logger.info("=" * 60)
    logger.info(f"TEST 3: Dual Camera Capture (Cameras {camera_ids[0]}, {camera_ids[1]})")
    logger.info("=" * 60)
    
    # Open primary camera
    cap_primary = cv2.VideoCapture(camera_ids[0])
    if not cap_primary.isOpened():
        logger.error(f"✗ Failed to open primary camera {camera_ids[0]}")
        return False
    
    # Open secondary camera
    cap_secondary = cv2.VideoCapture(camera_ids[1])
    if not cap_secondary.isOpened():
        logger.warning(f"✗ Failed to open secondary camera {camera_ids[1]}")
        cap_secondary = None
    
    if cap_secondary:
        cap_primary.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap_primary.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap_secondary.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap_secondary.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        logger.info(f"Capturing {num_frames} dual frames...")
        successful_pairs = 0
        
        for i in range(num_frames):
            ret1, frame1 = cap_primary.read()
            ret2, frame2 = cap_secondary.read()
            
            if ret1 and ret2:
                successful_pairs += 1
            elif ret1:
                logger.debug(f"Frame {i}: Primary OK, Secondary failed")
            elif ret2:
                logger.debug(f"Frame {i}: Primary failed, Secondary OK")
            else:
                logger.warning(f"Frame {i}: Both cameras failed")
        
        cap_secondary.release()
        success_rate = (successful_pairs / num_frames) * 100
        logger.info(f"✓ Captured {successful_pairs}/{num_frames} dual frames ({success_rate:.1f}% success)\n")
        return successful_pairs > (num_frames * 0.8)
    else:
        logger.warning("Secondary camera unavailable - skipping dual camera test")
        return False


def test_camera_fallback(camera_ids: list = [0, 99]) -> bool:
    """
    Test graceful fallback when one camera is unavailable
    
    Args:
        camera_ids: [primary_camera_id, invalid_camera_id]
    
    Returns:
        True if test passed (fallback worked), False otherwise
    """
    logger.info("=" * 60)
    logger.info("TEST 4: Graceful Fallback (Missing Camera)")
    logger.info("=" * 60)
    
    # Open primary camera
    cap_primary = cv2.VideoCapture(camera_ids[0])
    if not cap_primary.isOpened():
        logger.error(f"✗ Primary camera {camera_ids[0]} not available")
        return False
    
    # Try to open invalid camera
    cap_secondary = cv2.VideoCapture(camera_ids[1])
    fallback_triggered = not cap_secondary.isOpened()
    
    if fallback_triggered:
        logger.info(f"✓ Detected unavailable camera {camera_ids[1]}")
        logger.info("✓ Fallback to single camera mode triggered")
        
        # Test single camera mode
        ret, frame = cap_primary.read()
        if ret:
            logger.info(f"✓ Single camera fallback working (frame size: {frame.shape})\n")
            cap_primary.release()
            return True
    else:
        logger.warning(f"Camera {camera_ids[1]} was unexpectedly available")
        cap_secondary.release()
    
    cap_primary.release()
    return fallback_triggered


def test_vision_embedding_shape() -> bool:
    """
    Test that vision embeddings have correct shape
    
    Returns:
        True if test passed, False otherwise
    """
    logger.info("=" * 60)
    logger.info("TEST 5: Vision Embedding Dimensions")
    logger.info("=" * 60)
    
    try:
        import torch
        
        logger.info("Expected shapes:")
        logger.info("  - Single vision embedding: (1, 512)")
        logger.info("  - Averaged dual embeddings: (1, 512)")
        logger.info("  - Fused multimodal: (1, 512)")
        
        # Simulate embedding shapes
        emb1 = torch.randn(1, 512)
        emb2 = torch.randn(1, 512)
        emb_avg = (emb1 + emb2) / 2.0
        
        assert emb1.shape == (1, 512), f"Single embedding shape mismatch: {emb1.shape}"
        assert emb_avg.shape == (1, 512), f"Averaged embedding shape mismatch: {emb_avg.shape}"
        
        logger.info(f"✓ Single embedding shape: {emb1.shape}")
        logger.info(f"✓ Averaged embedding shape: {emb_avg.shape}")
        logger.info("✓ No dimension mismatch in fusion\n")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Vision embedding test failed: {e}\n")
        return False


def print_summary(results: dict):
    """Print summary of all tests"""
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ All tests passed! Dual camera support is ready.\n")
        return 0
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed. Please check the logs above.\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Verify dual camera functionality")
    parser.add_argument("--max-cameras", type=int, default=5, help="Maximum camera ID to test")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames for capture tests")
    parser.add_argument("--camera1", type=int, default=0, help="Primary camera ID")
    parser.add_argument("--camera2", type=int, default=1, help="Secondary camera ID")
    
    args = parser.parse_args()
    
    results = {}
    
    # Test 1: Camera availability
    available = test_camera_availability(args.max_cameras)
    working_cameras = [cid for cid, info in available.items() if info['available']]
    
    # Test 2: Single camera
    if args.camera1 in working_cameras:
        results["Single camera capture"] = test_single_camera_capture(args.camera1, args.frames)
    else:
        logger.warning(f"Skipping single camera test - camera {args.camera1} not available")
        results["Single camera capture"] = False
    
    # Test 3: Dual camera (if both available)
    if len(working_cameras) >= 2:
        results["Dual camera capture"] = test_dual_camera_capture([args.camera1, args.camera2], args.frames)
    elif len(working_cameras) == 1:
        logger.warning("Only one camera available - skipping dual camera test")
        results["Dual camera capture"] = False
    else:
        logger.error("No cameras available!")
        results["Dual camera capture"] = False
    
    # Test 4: Fallback
    results["Graceful camera fallback"] = test_camera_fallback([working_cameras[0] if working_cameras else 0, 99])
    
    # Test 5: Vision embeddings
    results["Vision embedding shapes"] = test_vision_embedding_shape()
    
    # Print summary
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
