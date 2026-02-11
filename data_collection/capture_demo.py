#!/usr/bin/env python3
"""
Simple capture scaffold for multimodal demonstration trials.

Records:
 - primary camera (MP4)
 - microphone (WAV, mono 16k)
 - robot log (CSV: timestamp, joint_pos...)
 - sensors (CSV)
 - meta.json describing the trial

This is a lightweight scaffold for pilots; adapt to your sensors and robot API.
"""
import argparse
import os
import time
import json
import wave
import threading
from collections import deque

import cv2
import numpy as np
import sounddevice as sd

# Placeholder robot/sensor logger functions — replace with real interfaces
def sample_robot_state():
    # return a dict of joint positions
    return {'t': time.time(), 'j1': 0.0, 'j2': 0.0, 'j3': 0.0}

def sample_sensors():
    return {'t': time.time(), 'pressure': 0.0, 'emg1': 0.0, 'emg2': 0.0}


def record_audio(out_path, duration, samplerate=16000, channels=1):
    wf = wave.open(out_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(samplerate)

    frames = []
    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        sd.sleep(int(duration * 1000))
    audio = np.concatenate(frames, axis=0)
    # convert to int16
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype('int16')
    wf.writeframes(audio_int16.tobytes())
    wf.close()


def split_speech_ambient(wav_path: str, speech_out: str, ambient_out: str, frame_ms: int = 30, threshold: float = 0.01):
    """Split a mono WAV into speech and ambient files using simple RMS VAD.

    - frame_ms: analysis window in milliseconds
    - threshold: RMS threshold (in normalized [-1,1]) above which frame considered speech
    """
    import wave as _wave
    import numpy as _np

    wf = _wave.open(wav_path, 'rb')
    nchan = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    sr = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    wf.close()

    if nchan != 1 or sampwidth != 2:
        # Only handle 16-bit mono WAV here — fallback: copy original to both outputs
        import shutil
        shutil.copy(wav_path, speech_out)
        shutil.copy(wav_path, ambient_out)
        return

    arr = _np.frombuffer(frames, dtype=_np.int16).astype(_np.float32) / 32767.0
    frame_len = int(sr * (frame_ms / 1000.0))
    if frame_len <= 0:
        frame_len = 512

    speech_chunks = []
    ambient_chunks = []
    for i in range(0, len(arr), frame_len):
        chunk = arr[i:i+frame_len]
        if chunk.size == 0:
            continue
        rms = _np.sqrt((chunk ** 2).mean())
        if rms >= threshold:
            speech_chunks.append(chunk)
        else:
            ambient_chunks.append(chunk)

    def _write_wav(path, chunks):
        wf = _wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        if len(chunks) == 0:
            # write 0.1s silence
            silence = (_np.zeros(int(sr * 0.1))).astype(_np.int16)
            wf.writeframes(silence.tobytes())
        else:
            data = _np.concatenate(chunks)
            ints = (_np.clip(data, -1.0, 1.0) * 32767).astype(_np.int16)
            wf.writeframes(ints.tobytes())
        wf.close()

    _write_wav(speech_out, speech_chunks)
    _write_wav(ambient_out, ambient_chunks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='Output trial folder')
    p.add_argument('--duration', type=float, default=30.0)
    p.add_argument('--webcam', type=int, default=0, help='Primary webcam index')
    p.add_argument('--webcam2', type=int, default=1, help='Secondary webcam index (optional)')
    p.add_argument('--fps', type=int, default=30)
    args = p.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    cam_file = os.path.join(out_dir, 'cam1.mp4')
    cam2_file = os.path.join(out_dir, 'cam2.mp4') if args.webcam2 is not None else None
    mic_file = os.path.join(out_dir, 'mic.wav')
    robot_file = os.path.join(out_dir, 'robot.csv')
    pressure_file = os.path.join(out_dir, 'pressure.csv')
    emg_file = os.path.join(out_dir, 'emg.csv')
    meta_file = os.path.join(out_dir, 'meta.json')

    # Start video capture threads for primary and optional secondary cameras
    cap1 = cv2.VideoCapture(args.webcam)
    if not cap1.isOpened():
        raise RuntimeError(f'Cannot open primary webcam {args.webcam}')
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer1 = cv2.VideoWriter(cam_file, fourcc, args.fps, (width1, height1))

    cap2 = None
    writer2 = None
    if args.webcam2 is not None:
        cap2 = cv2.VideoCapture(args.webcam2)
        if not cap2.isOpened():
            logger_warn = getattr(__import__('logging'), 'warning')
            logger_warn(f'Cannot open secondary webcam {args.webcam2}; continuing without it')
            cap2 = None
        else:
            width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or width1)
            height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or height1)
            writer2 = cv2.VideoWriter(cam2_file, fourcc, args.fps, (width2, height2))

    stop_flag = threading.Event()

    def video_thread_1():
        start = time.time()
        while not stop_flag.is_set() and (time.time() - start) < args.duration:
            ret, frame = cap1.read()
            if not ret:
                break
            writer1.write(frame)
            time.sleep(1.0 / args.fps)
        writer1.release()
        cap1.release()

    vt1 = threading.Thread(target=video_thread_1, daemon=True)
    vt1.start()

    vt2 = None
    if cap2 is not None:
        def video_thread_2():
            start = time.time()
            while not stop_flag.is_set() and (time.time() - start) < args.duration:
                ret, frame = cap2.read()
                if not ret:
                    break
                writer2.write(frame)
                time.sleep(1.0 / args.fps)
            writer2.release()
            cap2.release()

        vt2 = threading.Thread(target=video_thread_2, daemon=True)
        vt2.start()

    # Audio capture in parallel (blocking here for simplicity)
    print('Recording audio...')
    record_audio(mic_file, duration=args.duration, samplerate=16000, channels=1)
    print('Audio done')
    # Post-process audio into speech and ambient tracks
    speech_wav = os.path.join(out_dir, 'mic_speech.wav')
    ambient_wav = os.path.join(out_dir, 'mic_ambient.wav')
    try:
        split_speech_ambient(mic_file, speech_wav, ambient_wav, frame_ms=30, threshold=0.01)
    except Exception as e:
        print('Audio split failed:', e)

    # Start sensor sampling threads: robot, pressure (100Hz), emg (200Hz)
    robot_rows = []
    pressure_rows = []
    emg_rows = []

    def robot_sampler():
        start = time.time()
        while (time.time() - start) < args.duration:
            robot_rows.append(sample_robot_state())
            time.sleep(0.05)

    def pressure_sampler():
        start = time.time()
        while (time.time() - start) < args.duration:
            s = sample_sensors()
            pressure_rows.append({'t': s['t'], 'pressure': s['pressure']})
            time.sleep(0.01)  # 100 Hz

    def emg_sampler():
        start = time.time()
        while (time.time() - start) < args.duration:
            s = sample_sensors()
            emg_rows.append({'t': s['t'], 'emg1': s['emg1'], 'emg2': s['emg2'], 'emg3': s.get('emg3', 0.0)})
            time.sleep(0.005)  # 200 Hz

    rt = threading.Thread(target=robot_sampler, daemon=True)
    pt = threading.Thread(target=pressure_sampler, daemon=True)
    et = threading.Thread(target=emg_sampler, daemon=True)
    rt.start(); pt.start(); et.start()

    # wait for duration to elapse
    time.sleep(args.duration)

    # write CSVs
    import csv
    if len(robot_rows) > 0:
        with open(robot_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=robot_rows[0].keys())
            w.writeheader(); w.writerows(robot_rows)
    if len(pressure_rows) > 0:
        with open(pressure_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=pressure_rows[0].keys())
            w.writeheader(); w.writerows(pressure_rows)
    if len(emg_rows) > 0:
        with open(emg_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=emg_rows[0].keys())
            w.writeheader(); w.writerows(emg_rows)

    stop_flag.set()
    vt1.join()
    if vt2 is not None:
        vt2.join()

    meta = {
        'trial_id': os.path.basename(out_dir),
        'start_time': time.time(),
        'duration': args.duration,
        'camera_primary': cam_file,
        'camera_secondary': cam2_file,
        'audio': mic_file,
        'audio_speech': speech_wav,
        'audio_ambient': ambient_wav,
        'robot_log': robot_file,
        'pressure_log': pressure_file,
        'emg_log': emg_file,
        'notes': ''
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print('Done. Files saved in', out_dir)

if __name__ == '__main__':
    main()
