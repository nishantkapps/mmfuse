#!/usr/bin/env python3
"""
Simple capture scaffold for multimodal demonstration trials.

Records:
 - primary camera (MP4)
 - microphone (WAV, mono 16k)
 - robot log (CSV: timestamp, joint_pos...)
 - sensors (CSV)
 - meta.json describing the trial

Optional: --preview shows live view of cameras, audio level, and sensor data
(similar to demo_mock_streaming).
"""
import argparse
import csv
import os
import sys
import time
import json
import wave
import threading
from pathlib import Path

# Add repo root for encoders.asr_vosk
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import deque
import glob
import re
import yaml

# Configure Qt/OpenCV for display (similar to demo_mock_streaming)
if 'QT_QPA_PLATFORM' not in os.environ:
    if os.environ.get('XDG_SESSION_TYPE') == 'wayland' or os.environ.get('WAYLAND_DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    elif os.environ.get('DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    else:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

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
    """Blocking audio recording (used when --no-preview)."""
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
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype('int16')
    wf.writeframes(audio_int16.tobytes())
    wf.close()


def _annotate_capture_frame(frame, elapsed, duration, robot_state, pressure_val, emg_vals, audio_rms, transcript=""):
    """Add overlay to frame showing capture status (similar to demo_mock_streaming)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    thickness = 1

    cv2.rectangle(frame, (10, 10), (500, 220), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (500, 220), (0, 255, 0), 2)

    y = 35
    cv2.putText(frame, f"RECORDING | {elapsed:.1f}s / {duration:.1f}s", (20, y), font, font_scale, color, thickness)
    y += 30
    if robot_state:
        cv2.putText(frame, f"Robot: j1={robot_state.get('j1',0):.2f} j2={robot_state.get('j2',0):.2f} j3={robot_state.get('j3',0):.2f}", (20, y), font, font_scale, color, thickness)
    y += 30
    cv2.putText(frame, f"Pressure: {pressure_val:.2f}", (20, y), font, font_scale, color, thickness)
    y += 28
    cv2.putText(frame, f"EMG: {emg_vals[0]:.2f}, {emg_vals[1]:.2f}, {emg_vals[2]:.2f}", (20, y), font, font_scale, color, thickness)
    y += 30
    cv2.putText(frame, f"Audio RMS: {audio_rms:.4f}", (20, y), font, font_scale, (255, 255, 0), thickness)

    # Transcript overlay (bottom-left, same style as demo_mock_streaming)
    if transcript:
        status_y = h - 30
        t_y = status_y - 40
        max_width = 40
        words = transcript.split()
        line = ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 <= max_width:
                line = (line + " " + w).strip()
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        for i, ln in enumerate(reversed(lines)):
            cv2.putText(frame, ln, (20, t_y - i * 20), font, 0.6, (255, 255, 0), 1)

    return frame


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
    p.add_argument('--out', required=False, help='Output trial folder (overrides config)')
    p.add_argument('--duration', type=float, default=10.0)
    p.add_argument('--webcam', type=int, default=0, help='Primary webcam index')
    p.add_argument('--webcam2', type=int, default=1, help='Secondary webcam index (optional)')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'capture_config.yaml'), help='Path to capture config YAML')
    p.add_argument('--version', type=int, default=None, help='Dataset version number')
    p.add_argument('--participant', type=int, default=None, help='Participant/volunteer ID')
    p.add_argument('--session', type=int, default=None, help='Session number (for same participant)')
    p.add_argument('--lighting', type=str, default=None, help='Lighting condition (e.g. bright, dim, natural)')
    p.add_argument('--action', type=str, default=None, help='Action performed (e.g. reach, grasp, pick_place)')
    p.add_argument('--movement', type=str, default=None, help='Alias for --action (deprecated)')
    p.add_argument('--preview', action='store_true', default=True, help='Show live preview of capture (default)')
    p.add_argument('--no-preview', action='store_false', dest='preview', help='Run headless without display')
    p.add_argument('--use-asr', action='store_true', help='Enable Vosk ASR for live transcript (requires --vosk-model)')
    p.add_argument('--vosk-model', type=str, default=None, help='Path to Vosk model (e.g. models/vosk-model-small-en-in-0.4)')
    args = p.parse_args()

    # Load config for directory structure
    config_path = args.config
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    else:
        cfg = {}

    defaults = cfg.get('defaults', {})
    pattern = cfg.get('pattern', 'V{version:02d}/P{participant:03d}/S{session:02d}/L_{lighting}/A_{action}/T{trial:04d}')

    # Build capture context (for meta.json) - use args or config defaults
    def _sanitize(s):
        """Replace spaces/special chars for folder names."""
        return str(s).strip().replace(' ', '_').replace('/', '_') if s else ''

    action_val = _sanitize(args.action or args.movement or defaults.get('action', 'unspecified'))
    lighting_val = _sanitize(args.lighting or defaults.get('lighting', 'default'))
    capture_ctx = {
        'version': args.version if args.version is not None else defaults.get('version', 1),
        'participant': args.participant if args.participant is not None else defaults.get('participant', 1),
        'session': args.session if args.session is not None else defaults.get('session', 1),
        'lighting': lighting_val or 'default',
        'action': action_val or 'unspecified',
    }

    # If user provided explicit out path, use it. Otherwise build from pattern and args.
    if args.out:
        out_dir = args.out
    else:
        ctx = dict(capture_ctx)

        # Determine next trial number by globbing existing folders
        def pattern_to_glob(pat, ctx):
            def repl(m):
                key = m.group(1)
                if key == 'trial':
                    return '*'
                if key in ctx and ctx[key] not in (None, ''):
                    return str(ctx[key])
                return '*'
            return re.sub(r"\{(\w+)(?:[^}]*)\}", repl, pat)

        glob_pat = pattern_to_glob(pattern, ctx)
        cfg_base = cfg.get('base_dir', 'dataset')
        if not os.path.isabs(cfg_base):
            search_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', cfg_base))
        else:
            search_base = cfg_base
        full_glob = os.path.join(search_base, glob_pat)
        matches = glob.glob(full_glob)

        trial_nums = []
        trial_re = re.compile(r"T?(\d+)$")
        for m in matches:
            parts = m.rstrip('/').split(os.sep)
            last = parts[-1]
            mo = trial_re.search(last)
            if mo:
                trial_nums.append(int(mo.group(1)))

        next_trial = (max(trial_nums) + 1) if trial_nums else 1
        ctx['trial'] = next_trial

        try:
            out_dir = os.path.join(search_base, pattern.format(**ctx))
        except (KeyError, ValueError):
            out_dir = os.path.join(
                search_base,
                f"V{ctx['version']:02d}", f"P{ctx['participant']:03d}",
                f"S{ctx['session']:02d}", f"L_{ctx['lighting']}", f"A_{ctx['action']}",
                f"T{ctx['trial']:04d}"
            )

    os.makedirs(out_dir, exist_ok=True)

    cam_file = os.path.join(out_dir, 'cam1.mp4')
    cam2_file = os.path.join(out_dir, 'cam2.mp4') if args.webcam2 is not None else None
    mic_file = os.path.join(out_dir, 'mic.wav')
    robot_file = os.path.join(out_dir, 'robot.csv')
    pressure_file = os.path.join(out_dir, 'pressure.csv')
    emg_file = os.path.join(out_dir, 'emg.csv')
    meta_file = os.path.join(out_dir, 'meta.json')

    # Shared state for preview (updated by sensor threads, read by main loop)
    robot_rows = []
    pressure_rows = []
    emg_rows = []
    latest_robot = {}
    latest_pressure = 0.0
    latest_emg = [0.0, 0.0, 0.0]
    audio_buffer = deque()
    audio_rms = 0.0

    # Start sensor sampling threads
    def robot_sampler():
        start = time.time()
        while (time.time() - start) < args.duration:
            s = sample_robot_state()
            robot_rows.append(s)
            latest_robot.update(s)
            time.sleep(0.05)

    def pressure_sampler():
        nonlocal latest_pressure
        start = time.time()
        while (time.time() - start) < args.duration:
            s = sample_sensors()
            pressure_rows.append({'t': s['t'], 'pressure': s['pressure']})
            latest_pressure = s['pressure']
            time.sleep(0.01)

    def emg_sampler():
        nonlocal latest_emg
        start = time.time()
        while (time.time() - start) < args.duration:
            s = sample_sensors()
            emg_rows.append({'t': s['t'], 'emg1': s['emg1'], 'emg2': s['emg2'], 'emg3': s.get('emg3', 0.0)})
            latest_emg = [s['emg1'], s['emg2'], s.get('emg3', 0.0)]
            time.sleep(0.005)

    rt = threading.Thread(target=robot_sampler, daemon=True)
    pt = threading.Thread(target=pressure_sampler, daemon=True)
    et = threading.Thread(target=emg_sampler, daemon=True)
    rt.start()
    pt.start()
    et.start()

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
            print(f'Warning: Cannot open secondary webcam {args.webcam2}; continuing without it')
            cap2 = None
        else:
            width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or width1)
            height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or height1)
            writer2 = cv2.VideoWriter(cam2_file, fourcc, args.fps, (width2, height2))

    stop_flag = threading.Event()

    # Optional ASR for live transcript (same as demo_mock_streaming)
    asr = None
    if args.preview and args.use_asr and args.vosk_model:
        try:
            from encoders.asr_vosk import StreamingVosk
            asr = StreamingVosk(model_path=args.vosk_model, sample_rate=16000)
            print(f"Vosk ASR initialized from {args.vosk_model}")
        except Exception as e:
            print(f"ASR init failed: {e}. Continuing without transcript.")
            asr = None
    elif args.use_asr and not args.vosk_model:
        print("Warning: --use-asr requires --vosk-model. Skipping ASR.")

    if args.preview:
        # Preview mode: main loop with display, non-blocking audio
        target_samples = int(args.duration * 16000)
        def audio_callback(indata, frames_count, time_info, status):
            audio_buffer.extend(indata[:, 0].tolist())

        with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, blocksize=1024):
            if cap2 is not None:
                cv2.namedWindow('Capture - Camera 1', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Capture - Camera 2', cv2.WINDOW_NORMAL)
                try:
                    cv2.resizeWindow('Capture - Camera 1', 640, 480)
                    cv2.resizeWindow('Capture - Camera 2', 640, 480)
                except Exception:
                    pass
            else:
                cv2.namedWindow('Capture - Camera 1', cv2.WINDOW_NORMAL)
                try:
                    cv2.resizeWindow('Capture - Camera 1', 640, 480)
                except Exception:
                    pass

            print('Recording with preview (Ctrl+C or close window to stop early)...')
            start_time = time.time()
            frame_time = 1.0 / args.fps

            while (time.time() - start_time) < args.duration:
                loop_start = time.time()
                ret1, frame1 = cap1.read()
                if not ret1:
                    break

                writer1.write(frame1)
                # Compute audio RMS from buffer
                buf = list(audio_buffer)
                if len(buf) >= 1024:
                    recent = np.array(buf[-4000:], dtype=np.float32)
                    audio_rms = float(np.sqrt((recent ** 2).mean()))
                else:
                    audio_rms = 0.0

                # Get transcript from ASR (same as demo_mock_streaming)
                transcript = ""
                if asr is not None and len(buf) >= 1600:  # ~100ms at 16kHz
                    try:
                        chunk = np.array(buf[-8000:], dtype=np.float32)
                        transcript = asr.feed(chunk)
                    except Exception:
                        transcript = ""

                ann1 = _annotate_capture_frame(
                    frame1.copy(), time.time() - start_time, args.duration,
                    latest_robot, latest_pressure, latest_emg, audio_rms, transcript=transcript
                )
                cv2.imshow('Capture - Camera 1', ann1)

                frame2 = None
                if cap2 is not None:
                    ret2, frame2 = cap2.read()
                    if ret2:
                        writer2.write(frame2)
                        ann2 = _annotate_capture_frame(
                            frame2.copy(), time.time() - start_time, args.duration,
                            latest_robot, latest_pressure, latest_emg, audio_rms, transcript=transcript
                        )
                        cv2.imshow('Capture - Camera 2', ann2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

            # Write audio buffer to file
            buf = list(audio_buffer)
            if len(buf) > 0:
                audio_np = np.array(buf, dtype=np.float32)
                audio_int16 = (np.clip(audio_np, -1, 1) * 32767).astype('int16')
                with wave.open(mic_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_int16.tobytes())

        writer1.release()
        if writer2:
            writer2.release()
        cap1.release()
        if cap2:
            cap2.release()

        if asr is not None:
            try:
                asr.close()
            except Exception:
                pass

        rt.join()
        pt.join()
        et.join()

    else:
        # Headless: video threads + blocking audio (original behavior)
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

        def video_thread_2():
            if cap2 is None:
                return
            start = time.time()
            while not stop_flag.is_set() and (time.time() - start) < args.duration:
                ret, frame = cap2.read()
                if not ret:
                    break
                writer2.write(frame)
                time.sleep(1.0 / args.fps)
            writer2.release()
            cap2.release()

        vt1 = threading.Thread(target=video_thread_1, daemon=True)
        vt1.start()
        vt2 = threading.Thread(target=video_thread_2, daemon=True) if cap2 else None
        if vt2:
            vt2.start()

        print('Recording audio...')
        record_audio(mic_file, duration=args.duration, samplerate=16000, channels=1)
        print('Audio done')

        time.sleep(args.duration)
        stop_flag.set()
        vt1.join()
        if vt2:
            vt2.join()

        writer1.release()
        if writer2:
            writer2.release()
        cap1.release()
        if cap2:
            cap2.release()

        rt.join()
        pt.join()
        et.join()

    # Post-process audio and write outputs (common for both modes)
    speech_wav = os.path.join(out_dir, 'mic_speech.wav')
    ambient_wav = os.path.join(out_dir, 'mic_ambient.wav')
    try:
        split_speech_ambient(mic_file, speech_wav, ambient_wav, frame_ms=30, threshold=0.01)
    except Exception as e:
        print('Audio split failed:', e)

    # write CSVs
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

    meta = {
        'trial_id': os.path.basename(out_dir),
        'folder_path': out_dir,
        'volunteer_id': str(capture_ctx.get('participant', '')),
        'session_id': str(capture_ctx.get('session', '')),
        'lighting': capture_ctx.get('lighting', ''),
        'action': capture_ctx.get('action', ''),
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
        'notes': '',
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print('Done. Files saved in', out_dir)

if __name__ == '__main__':
    main()
