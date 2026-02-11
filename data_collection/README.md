# Data Collection Toolkit

This folder contains scripts and templates for collecting multimodal demonstration data (video, audio, sensors, robot logs) for training.

Contents:
- `capture_demo.py` — lightweight recording scaffold (video + audio + robot/sensor logging) per trial
- `meta_schema.json` — example metadata schema for each trial
- `consent_form.txt` — minimal consent template for volunteers
- `safety_checklist.md` — operator safety checklist

Quick start (pilot):
1. Install dependencies: `pip install opencv-python sounddevice numpy pyyaml` (plus drivers for sensors/robot as needed).
2. Run a short pilot: `python data_collection/capture_demo.py --out ./dataset/V01/S01/T0001 --duration 30 --webcam 0 --modelpath /path/to/vosk-model`
3. Verify generated `meta.json`, `cam.mp4`, `mic.wav`, and `robot.csv` files in the trial folder.

See `capture_demo.py` header for usage options and notes on synchronization and QC.
