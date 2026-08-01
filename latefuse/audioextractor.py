import whisper
import pandas as pd
import os
import json
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COMMANDS = ["start", "focus here", "stop", "left", "right", "up", "down", "perfect"]
AUDIO_FOLDER = "audios"
INPUT_CSV = "all_trajectories.csv"
OUTPUT_CSV = "labeled_all_trajectories.csv"
REVIEW_CSV = "labeling_review.csv"
PROGRESS_FILE = "labeling_progress.json"  # tracks which files are done

# Extra buffer (seconds) added around each command's time window.
# Covers transcription timing inaccuracies — Whisper segment timestamps
# are often 0.3–0.8 s off from the true spoken moment.
TIME_BUFFER = 0.5

# If two commands overlap after buffering, the one with the higher
# Whisper no_speech_prob is dropped.
NO_SPEECH_THRESHOLD = 0.6

# ─────────────────────────────────────────────
# 1. LOAD DATA  (resume from OUTPUT_CSV if it exists)
# ─────────────────────────────────────────────
if os.path.exists(OUTPUT_CSV) and os.path.exists(PROGRESS_FILE):
    print(f"⏩  Resuming from existing output: {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
else:
    print("🆕  Starting fresh labeling run.")
    df = pd.read_csv(INPUT_CSV)
    df["command"] = "idle"
    df["label_confidence"] = 0.0   # lets you filter low-confidence rows later

# Load already-processed files so we can skip them
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)   # {"done": ["file1.wav", ...]}
    print(f"   Already processed: {len(progress['done'])} file(s) — skipping them.")
else:
    progress = {"done": []}

# Load existing review rows so we append, not overwrite
if os.path.exists(REVIEW_CSV):
    review_rows = pd.read_csv(REVIEW_CSV).to_dict("records")
else:
    review_rows = []

# ─────────────────────────────────────────────
# 2. LOAD MODEL  (use "medium" unless accuracy is critical — large is slow)
# ─────────────────────────────────────────────
print("Loading Whisper model…")
model = whisper.load_model("large")

# ─────────────────────────────────────────────
# 3. COMMAND EXTRACTION  (word-level, with confidence)
# ─────────────────────────────────────────────
def extract_commands(result: dict) -> list[dict]:
    """
    Returns a list of dicts:
        cmd, start, end, confidence, raw_text
    Uses word-level timestamps when available so the window is tight.
    Falls back to segment-level timestamps otherwise.
    """
    found = []

    for seg in result["segments"]:
        seg_text = seg["text"].lower().strip()
        no_speech_prob = seg.get("no_speech_prob", 0.0)

        # Skip segments Whisper itself thinks are silence / noise
        if no_speech_prob > NO_SPEECH_THRESHOLD:
            continue

        # Prefer word-level timestamps (requires word_timestamps=True)
        words = seg.get("words", [])

        for cmd in COMMANDS:
            cmd_words = cmd.split()   # e.g. "focus here" → ["focus", "here"]

            # ── word-level match ──
            if words:
                for i in range(len(words) - len(cmd_words) + 1):
                    window = words[i : i + len(cmd_words)]
                    spoken = " ".join(w["word"].lower().strip() for w in window)

                    if spoken == cmd:
                        w_start = window[0]["start"]
                        w_end   = window[-1]["end"]
                        avg_prob = sum(w.get("probability", 1.0) for w in window) / len(window)

                        found.append({
                            "cmd":        cmd,
                            "start":      max(0.0, w_start - TIME_BUFFER),
                            "end":        w_end + TIME_BUFFER,
                            "confidence": round(avg_prob * (1 - no_speech_prob), 3),
                            "raw_text":   seg_text,
                        })

            # ── segment-level fallback ──
            elif cmd in seg_text:
                found.append({
                    "cmd":        cmd,
                    "start":      max(0.0, seg["start"] - TIME_BUFFER),
                    "end":        seg["end"] + TIME_BUFFER,
                    "confidence": round(1 - no_speech_prob, 3),
                    "raw_text":   seg_text,
                })

    # Remove duplicates: same cmd detected by both word & segment level
    seen, deduped = set(), []
    for item in found:
        key = (item["cmd"], round(item["start"], 1))
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


# ─────────────────────────────────────────────
# 4. PROCESS EACH AUDIO FILE
# ─────────────────────────────────────────────
all_files = sorted(
    f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(".wav")
)
remaining = [f for f in all_files if f not in progress["done"]]
print(f"\n📂  {len(all_files)} audio file(s) total — {len(remaining)} left to process.\n")

for file in remaining:
    audio_path = os.path.join(AUDIO_FOLDER, file)
    video_name = Path(file).stem + ".mp4"
    video_mask = df["video_name"] == video_name

    if not video_mask.any():
        print(f"⚠  No matching video rows for {file} — skipping")
        progress["done"].append(file)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
        continue

    print(f"▶  {file}")

    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        initial_prompt=(
            "The operator is giving movement commands: "
            "start, stop, left, right, up, down, focus here, perfect."
        ),
        # Suppress hallucinated filler phrases Whisper sometimes inserts
        suppress_tokens=[-1],
        condition_on_previous_text=False,   # avoids error propagation across segments
    )

    print(f"   Transcript: {result['text'].strip()}")

    commands = extract_commands(result)

    if not commands:
        print("   No commands detected.")
        review_rows.append({
            "file": file, "cmd": "NONE_DETECTED",
            "start": None, "end": None, "confidence": None, "raw_text": result["text"].strip()
        })
    else:
        for item in commands:
            cmd, start, end = item["cmd"], item["start"], item["end"]
            conf = item["confidence"]

            time_mask = (df["t"] >= start) & (df["t"] <= end)
            n_rows = (video_mask & time_mask).sum()

            if n_rows == 0:
                print(f"   ⚠  '{cmd}' [{start:.2f}s–{end:.2f}s] matched no trajectory rows")
            else:
                df.loc[video_mask & time_mask, "command"]          = cmd
                df.loc[video_mask & time_mask, "label_confidence"] = conf
                print(f"   ✓  '{cmd}' [{start:.2f}s–{end:.2f}s]  conf={conf:.2f}  rows={n_rows}")

            review_rows.append({
                "file": file, "cmd": cmd,
                "start": start, "end": end,
                "confidence": conf, "matched_rows": n_rows,
                "raw_text": item["raw_text"]
            })

    # ── Save after every file so a crash loses at most one file's work ──
    df.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(review_rows).to_csv(REVIEW_CSV, index=False)
    progress["done"].append(file)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"   💾  Progress saved ({len(progress['done'])}/{len(all_files)} files done)")

# ─────────────────────────────────────────────
# 5. FINAL SUMMARY  (files were already saved after each audio above)
# ─────────────────────────────────────────────
print(f"\n✅  All done — {len(progress['done'])}/{len(all_files)} file(s) processed.")
print(f"   Labeled dataset → {OUTPUT_CSV}")
print(f"   Review report   → {REVIEW_CSV}")
print(f"   Progress log    → {PROGRESS_FILE}  (delete this to restart from scratch)")

# ─────────────────────────────────────────────
# 6. QUICK LABEL SUMMARY
# ─────────────────────────────────────────────
print("\n── Label distribution ──────────────────")
summary = df.groupby("command").agg(
    rows        = ("command", "count"),
    avg_conf    = ("label_confidence", "mean"),
).sort_values("rows", ascending=False)
print(summary.to_string())

low_conf = df[(df["command"] != "idle") & (df["label_confidence"] < 0.4)]
if len(low_conf):
    print(f"\n⚠  {len(low_conf)} rows have confidence < 0.4 — consider reviewing or dropping them")
    print("   Filter with: df = df[df['label_confidence'] >= 0.4]")