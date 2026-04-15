import whisper
import pandas as pd
import os

# Load trajectory dataset
df = pd.read_csv("all_trajectories.csv")

# Initialize column
df["command"] = "idle"

# Load Whisper model
model = whisper.load_model("large")  # or "base"/"small" for speed

COMMANDS = ["start", "focus here", "stop", "left", "right", "up", "down", "perfect"]

audio_folder = "audios"

def extract_commands(result):
    found = []

    for seg in result["segments"]:
        text = seg["text"].lower()
        start = seg["start"]
        end = seg["end"]

        for cmd in COMMANDS:
            if cmd in text:
                found.append((cmd, start, end))

    return found


# Process each audio file
for file in os.listdir(audio_folder):
    if not file.lower().endswith(".wav"):
        continue

    audio_path = os.path.join(audio_folder, file)
    print(f"Processing audio: {file}")

    # Match audio → video_name
    # Example: vid1.wav → vid1.mp4
    video_name = file.replace(".wav", ".mp4")

    # Filter only rows for this video
    video_mask = df["video_name"] == video_name

    if not video_mask.any():
        print(f"No matching video found for {file}")
        continue

    # Transcribe
    result = model.transcribe(audio_path, word_timestamps=True)
    print("Text:", result["text"])

    commands = extract_commands(result)

    # Apply labels ONLY to this video's rows
    for cmd, start, end in commands:
        time_mask = (df["t"] >= start) & (df["t"] <= end)
        df.loc[video_mask & time_mask, "command"] = cmd


# Save final labeled dataset
df.to_csv("labeled_all_trajectories.csv", index=False)

print("Saved labeled dataset")