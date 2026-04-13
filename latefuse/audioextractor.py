import whisper
import time
import pandas as pd

# Load trajectory
df = pd.read_csv("trajectory.csv")

# Initialize column
df["command"] = "idle"



model = whisper.load_model("large")  # options: tiny, base, small, medium, large

result = model.transcribe("mic.wav", word_timestamps=True)

print("result: ", result["text"])

COMMANDS = ["start", "focus here", "stop", "left", "right", "up", "down", "perfect"]

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


dataset = []

commands = extract_commands(result)

# Label rows
for cmd, start, end in commands:
    mask = (df["t"] >= start) & (df["t"] <= end)
    df.loc[mask, "command"] = cmd

# Save updated CSV
df.to_csv("labeled_trajectory.csv", index=False)

