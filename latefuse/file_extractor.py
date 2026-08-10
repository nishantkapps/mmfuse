import os
import shutil

base_prefix = r"C:\V01\P"

video_out = "videos"
audio_out = "audios"

os.makedirs(video_out, exist_ok=True)
os.makedirs(audio_out, exist_ok=True)

vid_counter = 1

for p in range(33, 110):  # P29 → P109
    if p < 100:
        person_path = f"{base_prefix}0{p}"
    else:
        person_path = f"{base_prefix}{p}"

    if not os.path.exists(person_path):
        continue

    print(f"Scanning {person_path}")

    for root, dirs, files in os.walk(person_path):

        if "cam1.mp4" in files and "mic.wav" in files:

            cam_path = os.path.join(root, "cam1.mp4")
            mic_path = os.path.join(root, "mic.wav")

            vid_name = f"vid{vid_counter}.mp4"
            aud_name = f"vid{vid_counter}.wav"

            vid_out_path = os.path.join(video_out, vid_name)
            aud_out_path = os.path.join(audio_out, aud_name)

            print(f"Copying from {root}")
            print(f" → {vid_name}, {aud_name}")

            shutil.copy(cam_path, vid_out_path)
            shutil.copy(mic_path, aud_out_path)

            vid_counter += 1

print("Done extracting cam1.mp4 and mic.wav from all folders.")