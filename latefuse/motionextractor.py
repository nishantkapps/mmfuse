import cv2
import numpy as np
import pandas as pd
import os

video_folder = "videos"   # change this

# Define orange range
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([20, 255, 255])

def get_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < 500:
        return None, None, None

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2

    return (cx, cy), (x, y, w, h), mask


# --- GLOBAL STORAGE ---
all_data = []
video_id = 0
global_id = 0

# Loop through videos
for file in os.listdir(video_folder):
    if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    video_id += 1
    video_path = os.path.join(video_folder, file)

    print(f"Processing: {file}")

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Skipping (can't read)")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0

    # --- ORIGIN ---
    center, box, mask = get_center(frame)
    if center is None:
        print("No object detected in first frame, skipping")
        cap.release()
        continue

    origin_x, origin_y = center
    prev_cx, prev_cy = center

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center, box, mask = get_center(frame)

        if center is None:
            continue

        cx, cy = center

        dx = cx - prev_cx
        dy = cy - prev_cy

        X = cx - origin_x
        Y = cy - origin_y

        frame_id += 1
        t = frame_id / fps

        global_id += 1

        all_data.append({
            "id": global_id,
            "video_id": video_id,
            "video_name": file,
            "t": t,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dy": dy
        })

        prev_cx, prev_cy = cx, cy

    cap.release()

# --- SAVE ONCE ---
df = pd.DataFrame(all_data)
df.to_csv("all_trajectories.csv", index=False)

print("Saved all data to all_trajectories.csv")

cv2.destroyAllWindows()