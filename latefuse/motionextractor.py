import cv2
import numpy as np
import pandas as pd



cap = cv2.VideoCapture("cam1.mp4")

# Read first frame
ret, frame = cap.read()
if not ret:
    raise Exception("Failed to read video")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0

data = []
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

# --- ORIGIN ---
(center, box, mask) = get_center(frame)
if center is None:
    raise Exception("No orange object detected")

origin_x, origin_y = center

prev_cx, prev_cy = center

while True:
    ret, frame = cap.read()
    if not ret:
        break

    center, box, mask = get_center(frame)

    if center is None:
        print("Lost object")
        continue

    cx, cy = center
    x, y, w, h = box

    # Frame-to-frame motion
    dx = cx - prev_cx
    dy = cy - prev_cy

    # Global position
    X = cx - origin_x
    Y = cy - origin_y

    print(f"dx={dx:.2f}, dy={dy:.2f} | X={X:.2f}, Y={Y:.2f}")

    # Draw
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
    cv2.circle(frame, (origin_x, origin_y), 5, (255,0,0), -1)
    cv2.line(frame, (origin_x, origin_y), (cx, cy), (0,255,0), 2)

    prev_cx, prev_cy = cx, cy

    cv2.imshow("Tracking", frame)
    cv2.imshow("Mask", mask)

    t = frame_id / fps

    data.append({
        "t": t,
        "X": X,
        "Y": Y,
        "dx": dx,
        "dy": dy
    })

    frame_id += 1

    if cv2.waitKey(30) & 0xFF == 27:
        break
    
df = pd.DataFrame(data)
df.to_csv("trajectory.csv", index=False)
cap.release()
cv2.destroyAllWindows()