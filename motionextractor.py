import cv2
import numpy as np

cap = cv2.VideoCapture("cam1.mp4")

# Read first frame
ret, old_frame = cap.read()
if not ret:
    raise Exception("Failed to read video")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# --- STEP 1: Detect orange arm to get ROI ---
hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

lower_orange = np.array([5, 100, 100])
upper_orange = np.array([20, 255, 255])

mask = cv2.inRange(hsv, lower_orange, upper_orange)

kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    raise Exception("No orange object detected")

# Largest contour = arm
c = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)
origin_x = x + w // 2
origin_y = y + h // 2
# --- STEP 2: Use detected ROI ---
roi = old_gray[y:y+h, x:x+w]

p0 = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.3, minDistance=7)

if p0 is None:
    raise Exception("No features found in ROI")

# Convert to full image coordinates
p0[:, 0, 0] += x
p0[:, 0, 1] += y

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

    # Handle failure
    if p1 is None or st is None:
        print("Tracking lost")
        break

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Movement
    dx = np.mean(good_new[:, 0] - good_old[:, 0])
    dy = np.mean(good_new[:, 1] - good_old[:, 1])

    print(f"dx={dx:.2f}, dy={dy:.2f}")

    # Draw tracked points
    for pt in good_new:
        x_pt, y_pt = pt.ravel().astype(int)
        cv2.circle(frame, (x_pt, y_pt), 3, (0,255,0), -1)

    # Draw bounding box (initial one)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    cx = int(np.mean(good_new[:, 0]))
    cy = int(np.mean(good_new[:, 1]))

    X = cx - origin_x
    Y = cy - origin_y

    print(f"Global position: X={X:.2f}, Y={Y:.2f}")
    # Update
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()