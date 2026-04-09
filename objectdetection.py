import cv2
import numpy as np

cap = cv2.VideoCapture("cam1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 🟠 Orange color range (tweak if needed)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Clean noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Take largest contour (assumes it's the arm)
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 500:  # ignore tiny noise
            x, y, w, h = cv2.boundingRect(c)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # Center point
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()