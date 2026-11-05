import cv2
import numpy as np
import time

CALIBRATION_MODE = False
SCALE = 0.0680

alpha = 0.2

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

smoothed_width = 0
previous_width = 0
stable_counter = 0
locked_width = None

STABILITY_THRESHOLD = 2      # pixels
STABILITY_FRAMES = 15        # frames required to lock (~0.5 sec)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 1000:

            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            width, height = rect[1]

            if width > 0 and height > 0:

                grip_width_px = min(width, height)

                # smoothing
                if smoothed_width == 0:
                    smoothed_width = grip_width_px
                else:
                    smoothed_width = (1 - alpha) * smoothed_width + alpha * grip_width_px

                # ---- STABILITY CHECK ----
                if abs(smoothed_width - previous_width) < STABILITY_THRESHOLD:
                    stable_counter += 1
                else:
                    stable_counter = 0
                    locked_width = None

                previous_width = smoothed_width

                # Lock width if stable long enough
                if stable_counter > STABILITY_FRAMES and locked_width is None:
                    locked_width = smoothed_width
                    print("Width Locked!")

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # ---- DISPLAY ----
                if locked_width is not None:
                    real_width_cm = locked_width * SCALE
                    cv2.putText(frame,
                                f"LOCKED: {real_width_cm:.2f} cm",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2)
                else:
                    real_width_cm = smoothed_width * SCALE
                    cv2.putText(frame,
                                f"Measuring: {real_width_cm:.2f} cm",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2)

    cv2.imshow("Adaptive Prosthetic Vision", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()