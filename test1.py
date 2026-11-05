import cv2
import numpy as np
import serial
import time

# ---------------- SETTINGS ----------------
CAMERA_INDEX = 1
COM_PORT = 'COM16'      # CHANGE THIS
BAUD_RATE = 9600

SCALE = 0.0680
alpha = 0.2

STABILITY_THRESHOLD = 2
STABILITY_FRAMES = 15

NEUTRAL_LEVEL = -10   # MUST MATCH ARDUINO

# ---------------- SERIAL ----------------
ser = serial.Serial(COM_PORT, BAUD_RATE)
time.sleep(2)
print("Serial connected")

def read_from_nano():
    time.sleep(0.2)
    while ser.in_waiting:
        print("Nano says:", ser.readline().decode().strip())

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, 640)
cap.set(4, 480)

smoothed_width = 0
previous_width = 0
stable_counter = 0

locked_width = None
latched_width = None
value_sent = False

# ---------------- MAIN LOOP ----------------
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

    if contours and latched_width is None:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 1000:

            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            width, height = rect[1]

            if width > 0 and height > 0:

                grip_width_px = min(width, height)

                if smoothed_width == 0:
                    smoothed_width = grip_width_px
                else:
                    smoothed_width = (1 - alpha) * smoothed_width + alpha * grip_width_px

                if abs(smoothed_width - previous_width) < STABILITY_THRESHOLD:
                    stable_counter += 1
                else:
                    stable_counter = 0
                    locked_width = None

                previous_width = smoothed_width

                if stable_counter > STABILITY_FRAMES and locked_width is None:
                    locked_width = smoothed_width
                    print("Width Locked")

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                real_width_cm = smoothed_width * SCALE

                if locked_width is not None:
                    cv2.putText(frame,
                                f"LOCKED: {real_width_cm:.2f} cm",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2)
                    cv2.putText(frame,
                                "Press 'l' to LATCH",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2)
                else:
                    cv2.putText(frame,
                                f"Measuring: {real_width_cm:.2f} cm",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2)

    # ---------- LATCHED DISPLAY ----------
    if latched_width is not None:
        cv2.putText(frame,
                    f"LATCHED: {latched_width:.2f} cm",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2)

        if not value_sent:
            cv2.putText(frame,
                        "Press 's' to SEND",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    cv2.imshow("Adaptive Prosthetic Vision", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

    # -------- LATCH --------
    if key == ord('l'):
        if locked_width is not None:
            latched_width = locked_width * SCALE
            print("Latched:", latched_width)

    # -------- SEND --------
    if key == ord('s'):
        if latched_width is not None and not value_sent:

            # TUNE THIS GAIN IF NEEDED
            grip_level = int(30 - 4 * latched_width)

            grip_level = max(-30, min(30, grip_level))

            ser.write(f"G,{grip_level}\n".encode())
            ser.flush()

            print("Sent gripLevel:", grip_level)
            read_from_nano()

            value_sent = True

    # -------- RESET --------
    if key == ord('r'):

        ser.write(f"G,{NEUTRAL_LEVEL}\n".encode())
        ser.flush()

        print("Reset sent")
        read_from_nano()

        locked_width = None
        latched_width = None
        stable_counter = 0
        smoothed_width = 0
        previous_width = 0
        value_sent = False

cap.release()
cv2.destroyAllWindows()
ser.close()