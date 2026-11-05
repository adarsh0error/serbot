import cv2
import numpy as np
import serial
import time

# ============================================
# SETTINGS
# ============================================

CAMERA_INDEX = 1
COM_PORT = 'COM16'      # Change this
BAUD_RATE = 9600

SCALE = 0.0475
alpha = 0.2

STABILITY_THRESHOLD = 2
STABILITY_FRAMES = 15

# ============================================
# SERIAL SETUP
# ============================================

ser = serial.Serial(COM_PORT, BAUD_RATE)
time.sleep(2)
print("Serial connected.")

# ============================================
# CAMERA SETUP
# ============================================

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, 640)
cap.set(4, 480)

smoothed_width = 0
previous_width = 0
stable_counter = 0

locked_width = None       # auto lock from stability
latched_width = None      # manually latched by user
value_sent = False

# ============================================
# MAIN LOOP
# ============================================

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

                # Smoothing
                if smoothed_width == 0:
                    smoothed_width = grip_width_px
                else:
                    smoothed_width = (1 - alpha) * smoothed_width + alpha * grip_width_px

                # Stability Check
                if abs(smoothed_width - previous_width) < STABILITY_THRESHOLD:
                    stable_counter += 1
                else:
                    stable_counter = 0
                    locked_width = None

                previous_width = smoothed_width

                # Auto Lock
                if stable_counter > STABILITY_FRAMES and locked_width is None:
                    locked_width = smoothed_width
                    print("Width Locked (Auto)")

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

    # If latched
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
        else:
            cv2.putText(frame,
                        "Value Sent",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 0),
                        2)

    cv2.imshow("Adaptive Prosthetic Vision", frame)

    key = cv2.waitKey(1)

    # ESC to exit
    if key == 27:
        break

    # Latch width
    if key == ord('l'):
        if locked_width is not None:
            latched_width = locked_width * SCALE
            print("Latched:", latched_width)
        else:
            print("Nothing locked to latch!")

    # Send only latched value
    if key == ord('s'):
        if latched_width is not None and not value_sent:
            ser.write(f"{latched_width}\n".encode())
            print("Sent to Arduino:", latched_width)
            value_sent = True
        elif latched_width is None:
            print("No latched value to send!")
        else:
            print("Value already sent!")

    # Reset
    if key == ord('r'):

        ser.write("0\n".encode())
        print("System Reset Sent to Arduino")

        
        locked_width = None
        latched_width = None
        stable_counter = 0
        smoothed_width = 0
        previous_width = 0
        value_sent = False
        print("System Reset")

cap.release()
cv2.destroyAllWindows()
ser.close()