import cv2

cap = cv2.VideoCapture(1)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)  # <-- fixes the upside-down mount

    cv2.imshow("Corrected Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()