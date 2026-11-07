import cv2

for i in [0, 1]:
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imshow(f"Camera Index {i}", frame)
            print(f"Camera {i} opened successfully")

    cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()