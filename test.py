import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed")
else:
    print("Camera connected")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()