import cv2

cap = cv2.VideoCapture(0)
i = 0
count = 0
timestep = 0.5 # in seconds
frame = 30 * timestep
while cap.isOpened():
    try:
        ret, frame = cap.read()

        if ret == False:
            break

        if i == frame:
            count += 1
            cv2.imwrite("./cameraframes/Frame" + str(count) + ".jpg", frame)
            i = 0
        i += 1
        print(i)

    except KeyboardInterrupt:
        break
cap.release()
cv2.destroyAllWindows()
