from picamera2 import Picamera2
import cv2
import numpy as np
import time

# define green color range (HSV)
colorLower = (29, 70, 6)
colorUpper = (75, 255, 255)

# initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

time.sleep(1)

# video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('stoplight.avi', fourcc, 10, (640, 480))

while True:
    # capture frame
    frame = picam2.capture_array()

    # blur and convert to HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # create mask for green
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 0:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), -1)

    # write video
    out.write(frame)

    # show frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # press q to quit
    if key == ord("q"):
        break

out.release()
cv2.destroyAllWindows()
picam2.stop()
