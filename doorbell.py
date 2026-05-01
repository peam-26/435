# Import numerical and image processing packages
import numpy as np
import cv2

# Import timing and operating system packages
# time is used for delays between images and video length
# os is used to run the ffmpeg command for video conversion
import time
import os

# Import email and SMTP packages for sending the alert email
import smtplib
from smtplib import SMTP
from smtplib import SMTPException
import email
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Import Picamera2 tools for using the Raspberry Pi camera
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import Transform

"""
# This function was used for initial mask calibration
# It lets the user manually select points/regions on a calibration image
# It is commented out because the final mask points are now hard-coded below

def select_points(img): # for initial mask point selection
    points = []
    for i in range(0, 4): # number of points needed to form shape
        bbox = cv2.selectROI(img, False)
        print(bbox)
        points.append([bbox[0], bbox[1]])
    print(points)

    return points
"""

# This function takes an image, applies the selected region mask,
# then converts the result into a smaller grayscale blurred image
# The grayscale output is what the program uses for image comparison
def mask_image(img):

    # Create a black mask with the same height and width as the image
    # The mask starts fully black, meaning nothing is selected yet
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    # These are my final rectangle points for the area I want to monitor
    # I shifted the rectangle toward the left because my door is on the left side of the camera view
    # Points are ordered as top-left, top-right, bottom-right, bottom-left
    pts = np.array([[50, 50], [750, 50], [750, 700], [50, 700]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    # Apply the mask to the original image
    # Everything outside the selected region becomes black
    masked = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.resize(masked, (200, int(masked.shape[0] * 200 / masked.shape[1])))
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (9, 9), 0) # play with kernel size ... blur 
    return masked, gray

counter = 0
picam2 = Picamera2() #camera setup
camera_config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    transform=Transform(hflip=1, vflip=1)
)

picam2.configure(camera_config)
picam2.start()

time.sleep(1)

try:

    while True: #continuous
        if cv2.waitKey(1) == ord('q'):
            break

        #track how many detection cycles have happened
        counter += 1
        print(" ")
        print("----Times through loop since starting:", counter, "----")
        print(" ")

        # take a 1st and 2nd image to compare

        # Capture the first image
        # This acts as the "before" image
        picam2.capture_file("test1.jpg")
        time.sleep(2) #2 second wait time, determined through testing to decide how much time was adequate for object detection between images

        # Capture the second image
        # This acts as the "after" image
        picam2.capture_file("test2.jpg")
        
        print("Captured 1st & 2nd image for analysis...")

        test1 = cv2.imread("test1.jpg")
        time.sleep(3)

        test2 = cv2.imread("test2.jpg")

        # masked1 and masked2 show the selected region in color
        # gray1 and gray2 are the processed images used for comparison
        masked1, gray1 = mask_image(test1)
        masked2, gray2 = mask_image(test2)
        # compare the two images
        pixel_threshold = 50
        detector_total = np.uint64(0)
        detector = np.zeros((gray2.shape[0], gray2.shape[1]), dtype="uint8")

        # pixel by pixel comparison, loop through every pixel in the processed grayscale images
        for i in range(0, gray2.shape[0]):
            for j in range(0, gray2.shape[1]):

                # Compare the brightness of each pixel between image 1 and image 2
                # If the absolute difference is greater than the pixel threshold,
                # that pixel is marked as changed
                if abs(int(gray2[i, j]) - int(gray1[i, j])) > pixel_threshold:
                    detector[i, j] = 255


        # Add up all changed pixels
        # Larger detector_total means more motion/change happened in the masked region
        detector_total = np.uint64(np.sum(detector))
        print("detector_total = ", detector_total)
        print(" ")

        # If the total amount of change is high enough, trigger the smart doorbell
        # I chose 45000 because no motion was near 0 and walking into frame was over 100000, found through testing
        if detector_total > 45000:

            print("Smart Doorbell has detected someone/something at the door!")

            timestr = time.strftime("doorbell-%Y%m%d-%H%M%S")
            encoder = H264Encoder()
            picam2.start_recording(encoder, FileOutput(f"{timestr}.h264"))

            # Record for 7 seconds after detection
            # This gives enough time to capture the event after motion is detected
            time.sleep(7)
            picam2.stop_recording()

            print("Finished recording...converting to mp4...")
            
            # Convert the h264 video into mp4 format using ffmpeg
            command3 = f'ffmpeg -framerate 30 -i {timestr}.h264 -c copy {timestr}.mp4'
            os.system(command3)

            print("Finished converting file...available for viewing")

        
            cv2.imwrite("gray1.jpg", gray1)
            cv2.imwrite("gray2.jpg", gray2)
            cv2.imwrite("masked1.jpg", masked1)
            cv2.imwrite("masked2.jpg", masked2)

            fullDirectory = '/home/pi/Documents/ENME435/HW/HW7/' + timestr + '.mp4'

           # command4 = '/home/pi/dropbox_uploader.sh upload ' + fullDirectory + ' /'
           # os.system(command4)


            smtpUser = 'peam.affiliate@gmail.com'
            smtpPass = 'xirr uwbq bqzy skte'

            toAdd = 'peampats@gmail.com'
            fromAdd = smtpUser

            f_time = datetime.now().strftime('%a %d %b @ %H:%M')
            subject = 'Smart Doorbell Images Detected: ' + f_time

            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = fromAdd
            msg['To'] = toAdd
            
            msg.preamble = 'Image @ ' + f_time

            #body = email.mime.Text.MIMEText('Smart Doorbell video: ' + f_time)

            body = MIMEText('Motion Detected! - ' + f_time)
            msg.attach(body)

            fp = open('test1.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            fp = open('test2.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            fp = open('gray1.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            fp = open('gray2.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            fp = open('masked1.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            fp = open('masked2.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(smtpUser, smtpPass)
            s.sendmail(fromAdd, toAdd, msg.as_string())
            s.quit()

            print("Email Sent")

        else:
            # If the total change is not high enough, no detection is triggered
            print("No Detections")

except KeyboardInterrupt:
    print("Stopped")

picam2.stop()
