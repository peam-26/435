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

    # final selected points for porch
    # pts = np.array([[553, 707], [700, 650], [843, 550], [833, 124], [1100, 109], [1100, 619], [906, 700]], dtype=np.int32) # mitchell's array

    # These are my final rectangle points for the area I want to monitor
    # I shifted the rectangle toward the left because my door is on the left side of the camera view
    # Points are ordered as top-left, top-right, bottom-right, bottom-left
    pts = np.array([[50, 50], [750, 50], [750, 700], [50, 700]], dtype=np.int32)

    # Fill the selected polygon white on the mask
    # White areas are kept and black areas are ignored
    cv2.fillConvexPoly(mask, pts, 255)

    # pts = np.array([[553, 707], [300, 600], [400, 590], [550, 650]], dtype=np.int32) # walkway coordinates
    # cv2.fillConvexPoly(mask, pts, 255)

    # Apply the mask to the original image
    # Everything outside the selected region becomes black
    masked = cv2.bitwise_and(img, img, mask=mask)

    # Resize the masked image so the comparison runs faster
    # The width is set to 200 pixels while keeping the same aspect ratio
    gray = cv2.resize(masked, (200, int(masked.shape[0] * 200 / masked.shape[1])))

    # Convert the resized image to grayscale
    # The algorithm only compares brightness changes, not color changes
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale image to reduce noise and small lighting changes
    # The 11 by 11 kernel smooths the image but still keeps large motion visible
    gray = cv2.GaussianBlur(gray, (9, 9), 0) # play with kernel size

    # Return both images:
    # masked = useful for showing the region of interest
    # gray = used by the actual detection comparison
    return masked, gray

# Counter variable for analysis
# This tracks how many detection loops have run
counter = 0

# Mask calibration
# These lines can be used if I want to load a calibration image and manually pick mask points
# img = cv2.imread("calibration.jpg")
# select_points(img)

# Create the Picamera2 camera object
picam2 = Picamera2()

# Configure the Pi camera
# Resolution is 1280 by 720
# hflip and vflip are used because of how the camera is physically mounted
camera_config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    transform=Transform(hflip=1, vflip=1)
)

# Apply the camera configuration and start the camera
picam2.configure(camera_config)
picam2.start()

# Give the camera one second to initialize before taking images
time.sleep(1)

try:
    # Main loop
    # The smart doorbell keeps running until the user stops it
    while True:

        # Allows the user to stop the loop by pressing q
        if cv2.waitKey(1) == ord('q'):
            break

        # Increase and print the loop counter
        # This helps track how many detection cycles have happened
        counter += 1
        print(" ")
        print("----Times through loop since starting:", counter, "----")
        print(" ")

        # take a 1st and 2nd image to compare

        # Capture the first image
        # This acts as the "before" image
        picam2.capture_file("test1.jpg")

        # Wait 2 seconds before taking the second image
        # This delay gives a person or object time to enter the detection area
        time.sleep(2)

        # Capture the second image
        # This acts as the "after" image
        picam2.capture_file("test2.jpg")
        
        print("Captured 1st & 2nd image for analysis...")

        # mask images

        # Read the first and second images back into OpenCV
        test1 = cv2.imread("test1.jpg")

        # Small delay before reading the second image
        time.sleep(3)

        test2 = cv2.imread("test2.jpg")

        # Apply the mask/preprocessing function to both images
        # masked1 and masked2 show the selected region in color
        # gray1 and gray2 are the processed images used for comparison
        masked1, gray1 = mask_image(test1)
        masked2, gray2 = mask_image(test2)

        # compare the two images

        # Pixel threshold controls how different a pixel must be before it counts as changed
        # A value of 50 ignores small lighting/camera noise but still detects larger motion
        pixel_threshold = 50

        # Initialize the total detection score
        detector_total = np.uint64(0)

        # Create a blank detector image
        # White pixels will represent areas where enough change was detected
        detector = np.zeros((gray2.shape[0], gray2.shape[1]), dtype="uint8")

        # pixel by pixel comparison

        # Loop through every pixel in the processed grayscale images
        for i in range(0, gray2.shape[0]):
            for j in range(0, gray2.shape[1]):

                # Compare the brightness of each pixel between image 1 and image 2
                # If the absolute difference is greater than the pixel threshold,
                # that pixel is marked as changed
                if abs(int(gray2[i, j]) - int(gray1[i, j])) > pixel_threshold:
                    detector[i, j] = 255

        # sum the detector array

        # Add up all changed pixels
        # Larger detector_total means more motion/change happened in the masked region
        detector_total = np.uint64(np.sum(detector))
        print("detector_total = ", detector_total)
        print(" ")

        # If the total amount of change is high enough, trigger the smart doorbell
        # I chose 45000 because no motion was near 0 and walking into frame was over 100000
        if detector_total > 45000:

            print("Smart Doorbell has detected someone/something at the door!")

            # define a unique name for the new video file

            # Create a timestamped file name so each detection video is unique
            timestr = time.strftime("doorbell-%Y%m%d-%H%M%S")

            # Set up the H264 video encoder
            encoder = H264Encoder()

            # Start recording a video to an h264 file
            picam2.start_recording(encoder, FileOutput(f"{timestr}.h264"))

            # Record for 7 seconds after detection
            # This gives enough time to capture the event after motion is detected
            time.sleep(7)

            # Stop recording after 7 seconds
            picam2.stop_recording()

            print("Finished recording...converting to mp4...")
            
            # Convert the h264 video into mp4 format using ffmpeg
            # MP4 is easier to open and show for the assignment video
            command3 = f'ffmpeg -framerate 30 -i {timestr}.h264 -c copy {timestr}.mp4'
            os.system(command3)

            print("Finished converting file...available for viewing")

            # write masked images to file

            # Save the processed grayscale images
            # These show what the algorithm compared internally
            cv2.imwrite("gray1.jpg", gray1)
            cv2.imwrite("gray2.jpg", gray2)

            # Save the masked color images
            # These show the user-defined detection area
            cv2.imwrite("masked1.jpg", masked1)
            cv2.imwrite("masked2.jpg", masked2)

            # upload video file to the cloud

            # Path to the MP4 video file
            # The upload section is currently commented out below
            fullDirectory = '/home/pi/Documents/ENME435/HW/HW7/' + timestr + '.mp4'

           # command4 = '/home/pi/dropbox_uploader.sh upload ' + fullDirectory + ' /'
           # os.system(command4)

            # send email to user

            # Sender Gmail account
            smtpUser = 'peam.affiliate@gmail.com'

            # Gmail app password for the sender account
            # Paste your app password here locally before running
            smtpPass = 'PASTE_YOUR_APP_PASSWORD_HERE'

            # Email recipient and sender
            toAdd = 'peampats@gmail.com'
            fromAdd = smtpUser

            # Create a timestamp for the email
            f_time = datetime.now().strftime('%a %d %b @ %H:%M')

            # Email subject line
            subject = 'Smart Doorbell Images Detected: ' + f_time

            # Create a multipart email so text and images can be attached together
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = fromAdd
            msg['To'] = toAdd
            
            # Email preamble
            msg.preamble = 'Image @ ' + f_time

            #body = email.mime.Text.MIMEText('Smart Doorbell video: ' + f_time)

            # Email body text
            body = MIMEText('Motion Detected! - ' + f_time)
            msg.attach(body)

            # Attach the original first image
            fp = open('test1.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            # Attach the original second image
            fp = open('test2.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            # Attach the processed grayscale first image
            fp = open('gray1.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            # Attach the processed grayscale second image
            fp = open('gray2.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            # Attach the masked first image
            fp = open('masked1.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            # Attach the masked second image
            fp = open('masked2.jpg', 'rb')
            img = MIMEImage(fp.read())
            fp.close()
            msg.attach(img)

            # Connect to Gmail SMTP server
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.ehlo()

            # Start TLS encryption for secure login
            s.starttls()
            s.ehlo()

            # Log in to the sender Gmail account
            s.login(smtpUser, smtpPass)

            # Send the email from the sender account to the recipient
            s.sendmail(fromAdd, toAdd, msg.as_string())

            # Close the SMTP connection
            s.quit()

            print("Email Sent")

        else:
            # If the total change is not high enough, no detection is triggered
            print("No Detections")

# Allows the user to stop the program safely with Ctrl+C
except KeyboardInterrupt:
    print("Stopped")

# Stop the Pi camera when the program ends
picam2.stop()
