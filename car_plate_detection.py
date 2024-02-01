


import cv2

# Path to the Haar Cascade classifier XML file for license plate detection
harcascade = r"C:\Users\tanji\Desktop\myPW\end to endprojects\AI-CNN\car-plate\model\haarcascade_russian_plate_number.xml"

# Accessing the webcam (index 0) using VideoCapture
cap = cv2.VideoCapture(0)

# Setting the dimensions for the video stream
# The parameters 3 and 4 correspond to CV_CAP_PROP_FRAME_WIDTH and CV_CAP_PROP_FRAME_HEIGHT, respectively.
# suitable for webcam have a resolution of 640x480 pixels or more
cap.set(3, 640) # width
cap.set(4, 480) # height

# # 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# # 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# # 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
# # 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# # 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# # 5. CV_CAP_PROP_FPS Frame rate.
# # 6. CV_CAP_PROP_FOURCC 4-character code of codec.
# # 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
# # 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# # 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# # 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# # 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# # 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
# # 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
# # 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
# # 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
# # 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# # 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
# # 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

# Minimum area for a detected region to be considered a license plate
min_area = 500
count = 1

while True:
    # Reading frames from the webcam
    success, img = cap.read()

    # Creating a license plate classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)
    
    # Converting the frame to grayscale for easier processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting license plates in the grayscale image
    # The parameter 1.1 refers to the scaleFactor. This parameter specifies how much the image size is reduced at each image scale. 
    # In simple terms, it helps in resizing the image for multiscale detection
    # The parameter 4 refers to minNeighbors. This parameter specifies how many neighbors each candidate rectangle should have to retain it. 
    # It helps in filtering out false positives
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Iterating through the detected plates
    for (x, y, w, h) in plates:
        area = w * h

        # Checking if the detected area is larger than the minimum area
        if area > min_area :
            # Drawing a rectangle around the detected license plate
           
            #(x, y): These are the coordinates of the top-left corner of the rectangle.
            #(x+w, y+h): These are the coordinates of the bottom-right corner of the rectangle.
            #(0, 255, 0): This is the color of the rectangle in BGR format. In this case, it's a green rectangle because it's represented as (0, 255, 0).
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Adding a label for the detected plate
            # (x, y-5): These are the coordinates specifying where the text will be placed. (x, y-5) indicates the position slightly above the top-left corner of the rectangle (specifically, 5 pixels above y).
            #1: This parameter is the font scale factor. It denotes the size of the font. Here, it's set to 1, indicating the standard size.
            #(255, 0, 255): This is the color of the text in BGR format. (255, 0, 255) represents magenta, with intensities of blue (255), green (0), and red (255).
            #2: This parameter denotes the thickness of the text stroke
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extracting the region of interest (ROI) for the license plate
            img_roi = img[y: y+h, x:x+w]
            
            # Displaying the ROI in a separate window
            cv2.imshow("ROI", img_roi)

    # Displaying the result with detected plates
    cv2.imshow("Result", img)

    # Saving the plate when 'esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  #ascii of esc is 27
        # Writing the plate image to a file with a count in the filename
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        
        # Displaying a message that the plate is saved
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        
        # Waiting for 500 milliseconds before continuing
        cv2.waitKey(500)
        
        # Incrementing the count for the next saved plate
        count += 1
