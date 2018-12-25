# writing a simple face detection proram
# import modules
import cv2
import numpy as np


# Now we load in the facial classifiers
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

# now we create a video capture object to capture videos from webcam
cap = cv2.VideoCapture(0);

# we create a loop to capture video frame by frame
while True:
    ret,image = cap.read();
    # now we need to convert the image to gray for the classifier to work
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    # now we create list to store the faces we detected in image frame by frame
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    # now we detect multiple faces and draw rectangle on them
    for (x,y,w,h) in faces:
             # now we draw rectangle on the original image not the gray
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)


        # now we show in a window
    cv2.imshow('Frame', image)

        # Now to close the program we need to specify a waitkey and compare to a key to close the frame
    k= cv2.waitKey(30) & 0xFF == ord('q')
    if k == 27:
        break


# Now we release the camera and destroy all of it







