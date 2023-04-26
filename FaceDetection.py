# reference: https://www.geeksforgeeks.org/opencv-python-program-face-detection/

import cv2 
import numpy as np 
import tensorflow as tf 


# https://stackoverflow.com/questions/74670850/face-detection-without-cutting-the-head
def increase_rectangle_size(p, inc_per):
    x = (p[0] - p[2]) * inc_per // 100
    y = (p[1] - p[3]) * inc_per // 100

    new_points = [p[0] + x, p[1] + y, p[2] - x, p[3] - y]

    return [(i > 0) * i for i in new_points]  # Negative numbers to zeros.


#it will reaturn a rectangle which will be used in GrabCut. 
def face_detect(image):

    # so far, we would like to segement the face area. If we want any other part of body, we can change the xml files. 

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # image_path = "Images/style/myface.jpg"
    # box_image = cv2.imread(image_path)


    content_copy  = tf.identity(image)


    # we would like to change the image type 
    content_copy = tf.image.convert_image_dtype(content_copy, tf.uint8)
    content_copy = np.squeeze(content_copy)


    # box_image=np.uint8(image*255.)


    # make it grayscale 
    gray = cv2.cvtColor(content_copy, cv2.COLOR_BGR2GRAY)

    # Detects faces 
    faces = face_cascade.detectMultiScale(gray)


    # we want the rectangle for GrabCut 
    rectangle = (faces[1][0], faces[1][1], faces[1][2], faces[1][3])


    return rectangle



# the follwing is draw rectangle in the face. 
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(content_copy,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = content_copy[y:y+h, x:x+w]
        # cv2_imshow(roi_color)
        # break


    # in case we want change the eyes part 
    # eyes = eye_cascade.detectMultiScale(roi_gray) 

    # #To draw a rectangle in eyes
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)





