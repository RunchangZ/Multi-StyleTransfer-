# https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/


import cv2 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

import FaceDetection as faceD 
import os 





def grabCut(image):

    # based on the way we load image, we have to squeeze the first axis first
    image = np.squeeze(image) 
    #also, we need to change the datatype
    image=np.uint8(image*255.)


    mask = np.zeros(image.shape[:2], np.uint8)

    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)


    # we find the area that we want to cut by openCV face detection. 
    rectangle = faceD(image)

    cv2.grabCut(image, mask, rectangle,backgroundModel, foregroundModel,3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')

    return mask2 

    # # The final mask is multiplied with
    # # the input image to give the
    image = image * mask2[:, :, np.newaxis]


    # in case we want to see the segmented image.
    plt.imshow(image)
    plt.colorbar()
    plt.show()


    # in case we want to save the image 
    os.chdir("Images/style")
    cv2.imwrite("input_content.jpg", image)