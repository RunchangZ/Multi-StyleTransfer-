import cv2 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib as mpl
import processing as pr 
import hyperparameters as hp


def main():
    

    content_path  = "Images/style/taylor.jpg"
    style_path = "Images/style/ayan_face1/ben-crop.png"

    content_image = pr.load_img(content_path)
    style_image = pr.load_img(style_path)


# show the images
    plt.subplot(1, 2, 1)
    pr.imshow(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    pr.imshow(style_image, 'Style Image')




# we woul




