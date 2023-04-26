
import hyperparameters as hp
import cv2 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib as mpl


#show the image in matplotlib. 
def imshow(image, title=None):
    if len(image.shape) > 3:
       image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)


# open the image with given path, with tensorflow
def load_img(path):

    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    new_shape = tf.cast(shape *  512 / max(shape), tf.int32)
    image = tf.image.resize(image, new_shape)

    return image[tf.newaxis, :]






