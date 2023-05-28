import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

class MHStyleMagenta:
    def __init__(self) -> None:
        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def preprocess(self, image):
        return tf.image.resize(image[np.newaxis, ...] / 255., (256, 256))
    
    def __call__(self, content_image, style_image):
        content_image = self.preprocess(content_image)
        style_image = self.preprocess(style_image)
        stylized_image = self.model(tf.constant(content_image), tf.constant(style_image))[0]
        stylized_image = tf.squeeze(stylized_image)
        return stylized_image

if __name__ == '__main__':
    ##  Load images...
    content_image_path = 'dog.jpg'
    style_image_path = 'style.jpg'
    content_image = plt.imread(content_image_path)
    style_image = plt.imread(style_image_path)
    ##  Initialize model...
    model = MHStyleMagenta()
    stylized_image = model(content_image, style_image)
    plt.imshow(stylized_image)
    plt.axis('off')
    plt.show()
    plt.imsave('stylized_dog.jpg', stylized_image.numpy())

