#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                  Title: Github Model                                ##
##                                   Date: 2023/05/13                                  ##
##                                                                                     ##
#########################################################################################

##  Description: Load the pretrained model from github


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, ELU, Dropout, Lambda, Conv2D
from tensorflow.keras.models import Sequential
import logging as log
import os

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  GPU...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    log.info(f'GPUs: {gpus}')
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    except RuntimeError as e:
        log.error(e)

def generateModel(input_shape=(160, 320, 3)):
    model = Sequential()
    # Input normalization layer
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name='lambda_norm'))

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding="valid", name='conv1'))
    model.add(ELU(name='elu1'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding="valid", name='conv2'))
    model.add(ELU(name='elu2'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding="valid", name='conv3'))
    model.add(ELU(name='elu3'))

    # 3x3 Convolutional layers with stride of 1x1
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid", name='conv4'))
    model.add(ELU(name='elu4'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid", name='conv5'))
    model.add(ELU(name='elu5'))

    # Flatten before passing to Fully Connected layers
    model.add(Flatten())
    # Three fully connected layers
    model.add(Dense(100, name='fc1'))
    model.add(Dropout(.5, name='do1'))
    model.add(ELU(name='elu6'))
    model.add(Dense(50, name='fc2'))
    model.add(Dropout(.5, name='do2'))
    model.add(ELU(name='elu7'))
    model.add(Dense(10, name='fc3'))
    model.add(Dropout(.5, name='do3'))
    model.add(ELU(name='elu8'))

    # Output layer with tanh activation 
    model.add(Dense(1, activation='tanh', name='output'))
    return model

def loadModel(weights_path):
    model = generateModel((66,200,3))
    model.load_weights(weights_path)
    model.compile(loss='mae', optimizer='adam')
    return model



if __name__ == '__main__':
    weights_path = os.path.join('DAVE2-Keras-master', 'model.h5')
    model = loadModel(weights_path)
    model.summary()
    log.info('Model loaded successfully')


