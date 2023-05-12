#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                 Title: Dave2 Model                                  ##
##                                   Date: 2023/05/11                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, ELU, Dropout, Lambda, Conv2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging as log
from PIL import Image
from mh_dave2_data import prepareDataset

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

if __name__ == "__main__":
    model = generateModel()
    model.compile(optimizer="adam", loss="mse")
    model.summary()