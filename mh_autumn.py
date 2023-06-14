from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Lambda
import tensorflow as tf

def generateModel(input_shape, dropout_prob=0.2, batch_norm=False, whitening=False, is_training=True):
    ''' Implements the ConvNet model from the NVIDIA paper '''
    x = Input(shape=input_shape, name='x')
    x_image = x

    h_conv1 = Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu')(x_image)
    if batch_norm:
        h_conv1 = BatchNormalization()(h_conv1, training=is_training)

    h_conv2 = Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu')(h_conv1)

    h_conv3 = Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu')(h_conv2)
    if batch_norm:
        h_conv3 = BatchNormalization()(h_conv3, training=is_training)

    h_conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(h_conv3)

    h_conv5 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(h_conv4)
    if batch_norm:
        h_conv5 = BatchNormalization()(h_conv5, training=is_training)

    h_fc1 = Dense(1164, activation='relu')(Flatten()(h_conv5))
    if batch_norm:
        h_fc1 = BatchNormalization()(h_fc1, training=is_training)
    h_fc1_drop = Dropout(dropout_prob)(h_fc1)

    h_fc2 = Dense(100, activation='relu', name='fc2')(h_fc1_drop)
    if batch_norm:
        h_fc2 = BatchNormalization()(h_fc2, training=is_training)
    h_fc2_drop = Dropout(dropout_prob)(h_fc2)

    h_fc3 = Dense(50, activation='relu', name='fc3')(h_fc2_drop)
    if batch_norm:
        h_fc3 = BatchNormalization()(h_fc3, training=is_training)
    h_fc3_drop = Dropout(dropout_prob)(h_fc3)

    h_fc4 = Dense(10, activation='relu', name='fc4')(h_fc3_drop)
    if batch_norm:
        h_fc4 = BatchNormalization()(h_fc4, training=is_training)
    h_fc4_drop = Dropout(dropout_prob)(h_fc4)

    y = Dense(1)(h_fc4_drop)

    model = Model(inputs=x, outputs=y)
    return model

if __name__ == '__main__':
    model = generateModel((160, 320, 3))
    model.summary()