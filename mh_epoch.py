from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
# from keras.layers.advanced_activations import ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

def generateModel(input_shape=(160, 320, 3)):
    img_input = Input(input_shape)

    x = Conv2D(32, 3, 3, activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, 3, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, 3, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    y = Flatten()(x)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(inputs=img_input, outputs=y)
    return model


def build_InceptionV3(image_size=None,weights_path=None):
    image_size = image_size or (299, 299)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )
    bottleneck_model = InceptionV3(weights='imagenet',include_top=False, 
                                   input_tensor=Input(input_shape))
    for layer in bottleneck_model.layers:
        layer.trainable = False

    x = bottleneck_model.input
    y = bottleneck_model.output
    # There are different ways to handle the bottleneck output
    y = GlobalAveragePooling2D()(x)
    #y = AveragePooling2D((8, 8), strides=(8, 8))(x)
    #y = Flatten()(y)
    #y = BatchNormalization()(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(input=x, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
    return model

if __name__ == '__main__':
    model = generateModel()
    model.summary()