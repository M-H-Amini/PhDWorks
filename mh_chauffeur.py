from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SpatialDropout2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import RootMeanSquaredError

def generateModel(input_shape, W_l2=0.0001):
    model = Sequential([
            Conv2D(16, (5, 5),
                input_shape=input_shape,
                kernel_initializer='he_normal',
                activation='relu',
                padding='same'),
            SpatialDropout2D(0.1),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(20, (5, 5),
                kernel_initializer='he_normal',
                activation='relu',
                padding='same'),
            SpatialDropout2D(0.1),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(40, (3, 3),
                kernel_initializer='he_normal',
                activation='relu',
                padding='same'),
            SpatialDropout2D(0.1),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(60, (3, 3),
                kernel_initializer='he_normal',
                activation='relu',
                padding='same'),
            SpatialDropout2D(0.1),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(80, (2, 2),
                kernel_initializer='he_normal',
                activation='relu',
                padding='same'),
            SpatialDropout2D(0.1),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (2, 2),
                kernel_initializer='he_normal',
                activation='relu',
                padding='same'),
            Flatten(),
            Dropout(0.5),
            Dense(1,
                # kernel_initializer='he_normal',
                # kernel_regularizer=l2(W_l2)
                )
        ])
    return model


if __name__ == '__main__':
    model = generateModel((160, 320, 3))
    model.compile(loss='mse', optimizer=SGD(lr=0.001), metrics=[RootMeanSquaredError()])
    model.summary()