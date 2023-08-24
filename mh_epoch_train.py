#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                 Title: Dave2 Training                               ##
##                                   Date: 2023/05/11                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging as log
from PIL import Image
from mh_dave2_data import imageGenerator
from mh_epoch import generateModel
from mh_ds import loadDataset


log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  GPU...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    log.info(f'GPUs: {gpus}')
    try:
        tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
    except RuntimeError as e:
        log.error(e)
else:
    log.info('No GPUs found')


dataset = 'udacitybeamng'  ##  'udacity' or 'beamng'
model_name = f'mh_epoch_{dataset}'
epochs = 50
batch_size = 32

##  Dataset...
X_train, y_train, X_val, y_val, X_test, y_test = loadDataset(dataset)

log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
log.info(f'X_valid shape: {X_val.shape}, y_valid shape: {y_val.shape}')
log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
datagen = tf.keras.preprocessing.image.ImageDataGenerator()

datagen_train = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=42, ignore_class_split=True)
datagen_val = datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=True, seed=42, ignore_class_split=True)
datagen_test = datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=True, seed=42, ignore_class_split=True)

##  Model...
image_shape = X_train.shape[1:]
model = generateModel(image_shape)
model.compile(optimizer='adam', loss='mae')
model.summary()
ckpt = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
history = model.fit(datagen_train, epochs=epochs, validation_data=datagen_val, callbacks=[ckpt])

##  Training...
model = tf.keras.models.load_model(model_name)
log.info('\033[92m' + 'Model loaded!' + '\033[0m')

##  Evaluation...
log.info('\033[92m' + 'Evaluating...' + '\033[0m')
loss_train = model.evaluate(X_train, y_train, batch_size=batch_size)
log.info(f'Train Loss: {loss_train}')
loss_val = model.evaluate(X_val, y_val, batch_size=batch_size)
log.info(f'Validation Loss: {loss_val}')
loss_test = model.evaluate(X_test, y_test, batch_size=batch_size)
log.info(f'Test Loss: {loss_test}')

##  Plotting training history...
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(epochs))
plt.ylim([0, 1])
plt.title('Training History')
plt.savefig(f'{model_name}.pdf')
plt.show()

