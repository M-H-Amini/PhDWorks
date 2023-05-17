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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging as log
from PIL import Image
from mh_dave2_data import imageGenerator
from mh_dave2_data import prepareDataset as prepareDatasetUdacity
from mh_beamng_ds import prepareDataset as prepareDatasetBeamNG
from mh_dave2_model import generateModel

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  GPU...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    log.info(f'GPUs: {gpus}')
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    except RuntimeError as e:
        log.error(e)
else:
    log.info('No GPUs found')

dataset = 'beamng'  ##  'udacity' or 'beamng'
model_name = f'mh_dave2_{dataset}'
epochs = 20
batch_size = 128

##  Dataset...
if dataset == 'udacity':
    dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
    dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
    train_cols = ['center']
    transform = lambda x: x[70:136, 100:300, :]
    reduce_ratio = 0.7
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetUdacity(dataset_folder, dataset_csv, train_cols, reduce_ratio=reduce_ratio, test_size=0.1, val_size=0.1, transform=transform, show=True)
elif dataset == 'beamng':
    json_folder = 'ds_beamng'
    test_size = 0.1
    val_size = 0.1
    step = 15
    transform = lambda x: x[130-66:130, 60:260, :]  ##  Crop the image
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetBeamNG(json_folder, step=step, test_size=test_size, val_size=val_size, random_state=28, transform=transform, show=False)
else:
    raise ValueError('Dataset not found')

log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
log.info(f'X_valid shape: {X_val.shape}, y_valid shape: {y_val.shape}')
log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
gen_train = imageGenerator(X_train, y_train, batch_size=batch_size)
gen_val = imageGenerator(X_val, y_val, batch_size=batch_size)

##  Model...
image_shape = X_train.shape[1:]
model = generateModel(image_shape)
model.compile(optimizer=Adam(lr=1e-4), loss='mae')
ckpt = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
##  Training...
history = model.fit(gen_train, epochs=epochs, steps_per_epoch=len(X_train)//batch_size, validation_data=gen_val, validation_steps=len(X_val)//batch_size, callbacks=[ckpt])

##  Plotting training history...
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(epochs))
plt.title('Training History')
plt.savefig(f'{model_name}.pdf')
plt.show()

