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
from mh_dave2_data import prepareDataset, imageGenerator
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


##  Dataset...
dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
train_cols = ['center']
transform = lambda x: x[60:150, :, :]
# train_cols = ['center', 'left', 'right']
X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDataset(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.2, test_size=0.1, val_size=0.1, transform=transform, show=True)
log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
log.info(f'X_valid shape: {X_val.shape}, y_valid shape: {y_val.shape}')
log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
gen_train = imageGenerator(X_train, y_train, batch_size=128)
gen_val = imageGenerator(X_val, y_val, batch_size=128)

##  Model...
image_shape = X_train.shape[1:]
model = generateModel(image_shape)
model.compile(optimizer=Adam(lr=1e-4), loss='mae')
ckpt = ModelCheckpoint('mh_dave2', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
##  Training...
history = model.fit(gen_train, epochs=20, steps_per_epoch=len(X_train)//128, validation_data=gen_val, validation_steps=len(X_val)//128, callbacks=[ckpt])

##  Plotting training history...
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training History')
plt.savefig('mh_dave2_train.pdf')
plt.show()

