#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                Title: Dave2 Evaluation                              ##
##                                   Date: 2023/05/11                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging as log
from mh_dave2_data import prepareDataset, imageGenerator

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
model = load_model('mh_dave2')
log.info('Model loaded successfully')

##  Evaluation...
log.info('Evaluating model on train, val and test sets...')
loss_train = model.evaluate(X_train, y_train, steps=len(X_train)//128)
log.info(f'Train Loss: {loss_train}')
loss_val = model.evaluate(X_val, y_val, steps=len(X_val)//128)
log.info(f'Val Loss: {loss_val}')
loss_test = model.evaluate(X_test, y_test, steps=len(X_test)//128)
log.info(f'Test Loss: {loss_test}')
