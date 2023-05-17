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
from mh_dave2_data import prepareDataset as prepareDatasetUdacity
from mh_beamng_ds import prepareDataset as prepareDatasetBeamNG
from git_dave2_model import loadModel
from functools import partial

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

def evaluate(train, val, test, model, verbose=True):
    res = {}
    if train:
        loss_train = model.evaluate(train[0], train[1], steps=len(train[0])//128)
        verbose and log.info(f'Train Loss: {loss_train}')
        res['train'] = loss_train
    if val:
        loss_val = model.evaluate(val[0], val[1], steps=len(val[0])//128)
        verbose and log.info(f'Val Loss: {loss_val}')
        res['val'] = loss_val
    if test:
        loss_test = model.evaluate(test[0], test[1], steps=len(test[0])//128)
        verbose and log.info(f'Test Loss: {loss_test}')
        res['test'] = loss_test
    return res
 

##  Evaluation on Udacity Dataset...

##  Dataset...
dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
train_cols = ['center']
# train_cols = ['center', 'left', 'right']
transform = lambda x: x[70:136, 100:300, :]
X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetUdacity(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.7, test_size=0.1, val_size=0.1, transform=transform, show=False)
log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
log.info(f'X_valid shape: {X_val.shape}, y_valid shape: {y_val.shape}')
log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


log.info('Evaluating model on train, val and test sets...')
eval = partial(evaluate, train=(X_train, y_train), val=(X_val, y_val), test=(X_test, y_test), verbose=False)
models = {'Dave2Scratch': load_model('mh_dave2'), 'Dave2Git': loadModel('DAVE2-Keras-master/model.h5')}
losses = {model_type: eval(model=model) for model_type, model in models.items()}
df_eval = pd.DataFrame(losses, index=['train', 'val', 'test']).T
df_eval.to_latex('eval_udacity.tex', float_format='%.3f')
print('Evaluation on Udacity Dataset...')
print(df_eval)

##  Evaluation on BeamNG Dataset...
transform = lambda x: x[130-66:130, 60:260, :]  ##  Crop the image
json_folder = 'ds_beamng'
test_size = 0.1
val_size = 0.1
step = 15
X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetBeamNG(json_folder, step=step, test_size=test_size, val_size=val_size, random_state=28, transform=transform, show=False)
X = np.concatenate((X_train, X_val, X_test), axis=0)
y = np.concatenate((y_train, y_val, y_test), axis=0)
log.info(f'X shape: {X.shape}, y shape: {y.shape}')
log.info('Evaluating model on BeamNG dataset...')
eval = partial(evaluate, train=(X, y), val=None, test=None, verbose=False)
models = {'Dave2Scratch': load_model('mh_dave2'), 'Dave2Git': loadModel('DAVE2-Keras-master/model.h5')}
losses = {model_type: eval(model=model) for model_type, model in models.items()}
df_eval = pd.DataFrame(losses, index=['train']).T
df_eval.to_latex('eval_beamng.tex', float_format='%.3f')
print('Evaluation on BeamNG Dataset...')
print(df_eval)
