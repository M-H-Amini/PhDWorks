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
from git_dave2_model import loadModel as loadModelGit
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

def loadModel(model='git'):
    if 'udacity' in model.lower():
        return load_model('mh_dave2_udacity')
    elif 'git' in model.lower():
        return loadModelGit('DAVE2-Keras-master/model.h5')
    elif 'beamng' in model.lower():
        return load_model('mh_dave2_beamng')

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
 

##  Datasets...

##  Udacity Dataset...
dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
train_cols = ['center']
# train_cols = ['center', 'left', 'right']
transform_u = lambda x: x[70:136, 100:300, :]
X_train_u, y_train_u, meta_train_u, X_val_u, y_val_u, meta_val_u, X_test_u, y_test_u, meta_test_u = prepareDatasetUdacity(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.7, test_size=0.1, val_size=0.1, transform=transform_u, show=False)
X_udacity = np.concatenate((X_train_u, X_val_u, X_test_u), axis=0)
y_udacity = np.concatenate((y_train_u, y_val_u, y_test_u), axis=0)
log.info(f'X_udacity shape: {X_udacity.shape}, y_udacity shape: {y_udacity.shape}')
log.info(f'X_train shape: {X_train_u.shape}, y_train shape: {y_train_u.shape}')
log.info(f'X_valid shape: {X_val_u.shape}, y_valid shape: {y_val_u.shape}')
log.info(f'X_test shape: {X_test_u.shape}, y_test shape: {y_test_u.shape}')

##  BeamNG Dataset...
transform_b = lambda x: x[130-66:130, 60:260, :]  ##  Crop the image
json_folder = 'ds_beamng'
test_size = 0.1
val_size = 0.1
step = 15
X_train_b, y_train_b, meta_train_b, X_val_b, y_val_b, meta_val_b, X_test_b, y_test_b, meta_test_b = prepareDatasetBeamNG(json_folder, step=step, test_size=test_size, val_size=val_size, random_state=28, transform=transform_b, show=False)
X_beamng = np.concatenate((X_train_b, X_val_b, X_test_b), axis=0)
y_beamng = np.concatenate((y_train_b, y_val_b, y_test_b), axis=0)
log.info(f'X_beamng shape: {X_beamng.shape}, y_beamng shape: {y_beamng.shape}')
log.info(f'X_train shape: {X_train_b.shape}, y_train shape: {y_train_b.shape}')
log.info(f'X_valid shape: {X_val_b.shape}, y_valid shape: {y_val_b.shape}')
log.info(f'X_test shape: {X_test_b.shape}, y_test shape: {y_test_b.shape}')

log.info('Evaluating models...')

models = {key: loadModel(key) for key in ['Dave2Udacity', 'Dave2BeamNG', 'Dave2Git']}
ds = {
        'udacity': {
            'total': (X_udacity, y_udacity),
            'test': (X_test_u, y_test_u),
        },
        'beamng': {
            'total': (X_beamng, y_beamng),
            'test': (X_test_b, y_test_b),
        }
    }
df_eval = {key: [] for key in models.keys()}

for i, model_name in enumerate(models.keys()):
    model = models[model_name]
    for j, ds_name in enumerate(ds.keys()):
        if i == j:
            df_eval[model_name].append(evaluate(train=None, val=None, test=ds[ds_name]['test'], model=model, verbose=False)['test'])
        else:
            df_eval[model_name].append(evaluate(train=ds[ds_name]['total'], val=None, test=None, model=model, verbose=False)['train'])

df_eval = pd.DataFrame(df_eval, index=['UdacityJungle', 'Beamng']).T
df_eval.to_latex('eval_dave2_offline.tex', float_format='%.3f')
