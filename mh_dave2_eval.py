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
from mh_ds import loadDataset
from git_dave2_model import loadModel as loadModelGit
from functools import partial
import os

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
    if 'git' in model.lower():
        return loadModelGit('DAVE2-Keras-master/model.h5')
    else:
        return load_model(model)

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
ds_names = ['udacity', 'beamng', 'saevae', 'cycle']  #  , 'dclgan', 'saevae', 'magenta']
df_index = ['UdacityJungle', 'BeamNG', 'SAEVAE', 'Cycle']
ds = [loadDataset(ds_name) for ds_name in ds_names]
func_map = lambda x: {'total': (np.concatenate((x[0], x[4]), axis=0), np.concatenate((x[1], x[5]), axis=0)), 'test': (x[4], x[5])}
ds = map(func_map, ds)
ds = {ds_name: ds_ for ds_name, ds_ in zip(ds_names, ds)}
log.info('Evaluating models...')

model_folders = []
model_names = []
##  Loading dave2 models...
# for model in os.listdir('models'):
#     # if 'autumn' in model:
#     os.path.isdir((model_folder := os.path.join('models', model))) and (model_folders.append(model_folder) or model_names.append(model_folder[10:]))

model_folders.extend(['mh_cnn_udacity', 'mh_chauffeur_udacity', 'mh_epoch_udacity', 'mh_autumn_udacity', 'mh_dave2_udacity', 'mh_dave2_beamng'])
model_names.extend(['CNNUdacity', 'ChauffeurUdacity', 'EpochUdacity', 'AutumnUdacity', 'Dave2Udacity', 'Dave2BeamNG'])


models = {key: loadModel(key) for key in model_folders}
df_eval = {key: [] for key in models.keys()}

for i, model_key in enumerate(models.keys()):
    model = models[model_key]
    model_name = model_names[i]
    for j, ds_name in enumerate(ds.keys()):
        if 'beamng' in model_key.lower() and 'beamng' in ds_name.lower():  ##  To use test part of beamng dataset for beamng models
            print(f'{model_name} on test part of {ds_name}, shape: {ds[ds_name]["test"][0].shape}')
            df_eval[model_key].append(evaluate(train=None, val=None, test=ds[ds_name]['test'], model=model, verbose=False)['test'])
        else:
            print(f'{model_name} on total part of {ds_name}')
            df_eval[model_key].append(evaluate(train=ds[ds_name]['total'], val=None, test=None, model=model, verbose=False)['train'])

df_eval = pd.DataFrame(df_eval, index=df_index).T
df_eval.rename(index={model_folders[i]:model_names[i] for i in range(len(model_folders))}, inplace=True)

df_eval.to_latex('eval_dave2_offline.tex', float_format='%.3f')
df_eval.to_csv('eval_dave2_offline.csv', float_format='%.3f')
