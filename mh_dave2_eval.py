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
ds_names = ['udacity', 'beamng', 'saevae']  #  , 'dclgan', 'saevae', 'magenta']
df_index = ['UdacityJungle', 'BeamNG', 'SAEVAE']
ds = [loadDataset(ds_name) for ds_name in ds_names]
func_map = lambda x: {'total': (np.concatenate((x[0], x[4]), axis=0), np.concatenate((x[1], x[5]), axis=0)), 'test': (x[4], x[5])}
ds = map(func_map, ds)
ds = {ds_name: ds_ for ds_name, ds_ in zip(ds_names, ds)}
log.info('Evaluating models...')

models = {key: loadModel(key) for key in ['Dave2Udacity', 'Dave2BeamNG']}
df_eval = {key: [] for key in models.keys()}

for i, model_name in enumerate(models.keys()):
    model = models[model_name]
    for j, ds_name in enumerate(ds.keys()):
        if i == j and i < 2:  ##  i < 2 to test all of DCLGAN for Dave2Git
            print(f'{model_name} on test part of {ds_name}')
            df_eval[model_name].append(evaluate(train=None, val=None, test=ds[ds_name]['test'], model=model, verbose=False)['test'])
        else:
            print(f'{model_name} on total part of {ds_name}')
            df_eval[model_name].append(evaluate(train=ds[ds_name]['total'], val=None, test=None, model=model, verbose=False)['train'])

df_eval = pd.DataFrame(df_eval, index=df_index).T
df_eval.to_latex('eval_dave2_offline.tex', float_format='%.3f')
