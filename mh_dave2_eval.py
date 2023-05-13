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
from mh_dave2_data import prepareDataset
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
    loss_train = model.evaluate(train[0], train[1], steps=len(train[0])//128)
    verbose and log.info(f'Train Loss: {loss_train}')
    loss_val = model.evaluate(val[0], val[1], steps=len(val[0])//128)
    verbose and log.info(f'Val Loss: {loss_val}')
    loss_test = model.evaluate(test[0], test[1], steps=len(test[0])//128)
    verbose and log.info(f'Test Loss: {loss_test}')
    return {'train': loss_train, 'val': loss_val, 'test': loss_test}
 

##  Dataset...
dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
train_cols = ['center']
# train_cols = ['center', 'left', 'right']
transform = lambda x: x[70:136, 100:300, :]
X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDataset(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.7, test_size=0.1, val_size=0.1, transform=transform, show=False)
log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
log.info(f'X_valid shape: {X_val.shape}, y_valid shape: {y_val.shape}')
log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


##  Evaluation...
log.info('Evaluating model on train, val and test sets...')
eval = partial(evaluate, train=(X_train, y_train), val=(X_val, y_val), test=(X_test, y_test), verbose=False)
models = {'mh': load_model('mh_dave2'), 'git': loadModel('DAVE2-Keras-master/model.h5')}
losses = {model_type: eval(model=model) for model_type, model in models.items()}
df_eval = pd.DataFrame(losses, index=['train', 'val', 'test']).T
df_eval.to_latex('eval.tex', float_format='%.3f')
print(df_eval)
