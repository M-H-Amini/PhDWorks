#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                 Title: Training Dave2                               ##
##                                   Date: 2023/05/11                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 


import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, BatchNormalization, Activation, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging as log
from PIL import Image

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plotHistogram(df, col='steering', show=True):
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.hist(df['steering'], bins=100)
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    show and plt.show()


##  Dataset...
dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
train_cols = ['center']  ##  ['center', 'left', 'right']
df = pd.read_csv(dataset_csv, header=None, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
df[train_cols] = df[train_cols].applymap(lambda x: os.path.join(dataset_folder, x.split('\\')[-1]))
df[train_cols] = df[train_cols].applymap(lambda x: np.asarray(Image.open(x)))

##  Reducing 0 steering angles...
log.info(f'Imbalanced dataset size: {len(df)}')
plotHistogram(df)
reduce_ratio = 0.9
df = df.drop(df[df['steering'] == 0].sample(frac=reduce_ratio, random_state=28).index).reset_index(drop=True)
log.info(f'Balanced dataset size: {len(df)}')
plotHistogram(df)

##  Splitting dataset...
train_df, test_df = train_test_split(df, test_size=0.2, random_state=28)
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=28)
log.info(f'Train dataset size: {len(train_df)}')
log.info(f'Valid dataset size: {len(valid_df)}')
log.info(f'Test dataset size: {len(test_df)}')
X_train, y_train = train_df[train_cols].values, train_df['steering'].values
X_valid, y_valid = valid_df[train_cols].values, valid_df['steering'].values
X_test, y_test = test_df[train_cols].values, test_df['steering'].values
X_train = np.array([xx for x in X_train for xx in x])  ##  (N, 160, 320, 3)
X_valid = np.array([xx for x in X_valid for xx in x])  ##  (N, 160, 320, 3)
X_test = np.array([xx for x in X_test for xx in x])  ##  (N, 160, 320, 3)
