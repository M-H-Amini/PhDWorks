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
import tensorflow_datasets as tfds
from tqdm import tqdm 
import cv2
import os

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
def loadDataset(dataset='udacity'):
    if dataset == 'udacity':
        dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
        dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
        train_cols = ['center']
        # train_cols = ['center', 'left', 'right']
        transform_u = lambda x: x[70:136, 100:300, :] / 255.
        X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetUdacity(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.7, test_size=0.1, val_size=0.1, transform=transform_u, show=False)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'beamng':
        transform_b = lambda x: x[130-66:130, 60:260, :]  / 255.
        json_folder = 'ds_beamng'
        test_size = 0.1
        val_size = 0.1
        step = 15
        X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDatasetBeamNG(json_folder, step=step, test_size=test_size, val_size=val_size, random_state=28, transform=transform_b, show=False)
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = np.concatenate((X_train[..., np.newaxis], X_train[..., np.newaxis], X_train[..., np.newaxis]), axis=-1)
        X_test = np.concatenate((X_test[..., np.newaxis], X_test[..., np.newaxis], X_test[..., np.newaxis]), axis=-1)
        X_train = np.array([tf.image.resize(x, (66, 200)).numpy() / 255. for x in X_train])
        X_test = np.array([tf.image.resize(x, (66, 200)).numpy() / 255. for x in X_test])
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = np.array([tf.image.resize(x, (66, 200)).numpy() / 255. for x in X_train])
        X_test = np.array([tf.image.resize(x, (66, 200)).numpy() / 255. for x in X_test])
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'cats_vs_dogs':
        split = ['train[:70%]', 'train[70%:]']
        ds_train, ds_test = tfds.load(name='cats_vs_dogs', split=split, as_supervised=True)
        ds_train = ds_train.map(lambda x, y: (tf.image.resize(x, (66, 200)), y))
        ds_test = ds_test.map(lambda x, y: (tf.image.resize(x, (66, 200)), y))
        X_train = np.array([x.numpy()/255. for x, y in ds_train])
        y_train = np.array([y.numpy() for x, y in ds_train])
        X_test = np.array([x.numpy()/255. for x, y in ds_test])
        y_test = np.array([y.numpy() for x, y in ds_test])
        return X_train, y_train, None, None, X_test, y_test
    elif dataset == 'fake_gan':
        folder = 'ds_dclgan/images/fake_A'
        transform_b = lambda x: x[130-66:130, 50:250, :]  / 255.
        X_train = []
        for i in tqdm(os.listdir(folder)):
            if i.endswith('.png'):
                img = cv2.imread(os.path.join(folder, i))[..., ::-1]
                img = transform_b(img)
                X_train.append(img)
        X_train = np.array(X_train)
        X_test = X_train[-100:]
        X_train = X_train[:-100]
        return X_train, None, None, None, X_test, None

    
if __name__ == '__main__':
    # X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('udacity')
    # X_train, y_train, _, _, X_test, y_test = loadDataset('cats_vs_dogs')
    # X_train, y_train, _, _, X_test, y_test = loadDataset('mnist')
    X_train, _, _, _, _, _ = loadDataset('fake')
    print('X_train.shape:', X_train.shape)
    X_train, _, _, _, _, _ = loadDataset('udacity')
    print('X_train.shape:', X_train.shape)

    # print('y_train.shape:', y_train.shape)
    # print('X_test.shape:', X_test.shape)
    # print('y_test.shape:', y_test.shape)

