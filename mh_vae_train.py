from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_vae import MHVAE
from mh_dave2_data import prepareDataset as prepareDatasetUdacity
from mh_utils import buildQ, buildP
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  Loading data...
dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
train_cols = ['center']
transform_u = lambda x: x[70:136, 100:300, :]
X_train_u, y_train_u, meta_train_u, X_val_u, y_val_u, meta_val_u, X_test_u, y_test_u, meta_test_u = prepareDatasetUdacity(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.7, test_size=0.1, val_size=0.1, transform=transform_u, show=False)
X_test_u = np.concatenate((X_val_u, X_test_u), axis=0)
y_test_u = np.concatenate((y_val_u, y_test_u), axis=0)
X_train_u = X_train_u.astype('float32') / 255.
X_test_u = X_test_u.astype('float32') / 255.

##  Building models...
latent_dim = 20

model_q = buildQ()
model_q.summary()

model_p = buildP()
model_p.summary()

###  MHVAE model...
model = MHVAE(input_dim=(66, 200, 3), latent_dim=latent_dim, model_p=model_p, model_q=model_q, regularization_const=100000, train_visualize=True)
model.compile(optimizer='adam', run_eagerly=True)
model.load_weights('mh_cvae_weights.h5')
log.info('\033[92m' + 'Model loaded!' + '\033[0m')
model.fit(X_train_u, epochs=50, batch_size=32)
model.generateGIF('mh_cvae3.gif')
model.save_weights('mh_cvae_weights.h5')