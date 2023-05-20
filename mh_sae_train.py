from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_sae import MHAE
from mh_dave2_data import prepareDataset as prepareDatasetUdacity
from mh_utils import buildQ, buildP
from mh_ds import loadDataset
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  Loading data...
X_train_u, y_train_u, X_val_u, y_val_u, X_test_u, y_test_u = loadDataset('udacity')

##  Building models...
latent_dim = 20

model_q = buildQ(type_='sae')
model_q.summary()

model_p = buildP()
model_p.summary()

###  MHVAE model...
model = MHAE(input_dim=(66, 200, 3), latent_dim=latent_dim, model_p=model_p, model_q=model_q, train_visualize=True)
model.compile(optimizer='adam', run_eagerly=True)
model.load_weights('mh_csae_weights.h5')
log.info('\033[92m' + 'Model loaded!' + '\033[0m')
model.fit(X_train_u, epochs=50, batch_size=32)
model.generateGIF('mh_csae.gif')
model.save_weights('mh_csae_weights.h5')