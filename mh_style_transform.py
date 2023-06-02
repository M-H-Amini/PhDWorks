import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mh_style_magenta import MHStyleMagenta
from mh_ds import loadDataset
from mh_ds import visualize as vis
import logging as log
import os
from PIL import Image
from tqdm import tqdm

def visualize(X, X_hat, show=True, output_name=None):
    h, w = X.shape[1:3]
    indices = np.random.permutation(len(X))
    img = np.zeros((h*4, w*8, 3))
    for i in range(4):
        for j in range(4):
            img[i*h:(i+1)*h, 2*j*w:(2*j+1)*w, :] = X[indices[i*4+j]]
            img[i*h:(i+1)*h, (2*j+1)*w:(2*j+2)*w, :] = X_hat[indices[i*4+j]]
    plt.figure(figsize=(16, 8))
    plt.grid(False)
    plt.imshow(img)
    show and plt.show()
    output_name and plt.savefig(output_name)

    
X_train, y_train, X_val, y_val, X_test, y_test = loadDataset('beamng', resize=True)
X = np.concatenate((X_train, X_val, X_test), axis=0)[:2000]
h,w = X_train.shape[1:3]

##  Load images...
style_image_path = 'UdacityDS/style_udacity_0.jpg'
# transform_u = lambda x: x[70:136, 100:300, :] / 255.
style_image = plt.imread(style_image_path)[70:136, 100:300, :].astype(np.float32) / 255.0
plt.imshow(style_image)
plt.show()
##  Load model...
model = MHStyleMagenta()
stylized_images = []
batch_size = 64
for i in tqdm(range(1 + len(X) // batch_size)):
    stylized_images.append(model(X[i*batch_size:(i+1)*batch_size], style_image, normalize=False, resize=True))  ##  normalize=False because images are already normalized in the dataset... 

stylized_images = np.concatenate(stylized_images, axis=0)
stylized_images = tf.image.resize(stylized_images, (h, w)).numpy()
print('stylized_images.shape:', stylized_images.shape)
visualize(X, stylized_images, output_name='stylized_images_beamng.png')

##  Save stylized images...
output_folder = 'ds_beamng_style_magenta'
os.makedirs(output_folder, exist_ok=True)
X_hat = stylized_images
y = np.concatenate((y_train, y_val, y_test), axis=0)
df_dict = {'img': [], 'steer': []}
for i in tqdm(range(len(X_hat))):
    img = (X_hat[i] * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img_name = f'img_{i}.jpg'
    img.save(os.path.join(output_folder, img_name))
    df_dict['img'].append(img_name)
    df_dict['steer'].append(y[i])
df = pd.DataFrame(df_dict)
df.to_csv(os.path.join(output_folder, 'labels.csv'), index=False)
vis(X, X_hat, 'style_magenta.pdf')
log.info('\033[92m' + 'Dataset generated!' + '\033[0m')
