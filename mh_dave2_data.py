#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                 Title:  Dave2 Dataset                               ##
##                                   Date: 2023/05/11                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import logging as log
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plotHistogram(df, col='steering', show=True, output_name=None):
    """Plots histogram of a column of a dataframe.

    Args:
        df (pandas.DataFrame): Dataframe.
        col (str, optional): Column name. Defaults to 'steering'.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.hist(df[col], bins=100)
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    output_name and plt.savefig(output_name)
    show and plt.show()
    plt.close()

def readDataset(dataset_csv, dataset_folder, train_cols, transform=lambda x: x):
    """Gets dataset folder and csv file and returns a dataframe of images and steering angles.

    Args:
        dataset_csv (str): Path to csv file.
        dataset_folder (str): Path to dataset folder.
        train_cols (list): List of columns to be read from csv file.
        transform (function, optional): Transformation function. Defaults to lambda x: x.

    Returns:
        df (pandas.DataFrame): Dataframe of images and steering angles.
    """
    df = pd.read_csv(dataset_csv, header=None, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    df[train_cols] = df[train_cols].applymap(lambda x: os.path.join(dataset_folder, x.split('\\')[-1]))
    df[['center_path', 'left_path', 'right_path']] = df[['center', 'left', 'right']]
    df_flip = df.copy()  ##  Copying the dataframe to flip it later...
    df[train_cols] = df[train_cols].applymap(lambda x: np.asarray(Image.open(x)))
    ##  Flipping each image and negating its steering angle...
    df_flip[train_cols] = df_flip[train_cols].applymap(lambda x: np.asarray(Image.open(x).transpose(Image.FLIP_LEFT_RIGHT)))
    df_flip['steering'] = df_flip['steering'].apply(lambda x: -x)
    ##  Concatenating the original and flipped dataframes...
    df = pd.concat([df, df_flip], ignore_index=True)
    ##  Applying the transformation function (e.x. cropping)...
    df[train_cols] = df[train_cols].applymap(transform)
    return df

def balanceDataset(df, col='steering', reduce_ratio=0.9, show=True):
    """Balances a dataframe by reducing the number of rows with 0 steering angle.

    Args:
        df (pandas.DataFrame): Dataframe.
        col (str, optional): Column name. Defaults to 'steering'.
        reduce_ratio (float, optional): Ratio of rows with 0 steering angle to be reduced. Defaults to 0.9.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        df (pandas.DataFrame): Balanced dataframe.
    """
    ##  Reducing 0 steering angles...
    log.info(f'Imbalanced dataset size: {len(df)}')
    plotHistogram(df, show=show, output_name='dataset_hist_imbalanced.pdf')
    df = df.drop(df[df['steering'] == 0].sample(frac=reduce_ratio, random_state=28).index).reset_index(drop=True)
    log.info(f'Balanced dataset size: {len(df)}')
    plotHistogram(df, show=show, output_name='dataset_hist_balanced.pdf')
    return df

def splitDataset(df, train_cols, test_size=0.2, val_size=0.2, random_state=42):
    """Splits a dataframe into train, validation and test sets.

    Args:
        df (pandas.DataFrame): Dataframe.
        train_cols (list): List of columns to be used as features.
        test_size (float, optional): Ratio of test set size to the whole dataset. Defaults to 0.2.
        val_size (float, optional): Ratio of validation set size to the train set. Defaults to 0.2.
        random_state (int, optional): Random state. Defaults to 28.

    Returns:
        X_train (numpy.ndarray): Train set features.
        y_train (numpy.ndarray): Train set labels.
        meta_train (list): Train set metadata.
        X_val (numpy.ndarray): Validation set features.
        y_val (numpy.ndarray): Validation set labels.
        meta_val (list): Validation set metadata.
        X_test (numpy.ndarray): Test set features.
        y_test (numpy.ndarray): Test set labels.
        meta_test (list): Test set metadata.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    log.info(f'Train dataset size: {len(train_df)}')
    log.info(f'Val dataset size: {len(val_df)}')
    log.info(f'Test dataset size: {len(test_df)}')
    X_train, y_train, meta_train = train_df[train_cols].values, train_df['steering'].values, train_df[[col+'_path' for col in train_cols]].values
    X_val, y_val, meta_val = val_df[train_cols].values, val_df['steering'].values, val_df[[col+'_path' for col in train_cols]].values
    X_test, y_test, meta_test = test_df[train_cols].values, test_df['steering'].values, test_df[[col+'_path' for col in train_cols]].values
    X_train = np.array([xx for x in X_train for xx in x])  ##  (N, 160, 320, 3)
    y_train = np.array([y for y in y_train for _ in range(len(train_cols))])  ##  (N, 1)
    meta_train = [mm for m in meta_train for mm in m]  ##  (N,)
    X_val = np.array([xx for x in X_val for xx in x])  ##  (N, 160, 320, 3)
    y_val = np.array([y for y in y_val for _ in range(len(train_cols))])  ##  (N, 1)
    meta_val = [mm for m in meta_val for mm in m]  ##  (N,)
    X_test = np.array([xx for x in X_test for xx in x])  ##  (N, 160, 320, 3)
    y_test = np.array([y for y in y_test for _ in range(len(train_cols))])  ##  (N, 1)
    meta_test = [mm for m in meta_test for mm in m]  ##  (N,)
    return X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test

def saveDataset(output_folder=None):
    target_folder = 'UdacityDS/self_driving_car_dataset_jungle/'
    img_folder = os.path.join(target_folder, 'IMG')
    csv_file = os.path.join(target_folder, 'driving_log.csv')
    df = pd.read_csv(csv_file, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    df['center'] = df['center'].apply(lambda x: os.path.join(img_folder, x.split("\\")[-1]))
    df['left'] = df['left'].apply(lambda x: os.path.join(img_folder, x.split("\\")[-1]))
    df['right'] = df['right'].apply(lambda x: os.path.join(img_folder, x.split("\\")[-1]))
    non_zero = df[df['steering'] != 0]
    zero_df = df[df['steering'] == 0].sample(frac=0.05)
    df_train = pd.concat([non_zero, zero_df])
    df_train = df_train[(df_train['steering'] < 0.99) & (df_train['steering'] > -0.99)]
    df_train = df_train[['center', 'steering']]
    df_train.columns = ['img', 'steering']
    df_flip = df_train.copy()
    df_flip['steering'] = -df_flip['steering']
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
        for i in tqdm(range(len(df_train))):
            img = Image.open(df_train['img'].iloc[i])
            img.save(os.path.join(output_folder, 'images', os.path.basename(df_train['img'].iloc[i])))
            df_train['img'].iloc[i] = os.path.basename(df_train['img'].iloc[i])
        for i in tqdm(range(len(df_flip))):
            img = Image.open(df_flip['img'].iloc[i])
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(os.path.join(output_folder, 'images', os.path.basename(df_flip['img'].iloc[i])[:-4] + '_flip.png'))
            df_flip['img'].iloc[i] = os.path.basename(df_flip['img'].iloc[i])[:-4] + '_flip.png'
        df_train = pd.concat([df_train, df_flip])
        df_train.to_csv(os.path.join(output_folder, 'ds_udacity.csv'), index=False)

def prepareDataset(dataset_folder, test_size=0.2, val_size=0.2, random_state=28, x_transform=lambda x:x, y_transform=lambda y:y, show=True):
    df = pd.read_csv(os.path.join(dataset_folder, 'ds_udacity.csv'))
    log.info(f'Reading {len(df)} images...')
    X = np.array(list(map(lambda x: x_transform(cv2.imread(os.path.join(dataset_folder, 'images', x))), df['img'].values)))[...,::-1]
    y = df['steering'].values
    y = np.array(list(map(lambda y: y_transform(y), y)))
    df['steering'] = y
    log.info(f'X shape: {X.shape}, y shape: {y.shape}')
    show and plotHistogram(df, 'steering', output_name='ds_udacity_hist.pdf', show=show)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state)
    log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    log.info(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    df.to_csv(os.path.join(dataset_folder, 'ds_beamng_normalized.csv'), index=False)
    return X_train, y_train, X_val, y_val, X_test, y_test

def validate(X, y, meta, dataset_folder, dataset_csv, n=5, transform=lambda x:x, show=True):
    """Validates a dataset.

    Args:
        X (numpy.ndarray): Images.
        y (numpy.ndarray): Steering angles.
        meta (list): Metadata (Image paths).
        dataset_folder (str): Path to dataset folder.
        dataset_csv (str): Path to csv file.
        n (int, optional): Number of samples to be validated. Defaults to 5.
        transform (function, optional): Transform function. Defaults to lambda x:x.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """

    df = pd.read_csv(dataset_csv, header=None, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    df[['center', 'left', 'right']] = df[['center', 'left', 'right']].applymap(lambda x: os.path.join(dataset_folder, x.split('\\')[-1]))
    for i in range(n):
        index = np.random.randint(len(X))
        img = X[index]
        steering = y[index]
        img_path = meta[index]
        img_gt = transform(np.asarray(Image.open(img_path)))  ##  Ground truth image
        steering_gt = df[df.apply(lambda x: img_path in x.values, axis=1)]['steering'].values[0]
        assert steering == steering_gt or steering == -steering_gt, f'Steering angle {steering} is not equal to its version in the dataset.'
        assert np.all(img == img_gt) or np.all(img == img_gt[:, ::-1, :]), f'Image {img_path} is not equal to its version in the dataset.'
        if show: 
            plt.figure(figsize=(10, 5))
            plt.imshow(img)
            plt.title(f'Steering angle: {steering}\nGround truth steering angle: {steering_gt}\nImage path: {img_path}')
            plt.show()
            plt.close()
    log.info('\033[92m' + 'Validation passed!' + '\033[0m')

def imageGenerator(X, y, brightness_range=(0.4, 0.6), batch_size=32, shuffle=True):
    """Generates and augments images. Augmentation includes brightness adjustment.

    Args:
        X (numpy.ndarray): Images.
        y (numpy.ndarray): Steering angles.
        brightness_range (tuple, optional): Brightness range. Defaults to (0.2, 0.8).
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        generator: Generator of images.
    """
    if brightness_range is None:
        datagen = ImageDataGenerator()
    else:
        datagen = ImageDataGenerator(
            brightness_range=brightness_range,
        )
    datagen.fit(X)
    return datagen.flow(X, y, batch_size=batch_size, shuffle=shuffle)
    
def visualizeGenerator(gen):
    """Visualizes a generator.

    Args:
        gen (generator): Generator of images.
    """
    X, y = gen.next()
    X = X.astype(np.float32) / 255.
    plt.figure(figsize=(10, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(X[i])
        plt.title(f'Steering angle: {round(y[i], 2)}')
        plt.axis('off')
    plt.savefig('dataset.pdf', bbox_inches='tight')
    plt.show()
    

if __name__  == '__main__':
    ##  Dataset...
    # dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
    # dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
    # transform = lambda x: x[60:160, :, :]  ##  Crop the image
    # # transform = lambda x: x  ##  No transform
    # train_cols = ['center']  ##  Or it can be ['center', 'left', 'right']
    # # train_cols = ['center', 'left', 'right']
    # X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDataset(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.9, transform=transform, show=True)
    # log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    # log.info(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    # log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    # validate(X_train, y_train, meta_train, dataset_folder, dataset_csv, n=5, transform=transform, show=False)
    
    # saveDataset(output_folder='ds_udacity')  ##  To save the dataset...
    X_train, y_train, X_val, y_val, X_test, y_test = prepareDataset('ds_udacity')
    log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    log.info(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    gen_train = imageGenerator(X_train, y_train, (0.6, 1.), batch_size=16, shuffle=True)
    visualizeGenerator(gen_train)
