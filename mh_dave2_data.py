#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                 Title:  Dave2 Dataset                               ##
##                                   Date: 2023/05/11                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import logging as log
import seaborn as sns
import pandas as pd
import numpy as np
import os

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plotHistogram(df, col='steering', show=True):
    """Plots histogram of a column of a dataframe.

    Args:
        df (pandas.DataFrame): Dataframe.
        col (str, optional): Column name. Defaults to 'steering'.
        show (bool, optional): Whether to show the plot. Defaults to True.
    """
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.hist(df['steering'], bins=100)
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
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
    plotHistogram(df, show=show)
    df = df.drop(df[df['steering'] == 0].sample(frac=reduce_ratio, random_state=28).index).reset_index(drop=True)
    log.info(f'Balanced dataset size: {len(df)}')
    plotHistogram(df, show=show)
    return df

def splitDataset(df, train_cols, test_size=0.2, val_size=0.2, random_state=28):
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

def prepareDataset(dataset_folder, dataset_csv, train_cols, reduce_ratio=0.9, test_size=0.2, val_size=0.2, random_state=28, transform=lambda x:x, show=True):
    """Prepares a dataset for training.

    Args:
        dataset_folder (str): Path to dataset folder.
        dataset_csv (str): Path to csv file.
        train_cols (list): List of columns to be used as features.
        reduce_ratio (float, optional): Ratio of rows with 0 steering angle to be reduced. Defaults to 0.9.
        test_size (float, optional): Ratio of test set size to the whole dataset. Defaults to 0.2.
        val_size (float, optional): Ratio of validation set size to the train set. Defaults to 0.2.
        random_state (int, optional): Random state. Defaults to 28.
        transform (function, optional): Transform function. Defaults to lambda x:x.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        X_train (numpy.ndarray): Train set features.
    """

    df = readDataset(dataset_csv, dataset_folder, train_cols, transform=transform)
    df = balanceDataset(df, reduce_ratio=reduce_ratio, show=show)
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = splitDataset(df, train_cols, test_size=test_size, val_size=val_size, random_state=random_state)
    return X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test

def validate(X, y, meta, dataset_folder, dataset_csv, n=5, transform=lambda x:x, show=True):
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
    


if __name__  == '__main__':
    ##  Dataset...
    dataset_folder = 'UdacityDS/self_driving_car_dataset_jungle/IMG'
    dataset_csv = 'UdacityDS/self_driving_car_dataset_jungle/driving_log.csv'
    train_cols = ['center']  ##  Or it can be ['center', 'left', 'right']
    transform = lambda x: x[60:150, :, :]
    # train_cols = ['center', 'left', 'right']
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDataset(dataset_folder, dataset_csv, train_cols, transform=transform, show=False)
    log.info(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    log.info(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    log.info(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    validate(X_train, y_train, meta_train, dataset_folder, dataset_csv, n=5, transform=transform, show=True)