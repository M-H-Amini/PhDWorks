import logging as log
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import json
import os
from mh_dave2_data import plotHistogram, splitDataset, imageGenerator, visualizeGenerator

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def json2CSV(json_folder, step=5):
    jsons = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    log.debug(f'Found {len(jsons)} json files in {json_folder}')
    cols = list(json.load(open(os.path.join(json_folder, jsons[0]))).keys())
    df = pd.DataFrame(columns=cols)
    for json_file in tqdm(jsons):
        json_path = os.path.join(json_folder, json_file)
        df_json = pd.read_json(json_path)
        ##  To remove first and last 10% of the data (black frames and grass frames)
        a, b = int(0.1 * len(df_json)), int(0.9 * len(df_json))
        df_json = df_json.iloc[a:b:step]
        ##  Concatenating df_json to df...
        df = pd.concat([df, df_json], ignore_index=True)
    ##  Converting all columns other than is_oob and img to float32...
    for col in df.columns:
        if col not in ['is_oob', 'img']:
            df[col] = df[col].astype(np.float32)
    ##  Saving df to csv...
    output_csv = os.path.join(json_folder, 'ds_beamng.csv')
    df.to_csv(output_csv, index=False)
    log.info(f'The resulting csv file saved to {output_csv}!')
    return df

def readDataset(json_folder, step=15, transform=lambda x:x):
    """Reads the dataset from json files. It also balances the dataset by flipping the images and steering angles.

    Args:
        json_folder (str): Path to json folder.
        step (int, optional): Step to skip frames. Defaults to 15.
        transform (function, optional): Transform function to be applied to images. Defaults to lambda x:x.

    Returns:
        df (pandas.DataFrame): Dataframe containing the dataset.
    """
    df = json2CSV(json_folder, step=step)
    df['steering'] = df['steering_input']  ##  Renaming steering_input to steering to be consistent with Udacity dataset...
    df = df[['img', 'steering']]  ##  Dropping other columns...
    log.info(f'Reading {len(df)} images...')
    df[['img_path']] = df[['img']]
    df_flip = df.copy()  ##  Coppying the dataframe to flip it later...
    df['img'] = df['img'].apply(lambda x: np.asarray(Image.open(os.path.join(json_folder, 'img', x))))
    df_flip['img'] = df_flip['img'].apply(lambda x: np.asarray(Image.open(os.path.join(json_folder, 'img', x)).transpose(Image.FLIP_LEFT_RIGHT)))
    df_flip['steering'] = -df_flip['steering']
    ##  Concatenating df and df_flip...
    df = pd.concat([df, df_flip], ignore_index=True)
    df['img'] = df['img'].apply(transform)
    return df


def prepareDataset(json_folder, step=15, test_size=0.2, val_size=0.2, random_state=28, transform=lambda x:x, show=True):
    """Prepares a dataset for training.

    Args:
        json_folder (str): Path to json folder.
        step (int, optional): Step to skip frames. Defaults to 15.
        test_size (float, optional): Ratio of test set size to the whole dataset. Defaults to 0.2.
        val_size (float, optional): Ratio of validation set size to the train set. Defaults to 0.2.
        random_state (int, optional): Random state. Defaults to 28.
        transform (function, optional): Transform function to be applied to images. Defaults to lambda x:x.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        X_train (numpy.ndarray): Train set features.
    """
    df = readDataset(json_folder, step=step, transform=transform)
    plotHistogram(df, 'steering', output_name='ds_beamng_hist.pdf', show=show)
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = splitDataset(df, ['img'], test_size=test_size, val_size=val_size, random_state=random_state)
    return X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test

if __name__ == '__main__':
    json_folder = 'ds_beamng'
    test_size = 0.1
    val_size = 0.1
    step = 15
    transform = lambda x: x[130-66:130, 60:260, :]  ##  Crop the image
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = prepareDataset(json_folder, step=step, test_size=test_size, val_size=val_size, random_state=28, transform=transform, show=True)
    gen_train = imageGenerator(X_train, y_train, (0.4, 0.8), batch_size=16, shuffle=True)
    visualizeGenerator(gen_train)
    