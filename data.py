from pathlib import Path
from zipfile import ZipFile

import matplotlib.image as mpimage
import numpy as np
import os
import pandas as pd
import shutil

def _download_dataset():
    """
    Download data from kaggle and unzip it.
    The data is put into the sign_data folder
    """
    os.system('kaggle datasets download -d robinreni/signature-verification-dataset')
    data_file = 'signature-verification-dataset.zip'
    file = ZipFile(data_file)
    file.extractall()
    os.remove(data_file)
    shutil.rmtree('sign_data/sign_data')



def get_data_dataframe(data_dir:str=None) -> pd.DataFrame:
    """Iterate through the data directory and store all filenames in a pandas DataFrame

    Args:
        data_dir (str, optional): Directory path. Defaults to None, which looks in 'sign_data' directory for data.

    Returns:
        pd.DataFrame: Columns are type, person, label, and filename
    """

    info = []
    data_dir = data_dir or Path() / 'sign_data'
    for type in data_dir.iterdir():
        if type.is_dir():
            for subfolder in type.iterdir():
                try:
                    person, label = subfolder.name.split('_')
                except ValueError:
                    person = subfolder.name
                    label = 'real'
                person = int(person)
                for file in subfolder.iterdir():
                    h, w, c = mpimage.imread(file).shape
                    info.append((type.name, person, label, h, w, c, str(file)))
    df = pd.DataFrame(info, columns=['type', 'person', 'label', 'height', 'width', 'channel', 'filename'])
    return df