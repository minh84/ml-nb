from urllib.request import urlretrieve
from os.path import isfile, isdir, basename
from tqdm import tqdm

import pandas as pd

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download(url):
    filename = basename(url)

    if not isfile(filename):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(filename)) as pbar:
            urlretrieve(
                url,
                filename,
                pbar.hook)

def load_hitters():
    # download data to current dir
    url = 'https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Hitters.csv'
    maybe_download(url)

    return pd.read_csv('./Hitters.csv', na_filter=True)

