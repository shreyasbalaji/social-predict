import os
import numpy as np
import scipy.ndimage
from gen_data_matrix import load_data_matrix

pwd = os.path.dirname(__file__)
imn = np.load('image_mean.npy')
isd = np.load('image_std.npy')

def load(dirname):
    data_matrix = load_data_matrix(dirname)
    n = data_matrix.shape[0]
    image_tensor = np.empty((n, 300, 300, 3), dtype=np.uint8)
    for i in range(n):
        iid = "%04d" % (i+1)
        fname = os.path.join(dirname, f'{iid}.jpg')
        image_tensor[i] = (scipy.ndimage.imread(fname) - imn) / isd
    return (image_tensor, data_matrix)

def stats(dirname):
    data_matrix = load_data_matrix(dirname)
    n = data_matrix.shape[0]
    print(n)
    image_tensor = np.empty((n, 300, 300, 3), dtype=np.uint8)
    for i in range(n):
        iid = "%04d" % (i+1)
        fname = os.path.join(dirname, f'{iid}.jpg')
        image_tensor[i] = scipy.ndimage.imread(fname)
    return (np.mean(image_tensor, axis=0), np.std(image_tensor, axis=0))
