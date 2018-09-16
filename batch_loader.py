import os
import scipy
from gen_data_matrix import load_data_matrix

pwd = os.dirname(__file__)

def load(dirname):
    data_matrix = load_data_matrix(dirname)
    n = data_matrix.shape[0]
    image_tensor = np.empty((n, 300, 300, 3), dtype=uint8)
    nid = "%04s" % n
    for i in range(n):
        fname = os.path.join(dirname, f'{nid}.jpg')
        image_tensor[i] = scipy.ndimage.imread(fname)
    return (image_tensor, data_matrix)


def stats(dirname):
    data_matrix = load_data_matrix(dirname)
    n = data_matrix.shape[0]
    image_tensor = np.empty((n, 300, 300, 3), dtype=uint8)
    nid = "%04s" % n
    for i in range(n):
        fname = os.path.join(dirname, f'{nid}.jpg')
        image_tensor[i] = scipy.ndimage.imread(fname)
    return (np.mean(image_tensor, axis=0), np.stdev(image_tensor, axis=0))
