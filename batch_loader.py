import os
import scipy

pwd = os.dirname(__file__)

def load(dirname):
    data_matrix = None # TODO
    n = data_matrix.shape[0]
    image_tensor = np.empty((n, 300, 300, 3), dtype=uint8)
    nid = "%04s" % n
    for i in range(n):
        fname = os.path.join(dirname, f'{nid}.jpg')
        image_tensor[i] = scipy.ndimage.imread(fname)
    return (image_tensor, data_matrix)
