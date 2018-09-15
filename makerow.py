import numpy as np
from PIL import Image


def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    np.pad(data, ((0, 640 - data.shape[0]), (0, 640 - data.shape[1])), 'constant', constant_values=(0, 0))
    return data

print(load_image("eeifshemaisrael/2017-04-11_22-23-03_UTC.jpg"))
