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

def load_with_more(dirname):
    data_matrix = load_data_matrix(dirname)
    n = data_matrix.shape[0]
    image_tensor = np.empty((n, 300, 300, 3), dtype=np.uint8)
    for i in range(n):
        iid = "%04d" % (i+1)
        fname = os.path.join(dirname, f'{iid}.jpg')
        image_tensor[i] = (scipy.ndimage.imread(fname) - imn) / isd

    y = data_matrix[:, 393]
    features = data_matrix[:, 384:393]
    wordvecs = data_matrix[:, :384]
    return (image_tensor, wordvecs, features, y)

def setup_keras():
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(384,))  # Returns a placeholder tensor

    # A layer instance is callable on a tensor, and returns a tensor.
    x = keras.layers.Dense(128, bias_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    predictions = keras.layers.Dense(8, activation='softmax')(x)

    # Instantiate the model given inputs and outputs.
    model = keras.Model(inputs=inputs, outputs=predictions)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Trains for 5 epochs
    model.fit(data, labels, batch_size=32, epochs=5)

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
