import numpy as np
import tensorflow as tf
from SP_API.SPCore import SPCore, CKPT_PATH, DATA_PATH
from SP_API.train import train_SP
from SP_API.validation import validate_SP
from SP_API.eval import eval_SP

print('Loading dataset ...')
ds_imgs = np.load('../../../mnt/images.npy')
ds_attribs = np.load('../../../mnt/vectors.npy')
ds_labels = np.load('../../../mnt/results.npy')
ds_embeddings = np.load('../../../mnt/wordvec.npy')

# X_data = zip(ds_imgs, ds_attribs)
# y_data = ds_labels

X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels = \
    SPCore.generate_split(x_imgs=ds_imgs, x_attribs=ds_attribs, x_embs = ds_embeddings, y_labels=ds_labels, validation_split=0.2)

tf.reset_default_graph()


print('Initiating training ...')
train_SP(X_train_imgs, X_train_attribs, X_train_embs, y_train_labels)

print('Initiating validation ...')
validate_SP(X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels)

# print('Initiating evaluation ...')
# eval_SP(X_valid_imgs[0], X_valid_attribs[0], X_valid_embs[0])

