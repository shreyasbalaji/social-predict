import numpy as np
import tensorflow as tf
from SP_API.SPCore import SPCore, CKPT_PATH, DATA_PATH
from SP_API.train import train_SP
from SP_API.validation import validate_SP
from SP_API.eval import eval_SP
from SP_API.prog_train import prog_train_SP


print('Loading dataset ...')
ds_imgs = np.load('../../../mnt/images.npy')
ds_attribs = np.load('../../../mnt/vectors.npy')
ds_labels = np.load('../../../mnt/results.npy')
ds_embeddings = np.load('../../../mnt/wordvec.npy')

# X_data = zip(ds_imgs, ds_attribs)
# y_data = ds_labels

tf.reset_default_graph()


# print('Initiating evaluation ...')
# eval_SP(X_valid_imgs[0], X_valid_attribs[0], X_valid_embs[0])

def do_split_data(ds_imgs, ds_attribs, ds_labels, ds_embeddings, valid_split=0.2):
    X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels = \
        SPCore.generate_split(x_imgs=ds_imgs, x_attribs=ds_attribs, x_embs=ds_embeddings, y_labels=ds_labels,
                              validation_split=valid_split)
    return X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels



def do_training(ds_imgs, ds_attribs, ds_labels, ds_embeddings, bsz, learning_rate, ckpt_path, img_dim, epochs, valid_split=0.0):
    print('Processing data ...')
    X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, X_valid_imgs, X_valid_attribs, \
                        X_valid_embs, y_valid_labels = do_split_data(ds_imgs, ds_attribs, ds_labels, ds_embeddings,
                        valid_split=valid_split)

    print('Initiating training ...')
    train_SP(X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, bsz, learning_rate, ckpt_path, img_dim, epochs)


def do_validation(ds_imgs, ds_attribs, ds_labels, ds_embeddings, bsz, learning_rate, ckpt_path, img_dim, valid_split=1.0):
    print('Processing data ...')
    X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, X_valid_imgs, X_valid_attribs, \
                        X_valid_embs, y_valid_labels = do_split_data(ds_imgs, ds_attribs, ds_labels, ds_embeddings,
                        valid_split=valid_split)

    print('Initiating validation ...')
    validate_SP(X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels, bsz, learning_rate, ckpt_path, img_dim)

def do_training_validation(ds_imgs, ds_attribs, ds_labels, ds_embeddings, bsz, learning_rate, ckpt_path, img_dim, epochs, valid_split=0.2):
    print('Processing data ...')
    X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, X_valid_imgs, X_valid_attribs, \
                        X_valid_embs, y_valid_labels = do_split_data(ds_imgs, ds_attribs, ds_labels, ds_embeddings,
                        valid_split=valid_split)

    print('Initiating training ...')
    train_SP(X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, bsz, learning_rate, ckpt_path, img_dim, epochs)

    print('Initiating validation ...')
    validate_SP(X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels)

def do_progressive_training_validation(ds_imgs, ds_attribs, ds_labels, ds_embeddings, bsz, learning_rate, ckpt_path, img_dim, epochs, valid_split=0.2):
    X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, X_valid_imgs, X_valid_attribs, \
                        X_valid_embs, y_valid_labels = do_split_data(ds_imgs, ds_attribs, ds_labels, ds_embeddings,
                        valid_split=valid_split)

    print('Initiating progressive training ...')
    prog_train_SP(X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, bsz, learning_rate, ckpt_path, img_dim, epochs)

    print('Initiating progressive validation ...')
    validate_SP(X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels)


def do_evaluation(X_eval_img, X_eval_attribs, X_eval_embs, ckpt_path, img_dim):
    eval_SP(X_eval_img, X_eval_attribs, X_eval_embs, ckpt_path, img_dim)