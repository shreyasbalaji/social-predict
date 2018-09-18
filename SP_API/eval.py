import tensorflow as tf
import numpy as np
from keras import backend as K
from SP_API.SPCore import SPCore, CKPT_PATH, DATA_PATH


def eval_SP(X_eval_img, X_eval_attribs, X_eval_embs, ckpt_path, img_dim):

    print('Initiating SPCore ...')
    core = SPCore(ckpt_path=ckpt_path, img_dim=img_dim)

    with tf.Session() as sess:
        K.set_session(sess)
        # Initialize TF Saver
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        X_img_imp, X_attribs_inp, X_embs_inp = np.reshape(X_eval_img, (1, 300, 300, 3)), \
                                               np.reshape(X_eval_attribs, (1, 9)), np.reshape(X_eval_embs, (1, 384))

        out = sess.run([core.z],
                       feed_dict={
                           core.x_img: X_img_imp,
                           core.x_attribs:  X_attribs_inp,
                           core.x_embs: X_embs_inp
                       })

        print("Evaluation Result {}".format(np.reshape(out, (1))[0]))
        return out


