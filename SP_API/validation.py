import tensorflow as tf
from keras import backend as K
from SP_API.SPCore import SPCore, CKPT_PATH, DATA_PATH


def validate_SP(X_valid_imgs, X_valid_attribs, X_valid_embs, y_valid_labels, bsz, learning_rate, ckpt_path, img_dim):

    print('Initiating SPCore ...')
    core = SPCore(bsz=bsz, learning_rate=learning_rate, ckpt_path=ckpt_path, img_dim=img_dim)


    # Initialize TF Session
    sess = tf.Session()
    K.set_session(sess)

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    total_loss = 0.0

    for dp in range(len(X_valid_imgs)//bsz):
        dp_loss = 0.0

        X_batch_img, X_batch_attribs, X_batch_embs, y_batch = SPCore.get_next_batch(X_valid_imgs,
                                            X_valid_attribs, X_valid_embs, y_valid_labels, bsz=bsz)

        out = sess.run([core.z], feed_dict={
                            core.x_img:  X_batch_img,
                            core.x_attribs:  X_batch_attribs,
                            core.x_embs: X_batch_embs,
                })

        dp_loss = out[0][dp][0] - y_batch[dp][0]
        total_loss += dp_loss

        if dp % 10 == 0:
            print("Test {} Test Var {} Avg Var {}".format(dp, dp_loss, total_loss/len(X_valid_imgs)))

    # Display metrics
    validation_rmse = core.compute_error(out, y_batch)
    print('Validation RMSE: {}'.format(validation_rmse))






