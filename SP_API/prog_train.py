import tensorflow as tf
from keras import backend as K
from SP_API.SPCore import SPCore, CKPT_PATH, DATA_PATH


def prog_train_SP(X_train_imgs, X_train_attribs, X_train_embs, y_train_labels, bsz, learning_rate, ckpt_path, img_dim, epochs):

    tf.reset_default_graph()

    print('Initiating SPCore ...')
    core = SPCore(bsz=bsz, learning_rate=learning_rate, ckpt_path=ckpt_path, img_dim=img_dim)

    # Initialize TF Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    K.set_session(sess)
    # Initialize TF Saver
    saver = tf.train.Saver()
    saver.restore(sess, CKPT_PATH)

    # Start Training
    for ep in range(1, epochs + 1):

        final_out_loss, final_out_err = 0.0, 0.0

        num_batches = len(X_train_imgs)//50
        for i in range(num_batches):
            X_batch_img, X_batch_attribs, X_batch_embs, y_batch = SPCore.get_next_batch(X_train_imgs,
                                                    X_train_attribs, X_train_embs, y_train_labels, bsz=bsz)
            # print("HERE", X_batch_img.shape, X_batch_attribs.shape, y_batch.shape, core.x_img.shape)

            # Fit!
            # loss, acc, _ = sess.run([core.final_loss, core.final_loss_metric, core.train_op],
            loss, z, _ = sess.run([core.final_loss, core.z, core.train_op],
                    feed_dict={
                        core.x_img: X_batch_img,
                        core.x_attribs: X_batch_attribs,
                        core.x_embs: X_batch_embs,
                        core.out_labels: y_batch
                    })
            final_out_loss += loss
            # final_out_acc += acc
            batch_rmse = core.compute_error(z, y_batch)
            final_out_err += batch_rmse


        print("Epoch {} Loss {} Err {} ".format(ep, final_out_loss / num_batches, final_out_err / num_batches))
        # print("Epoch {} Loss {}".format(ep, final_out_loss / num_batches))

        # Save Model
        saver.save(sess, CKPT_PATH)