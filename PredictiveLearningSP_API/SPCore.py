import tensorflow as tf
from keras import backend as K
import numpy as np
import time


from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten, Input
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D, Conv2D, Conv3D, MaxPooling3D, Activation
from keras.preprocessing import image

DATA_PATH = "SP_API/data/"
CKPT_PATH = "SP_API/log/model.ckpt"

pld = [9, 10, 10]
pla = ['relu', 'relu', 'sigmoid']

dld = [30, 20, 10, 10, 1]
dla = ['relu', 'relu', 'relu', 'relu', 'softmax']

eld = [384, 60, 10]
ela = ['relu', 'relu', 'sigmoid']

IMG_DIM = 300


class SPCore():
    def __init__(self, img_dim=IMG_DIM, ckpt_path=CKPT_PATH, primary_layer_dims=pld, primary_layer_activations=pla,
                 deep_layer_dims=dld, deep_layer_activations=dla, embedding_layer_dims=eld, embedding_layer_activations=ela, verbose=0, restore_mode=True, bsz = 50):

        tf.reset_default_graph()


        # self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = X_train, y_train, X_valid, y_valid, X_test, y_test
        self.ckpt_path, self.verbose = ckpt_path, verbose
        self.pld, self.pla, self.dld, self.dla = pld, pla, dld, dla
        self.primary_layer_dims, self.deep_layer_dims, self.embedding_layer_dims = primary_layer_dims, deep_layer_dims, embedding_layer_dims
        self.img_dim, self.num_vars, self.bsz = img_dim, 1, bsz
        # self.x_img, self.x_attribs, self.y_img, self.y_attribs, self.y, self.z, self.out_labels = \
        #     self.create_placeholders(img_in_dim=img_dim, attribs_in_dim=9, img_out_dim=10,
        #                              attribs_out_dim=10, final_out_dim=1, bsz=self.bsz)
        self.x_img, self.x_attribs, self.x_embs, self.z, self.out_labels = self.create_placeholders(
                    img_in_dim=img_dim, attribs_in_dim=9, embs_in_dim=384, img_out_dim=10, attribs_out_dim=10,
                    embs_out_dim = 10, final_out_dim=1, bsz=self.bsz)


        self.p_primary, self.n_l_primary = self.initialize_parameters(primary_layer_dims, primary_layer_activations)
        self.p_embedding, self.n_l_embedding = self.initialize_parameters(embedding_layer_dims, embedding_layer_activations)
        self.p_deep, self.n_l_deep = self.initialize_parameters(deep_layer_dims, deep_layer_activations)




        # self.convolution_net = self.convolution_net()
        # self.primary_net = self.primary_net()
        # self.deep_net = self.deep_net()

        self.build()


        # Build Losses
        self.final_loss = self.build_losses()


        # Build Optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.0001, self.global_step, 10000, 0.95,
                                                    staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Build Metrics
        # self.final_loss_metric = self.build_metrics()


        # Build Train Ops
        self.train_op = self.opt.minimize(self.final_loss, global_step=self.global_step)


    def build(self):
        # self.y_img = self.convolution_net
        # self.y_attribs = self.primary_net
        # print("MORE: ", self.y_img.shape, self.y_attribs.shape)
        # self.y = self.merge_l(self.y_img, self.y_attribs)
        # print(self.y.shape)
        # self.z = self.deep_net
        # y_c = tf.concat([self.convolution_net(), self.primary_net()], axis=1)
        self.num_vars = 1
        y_c = tf.concat([self.convolution_net(), self.primary_net(), self.embedding_net()], axis=1)
        self.z = self.deep_net(y_c)




    def convolution_net(self):
        # mean_px = self.x_img.mean().astype(np.float32)
        # std_px = self.x_img.std().astype(np.float32)
        # std_px = self.x_img.std().astype(np.float32)
        #
        # def standardize(x):
        #     return (x - mean_px) / std_px

        # model = Sequential([
        #     Convolution3D(32, kernel_size=(32, 32, 3), input_shape=(self.img_dim, self.img_dim, 3, 1),
        #                   border_mode='same', data_format="channels_last", activation='relu'),
        #     BatchNormalization(axis=1),
        #     # Convolution3D(32, (32, 32, 3), activation='relu'),
        #     # BatchNormalization(axis=1),
        #     Convolution3D(32, (16, 16, 3), activation='relu'),
        #     MaxPooling3D(),
        #     BatchNormalization(axis=1),
        #     Convolution3D(64, (4, 4, 3), activation='relu'),
        #     BatchNormalization(axis=1),
        #     Convolution3D(64, (2, 2, 3), activation='relu'),
        #     MaxPooling3D(),
        #     Flatten(),
        #     BatchNormalization(axis=1),
        #     Dense(512, activation='relu'),
        #     BatchNormalization(axis=1),
        #     Dense(10, activation='sigmoid')
        # ])
        # model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # model = Sequential()
        # model.add(Conv3D(32, (32, 32, 3), input_shape=(self.img_dim, self.img_dim, 3, 1),
        #                         border_mode='same', data_format="channels_last", activation='relu'))

        # print(self.x_img.shape)
        model = Sequential()
        # model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(self.img_dim, self.img_dim, 3, 1), padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(self.img_dim, self.img_dim, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(Activation('softmax'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('softmax'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        convolution_out = model(self.x_img)
        return convolution_out



    def primary_net(self):
        primary_out = self.forward_propagation(self.x_attribs, self.p_primary, self.n_l_primary)
        return primary_out

    # def merge_l(self, conv_img, attribs_vec):
    #     self.y = tf.concat([tf.to_float(conv_img), tf.to_float(attribs_vec)], axis=1)
    #     # print("Y: ", self.y.shape)

    def embedding_net(self):
        embedding_out = self.forward_propagation(self.x_embs, self.p_embedding, self.n_l_primary)
        return embedding_out

    def deep_net(self, y_concat):
        deep_out = self.forward_propagation(y_concat, self.p_deep, self.n_l_deep)
        return deep_out





    def build_losses(self):

        # self.z = tf.expand_dims(self.z, 0)
        final_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.z, labels=self.out_labels), name='SP_Network_Loss')

        return final_loss


    def build_metrics(self):
        """
        Build accuracy metrics for each of the sub-networks.
        """
        final_network_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.z, 1),
                                                      self.out_labels),
                                             tf.float32), name='Final_Network_Accuracy')

        return final_network_metric




    def create_placeholders(self, img_in_dim=None, attribs_in_dim=None, embs_in_dim=None, img_out_dim=None,
                            attribs_out_dim=None, embs_out_dim=None, final_out_dim=None, bsz=None):
        """
        :param input_dim: int
        :param output_dim: int
        :return: tuple
        """

        x_img = tf.placeholder(shape=(None, img_in_dim, img_in_dim, 3), dtype=tf.float32, name="x_img")
        # print(x_img.shape)
        x_attribs = tf.placeholder(shape=(None, attribs_in_dim), dtype=tf.float32, name="x_attribs")
        x_embs = tf.placeholder(shape=(None, embs_in_dim), dtype=tf.float32, name="x_embs")
        # y_img = tf.placeholder(shape=(None, img_out_dim), dtype=tf.float32, name="y_img")
        # y_attribs = tf.placeholder(shape=(None, attribs_out_dim), dtype=tf.float32, name="y_atribs")
        # y = tf.placeholder(shape=(None, img_out_dim + attribs_out_dim), dtype=tf.float32, name="y")
        z = tf.placeholder(shape=(None, final_out_dim), dtype=tf.float32, name="z")
        out_labels = tf.placeholder(shape=(None, final_out_dim), dtype=tf.float32, name="out_labels")

        # return x_img, x_attribs, y_img, y_attribs, y, z, out_labels
        return x_img, x_attribs, x_embs, z, out_labels




    def initialize_parameters(self, layer_dims, layer_activations=None, start=1):
        """
        creates all w, b, and activation functions
        :param layer_dims: list of ints eq to number of neurons in each layer including input and output
        :param layer_activations: list of strings of activation function names
        :return: parameters dictionary of all w, b, l and number of layers in
        """


        parameters = {}
        n_layers = len(layer_dims)

        if layer_activations != None:
            assert (n_layers == len(layer_activations))

        for i in range(1, n_layers):
            parameters['w' + str(self.num_vars)] = tf.get_variable('w' + str(self.num_vars), [layer_dims[i - 1], layer_dims[i]],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       dtype=tf.float32)
            parameters['b' + str(self.num_vars)] = tf.get_variable('b' + str(self.num_vars), [layer_dims[i]],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       dtype=tf.float32)
            parameters['l' + str(self.num_vars)] = layer_activations[i] if layer_activations != None else 'linear'

            self.num_vars +=1

        # set activation func for output layer to linear for loss function
        parameters['l' + str(n_layers)] = 'linear'

        return parameters, n_layers



    def activate(self, a, activation_f):
        """
        Applies activation function on previous output
        :param a: tensor
        :param activation_f: string - activation name
        :return: tensor
        """
        if activation_f == "linear":
            return tf.cast(a, tf.float32)
        elif activation_f == 'relu':
            return tf.cast(tf.nn.relu(a), tf.float32)
        elif activation_f == 'sigmoid':
            return tf.cast(tf.nn.sigmoid(a), tf.float32)
        elif activation_f == 'softmax':
            return tf.cast(tf.nn.softmax(a), tf.float32)
        else:
            raise ValueError("Not a valid activation function name")



    def forward_propagation(self, inp, parameters, n_layers):
        """
        Calculates the output of the neural net given input inp
        :param inp: tensor
        :param parameters: model parameters dictionary
        :param n_layers: int number of layers
        :return: output tensor
        """
        a = None
        z = tf.cast(inp, tf.float32)

        for l in range(self.num_vars, self.num_vars + n_layers - 1):
            a = tf.add(tf.matmul(z, parameters['w' + str(l)]), parameters['b' + str(l)])
            z = self.activate(a, parameters['l' + str(l)])
            self.num_vars += 1

        return z



    # def compute_loss(self, y, pred, size, loss_f='reduce_sum'):
    #     """
    #     Calculates loss of net with test/validation label
    #     :param y: label
    #     :param pred: output prediction
    #     :param size: int data points
    #     :param loss_f: string of name of loss function
    #     :return: loss according to loss-f
    #     """
    #     if loss_f == "reduce_sum":
    #         loss = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * int(size))
    #         return loss

    # def predict_model(self, data, parameters, n_layers):
    #     """
    #     Runs data through Neural Net
    #     :param data: dataset of input arrays
    #     :param parameters: dictionary of trained model parameters
    #     :param n_layers: int number of layers
    #     :return: output array prediction labels
    #     """
    #     init = tf.global_variables_initializer()
    #     with tf.Session() as sess:
    #         sess.run(init)
    #
    #         self.x_img, self.x_attribs = data
    #
    #         fprop_result = self.forward_propagation(data, parameters, n_layers)
    #         prediction = fprop_result.eval()
    #
    #     return prediction

    def compute_error(self, predictions, labels, error_f='rmse'):
        """
        Root Mean Squared Error
        :param predictions: np array
        :param labels: np array
        :param error_f: string of error function name
        :return: error according to error_f
        """
        prediction_loss = 0

        if error_f == 'rmse':
            prediction_size = len(predictions)
            prediction_loss = np.sqrt(np.sum(np.square(labels - predictions)) / prediction_size)

        return prediction_loss



    @staticmethod
    def get_next_batch(x_imgs, x_attribs, x_embs, y_labels, bsz):
        """
        Selectes random set of batch_size elements from x_data and y_data
        :param x_data: np matrix
        :param y_data: np matrix/array
        :param batch_size: int
        :return: batch_x, batch_y
        """

        batch_x_img = []
        batch_x_attribs = []
        batch_x_embs = []
        batch_y = []

        if len(y_labels) == 0:

            batch_x_img, batch_x_attribs, batch_x_embs= x_imgs, x_attribs, x_embs

            return np.array(batch_x_img), np.array(batch_x_attribs), np.array(batch_x_embs)

        else:

            if bsz == None:
                for i in range(len(x_imgs)):
                    batch_x_img = x_imgs
                    batch_x_attribs = x_attribs
                    batch_x_embs = x_embs
                    batch_y = y_labels
            else:
                idx = np.random.randint(0, int(x_imgs.shape[0]), bsz)

                for i in range(bsz):
                    batch_x_img.append(x_imgs[idx[i]])
                    batch_x_attribs.append(x_attribs[idx[i]])
                    batch_x_embs.append(x_embs[idx[i]])
                    batch_y.append(y_labels[idx[i]])


            return np.array(batch_x_img), np.array(batch_x_attribs), np.array(batch_x_embs), np.array(batch_y)



    @staticmethod
    def generate_split(x_imgs, x_attribs, y_labels, x_embs, validation_split):
        # """
        # Splits any X, y, data into train, validation, test
        # :param x_data: X array
        # :param y_data: y array
        # :param validation_split: float percent of validation size, 0 if split not required
        # :param test_split: float percent of test size, 0 if split not required
        # :return: X_train, y_train, X_valid, y_valid, X_test, y_test
        # """
        num_valid = int(x_imgs.shape[0] * validation_split)
        validInds = np.random.randint(0, x_imgs.shape[0], num_valid)

        X_train_imgs, X_train_attribs, X_train_embs, X_valid_imgs, X_valid_attribs, X_valid_embs, \
            y_train_labels, y_valid_labels = [], [], [], [], [], [], [], []

        for i in range(x_imgs.shape[0]):
            if i in validInds:
                X_valid_imgs.append(x_imgs[i])
                X_valid_attribs.append(x_attribs[i])
                X_valid_embs.append(x_embs[i])
                y_valid_labels.append([y_labels[i]])
            else:
                X_train_imgs.append(x_imgs[i])
                X_train_attribs.append(x_attribs[i])
                X_train_embs.append(x_embs[i])
                y_train_labels.append([y_labels[i]])

        return np.array(X_train_imgs), np.array(X_train_attribs), np.array(X_train_embs), np.array(y_train_labels), \
               np.array(X_valid_imgs), np.array(X_valid_attribs), np.array(X_valid_embs), np.array(y_valid_labels)


