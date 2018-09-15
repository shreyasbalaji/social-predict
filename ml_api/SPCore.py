import tensorflow as tf
import numpy as np
import time




class SP():
    def __init__(self, config, log_path, npi_core_dim=256, npi_core_layers=2, verbose=0):

        self.core, self.state_dim, self.program_dim = core, core.state_dim, core.program_dim
        self.bsz, self.npi_core_dim, self.npi_core_layers = core.bsz, npi_core_dim, npi_core_layers
        self.env_in, self.arg_in, self.prg_in = core.env_in, core.arg_in, core.prg_in
        self.state_encoding, self.program_embedding = core.state_encoding, core.program_embedding
        self.num_args, self.arg_depth = config["ARGUMENT_NUM"], config["ARGUMENT_DEPTH"]
        self.num_progs, self.key_dim = config["PROGRAM_NUM"], config["PROGRAM_KEY_SIZE"]
        self.log_path, self.verbose = log_path, verbose

        # Setup Label Placeholders
        self.y_term = tf.placeholder(tf.int64, shape=[None], name='Termination_Y')
        self.y_prog = tf.placeholder(tf.int64, shape=[None], name='Program_Y')
        self.y_args = [tf.placeholder(tf.int64, shape=[None, self.arg_depth],
                                      name='Arg{}_Y'.format(str(i))) for i in range(self.num_args)]

        self.convolition_net = self.convolution_net()

        self.primary_net = self.primary_net()

        self.deep_net = self.deep_net()





        # Build Losses
        self.t_loss, self.p_loss, self.a_losses = self.build_losses()
        self.default_loss = 2 * self.t_loss + self.p_loss
        self.arg_loss = 0.25 * sum([self.t_loss, self.p_loss]) + sum(self.a_losses)

        # Build Optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.0001, self.global_step, 10000, 0.95,
                                                        staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Build Metrics
        self.p_metric = self.build_metrics()


        # Build Train Ops
        self.default_train_op = self.opt.minimize(self.default_loss, global_step=self.global_step)
        self.arg_train_op = self.opt.minimize(self.arg_loss, global_step=self.global_step)



    def SP_core(self):


    def convolution_net(self):

        return convolution_out

    def primary_net(self):

        return primary_out

    def deep_net(self):

        return deep_out



    # def build_losses(self):
    #     """
    #     Build separate loss computations, using the logits from each of the sub-networks.
    #     """
    #     # Termination Network Loss
    #     termination_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         logits=self.terminate, labels=self.y_term), name='Termination_Network_Loss')
    #
    #     # Program Network Loss
    #     program_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         logits=self.program_distribution, labels=self.y_prog), name='Program_Network_Loss')
    #
    #     # Argument Network Losses
    #     arg_losses = []
    #     for i in range(self.num_args):
    #         arg_losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #             logits=self.arguments[i], labels=self.y_args[i]), name='Argument{}_Network_Loss'.format(str(i))))
    #
    #     return termination_loss, program_loss, arg_losses
    #
    # def build_metrics(self):
    #     """
    #     Build accuracy metrics for each of the sub-networks.
    #     """
    #     term_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.terminate, 1),
    #                                                   self.y_term),
    #                                          tf.float32), name='Termination_Accuracy')
    #
    #     program_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.program_distribution, 1),
    #                                                      self.y_prog),
    #                                             tf.float32), name='Program_Accuracy')
    #
    #     arg_metrics = []
    #     for i in range(self.num_args):
    #         arg_metrics.append(tf.reduce_mean(
    #             tf.cast(tf.equal(tf.argmax(self.arguments[i], 1), tf.argmax(self.y_args[i], 1)),
    #                     tf.float32), name='Argument{}_Accuracy'.format(str(i))))
    #
    #     return term_metric, program_metric, arg_metrics

    def create_placeholders(img_dim = None, attrib_dim = None, output_dim = None):
        """
        :param input_dim: int
        :param output_dim: int
        :return: tuple
        """

        x_img = tf.placeholder(shape=(None, img_dim), dtype=tf.float32, name="X_img")
        x_attribs = tf.placeholder(shape=(None, attrib_dim), dtype=tf.float32, name="X_attribs")
        y = tf.placeholder(shape=(None, output_dim), dtype=tf.float32, name="Y")

        return x_img, x_attribs, y


    ################################

    def initialize_parameters(layer_dims, layer_activations=None):
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

        for i in range(1, n_layers + 1):
            parameters['w' + str(i)] = tf.get_variable('w' + str(i), [layer_dims[i - 1], layer_dims[i]],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       dtype=tf.float32)
            parameters['b' + str(i)] = tf.get_variable('b' + str(i), [layer_dims[i]],
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       dtype=tf.float32)
            parameters['l' + str(i)] = layer_activations[i] if layer_activations != None else 'linear'

        # set activation func for output layer to linear for loss function
        parameters['l' + str(n_layers)] = 'linear'

        return parameters, n_layers

    def activate(a, activation_f):
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
        else:
            raise ValueError("Not a valid activation function name")

    def forward_propagation(inp, parameters, n_layers):
        """
        Calculates the output of the neural net given input inp
        :param inp: tensor
        :param parameters: model parameters dictionary
        :param n_layers: int number of layers
        :return: output tensor
        """
        a = None
        z = tf.cast(inp, tf.float32)

        for l in range(1, n_layers):
            a = tf.add(tf.matmul(z, parameters['w' + str(l)]), parameters['b' + str(l)])
            z = activate(a, parameters['l' + str(l)])

        return z

    def compute_loss(y, pred, size, loss_f='reduce_sum'):
        """
        Calculates loss of net with test/validation label
        :param y: label
        :param pred: output prediction
        :param size: int data points
        :param loss_f: string of name of loss function
        :return: loss according to loss-f
        """
        if loss_f == "reduce_sum":
            loss = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * int(size))
            return loss

    def predict_model(data, parameters, n_layers):
        """
        Runs data through Neural Net
        :param data: dataset of input arrays
        :param parameters: dictionary of trained model parameters
        :param n_layers: int number of layers
        :return: output array prediction labels
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            fprop_result = forward_propagation(data, parameters, n_layers)
            prediction = fprop_result.eval()

        return prediction

    def compute_error(predictions, labels, error_f='rmse'):
        """
        Root Mean Squared Error
        :param predictions: np array
        :param labels: np array
        :param error_f: string of error function name
        :return: error according to error_f
        """
        prediction_loss = 0

        if error_f == 'rmse':
            prediction_size = predictions.shape[0]
            prediction_loss = np.sqrt(np.sum(np.square(labels - predictions)) / prediction_size)

        return prediction_loss

    def get_next_batch(x_data, y_data, batch_size):
        """
        Selectes random set of batch_size elements from x_data and y_data
        :param x_data: np matrix
        :param y_data: np matrix/array
        :param batch_size: int
        :return: batch_x, batch_y
        """
        idx = np.random.randint(0, int(x_data.shape[0]), batch_size)
        batch_x = []
        batch_y = []

        for i in range(batch_size):
            batch_x.append(x_data[idx[i]])
            batch_y.append(y_data[idx[i]])

        return np.array(batch_x), np.array(batch_y)

    def generate_split(x_data, y_data, validation_split, test_split):
        """
        Splits any X, y, data into train, validation, test
        :param x_data: X array
        :param y_data: y array
        :param validation_split: float percent of validation size, 0 if split not required
        :param test_split: float percent of test size, 0 if split not required
        :return: X_train, y_train, X_valid, y_valid, X_test, y_test
        """
        num_test = int(x_data.shape[0] * test_split)
        testInds = np.random.randint(0, x_data.shape[0], num_test)

        num_val = int(x_data.shape[0] * validation_split)
        valInds = []
        for i in range(num_val):
            val_Ind = np.random.randint(0, x_data.shape[0])
            found = False
            while not found:
                if val_Ind in testInds:
                    val_Ind = np.random.randint(0, x_data.shape[0])
                else:
                    found = True
            valInds.append(val_Ind)

        X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []
        for i in range(x_data.shape[0]):
            if i in testInds:
                X_test.append(x_data[i])
                y_test.append([y_data[i]])
            elif i in valInds:
                X_valid.append(x_data[i])
                y_valid.append([y_data[i]])
            else:
                X_train.append(x_data[i])
                y_train.append([y_data[i]])

        return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid), \
               np.array(X_test), np.array(y_test)

    def model(X_train, y_train, X_valid, y_valid, X_test, y_test, input_dim, output_dim, layer_dims, learning_rate,
              num_epochs, display_step, batch_size, train_data_size, valid_data_size, test_data_size):
        """
        Tensorflow Dense Neural Network (DNN) Core Module
        :param X_train: training input
        :param y_train: training labels
        :param X_valid: validation input
        :param y_valid: validation labels
        :param X_test: testing input
        :param y_test: testing labels
        :param input_dim: int size of input
        :param output_dim: int size of output
        :param layer_dims: array of ints of number of neurons per layer
        :param learning_rate: float rate
        :param num_epochs: number of epochs of training
        :param display_step: int display logs every display_step step
        :param batch_size: int size of batch
        :param train_data_size: int number of training elements
        :param valid_data_size: int number of validation elements
        :param test_data_size: int number of testing elements
        """

        # Create placeholdes
        X, y = create_placeholders(input_dim, output_dim)

        # Init all params
        p, n_l = initialize_parameters(layer_dims)

        # Set prediction and loss functions
        train_pred = forward_propagation(X, p, n_l)
        train_loss = compute_loss(y, train_pred, train_data_size, 'reduce_sum')

        valid_pred = forward_propagation(X, p, n_l)
        valid_loss = compute_loss(y, valid_pred, valid_data_size, 'reduce_sum')

        # Set Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)

        # Train Model
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            # Run initializer
            sess.run(init)

            # Fit training data
            for ep in range(num_epochs):

                epoch_train_loss = 0
                epoch_valid_loss = 0
                num_batches = int(float(train_data_size) / batch_size)

                for batch in range(num_batches):
                    X_batch, y_batch = get_next_batch(X_train, y_train, batch_size)
                    _, batch_train_loss, batch_valid_loss = sess.run([optimizer, train_loss, valid_loss],
                                                                     feed_dict={X: X_batch, y: y_batch})

                    epoch_train_loss += batch_train_loss / num_batches
                    epoch_valid_loss += batch_valid_loss / num_batches

                # Display logs
                if (ep + 1) % display_step == 0:
                    print("Epoch:  {},   Training Loss:  {},   Validation Loss:  {}".format(ep + 1,
                                                                                            epoch_train_loss,
                                                                                            epoch_valid_loss))

            # Display final loss
            print("Optimization Finished!")
            final_train_loss = sess.run(train_loss, feed_dict={X: X_train, y: y_train})
            final_valid_loss = sess.run(valid_loss, feed_dict={X: X_valid, y: y_valid})
            print("Final Training Loss:  {},   Final Validation Loss:  {}".format(final_train_loss, final_valid_loss),
                  '\n')

            print("Parameters have been trained, testing model and getting metrics...")

            # Display model predictions
            predictions_final = predict_model(X_train, p, len(layer_dims))
            print("Predictions: ", predictions_final)

            # Display metrics
            train_rmse = compute_error(predictions_final, y_train, 'rmse')
            # test_rmse = compute_error(predict_model(X_test, p, len(layer_dims)), y_test, 'rmse)
            print('Train RMSE: {:.4f}'.format(train_rmse))
            # print('Test RMSE: {:.4f}'.format(test_rmse))