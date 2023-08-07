import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input,ReLU
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.layers import concatenate, BatchNormalization
import numpy as np


def coverage_test(model_prob_input, model_labels, nbins=50):
    """
    Coverage test for validation data
    :param model_prob_input: estimated model posterior probability (K/(1+K)) for validation data
    :param model_labels: labels for validation data
    :param nbins: number of bins in coverage test
    :return: expected probability (binned), empirical probability (true model fraction), model label count
    """
    probability_array = np.linspace(0, 1, nbins)
    label_fraction = []
    label_count = []

    for i in range(len(probability_array) - 1):
        bin_indices = np.where(((model_prob_input < probability_array[i + 1]) &
                                (model_prob_input > probability_array[i])))

        labels_in_prob_bin = model_labels[bin_indices]

        label_fraction.append(float(labels_in_prob_bin.mean()))
        label_count.append(float(len(labels_in_prob_bin)))

    label_fraction = np.array(label_fraction)
    label_count = np.array(label_count)

    probability_bin_centre = 0.5 * (probability_array[:-1] + probability_array[1:])[np.where(label_count > 0)]
    label_fraction = label_fraction[np.where(label_count > 0)]
    label_count = label_count[np.where(label_count > 0)]

    return probability_bin_centre, label_fraction, label_count


def smooth_sign(x, k =100.):
    """
    Gives the option of a smooth approximation to the sign function
    :param x: input
    :param k: smoothness parameter
    :return: smooth sign function
    """
    
    return 2.*tf.math.sigmoid(k*x) - 1


def parity_odd_power(x, alpha=2):
    """
    Parity odd power (POP) function
    :param x: input
    :param alpha: power
    :return: POP output
    """
    return x*(tf.math.abs(x)**(alpha-1))


def leaky_parity_odd_power(x, alpha=2):
    """
    Leaky parity odd power function (lPOP)
    :param x: input
    :param alpha: power (need not be an integer)
    :return: lPOP output
    """
    return x + parity_odd_power(x, alpha)


class POPExpLoss(tf.keras.losses.Loss):
    """
    Custom exponential loss
    """

    def call(self, model_label, model_pred):
        model_pred = leaky_parity_odd_power(model_pred, alpha=1)
        model_pred = tf.clip_by_value(model_pred, -50, 50)
        loss_val = tf.math.exp((0.5 - model_label) * (model_pred) )
        return tf.reduce_mean(loss_val)
    
    
class ExpLoss(tf.keras.losses.Loss):
    """
    Custom exponential loss
    """

    def call(self, model_label, model_pred):
        model_pred = tf.clip_by_value(model_pred, -50, 50)
        loss_val =  tf.math.exp((0.5 - model_label) * (model_pred) )
        return tf.reduce_mean(loss_val)


def get_lr_metric(optimizer):
    """
    Learning rate on-the-fly
    :param optimizer:
    :return: learning rate
    """
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)  # I use ._decayed_lr method instead of .lr
    return lr


class EvidenceNetworkSimple:
    """
    A simple dense network (with some skip connections)
    This becomes the evidence network if: (i) combined with the appropriate loss function (e.g. lPOP),
    (ii) if the data are drawn from the correction distributions, and (iii) if the network output
    is validated (e.g. with a coverage test)
    """

    def __init__(self, input_size, weight_init='he_normal', layer_width=100, added_layers=3,
                 learning_rate=1e-4, decay_rate=None, batch_norm_flag=1, residual_flag=0, alpha=None, first_layer_width=None):
        """
        :param input_size:
        :param weight_init:
        :param layer_width:
        :param added_layers:
        :param learning_rate:
        :param decay_rate:
        :param batch_norm_flag:
        :param residual_flag:
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.layer_width = layer_width
        self.weight_init = weight_init
        self.decay_rate = decay_rate
        self.added_layers = added_layers
        self.batch_norm_flag = batch_norm_flag
        self.residual_flag = residual_flag
        self.alpha = alpha
        if first_layer_width is None:
            self.first_layer_width = int(input_size*1.1) + 20
        else:
            self.first_layer_width = first_layer_width

        print('Hyper-parameters:', input_size, weight_init, layer_width, added_layers,
              learning_rate, decay_rate, batch_norm_flag, residual_flag)

    def simple_layer(self, x_in):
        x_out = Dense(self.layer_width, kernel_initializer=self.weight_init)(x_in)
        x_out = LeakyReLU(alpha=0.1)(x_out)
        if self.batch_norm_flag == 1:
            x_out = BatchNormalization()(x_out)
        return x_out

    def residual_block(self, x_in):
        x_out = Dense(self.layer_width, kernel_initializer=self.weight_init)(x_in)
        x_out = LeakyReLU(alpha=0.1)(x_out)
        if self.batch_norm_flag == 1:
            x_out = BatchNormalization()(x_out)
            
        x_out = Dense(self.layer_width, kernel_initializer=self.weight_init)(x_out)
        x_out = LeakyReLU(alpha=0.1)(x_out) + x_in
        if self.batch_norm_flag == 1:
            x_out = BatchNormalization()(x_out)
             
        return x_out

    def model(self):

        input_data = (Input(shape=(self.input_size,)))

        x1 = Dense(self.first_layer_width, input_dim=self.input_size, kernel_initializer=self.weight_init)(input_data)
        x_inner = LeakyReLU(alpha=0.1)(x1)
        x_inner = BatchNormalization()(x_inner)
        x_inner = Dense(self.layer_width, input_dim=self.input_size, kernel_initializer=self.weight_init)(x_inner)
        x_inner = LeakyReLU(alpha=0.1)(x_inner)
        x_inner = BatchNormalization()(x_inner)

        for i in range(self.added_layers):
            x_inner = self.residual_block(x_inner)

        x_out = Dense(self.layer_width, kernel_initializer=self.weight_init)(x_inner)
        x_out = LeakyReLU(alpha=0.1)(x_out)
        x_out = Dense(1, kernel_initializer=self.weight_init)(x_out)
        x_out = 0.1*x_out + 0.001
        x_out = leaky_parity_odd_power(x_out, alpha=self.alpha)

        dense_model = Model(input_data, x_out)
#         dense_model.summary()

        if self.decay_rate is not None:
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate,
                                                                decay_steps=5000,
                                                                decay_rate=self.decay_rate)
            optimizer = optimizers.Adam(learning_rate=lr_schedule)
            lr_metric = get_lr_metric(optimizer)
            dense_model.compile(optimizer=optimizer,
                                loss=ExpLoss(),
                                metrics=[lr_metric])
        else:
            dense_model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                                loss=ExpLoss())

        return dense_model


class CustomCallback(callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print("Loss: {:.5e}. Val loss: {:.5e}".format(logs['loss'], logs['val_loss']))