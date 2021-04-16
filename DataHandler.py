import numpy as np
import pandas as pd
import tensorflow as tf

# Loading the MNIST Dataset
class MNIST():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
