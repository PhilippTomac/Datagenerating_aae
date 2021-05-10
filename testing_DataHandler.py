import pandas as pd
import tensorflow as tf
import numpy as np

# from lib.DataHandler import MNIST
from lib.DataHandler import MNIST


'''
Testing how to prepre the dataset
    - split into trainig, test and validation
    - creating supervised, unsupervised and semi supervised data
    - using the DataHandler of A3/A4
    - setting datapoints as normal, anomaly and unknown
'''
# ----------------------------------------------------------------------------------------------------------------------
# Loading Data
print('Process Data with DataHandler')
mnist = MNIST(random_state=1993)

anomaly = [4]
delete_labels = [9]
drop = [0, 2, 3, 5, 6, 7, 8]
include = [1, 4, 9]

# Traingins Data
x_train, y_train = mnist.get_experiment_data('train', anomaly, drop, include, delete_labels)
print(x_train.shape)
print(y_train.shape)

# Testdata
x_test, y_test = mnist.get_experiment_data('test', anomaly, drop, include, delete_labels)
print(x_test.shape)
print(y_test.shape)

# Validation data
x_val, y_val = mnist.get_experiment_data('val', anomaly, drop, include, delete_labels)
print(x_val.shape)
print(y_val.shape)