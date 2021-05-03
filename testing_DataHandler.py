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
print('Process Data with DataHandler')
mnist = MNIST(random_state=1993)
normal = [0, 1, 2, 3, 5, 6, 7]
anomaly = [4, 9]
include = [8]


x_train_a, y_train_a = mnist.get_semisupervised_data('train', [4, 9], [5, 6, 7, 8], [0, 1, 2, 3, 4, 9])
x_train_unk, y_train_unk = mnist.get_data_unsupervised('train', None, [5, 6, 7, 8])
print(x_train_a.shape)
print(y_train_a.shape)

print(x_train_unk.shape)
print(y_train_unk.shape)

x_train = np.concatenate([x_train_a, x_train_unk], axis=0)
y_train = y_train_a

possible_digits = np.unique(y_train).tolist()
possible_digits = possible_digits
n_samples = len(y_train)
print(possible_digits)
print(n_samples)










