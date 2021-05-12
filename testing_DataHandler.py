import pandas as pd
import tensorflow as tf
import numpy as np

# from lib.DataHandler import MNIST
from numpy import newaxis

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
drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
include = [4, 9]

# Traingins Data
x_train, y_train = mnist.get_supervised_data('train', drop, include)
print(x_train.shape)
print(y_train.shape)

new_x = x_train[:, :, np.newaxis]
print(new_x.shape)
label = list(y_train)
print(len(label))

for i in range(len(label)):
    new_x = np.array([x_train], [label[1]])

print(new_x.shape)






