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
anomaly = [4]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [2, 4]


x_train, y_train = mnist.get_anomdata_nolabels('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)









