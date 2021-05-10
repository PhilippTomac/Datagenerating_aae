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

print('Referenz 1 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 2, 3, 4, 5, 6, 7, 8]
include = [1, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = mnist.get_semisupervised_data('test', anomaly, drop, include)
print(x_test.shape)
print(y_test.shape)

x_val, y_val = mnist.get_semisupervised_data('val', anomaly, drop, include)
print(x_val.shape)
print(y_val.shape)


print('Referenz 2 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = mnist.get_semisupervised_data('test', anomaly, drop, include)
print(x_test.shape)
print(y_test.shape)

x_val, y_val = mnist.get_semisupervised_data('val', anomaly, drop, include)
print(x_val.shape)
print(y_val.shape)

print('Referenz 3 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)


print('Referenz 4 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)



print('Test 1 --------------------------------------------------------------------------------')
# Test 1
anomaly = [7]
drop = [0, 1, 3, 4, 5, 6, 8, 9]
include = [2, 7]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = mnist.get_semisupervised_data('test', anomaly, drop, include)
print(x_test.shape)
print(y_test.shape)

x_val, y_val = mnist.get_semisupervised_data('val', anomaly, drop, include)
print(x_val.shape)
print(y_val.shape)

# Test 2
print('Test 2 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = mnist.get_semisupervised_data('test', anomaly, drop, include)
print(x_test.shape)
print(y_test.shape)

x_val, y_val = mnist.get_semisupervised_data('val', anomaly, drop, include)
print(x_val.shape)
print(y_val.shape)

print('Test 3 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = mnist.get_semisupervised_data('test', anomaly, drop, include)
print(x_test.shape)
print(y_test.shape)

x_val, y_val = mnist.get_semisupervised_data('val', anomaly, drop, include)
print(x_val.shape)
print(y_val.shape)


print('Test 4 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = mnist.get_semisupervised_data('test', anomaly, drop, include)
print(x_test.shape)
print(y_test.shape)

x_val, y_val = mnist.get_semisupervised_data('val', anomaly, drop, include)
print(x_val.shape)
print(y_val.shape)


print('Test 5 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)



print('Test 6 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)



print('Test 7 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)


print('Test 8 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)

print('Test 9 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)


print('Test 10 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)


print('Test 11 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)


print('Test 12 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)



print('Test 13 --------------------------------------------------------------------------------')
anomaly = [9]
drop = [0, 1, 3, 5, 6, 7, 8, 9]
include = [4, 9]

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include, 0)
print(x_train.shape)
print(y_train.shape)

x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, [4])
print(x_train.shape)
print(y_train.shape)