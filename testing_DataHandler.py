import tensorflow as tf
import numpy as np

from lib.DataHandler import MNIST, ExperimentConfig


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
# print("-----------------------------------------------Training--------------------------------------------------------")
#
# x_train, y_train = mnist.get_target_classifier_data('train', list(range(6, 9)), list(range(0, 5)))
# print(x_train.shape)
# print(y_train.shape)
#
# possible_digits = np.unique(y_train).tolist()
# n_samples = len(y_train)
# print(possible_digits)
# print(n_samples)
#
# print("--------------------------------------------------Test------------------------------------------------------")
# x_test, y_test = mnist.get_target_classifier_data('test', list(range(6, 9)), list(range(0, 5)))
# print(x_test.shape)
# print(y_test.shape)
#
# possible_digits = np.unique(y_test).tolist()
# n_samples = len(y_test)
# print(possible_digits)
# print(n_samples)
#
#
# print("------------------------------------------------Validation----------------------------------------------------")
# x_val, y_val = mnist.get_target_classifier_data('val', list(range(6, 9)), list(range(0, 5)))
# print(x_val.shape)
# print(y_val.shape)
#
# possible_digits = np.unique(y_val).tolist()
# n_samples = len(y_val)
# print(possible_digits)
# print(n_samples)


print("---------------------------------------------------------------------------------------------------------------")
print("-----------------------------------------------Training--------------------------------------------------------")
x_train, y_train = mnist.get_alarm_data('train',  list(range(6, 9)), None, list(range(0, 9)))
print(x_train.shape)
print(y_train.shape)

possible_digits = np.unique(y_train).tolist()
n_samples = len(y_train)
print(possible_digits)
print(n_samples)

print("-----------------------------------------------Test--------------------------------------------------------")
x_test, y_test = mnist.get_alarm_data('test',  list(range(6, 9)), None, list(range(0, 9)))
print(x_test.shape)
print(y_test.shape)

possible_digits = np.unique(y_test).tolist()
n_samples = len(y_test)
print(possible_digits)
print(n_samples)
print("-----------------------------------------------Validation------------------------------------------------------")
x_val, y_val = mnist.get_alarm_data('test',  list(range(6, 9)), None, list(range(0, 9)))
print(x_val.shape)
print(y_val.shape)

possible_digits = np.unique(y_val).tolist()
n_samples = len(y_val)
print(possible_digits)
print(n_samples)




# xtrain = mnist.x_train
# ytrain = mnist.y_train
#
# xtest = mnist.x_test
# ytest = mnist.y_test
#
# xval = mnist.x_val
# yval = mnist.y_val
# print('Shape of x_train and y_train')
# print(xtrain.shape)
# print(ytrain.shape)
# print('Shape of x_test and y_test')
# print(xtest.shape)
# print(ytest.shape)
# print('Shape of x_val and y_val')
# print(xval.shape)
# print(yval.shape)

# possible_digits = np.unique(mnist.y_train).tolist()
# n_samples = len(mnist.y_train)
# print(possible_digits)
# print(n_samples)


# ----------------------------------------------------------------------------------------------------------------------
# print('---------------------------------------------------------------------------------------------------------------')
# print('Process Data without DataHandler')
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
#
# # Flatten the dataset
# x_train = x_train.reshape((-1, 28 * 28))
# x_test = x_test.reshape((-1, 28 * 28))
#
# print('Shape of x_train and y_train')
# print(x_train.shape)
# print(y_train.shape)
#
# print('Shape of x_test and y_test')
# print(x_test.shape)
# print(y_test.shape)
#
# a_possible_digits = np.unique(y_train).tolist()
# a_n_samples = len(y_train)
# print(a_possible_digits)
# print(a_n_samples)

# ----------------------------------------------------------------------------------------------------------------------
# Set specific labels as normal and as anomalies



