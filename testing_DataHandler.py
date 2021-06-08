from matplotlib import pyplot as plt

from lib.DataHandler import MNIST
from lib import models
import tensorflow as tf



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

anomaly = [9]
drop = [0, 2, 3, 4, 5, 6, 7, 8]
include = [1, 9]

# ---------------------------------------------------------
# Traingins Data
print('Training Data...')
x_train, y_train, y_train_original = mnist.get_datasplit('train', anomaly, drop, include, None, None)
print(x_train.shape)
print(y_train.shape)
print(y_train_original.shape)


x_trai2n, y_trai2n, y_trai2n_original = mnist.get_datasplit('train', anomaly, drop, include, None, None)
print(x_trai2n.shape)
print(y_trai2n.shape)
print(y_trai2n_original.shape)


label_list = list(y_train)
classes = set(label_list)
print(classes)

# Generator
aae = models.AAE()
shape_noise = aae.shape_noise
generator = aae.noise_generator()

noise = tf.random.uniform([1, 100])
img = generator(noise, training=False)
plt.imshow(img[0, :, :, 0], cmap='gray')
plt.savefig('image')





