import numpy as np
from matplotlib import pyplot as plt, colors

from lib import models
import tensorflow as tf
from lib.DataHandler import MNIST

import glob
# import imageio
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

'''
Testing how to prepre the dataset
    - split into trainig, test and validation
    - creating supervised, unsupervised and semi supervised data
    - using the DataHandler of A3/A4
    - setting datapoints as normal, anomaly and unknown
'''
# -------------------------------------------------------------------------------------------------------------
anomaly = [8, 9]
# delete_y = [7]
# delete_x = [7]
drop = [1, 2, 3]
include = [0, 4, 5, 6, 7, 8, 9]
# -------------------------------------------------------------------------------------------------------------
# Traingins Data
# Setting the seed
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
mnist = MNIST(random_state=random_seed)

print('Training Data...')
x_train, y_train, y_train_original = mnist.get_datasplit('train', anomaly, drop, include,
                                                         None, None)
print(x_train.shape)
print(y_train.shape)
print(y_train_original.shape)

# -------------------------------------------------------------------------------------------------------------
# Testdata
print('Test Data...')
x_test, y_test, y_test_original = mnist.get_datasplit('test', anomaly, drop, include,
                                                      None, None)
print(x_test.shape)
print(y_test.shape)
print(y_test_original.shape)

# -------------------------------------------------------------------------------------------------------------
# Validation data
print('Validation Data...')
x_val, y_val, y_val_original = mnist.get_datasplit('val', anomaly, drop, include,
                                                   None, None)
print(x_val.shape)
print(y_val.shape)
print(y_val_original.shape)
# -------------------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------------------
# Generator
# aae = models.AAE()
# encoder = aae.create_encoder()
# decoder = aae.create_decoder()
#
# shape_noise = aae.shape_noise
# image_size = aae.image_size
# discriminator_input = aae.gan_discriminator_input
#
# generator = aae.noise_generator()
# # -------------------------------------------------------------------------------------------------------------
# training_data = []
# noise_label = []
# mean = 2 # 100
# stddev = 1 # 50
# epoch = 100
#
# noise = tf.random.normal([3, 4], mean=mean, stddev=stddev, seed=1993)
# print(noise)
# # z = encoder(noise, training=False)
# # blub = tf.random.normal([1, 2], mean=2, stddev=0.5, seed=1993)
# # new = np.concatenate((z, blub), axis=0)
# # img = decoder(new, training=False)
#
# fig = plt.figure()
# count, bins, ignored = plt.hist(noise, 30, density=True)
# plt.plot(bins, 1/(stddev * np.sqrt(2 * np.pi)) *
#                np.exp(- (bins - mean)**2 / (2 * stddev**2)),
#          linewidth=2, color='r')
# plt.show()

# plt.imshow(img, cmap='gray')
# plt.savefig('image_generated')
# training_data.append(img)
# # noise_label.append(10)
#
# noise_label = np.array(noise_label)
# training_data = np.array(training_data)
# training_data = training_data.reshape((-1, 28 * 28))
#
# print(training_data.shape)
# new_dataset = np.concatenate((x_train, training_data), axis=0)
# new_labels = np.concatenate((y_train, noise_label))

# -------------------------------------------------------------------------------------------------------------
# plot_data = tf.random.normal([1, 784], mean=mean, stddev=stddev, seed=1993)
# fig = plt.figure()
# count, bins, ignored = plt.hist(plot_data, 30, density=True)
# plt.plot(bins, 1/(stddev * np.sqrt(2 * np.pi)) *
#                np.exp(- (bins - mean)**2 / (2 * stddev**2)),
#          linewidth=2, color='r')
# plt.show()
# plt.savefig('image_distribution')
# -------------------------------------------------------------------------------------------------------------
# discriminator = aae.noise_discrimniator()
# decision = discriminator(img)
# print(decision)


# -------------------------------------------------------------------------------------------------------------
# data = encoder(training_data, training=False)
# label_list = []
# for i in range(epoch):
#     label_list.append(1)
#
# cmap = colors.ListedColormap(['blue'])
# # bounds = [0, 5, 10]
# # norm = colors.BoundaryNorm(bounds, cmap.N)
#
# fig, ax = plt.subplots()
# scatter = ax.scatter(training_data[:, 0], training_data[:, 1], c=label_list,
#                      alpha=0.9, s=2, cmap=cmap)
#
# legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
# ax.add_artist(legend)
#
# plt.savefig('generated_data_spread.png')
# plt.close('all')
# -------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------

