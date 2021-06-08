import numpy as np
from matplotlib import pyplot as plt, colors

from lib.DataHandler import MNIST
from lib import models
import tensorflow as tf
import matplotlib.pyplot as plt
'''
Testing how to prepre the dataset
    - split into trainig, test and validation
    - creating supervised, unsupervised and semi supervised data
    - using the DataHandler of A3/A4
    - setting datapoints as normal, anomaly and unknown
'''
# -------------------------------------------------------------------------------------------------------------
# Generator
aae = models.AAE()
encoder = aae.create_encoder()

shape_noise = aae.shape_noise
image_size = aae.image_size
generator = aae.noise_generator()
# -------------------------------------------------------------------------------------------------------------
training_data = []
mean = 3
stddev = 0.05
epoch = 100
for i in range(epoch):
    noise = tf.random.normal([1, 784], mean=mean, stddev=stddev, seed=1993)
    img = generator(noise, training=False)
    # plt.imshow(img[0, :, :, 0], cmap='gray')
    # plt.savefig('image_%d' % i)
    training_data.append(img)

training_data = np.array(training_data)
training_data = training_data.reshape((-1, 28 * 28))

print(training_data.shape)

# -------------------------------------------------------------------------------------------------------------
plot_data = tf.random.normal([1, 784], mean=mean, stddev=stddev, seed=1993)
count, bins, ignored = plt.hist(plot_data, 30, density=True)
plt.plot(bins, 1/(stddev * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mean)**2 / (2 * stddev**2) ),
         linewidth=2, color='r')
plt.show()

# -------------------------------------------------------------------------------------------------------------
data = encoder(training_data, training=False)
label_list = []
for i in range(epoch):
    label_list.append(1)

cmap = colors.ListedColormap(['blue'])
# bounds = [0, 5, 10]
# norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
scatter = ax.scatter(training_data[:, 0], training_data[:, 1], c=label_list,
                     alpha=0.9, s=2, cmap=cmap)

legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend)

plt.savefig('validation_latentspace.png')
plt.close('all')
# -------------------------------------------------------------------------------------------------------------
