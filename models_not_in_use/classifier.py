## Imports
# Matplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec, colors
from matplotlib.cm import get_cmap

import numpy as np
import tensorflow as tf
import time
from pathlib import Path

from lib import models
from lib.DataHandler import MNIST

# GPU:
# Reduce the hunger of TF when we're training on a GPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

# Setting the seed
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

anomaly = []
drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
include = [1, 9]

print("1. Loading and Preprocessing Data with DataHandler.py")
mnist = MNIST(random_state=random_seed)

x_train, y_train, y_train_o = mnist.get_datasplit('train', anomaly, drop, include, None, None)
x_test, y_test, y_test_o = mnist.get_datasplit('test', anomaly, drop, include, None, None)
x_val, y_val, y_val_o = mnist.get_datasplit('val', anomaly, drop, include, None, None)

print(x_train.shape)
x_train = x_train.reshape()



aae = models.AAE()
z_dim = aae.z_dim
h_dim = aae.h_dim
image_size = aae.image_size

encoder = aae.create_encoder()

decoder = aae.create_decoder()
classifier = aae.create_classifier()
classifier.summary()




batch = 256
stepsize = len(x_train) / batch

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = classifier.fit(x_train, y_train, batch_size=batch,
                                   steps_per_epoch=stepsize, epochs=50,
                                   validation_data=(x_test, y_test))

test_loss, test_acc = classifier.evaluate(x_val, y_val, verbose=2)

print(history.history.keys())

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.6, 1])
plt.legend(loc='lower right')

print(test_acc)
