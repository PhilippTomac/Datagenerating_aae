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


aae = models.AAE()
z_dim = aae.z_dim
h_dim = aae.h_dim
image_size = aae.image_size

encoder = aae.create_encoder()
decoder = aae.create_decoder()

decoder.load_weights('/home/fipsi/Documents/Code/Masterarbeit_GPU/SavedModels_Decoder/decoder_weights')

# ------------------------------------------------------------------------------------------------------
# One Datapoint
z = tf.random.normal([1, 2], mean=1.5, stddev=0.6)
decoder_out = decoder(z)

print(decoder_out.shape)
generated_image = decoder_out.numpy()

restore_image = generated_image.reshape(generated_image.shape[0], 28, 28)
print(restore_image.shape)

plt.imshow(restore_image[0], cmap='gray')
plt.savefig('test_generated.png')
plt.close('all')

# ------------------------------------------------------------------------------------------------------
# n Datapoints
data = []

for i  in range(100):
    z = tf.random.normal([1, 2], mean=1.5, stddev=0.6)
    data.append(z)

images = []
print(len(data))
for i in range(len(data)):
    decoder_out = decoder(data[i])
    decoder_out_numpy = decoder_out.numpy()
    restored_img = decoder_out_numpy.reshape(decoder_out_numpy.shape[0], 28, 28)
    images.append(restored_img)
print(images[2].shape)


plt.imshow(images[99][0], cmap='gray')
plt.savefig('test_generated_loop.png')
plt.close('all')





