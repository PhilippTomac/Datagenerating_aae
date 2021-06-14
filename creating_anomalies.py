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

z = tf.random.normal([1, 2], mean=2, stddev=0.1)
decoder_out = decoder(z)

print(decoder_out.shape)
