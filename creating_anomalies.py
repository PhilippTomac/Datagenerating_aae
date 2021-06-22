## Imports
# Matplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
from matplotlib import gridspec, colors
from matplotlib.cm import get_cmap

import numpy as np
import tensorflow as tf
import time
from pathlib import Path

from lib import models
from lib.DataHandler import MNIST

# ------------------------------------------------------------------------------------------------------
# GPU:
# Reduce the hunger of TF when we're training on a GPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

# ------------------------------------------------------------------------------------------------------
ROOT_PATH = Path.cwd()
# Path for images and results
output_dir = ROOT_PATH / 'Generated_Data/DataPoint_1'
output_dir.mkdir(exist_ok=True)
# ------------------------------------------------------------------------------------------------------
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------
# Create and load the trained models
aae = models.AAE()
z_dim = aae.z_dim
h_dim = aae.h_dim
image_size = aae.image_size

encoder = aae.create_encoder()
decoder = aae.create_decoder()
bad_decoder = aae.create_decoder()


encoder.load_weights('/home/fipsi/Documents/Code/Masterarbeit_GPU/TrainedModels/Encoder_6_7/encoder_weights')
decoder.load_weights('/home/fipsi/Documents/Code/Masterarbeit_GPU/TrainedModels/Decoder_6_7/decoder_weights')
# ------------------------------------------------------------------------------------------------------
# Creaing n Datapoints for new Dataset
data = []
images = []
dataset = []
labels = []

for i in range(1000):
    z = tf.random.normal([1, 2], mean=3, stddev=3)
    data.append(z)

for i in range(len(data)):
    decoder_out = decoder(data[i])
    decoder_out_numpy = decoder_out.numpy()
    restored_img = decoder_out_numpy.reshape(28, 28, decoder_out_numpy.shape[0]) * 255.
    images.append(restored_img)
    labels.append(7)

labels = np.array(labels)

# Saving the generated Data as new Dataset (datapoints and labels)
np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_Images', images)
np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_labels', labels)

a = np.load('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_Images.npy')


for i in range(10):
    plt.imshow(a[i], cmap='gray')
    plt.savefig(output_dir / ('test_generated_loop_%d.png' % i))
    plt.close('all')





