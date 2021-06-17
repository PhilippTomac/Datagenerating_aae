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
output_dir = ROOT_PATH / 'Generated_Data'
output_dir.mkdir(exist_ok=True)
# ------------------------------------------------------------------------------------------------------
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
# ------------------------------------------------------------------------------------------------------
# Get the Data
anomaly = []
drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
include = [1, 9]

print("1. Loading and Preprocessing Data with DataHandler.py")
mnist = MNIST(random_state=random_seed)


x_train, y_train = mnist.get_supervised_data('train', drop, include)
print(y_train.shape)
# ------------------------------------------------------------------------------------------------------
# Create and load the trained models
aae = models.AAE()
z_dim = aae.z_dim
h_dim = aae.h_dim
image_size = aae.image_size

encoder = aae.create_encoder()
decoder = aae.create_decoder()
bad_decoder = aae.create_decoder()


encoder.load_weights('/home/fipsi/Documents/Code/Masterarbeit_GPU/SavedModels_Encoder/encoder_weights')
decoder.load_weights('/home/fipsi/Documents/Code/Masterarbeit_GPU/SavedModels_Decoder/decoder_weights')

# ------------------------------------------------------------------------------------------------------
# Plotting latent space of the trained encoder
e_out = encoder(x_train, training=False)
label_list = list(y_train)
cmap = colors.ListedColormap(['blue', 'red'])
bounds = [0, 5, 10]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
scatter = ax.scatter(e_out[:, 0], e_out[:, 1], c=label_list, alpha=.9, s=2, cmap=cmap)

legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend)
plt.savefig(output_dir / 'latent_space.png')
plt.close('all')

# ------------------------------------------------------------------------------------------------------
# Creating One Datapoint
z = tf.random.normal([1, 2], mean=1.5, stddev=0.6)
# Generating one image from Code Layer z
decoder_out = decoder(z)
generated_image = decoder_out.numpy()
restore_image = generated_image.reshape(generated_image.shape[0], 28, 28)
plt.imshow(restore_image[0], cmap='gray')
plt.savefig(output_dir / 'test_generated.png')
plt.close('all')

# ------------------------------------------------------------------------------------------------------
# Creating one image with the untrained decoder for comparison
bad_decoder_out = bad_decoder(z)
bad_generated_image = bad_decoder_out.numpy()
bad_restore_image = bad_generated_image.reshape(bad_generated_image.shape[0], 28, 28)
plt.imshow(bad_restore_image[0], cmap='gray')
plt.savefig(output_dir / 'untrained_generated.png')
plt.close('all')

# ------------------------------------------------------------------------------------------------------
# Creaing n Datapoints for new Dataset
data = []
images = []
dataset = []
labels = []

for i in range(10):
    z = tf.random.normal([1, 2], mean=1.5, stddev=0.6)
    data.append(z)

for i in range(len(data)):
    decoder_out = decoder(data[i])
    x_train = tf.concat([x_train, decoder_out], axis=0)
    decoder_out_numpy = decoder_out.numpy()
    restored_img = decoder_out_numpy.reshape(28, 28, decoder_out_numpy.shape[0])
    images.append(restored_img)
    labels.append(9)

labels = np.array(labels)


# Saving the generated Data as new Dataset (datapoints and labels)
np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_Images', images)
np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_labels', labels)

a = np.load('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_Images.npy')

for i in range(len(a)):
    plt.imshow(a[i], cmap='gray')
    plt.savefig(output_dir / ('test_generated_loop_%d.png' % i))
    plt.close('all')







