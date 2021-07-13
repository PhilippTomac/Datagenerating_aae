## Imports
# Matplot
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
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
# Setting Seed for better comparison
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# ------------------------------------------------------------------------------------------------------
# Create and load the trained models
aae = models.AAE()
'''
:parameter
z_dim = 2 - Compression in middle layer
h_dim = 100 - Denselayer n-neurons
image_size = 784 
'''
z_dim = aae.z_dim
h_dim = aae.h_dim
image_size = aae.image_size

# Creating the needed models for the aae
encoder = aae.create_encoder()
decoder = aae.create_decoder()
bad_decoder = aae.create_decoder()

# Loading the weights fo the pretrained models in the new encoder and decoder
encoder.load_weights('/home/fipsi/Documents/Code/Masterarbeit_GPU/TrainedModels/Encoder_6_7/encoder_weights')
decoder.load_weights('/home/fipsi/Documents/Code/Masterarbeit_GPU/TrainedModels/Decoder_6_7/decoder_weights')
# ------------------------------------------------------------------------------------------------------
# Creaing n Datapoints for new Dataset
'''
:var
data -> list with all Datapoints from normal distribution
images -> list for the generated images
labels -> list for the labels of the genrated images
'''
data = []
images = []
labels = []

# Creating 1000 Datapoints
for i in range(1000):
    # look up where the points are in the latent space
    # set mean and stddev to get the needed information to create the data you need
    z = tf.random.normal([1, 2], mean=3, stddev=3)
    data.append(z)

# Generate images and the labels based on the data in the list data[]
# set the label of the images you generate
label = 7
for i in range(len(data)):
    decoder_out = decoder(data[i])
    # Convert image-tensor to numpy-array
    decoder_out_numpy = decoder_out.numpy()
    # Reshape the image to the original shape of 28x28 pixel with values between 0 and 255
    restored_img = decoder_out_numpy.reshape(28, 28, decoder_out_numpy.shape[0]) * 255.
    images.append(restored_img)
    labels.append(label)

# convert list to a numpy-array
labels = np.array(labels)

# ------------------------------------------------------------------------------------------------------
# Saving the generated Data as new Dataset (datapoints and labels)
'''
The images and labels are saved separately in the File DataPoint_x
x = change this to the label of the data was created
'''

np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_Images', images)
np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_labels', labels)


# Loading the data and plotting it the 10 first images for review
test_images = np.load('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_Images.npy')
for i in range(10):
    plt.imshow(test_images[i], cmap='gray')
    plt.savefig(output_dir / ('test_generated_loop_%d.png' % i))
    plt.close('all')





