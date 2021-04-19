# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from lib import models, DataHandler

import time
from pathlib import Path

# Reduce the hunger of TF when we're training on a GPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

ROOT_PATH = Path.cwd()

# Setting the seed
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# Path for images and results
output_dir = ROOT_PATH / 'output'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'semisupervised_aae_deterministic'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

# Data
# @TODO Wie mach ich das Ganze semi supervised?!
mnist = DataHandler.MNIST()
batch_size = 256
train_buf = 60000

train_dataset = tf.data.Dataset.from_tensor_slices((mnist.x_train, mnist.y_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)