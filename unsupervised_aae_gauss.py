# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from lib import models

import time
from pathlib import Path

from lib.DataHandler import MNIST

'''
Gauss unsupervised AAE: Here we assume that q(z|x) is a Gaussian distribution whose mean and variance is predicted 
by the encoder network: zi ∼ N(µi(x), σi(x)). In this case, the stochasticity in q(z) comes from both the 
data-distribution and the randomness of the Gaussian distribution at the experiment_results of the encoder. We can use the same 
re-parametrization trick of [Kingma and Welling, 2014] for back-propagation through the encoder network.
'''

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

# Ordner Path for images and results
output_dir = ROOT_PATH / 'experiment_results'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'unsupervisied_aae_guass'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'experiment_a4579'
latent_space_dir.mkdir(exist_ok=True)

# load MNIST dataset
print("Loading and Preprocessing Data with DataHandler.py")
mnist = MNIST(random_state=random_seed)

anomaly = [4, 5, 7, 9]
drop = None
include = list(range(0, 10))

# Traingins Data
x_train, y_train = mnist.get_semisupervised_data('train', anomaly, drop, include)
print(x_train.shape)
print(y_train.shape)

# Testdata
x_test, y_test = mnist.get_semisupervised_data('test', anomaly, drop, include)
print(x_test.shape)
print(y_test.shape)

# Validation data
x_val, y_val = mnist.get_semisupervised_data('val', anomaly, drop, include)
print(x_val.shape)
print(y_val.shape)

# Parameter
batch_size = 256
train_buf = 60000

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

# Creating the models
aae = models.AAE()
z_dim = aae.z_dim
h_dim = aae.h_dim
image_size = aae.image_size

encoder = aae.create_encoder_gauss()
decoder = aae.create_decoder_gauss()
discriminator = aae.create_discriminator_gauss()

# encoder.summary()
# decoder.summary()
# discriminator.summary()

# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()


def autoencoder_loss(inputs, reconstruction, loss_weight):
    return loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, loss_weight):
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_weight * (loss_fake + loss_real)


def generator_loss(fake_output, loss_weight):
    return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)


# Define cyclic learning rate
base_lr = 0.00025
max_lr = 0.0025

n_samples = 60000
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

# -------------------------------------------------------------------------------------------------------------
# Define optimizers

ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)


@tf.function
def train_step(batch_x):
    # -------------------------------------------------------------------------------------------------------------
    # Autoencoder Training
    with tf.GradientTape() as ae_tape:
        # Using Encoder Output for the data distribution
        z_mean, z_std = encoder(batch_x, training=True)

        # Using Gauss distribution with the encoder experiment_results to generate data
        gauss = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
        z = z_mean + (1e-8 + z_std) * gauss
        decoder_output = decoder(z, training=True)

        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape() as dc_tape:
        real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
        z_mean, z_std = encoder(batch_x, training=True)

        # Probabilistic with Gaussian posterior distribution
        gauss = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
        z = z_mean + (1e-8 + z_std) * gauss

        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(z, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Generator (Encoder)
    with tf.GradientTape() as gen_tape:
        z_mean, z_std = encoder(batch_x, training=True)

        # Probabilistic with Gaussian posterior distribution
        gauss = tf.random.normal(shape=z_mean.shape, mean=0, stddev=1)
        z = z_mean + (1e-8 + z_std) * gauss

        dc_fake = discriminator(z, training=True)

        # Generator loss
        gen_loss = generator_loss(dc_fake, gen_loss_weight)

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    return ae_loss, dc_loss, dc_acc, gen_loss


# -------------------------------------------------------------------------------------------------------------
# Training loop
n_epochs = 601
for epoch in range(n_epochs):
    start = time.time()

    # Learning rate schedule
    if epoch in [60, 100, 300]:
        base_lr = base_lr / 2
        max_lr = max_lr / 2
        step_size = step_size / 2

        print('learning rate changed!')

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_loss_avg = tf.metrics.Mean()
    epoch_dc_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()

    for batch, (batch_x) in enumerate(train_dataset):
        # -------------------------------------------------------------------------------------------------------------
        # Calculate cyclic learning rate
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
        ae_optimizer.lr = clr
        dc_optimizer.lr = clr
        gen_optimizer.lr = clr

        ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x)

        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_dc_acc_avg(dc_acc)
        epoch_gen_loss_avg(gen_loss)

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
          .format(epoch, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_ae_loss_avg.result(),
                  epoch_dc_loss_avg.result(),
                  epoch_dc_acc_avg.result(),
                  epoch_gen_loss_avg.result()))

    # -------------------------------------------------------------------------------------------------------------
    if epoch % 100 == 0:
        # Latent space of test set
        z_mean, z_std = encoder(x_test, training=False)

        label_list = list(y_test)

        fig = plt.figure()
        classes = set(label_list)
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        kwargs = {'alpha': 0.8, 'c': [colormap[i] for i in label_list]}
        ax = plt.subplot(111, aspect='equal')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles = [mpatches.Circle((0, 0), label=class_, color=colormap[i])
                   for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45), fancybox=True, loc='center left')
        plt.scatter(z_mean[:, 0], z_mean[:, 1], s=2, **kwargs)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')
