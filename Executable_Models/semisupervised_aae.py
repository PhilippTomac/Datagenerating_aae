# Imports
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import gridspec, colors
from PIL import Image
from matplotlib.cm import get_cmap

from lib import models, DataHandler

import time
from pathlib import Path
'''
TODO: 
 - Check if model is correctly implemented --> Training Process: Gradient
 - Start new test 
    - With known Anomalies its supervised and not semisupervised. Minimize the number of known Anomalies
 - Validate solution of the test
'''
# -------------------------------------------------------------------------------------------------------------

# Reduce the hunger of TF when we're training on a GPU
from lib.DataHandler import MNIST

try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available
    # -------------------------------------------------------------------------------------------------------------

ROOT_PATH = Path.cwd()

# Setting the seed
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
# -------------------------------------------------------------------------------------------------------------

# Path for images and results
output_dir = ROOT_PATH / 'output'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'semisupervised_aae'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'test_13'
latent_space_dir.mkdir(exist_ok=True)

print('Experiment', latent_space_dir, ':')

# -------------------------------------------------------------------------------------------------------------
MULTI_COLOR = True

# Data MNIST
print("Loading and Preprocessing Data with DataHandler.py")
mnist = MNIST(random_state=random_seed)

anomaly = [1, 5]
delete_y = [0, 2, 3, 4]
delete_x = [0, 2, 3, 4]
drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
include = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# ---------------------------------------------------------
# Traingins Data
print('Training Data...')
x_train, y_train, y_train_original = mnist.get_anomdata_nolabels('train', anomaly, drop, include,
                                                                 delete_y, delete_x)
print(x_train.shape)
print(y_train.shape)
print(y_train_original.shape)

# ---------------------------------------------------------
# Testdata
print('Test Data...')
x_test, y_test, y_test_original = mnist.get_anomdata_nolabels('test', anomaly, drop, include, delete_y)
print(x_test.shape)
print(y_test.shape)
print(y_test_original.shape)

# ---------------------------------------------------------
# Validation data
print('Validation Data...')
x_val, y_val, y_val_original = mnist.get_anomdata_nolabels('val', anomaly, drop, include, delete_y)
print(x_val.shape)
print(y_val.shape)
print(y_val_original.shape)
# ---------------------------------------------------------

batch_size = 256
train_buf = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)
# -------------------------------------------------------------------------------------------------------------

# creating the model parts
aae = models.AAE()
# Parameter
image_size = aae.image_size
n_labeled = aae.n_labeled
h_dim = aae.h_dim
z_dim = aae.z_dim

n_labels = 2

encoder_ae = aae.create_encoder_semi()
decoder = aae.create_decoder_sup_semi()
discriminator_labels = aae.create_discriminator_label(n_labels)
discriminator_style = aae.create_discriminator_style()


# -------------------------------------------------------------------------------------------------------------
# Same as  x_val_encoded, x_val_encoded_l = encoder_ae.predict(x_val)

x_val_encoded, _, _ = encoder_ae(x_val, training=False)
label_list = list(y_val_original)

if MULTI_COLOR is True:
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_val_encoded[:, 0], x_val_encoded[:, 1], c=label_list,
                         alpha=0.9, s=2, cmap="tab10")
else:
    cmap = colors.ListedColormap(['blue', 'red'])
    bounds = [0, 5, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_val_encoded[:, 0], x_val_encoded[:, 1], c=label_list,
                         alpha=0.9, s=2, cmap=cmap)

legend = ax.legend(*scatter.legend_elements(), loc="center left", title="Classes")
ax.add_artist(legend)


plt.savefig(latent_space_dir / 'Before_training_validation_latentspace.png')
plt.close('all')

# -------------------------------------------------------------------------------------------------------------

# Loss Function
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.
label_loss_weight = 1.

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
softmax = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
accuracy = tf.keras.metrics.BinaryAccuracy()


def autoencoder_loss(inputs, reconstruction, loss_weight):
    return loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, loss_weight):
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_weight * (loss_fake + loss_real)


def generator_loss(fake_output, loss_weight):
    return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)


def label_loss(label_input, label_reconstruction, label_loss_weight):
    return label_loss_weight * tf.nn.softmax_cross_entropy_with_logits(label_input, label_reconstruction)


# -------------------------------------------------------------------------------------------------------------

base_lr = 0.00025
max_lr = 0.0025

n_samples = x_train.shape[0]
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
label_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

n_epochs = 501

# -------------------------------------------------------------------------------------------------------------

# Training
# Training of the semi supervsied aae
@tf.function
# need data x and labels y
def train_step(batch_x, batch_y):
    with tf.GradientTape() as ae_tape:
        encoder_z, _, encoder_softmax = encoder_ae(batch_x, training=True)
        decoder_input = tf.concat([encoder_z, encoder_softmax], axis=1)
        decoder_output = decoder(decoder_input)

        # Autoencoder Loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder_ae.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder_ae.trainable_variables))

    # Training of the AE is done; Output is a reconstructed Image of the Input
    # no Labels need to be given to the encoder

    # Training the Discriminators
    # There are 2 Discriminators:
    #       one for the style variable
    #       one for the labels Cat(y)

    with tf.GradientTape() as dc_tape1, tf.GradientTape() as dc_tape2:
        # Discriminator for the labels
        # Creating the Cat-Distribution for the labels
        # Create random num: batch_size labels between 0 and 9
        # TODO change np.random to tf.random
        # real_label_distribution = np.random.randint(low=0, high=n_labels, size=batch_size)
        # real_label_distribution = np.eye(n_labels)[real_label_distribution]

        real_label_distribution = tf.experimental.numpy.random.randint(low=0, high=n_labels, size=batch_size)
        real_label_distribution = tf.one_hot(real_label_distribution, n_labels)
        real_z_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)

        # genereated labels from the encoder/generator
        dc_y_fake = discriminator_labels(encoder_softmax)
        # input of the real labesl
        dc_y_real = discriminator_labels(real_label_distribution)

        # genereated style from the encoder/generator
        dc_z_fake = discriminator_style(encoder_z)
        # input of the real style distro
        dc_z_real = discriminator_style(real_z_distribution)

        # Calculating loss functions
        dc_y_loss = discriminator_loss(dc_y_real, dc_y_fake, dc_loss_weight)
        dc_z_loss = discriminator_loss(dc_z_real, dc_z_fake, dc_loss_weight)

        # Acc
        dc_y_acc = accuracy(tf.concat([tf.ones_like(dc_y_real), tf.zeros_like(dc_y_fake)], axis=0),
                            tf.concat([dc_y_real, dc_y_fake], axis=0))

        dc_z_acc = accuracy(tf.concat([tf.ones_like(dc_z_real), tf.zeros_like(dc_z_fake)], axis=0),
                            tf.concat([dc_z_real, dc_y_fake], axis=0))

        dc_y_grads = dc_tape1.gradient(dc_y_loss, discriminator_labels.trainable_variables)
        dc_optimizer.apply_gradients(zip(dc_y_grads, discriminator_labels.trainable_variables))

        dc_z_grads = dc_tape2.gradient(dc_z_loss, discriminator_style.trainable_variables)
        dc_optimizer.apply_gradients(zip(dc_z_grads, discriminator_style.trainable_variables))

        # Training Generator
        # one Generator(=Encoder) but 2 Outputs
        # So Generator must be trained for z and y
        #       --> gen_y_loss and gen_z_loss
        with tf.GradientTape() as gen_tape:  # , tf.GradientTape() as label_tape:
            encoder_z, _, encoder_y = encoder_ae(batch_x, training=True)
            dc_y_fake = discriminator_labels(encoder_y, training=True)
            dc_z_fake = discriminator_style(encoder_z, training=True)

            # Generator loss y
            gen_y_loss = generator_loss(dc_y_fake, gen_loss_weight)
            # Generator loss z
            gen_z_loss = generator_loss(dc_z_fake, gen_loss_weight)

            gen_loss = gen_z_loss + gen_y_loss

        gen_grads = gen_tape.gradient(gen_loss, encoder_ae.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, encoder_ae.trainable_variables))


        with tf.GradientTape() as label_tape:
            encoder_z, encoder_y, _ = encoder_ae(batch_x, training=True)

            '''
            TODO: Loss Function mit Labels einfügen damit semisupervised richtig funktioniert
            In the semi - supervised classification phase, the autoencoder updates
            q(y|x) to minimize the cross-entropy cost on a labeled mini-batch
            '''

            labels = tf.one_hot(batch_y, n_labels)
            l_loss = label_loss(labels, encoder_y, label_loss_weight)
            # l_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=encoder_y)
            # l_loss = tf.reduce_mean(l_loss)

        label_grads = label_tape.gradient(l_loss, encoder_ae.trainable_variables)
        label_optimizer.apply_gradients(zip(label_grads, encoder_ae.trainable_variables))

        return ae_loss, dc_y_loss, dc_y_acc, dc_z_loss, dc_z_acc, gen_loss, l_loss

# -------------------------------------------------------------------------------------------------------------

for epoch in range(n_epochs):
    start = time.time()

    if epoch in [60, 120, 240, 360]:
        base_lr = base_lr / 2
        max_lr = max_lr / 2
        step_size = step_size / 2

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_y_loss_avg = tf.metrics.Mean()
    epoch_dc_y_acc_avg = tf.metrics.Mean()
    epoch_dc_z_loss_avg = tf.metrics.Mean()
    epoch_dc_z_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()
    epoch_label_loss_avg = tf.metrics.Mean()

    for batch, (batch_x, batch_y) in enumerate(train_dataset):
        # -------------------------------------------------------------------------------------------------------------
        # Calculate cyclic learning rate
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
        ae_optimizer.lr = clr
        dc_optimizer.lr = clr
        gen_optimizer.lr = clr
        label_optimizer.lr = clr

        ae_loss, dc_y_loss, dc_y_acc, dc_z_loss, dc_z_acc, gen_loss, l_loss = train_step(batch_x, batch_y)

        epoch_ae_loss_avg(ae_loss)

        epoch_dc_y_loss_avg(dc_y_loss)
        epoch_dc_y_acc_avg(dc_y_acc)

        epoch_dc_z_loss_avg(dc_z_loss)
        epoch_dc_z_acc_avg(dc_z_acc)

        epoch_gen_loss_avg(gen_loss)
        epoch_label_loss_avg(l_loss)

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} AE_LOSS: {:.4f} DC_Y_LOSS: {:.4f} DC_Y_ACC: {:.4f} DC_Z_LOSS: {:.4f} '
          'DC_Z_ACC: {:.4f} GEN_LOSS: {:.4f} CLASSIFICATION_LOSS: {:.4f}' \
          .format(epoch, epoch_time,
                  epoch_ae_loss_avg.result(),
                  epoch_dc_y_loss_avg.result(),
                  epoch_dc_y_acc_avg.result(),
                  epoch_dc_z_loss_avg.result(),
                  epoch_dc_z_acc_avg.result(),
                  epoch_gen_loss_avg.result(),
                  epoch_label_loss_avg.result()))

    # -------------------------------------------------------------------------------------------------------------

    if epoch % 20 == 0:
        # Latent space of test set
        x_test_encoded, _, _ = encoder_ae(x_test, training=False)
        label_list = list(y_test_original)

        if MULTI_COLOR is True:
            fig, ax = plt.subplots()
            scatter = ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=label_list,
                                 alpha=.9, s=2, cmap="tab10")
        else:
            cmap = colors.ListedColormap(['blue', 'red'])
            bounds = [0, 5, 10]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots()
            scatter = ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=label_list,
                                 alpha=.9, s=2, cmap=cmap)

        legend = ax.legend(*scatter.legend_elements(), loc="center left", title="Classes")
        ax.add_artist(legend)

        # ax.set_xlim([-30, 30])
        # ax.set_ylim([-30, 30])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # # VALIDATION
        # Latent space of validation set
        if epoch == n_epochs - 1:
            # Same as  x_val_encoded, x_val_encoded_l = encoder_ae.predict(x_val)
            # Latent space of test set
            x_val_encoded, _, _ = encoder_ae(x_val, training=False)
            label_list = list(y_val_original)

            if MULTI_COLOR is True:
                fig, ax = plt.subplots()
                scatter = ax.scatter(x_val_encoded[:, 0], x_val_encoded[:, 1], c=label_list,
                                     alpha=0.9, s=2, cmap="tab10")
            else:
                cmap = colors.ListedColormap(['blue', 'red'])
                bounds = [0, 5, 10]
                norm = colors.BoundaryNorm(bounds, cmap.N)

                fig, ax = plt.subplots()
                scatter = ax.scatter(x_val_encoded[:, 0], x_val_encoded[:, 1], c=label_list,
                                     alpha=0.9, s=2, cmap=cmap)

            legend = ax.legend(*scatter.legend_elements(), loc="center left", title="Classes")
            ax.add_artist(legend)

            plt.savefig(latent_space_dir / 'validation_latentspace.png')
            plt.close('all')
