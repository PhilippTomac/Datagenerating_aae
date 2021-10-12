## Imports
# Matplot
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors

import numpy as np
import tensorflow as tf
import time
from pathlib import Path

from lib import models
from lib.DataHandler import MNIST

# -------------------------------------------------------------------------------------------------------------
# Reduce the hunger of TF when we're training on a GPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

# -------------------------------------------------------------------------------------------------------------
# Setting Seed for better comparison
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# -------------------------------------------------------------------------------------------------------------
# Creating Paths for the File-System
ROOT_PATH = Path.cwd()
# Path for images and results
# dir_name: var for the directory name where the images are going to be saved
dir_name = 'test_1hdim_DISTrain_false_acc'
output_dir = (ROOT_PATH / ('experiment_results/unsupervisied_aae/%s' % dir_name))
output_dir.mkdir(exist_ok=True)

# Directory for the generated samples
sampling_dir = output_dir / 'Sampling'
sampling_dir.mkdir(exist_ok=True)

# Visualisation if more aaes are trained parallel
print('Experiment', output_dir, ':')
# -------------------------------------------------------------------------------------------------------------
# Var for the plots
# If more then 2 classes are in the dataset, set the var MULTI_COLOR to True
multi_color = True

# Loading the MNIST Data
print("1. Loading and Preprocessing Data with DataHandler.py")
mnist = MNIST(random_state=random_seed)
# Selecting the needed Classes and categorize them
anomaly = [9]
drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
include = [1, 9]


#anomaly = [0, 1]
#delete_y = [2, 3]
#delete_x = [2, 3]
#drop = []
#include = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# -------------------------------------------------------------------------------------------------------------
# Creating the dataset
# Training Data
print('Training Data...')
x_train, y_train, y_train_original = mnist.get_datasplit('train', anomaly, drop, include)

# -------------------------------------------------------------------------------------------------------------
# Testdata
print('Test Data...')
x_test, y_test, y_test_original = mnist.get_datasplit('test', anomaly, drop, include)

# -------------------------------------------------------------------------------------------------------------
# Validation data
print('Validation Data...')
x_val, y_val, y_val_original = mnist.get_datasplit('val', anomaly, drop, include)

# -------------------------------------------------------------------------------------------------------------
# Creating the aae model
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

encoder = aae.create_encoder()
decoder = aae.create_decoder()
discriminator = aae.create_discriminator_style()

# -------------------------------------------------------------------------------------------------------------
# Shuffeling the training data and divide into batches
batch_size = 256
train_buf = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------
# Same as  x_val_encoded, x_val_encoded_l = encoder_ae.predict(x_val)
# Plotting the latent space before the training for comparison
x_val_encoded = encoder(x_val, training=False)
label_list = list(y_val_original)

if multi_color is True:
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_val_encoded[:, 0], x_val_encoded[:, 1], c=label_list,
                         alpha=.9, s=2, cmap="tab10")
else:
    cmap = colors.ListedColormap(['blue', 'red'])
    bounds = [0, 5, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_val_encoded[:, 0], x_val_encoded[:, 1], c=label_list,
                         alpha=0.9, s=2, cmap=cmap)

legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend)

plt.savefig(output_dir / 'Before_training_validation_latentspace.png')
plt.close('all')

# -------------------------------------------------------------------------------------------------------------
# Loss Function
# Weights can be changed for more or less effect in the trainingprocess
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


# -------------------------------------------------------------------------------------------------------------
# Circle Learning parameter
# Later the lr can be changes and used without circle
base_lr = 0.00025
max_lr = 0.0025

# Step size
n_samples = x_train.shape[0]
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

# Optimizier
ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

# -------------------------------------------------------------------------------------------------------------
# Training
# Training function of the unsupervsied aae
@tf.function
def train_step(batch_x):
    # Autoencoder
    with tf.GradientTape() as ae_tape:
        # Generating style z
        encoder_output = encoder(batch_x, training=True)
        # Creating Images with the style z
        decoder_output = decoder(encoder_output, training=True)

        # Autoencoder Loss:
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # Discriminator
    with tf.GradientTape() as dc_tape:
        # Creating the 'real_distribution' with a normal distribution
        real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
        # Generating style z
        encoder_output = encoder(batch_x, training=False)

        # Give the discriminator the 2 distributions and compare the outputs
        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    # Generator, also the encoder
    with tf.GradientTape() as gen_tape:
        encoder_output = encoder(batch_x, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Generator loss
        gen_loss = generator_loss(dc_fake, gen_loss_weight)

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    return ae_loss, dc_loss, dc_acc, gen_loss


# -------------------------------------------------------------------------------------------------------------
# Starting the training
n_epochs = 501

for epoch in range(n_epochs):
    start = time.time()
    # calculate new lr and step size at specific epochs
    if epoch in [60, 120, 240, 360]:
        base_lr = base_lr / 2
        max_lr = max_lr / 2
        step_size = step_size / 2

    # mean functions of the loss
    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_loss_avg = tf.metrics.Mean()
    epoch_dc_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()

    for batch, (batch_x) in enumerate(train_dataset):
        # -------------------------------------------------------------------------------------------------------------
        # Calculate cyclic learning rate
        # From the Git repo: ...
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
        # Setting the optimizers
        ae_optimizer.lr = clr
        dc_optimizer.lr = clr
        gen_optimizer.lr = clr

        # Calling the Train Function
        ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x)

        # Calucalting the average loss value
        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_dc_acc_avg(dc_acc)
        epoch_gen_loss_avg(gen_loss)

    epoch_time = time.time() - start
    # Terminal Output
    print('{:4d}: TIME: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
          .format(epoch, epoch_time,
                  epoch_ae_loss_avg.result(),
                  epoch_dc_loss_avg.result(),
                  epoch_dc_acc_avg.result(),
                  epoch_gen_loss_avg.result()))

    # -------------------------------------------------------------------------------------------------------------
    # Ploting the latent space every 100 epochs
    if epoch % 100 == 0:
        # Latent space of test set
        x_test_encoded = encoder(x_test, training=False)
        label_list = list(y_test_original)

        if multi_color is True:
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

        legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend)

        plt.savefig(output_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # Samling the data
        # Codesnippet from Alireza Makhzani - AAE
        x_points = np.linspace(-3, 3, 20).astype(np.float32)
        y_points = np.linspace(-3, 3, 20).astype(np.float32)

        nx, ny = len(x_points), len(y_points)
        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        for i, g in enumerate(gs):
            z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
            z = np.reshape(z, (1, 2))
            x = decoder(z, training=False).numpy()
            ax = plt.subplot(g)
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
        plt.savefig(sampling_dir / ('epoch_%d.png' % epoch))
        plt.close('all')
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # VALIDATION
        # Latent space of validation set
        # After the Training using the encoder to plot the latent space of the validation set to measure the performance
        # on not seen datapoints
        if epoch == n_epochs - 1:
            # Same as  x_val_encoded, x_val_encoded_l = encoder_ae.predict(x_val)
            # Latent space of test set
            x_val_encoded = encoder(x_val, training=False)
            label_list = list(y_val_original)

            if multi_color is True:
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

            legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
            ax.add_artist(legend)

            plt.savefig(output_dir / 'validation_latentspace.png')
            plt.close('all')

# Saving the trained decoder and encoder
#encoder.save_weights('/home/ptomac/Dokumente/Masterarbeit_AAE/TrainedModels/ExperimentModels/Encoder_67/encoder_weights', True)
#decoder.save_weights('/home/ptomac/Dokumente/Masterarbeit_AAE/TrainedModels/ExperimentModels/Decoder_67/decoder_weights', True)
