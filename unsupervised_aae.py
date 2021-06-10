# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from matplotlib import gridspec, colors
from matplotlib.cm import get_cmap
from lib import models

import time
from pathlib import Path

from lib.DataHandler import MNIST

import sys
sys.path.append('/home/fipsi/Documents/Code/Masterarbeit_GPU')

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
# -------------------------------------------------------------------------------------------------------------
ROOT_PATH = Path.cwd()
# Path for images and results
output_dir = ROOT_PATH / 'experiment_results'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'unsupervisied_aae_noise'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'test_noise_1_9'
latent_space_dir.mkdir(exist_ok=True)

generated_data_dir = latent_space_dir / 'generated_data'
generated_data_dir.mkdir(exist_ok=True)

sampling_dir = latent_space_dir / 'Sampling'
sampling_dir.mkdir(exist_ok=True)

multi_color = True
print('Experiment', latent_space_dir)

# Data MNIST
print("1. Loading and Preprocessing Data with DataHandler.py")
mnist = MNIST(random_state=random_seed)

anomaly = [9]
drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
include = [1, 9]

# anomaly = [8]
# # delete_y = [7]
# # delete_x = [7]
# drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# include = [3, 8]

# -------------------------------------------------------------------------------------------------------------
# Traingins Data
x_train, y_train, y_train_original = mnist.get_datasplit('train', anomaly, drop, include,
                                                         None, None, 5000)
# print(x_train.shape)
# print(y_train.shape)
# print(y_train_original.shape)

# -------------------------------------------------------------------------------------------------------------
# Testdata
x_test, y_test, y_test_original = mnist.get_datasplit('test', anomaly, drop, include,
                                                      None, None)
# print(x_test.shape)
# print(y_test.shape)
# print(y_test_original.shape)

# -------------------------------------------------------------------------------------------------------------
# Validation data
x_val, y_val, y_val_original = mnist.get_datasplit('val', anomaly, drop, include,
                                                   None, None)
# print(x_val.shape)
# print(y_val.shape)
# print(y_val_original.shape)

# -------------------------------------------------------------------------------------------------------------
# Creating the models
aae = models.AAE()
z_dim = aae.z_dim
h_dim = aae.h_dim
image_size = aae.image_size

encoder = aae.create_encoder()
decoder = aae.create_decoder()
discriminator = aae.create_discriminator_style()

# encoder.summary()
# decoder.summary()
# discriminator.summary()
# -------------------------------------------------------------------------------------------------------------
# TODO: Generating Data and adding it to the Dataset
print('Creating noisy images...')
generator = aae.noise_generator()
# Parameter for the generator to create Datapoints
mean = 100
stddev = 50
n_noise_img = 100
noise_dataset = []
original_noise_labels = []
anomalie_labels = []

# Creating Datapoints and the labels
for i in range(n_noise_img):
    # Normal distribution
    noise = tf.random.normal([1, image_size], mean=mean, stddev=stddev, seed=random_seed)
    # Generator creating an image
    img_noise = generator(noise, training=False)
    noise_dataset.append(img_noise)
    # Adding Labels
    # number 10 for generated images
    original_noise_labels.append(10)
    # Anomalie label
    anomalie_labels.append(1)

print('Creating noisy dataset...')
# Transforming the data to a dataset that can be used by the aae
noise_dataset = np.array(noise_dataset)
noise_dataset = noise_dataset.reshape((-1, 28*28))

original_noise_labels = np.array(original_noise_labels)
anomalie_labels = np.array(anomalie_labels)

# Plotting the Data Distribution
noise = tf.random.normal([1, image_size], mean=mean, stddev=stddev, seed=random_seed)
count, bins, ignored = plt.hist(noise, 30, density=True)
plt.plot(bins, 1/(stddev * np.sqrt(2 * np.pi)) *
               np.exp(- (bins - mean)**2 / (2 * stddev**2)),
         linewidth=2, color='r')
plt.savefig(generated_data_dir / 'generated_data_spread.png')
plt.close('all')

# Adding the generated data to the original data
x_train = np.concatenate((x_train, noise_dataset), axis=0)
y_train = np.concatenate((y_train, anomalie_labels))
y_train_original = np.concatenate((y_train_original, original_noise_labels))

x_test = np.concatenate((x_test, noise_dataset), axis=0)
y_test = np.concatenate((y_test, anomalie_labels))
y_test_original = np.concatenate((y_test_original, original_noise_labels))

x_val = np.concatenate((x_val, noise_dataset), axis=0)
y_val = np.concatenate((y_val, anomalie_labels))
y_val_original = np.concatenate((y_val_original, original_noise_labels))
# -------------------------------------------------------------------------------------------------------------

# Parameter
batch_size = 256
train_buf = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------
# Same as  x_val_encoded, x_val_encoded_l = encoder_ae.predict(x_val)
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

plt.savefig(latent_space_dir / 'Before_training_validation_latentspace.png')
plt.close('all')
# -------------------------------------------------------------------------------------------------------------


# loss functions
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


# learing rate
base_lr = 0.00025
max_lr = 0.0025

n_samples = x_train.shape[0]
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

n_epochs = 501

# Optimizier
ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)



# Training
@tf.function
def train_step(batch_x):
    # Autoencoder
    with tf.GradientTape() as ae_tape:
        encoder_output = encoder(batch_x, training=True)
        decoder_output = decoder(encoder_output, training=True)

        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # Discriminator
    with tf.GradientTape() as dc_tape:
        real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
        encoder_output = encoder(batch_x, training=True)

        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    # Generator (Encoder)
    with tf.GradientTape() as gen_tape:
        encoder_output = encoder(batch_x, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Generator loss
        gen_loss = generator_loss(dc_fake, gen_loss_weight)

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    return ae_loss, dc_loss, dc_acc, gen_loss


# Start the training
for epoch in range(n_epochs):
    start = time.time()

    if epoch in [60, 120, 240, 360]:
        base_lr = base_lr / 2
        max_lr = max_lr / 2
        step_size = step_size / 2

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
    print('{:4d}: TIME: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
          .format(epoch, epoch_time,
                  epoch_ae_loss_avg.result(),
                  epoch_dc_loss_avg.result(),
                  epoch_dc_acc_avg.result(),
                  epoch_gen_loss_avg.result()))

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

        # ax.set_xlim([-30, 30])
        # ax.set_ylim([-30, 30])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------
        # Samling the data
        # Code from Alireza Makhzani - AAE
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
        # # VALIDATION
        # Latent space of validation set
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

            plt.savefig(latent_space_dir / 'validation_latentspace.png')
            plt.close('all')