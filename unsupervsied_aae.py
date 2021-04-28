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
Deterministic unsupervised AAE:
Here we assume that q(z|x) is a deterministic function of x. In this case, the encoder is similar to the encoder 
of a standard autoencoder and the only source of stochasticity in q(z) is the data distribution, pd(x). 
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
output_dir = ROOT_PATH / 'output'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'unsupervisied_aae'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

# load MNIST dataset
# TODO: Datensatz MNIST aufbereiten fuer semi-supervised und in Datahandler Klasse auslagern
# TODO: DataHandler Importieren und Szenaroen für Plots überlegen
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
#
# # Flatten the dataset
# x_train = x_train.reshape((-1, 28 * 28))
# x_test = x_test.reshape((-1, 28 * 28))

mnist = MNIST(random_state=random_seed)
x_train, y_train = mnist.get_target_classifier_data('train', list(range(6, 9)), list(range(0, 5)))
print(x_train.shape)
print(y_train.shape)

x_test, y_test = mnist.get_target_classifier_data('test', list(range(6, 9)), list(range(0, 5)))
print(x_test.shape)
print(y_test.shape)


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

encoder = aae.create_encoder()
decoder = aae.create_decoder()
discriminator = aae.create_discriminator_style()

encoder.summary()
decoder.summary()
discriminator.summary()

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

n_samples = 60000
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

n_epochs = 601

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

for epoch in range(n_epochs):
    start = time.time()

    if epoch in [30, 60, 90]:
        base_lr = base_lr/2
        max_lr = max_lr/2
        step_size = step_size/2


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

    if epoch % 100 == 0:
        # Latent space of test set
        x_test_encoded = encoder(x_test, training=False)
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
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s=2, **kwargs)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')
