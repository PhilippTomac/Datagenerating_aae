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

train_dataset = tf.data.Dataset.from_tensor_slices(mnist.x_train)
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

# creating the model parts
aae = models.SemiSupervisedDeterministic()
# Parameter
image_size = aae.image_size
n_labels = aae.n_labels
n_labeled = aae.n_labeled
h_dim = aae.h_dim
z_dim = aae.z_dim

encoder_ae = aae.create_encoder_semi(False)
# hier Fehler: bzw Warning: Gradient cant be updated
generator_y = aae.create_encoder_semi(True)
decoder = aae.create_decoder_semi()
discriminator_labels = aae.create_discriminator_label()
discriminator_style = aae.create_discriminator_style()

encoder_ae.summary()
generator_y.summary()
decoder.summary()
discriminator_labels.summary()
discriminator_style.summary()

# Loss Function
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


base_lr = 0.00025
max_lr = 0.0025

n_samples = 60000
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

n_epochs = 601


# Training
# Training of the semi supervsied aae
# 1000 labeld data --> must b
@tf.function
# need data x and labels y
def train_step(batch_x):
    with tf.GradientTape() as ae_tape:
        encoder_z, encoder_y = encoder_ae(batch_x, training=True)
        decoder_input = tf.concat([encoder_z, encoder_y], axis=1)
        decoder_output = decoder(decoder_input)

        # Autoencoder Loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder_ae.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder_ae.trainable_variables + decoder.trainable_variables))
    # Training of the AE is done; Output is a reconstructed Image of the Input
    # no Labels need to be given to the encoder

    # Training the Discriminators
    # There are 2 Discriminators:
    #       one for the style variable
    #       one for the labels Cat(y)

    with tf.GradientTape() as dc_tape1, tf.GradientTape() as dc_tape2:
        # Discriminator for the labels
        # Creating the Cat-Distribution for the labels
        real_label_distribution = np.random.randint(low=0, high=10, size=batch_size)
        real_label_distribution = np.eye(10)[real_label_distribution]
        real_z_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)

        # genereated labels from the encoder/generator
        dc_y_fake = discriminator_labels(encoder_y)
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
        #       --> gen_y_loss and gen_y_loss
        with tf.GradientTape() as gen_tape:
            encoder_z, encoder_y = generator_y(batch_x, training=True)
            dc_y_fake = discriminator_labels(encoder_y, training=True)
            dc_z_fake = discriminator_style(encoder_z, training=True)

            # Generator loss y
            gen_y_loss = generator_loss(dc_y_fake, gen_loss_weight)
            # Generator loss z
            gen_z_loss = generator_loss(dc_z_fake, gen_loss_weight)

            gen_loss = gen_z_loss + gen_y_loss

        gen_grads = gen_tape.gradient(gen_loss, generator_y.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator_y.trainable_variables))

        return ae_loss, dc_y_loss, dc_y_acc, dc_z_loss, dc_z_acc, gen_loss


for epoch in range(n_epochs):
    start = time.time()

    if epoch in [60, 100, 300]:
        base_lr = base_lr / 2
        max_lr = max_lr / 2
        step_size = step_size / 2

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_y_loss_avg = tf.metrics.Mean()
    epoch_dc_y_acc_avg = tf.metrics.Mean()
    epoch_dc_z_loss_avg = tf.metrics.Mean()
    epoch_dc_z_acc_avg = tf.metrics.Mean()
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

        ae_loss, dc_y_loss, dc_y_acc, dc_z_loss, dc_z_acc, gen_loss = train_step(batch_x)

        epoch_ae_loss_avg(ae_loss)

        epoch_dc_y_loss_avg(dc_y_loss)
        epoch_dc_y_acc_avg(dc_y_acc)

        epoch_dc_z_loss_avg(dc_z_loss)
        epoch_dc_z_acc_avg(dc_z_acc)

        epoch_gen_loss_avg(gen_loss)

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
            .format(epoch, epoch_time,
                    epoch_time * (n_epochs - epoch),
                    epoch_ae_loss_avg.result(),
                    epoch_dc_y_loss_avg.result(),
                    epoch_dc_y_acc_avg.result(),
                    epoch_dc_z_loss_avg.result(),
                    epoch_dc_z_acc_avg.result(),
                    epoch_gen_loss_avg.result()))

    if epoch % 100 == 0:
        # Latent space of test set
        x_test_encoded, _ = encoder_ae(mnist.x_test, training=False)
        label_list = list(mnist.y_test)

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