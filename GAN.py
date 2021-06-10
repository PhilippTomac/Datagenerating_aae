import numpy as np
from matplotlib import pyplot as plt, colors

from lib import models
import tensorflow as tf
from lib.DataHandler import MNIST

import glob
# import imageio
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

import numpy as np
from matplotlib import pyplot as plt, colors

from lib import models
import tensorflow as tf
from lib.DataHandler import MNIST

import glob
# import imageio
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

'''
Testing how to prepre the dataset
    - split into trainig, test and validation
    - creating supervised, unsupervised and semi supervised data
    - using the DataHandler of A3/A4
    - setting datapoints as normal, anomaly and unknown
'''
# -------------------------------------------------------------------------------------------------------------
anomaly = [9]
# delete_y = [7]
# delete_x = [7]
drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
include = [9]
# -------------------------------------------------------------------------------------------------------------
# Traingins Data
# Setting the seed
random_seed = 1993
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
mnist = MNIST(random_state=random_seed)

print('Training Data...')
x_train, y_train, y_train_original = mnist.get_datasplit('train', anomaly, drop, include,
                                                         None, None, 5000)
print(x_train.shape)
print(y_train.shape)
# print(y_train_original.shape)

# -------------------------------------------------------------------------------------------------------------
# Testdata
print('Test Data...')
x_test, y_test, y_test_original = mnist.get_datasplit('test', anomaly, drop, include,
                                                      None, None)
# print(x_test.shape)
# print(y_test.shape)
# print(y_test_original.shape)

# -------------------------------------------------------------------------------------------------------------
# Validation data
print('Validation Data...')
x_val, y_val, y_val_original = mnist.get_datasplit('val', anomaly, drop, include,
                                                   None, None)
# print(x_val.shape)
# print(y_val.shape)
# print(y_val_original.shape)
# -------------------------------------------------------------------------------------------------------------
# Generator
aae = models.AAE()
encoder = aae.create_encoder()

shape_noise = aae.shape_noise
image_size = aae.image_size
discriminator_input = aae.gan_discriminator_input

generator = aae.noise_generator()
# -------------------------------------------------------------------------------------------------------------
training_data = []
noise_label = []
mean = 100
stddev = 50
epoch = 100

noise = tf.random.normal([1, 784], mean=mean, stddev=stddev, seed=1993)
img = generator(noise, training=False)
plt.imshow(img[0, :, :, 0], cmap='gray')
plt.savefig('image_generated')
training_data.append(img)
# noise_label.append(10)

noise_label = np.array(noise_label)
training_data = np.array(training_data)
training_data = training_data.reshape((-1, 28 * 28))

print(training_data.shape)
# -------------------------------------------------------------------------------------------------------------
plot_data = tf.random.normal([1, 784], mean=mean, stddev=stddev, seed=1993)
fig = plt.figure()
count, bins, ignored = plt.hist(plot_data, 30, density=True)
plt.plot(bins, 1/(stddev * np.sqrt(2 * np.pi)) *
               np.exp(- (bins - mean)**2 / (2 * stddev**2)),
         linewidth=2, color='r')
plt.show()
plt.savefig('image_distribution')
# -------------------------------------------------------------------------------------------------------------
discriminator = aae.noise_discrimniator()
decision = discriminator(img)
print(decision)


# -------------------------------------------------------------------------------------------------------------
# data = encoder(training_data, training=False)
# label_list = []
# for i in range(epoch):
#     label_list.append(1)
#
# cmap = colors.ListedColormap(['blue'])
# # bounds = [0, 5, 10]
# # norm = colors.BoundaryNorm(bounds, cmap.N)
#
# fig, ax = plt.subplots()
# scatter = ax.scatter(training_data[:, 0], training_data[:, 1], c=label_list,
#                      alpha=0.9, s=2, cmap=cmap)
#
# legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
# ax.add_artist(legend)
#
# plt.savefig('generated_data_spread.png')
# plt.close('all')
# -------------------------------------------------------------------------------------------------------------
# new_dataset = np.concatenate((x_train, training_data), axis=0)
# print(new_dataset.shape)
#
# new_labels = np.concatenate((y_train, noise_label))
# print(new_labels.shape)
# -------------------------------------------------------------------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 256

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([1, 784], mean=mean, stddev=stddev, seed=1993)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

train(train_dataset, EPOCHS)
