from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf

# Keras Imports
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


# ----------------------------------------------------------------------------------------------------------------------
# Encoder
class UnsupervisedDeterministic:
    def __init__(self):
        self.image_size = 784
        self.h_dim = 1000
        self.z_dim = 2

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.discriminator = self.create_discriminator()

    def create_encoder(self):
        inputs = tf.keras.Input(shape=(self.image_size,))
        x = tf.keras.layers.Dense(self.h_dim)(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        # Output Codelayer Z --> Verteilung q(z)
        encoded = tf.keras.layers.Dense(self.z_dim)(x)
        model = tf.keras.Model(inputs=inputs, outputs=encoded)
        return model

    # Decoder
    def create_decoder(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        # Output Erzeugtes Bild
        reconstruction = tf.keras.layers.Dense(self.image_size, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
        return model

    # Discriminator
    def create_discriminator(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        # Ein Wert wird hier  zurückgegeben
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction)
        return model


class SupervisedDeterministic:
    def __init__(self):
        self.image_size = 784
        self.h_dim = 1000
        self.z_dim = 2
        self.n_labels = 10

        self.encoder = self.create_encoder_s()
        self.decoder = self.create_decoder_s()
        self.discriminator = self.create_discriminator_s()

    def create_encoder_s(self):
        inputs = tf.keras.Input(shape=(self.image_size,))
        x = tf.keras.layers.Dense(self.h_dim)(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        encoded = tf.keras.layers.Dense(self.z_dim)(x)
        model = tf.keras.Model(inputs=inputs, outputs=encoded)
        return model

    def create_decoder_s(self):
        encoded = tf.keras.Input(shape=(self.z_dim + self.n_labels,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        reconstruction = tf.keras.layers.Dense(self.image_size, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
        return model

    def create_discriminator_s(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction)
        return model


class SemiSupervisedDeterministic:
    # Parameters
    def __init__(self):
        self.image_size = 784
        self.h_dim = 1000
        self.z_dim = 2
        self.n_labels = 10
        self.n_labeled = 1000

        # Createing the Modelparts of the Semi Supervised AAE


    # Function to create the encoder
    def create_encoder_semi(self, supervised=False):
        # Input is the Data x (image)
        inputs = tf.keras.Input(shape=(self.image_size,))
        x = tf.keras.layers.Dense(self.h_dim)(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        # 2 Outputs
        # Output Style variable z for reconstruction of the input
        encoded = tf.keras.layers.Dense(self.z_dim)(x)
        # Output of the labels with a softmax Cat(y)
        if supervised is False:
            encoded_labels = tf.keras.layers.Dense(self.n_labels, activation='softmax')(x)
        else:
            # Needed for training of the encoder
            encoded_labels = tf.keras.layers.Dense(self.n_labels)(x)

        model = tf.keras.Model(inputs=inputs, outputs=[encoded, encoded_labels], name='Encoder')
        return model

    def create_decoder_semi(self):
        # Input for decoder is the style variable z (encoded) and the encoded_labels form the Encoder
        encoded = tf.keras.Input(shape=(self.z_dim + self.n_labels,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        reconstruction = tf.keras.layers.Dense(self.image_size, activation='sigmoid')(x)
        # Output of the Model is
        model = tf.keras.Model(inputs=encoded, outputs=reconstruction, name='decoder')
        return model

    def create_discriminator_style(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction, name='disc_style')
        return model

    def create_discriminator_label(self):
        encoded = tf.keras.Input(shape=(self.n_labels,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction,name='disc_labels')
        return model
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# models for basic
# @ TODO add Implementation from colab

