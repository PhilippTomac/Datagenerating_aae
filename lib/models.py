import tensorflow as tf


class AAE:
    # Parameter
    def __init__(self):
        self.image_size = 784
        self.h_dim = 1000
        self.z_dim = 2
        self.labels = 2
        self.n_labeled = 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Encoders
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

    def create_encoder_semi(self):
        # Input is the Data x (image)
        inputs = tf.keras.Input(shape=(self.image_size))
        x = tf.keras.layers.Dense(self.h_dim)(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        # 2 Outputs
        # Output Style variable z for reconstruction of the input
        encoded = tf.keras.layers.Dense(self.z_dim, name='outputz')(x)
        # Output of the labels with a softmax Cat(y)
        encoded_labels_softmax = tf.keras.layers.Dense(self.labels, activation=tf.nn.softmax, name='softmaxlayer')(x)
        # Needed for training of the encoder
        encoded_labels = tf.keras.layers.Dense(self.labels, name='outputlabel')(x)

        model = tf.keras.Model(inputs=inputs, outputs=[encoded, encoded_labels, encoded_labels_softmax], name='Encoder')
        return model

    def create_encoder_cnn(self):
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        z = tf.keras.layers.Conv2D(filters=self.z_dim, kernel_size=3, strides=2, padding='same')(x)

        model = tf.keras.Model(inputs=inputs, outputs=z)
        return model

    def create_encoder_gauss(self):
        inputs = tf.keras.Input(shape=(self.image_size,))
        x = tf.keras.layers.Dense(self.h_dim)(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        mean = tf.keras.layers.Dense(self.z_dim)(x)
        stddev = tf.keras.layers.Dense(self.z_dim, activation='softplus')(x)
        model = tf.keras.Model(inputs=inputs, outputs=[mean, stddev])
        return model

    # ------------------------------------------------------------------------------------------------------------------
    # Decoders
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

    def create_decoder_sup_semi(self):
        encoded = tf.keras.Input(shape=(self.z_dim + self.labels,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        reconstruction = tf.keras.layers.Dense(self.image_size, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
        return model

    def create_decoder_gauss(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        reconstruction = tf.keras.layers.Dense(self.image_size)(x)
        model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
        return model

    # ------------------------------------------------------------------------------------------------------------------
    # Discrimininators
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

    def create_discriminator_label(self, n_labels):
        encoded = tf.keras.Input(shape=(n_labels,))
        x = tf.keras.layers.Dense(self.h_dim)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.h_dim)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction, name='disc_labels')
        return model

    def create_discriminator_gauss(self):
        encoded = tf.keras.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(self.h_dim / 4)(encoded)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dense(self.h_dim / 4)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        prediction = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=encoded, outputs=prediction)
        return model

    # @ TODO add Implementation from colab (GAN)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# models for basic
