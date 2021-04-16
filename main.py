# import
import models
import tensorflow as tf

try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

print('GAN Training')
gan = models.GAN()

gan.train(epochs=300, batch_size=32, sample_interval=200)

