# import
from lib import models
import tensorflow as tf

# ----------------------------------------------------------------------------------------------------------------------

# Limiting GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

# ----------------------------------------------------------------------------------------------------------------------
# Training AAE unsupervsied and deterministic
print('AAE Unsup Building')
aae_unsup = models.aae_unsup()

# ----------------------------------------------------------------------------------------------------------------------
# Training GAN
print('GAN Training')
gan = models.GAN()
gan.train(epochs=300, batch_size=32, sample_interval=200)

