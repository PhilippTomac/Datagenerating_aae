from lib.DataHandler import MNIST, ExperimentConfig
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
mnist = MNIST(random_state=42)
x = mnist.x_val
print(x)
# ----------------------------------------------------------------------------------------------------------------------
possible_digits = np.unique(mnist.y_train).tolist()
n_samples = len(mnist.y_test)
print(possible_digits)
print(n_samples)
# ----------------------------------------------------------------------------------------------------------------------
# Set specific labels as normal and as anomalies
# TODO aufteilen der Daten in anomaly, normal und unlabeled


