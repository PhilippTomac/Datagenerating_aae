import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ----------------------------------------------------------------------------------
from lib.DataHandler import MNIST

'''
Python File to visualise the MNIST Dataset for analysis.
The File was used to look for the difference and similarities between the digits.
It was helpful for the configuration of the Experiment
'''
# ----------------------------------------------------------------------------------

ROOT_PATH = Path.cwd()

output_dir = ROOT_PATH / 'output'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'mnist_analysis'
experiment_dir.mkdir(exist_ok=True)
# ----------------------------------------------------------------------------------
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784) / 255
# x_test = x_test.reshape(10000, 784) / 255

mnist = MNIST(random_state=1993)
x_train, y_train = mnist.get_supervised_data('train', None, list(range(0, 10)))
x_test, y_test = mnist.get_supervised_data('test', None, list(range(0, 10)))

print('Number of Datapoints of each label in the training Datasplit:')
print(np.unique(y_train, return_counts=True))
print('Number of Datapoints of each label in the test Datasplit:')
print(np.unique(y_test, return_counts=True))
# ----------------------------------------------------------------------------------
# plot of ervy number with label
fig = plt.figure()
for i in range(10):
    n_image = np.argwhere(y_train == i)[0]
    plottable_image = np.reshape(x_train[n_image], (28, 28))
    ax = fig.add_subplot(2, 10, i+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(plottable_image, cmap='gray_r')
    # plt.title('Label: %i' % y_train[i])
plt.savefig(experiment_dir / 'every_number')
plt.close('all')
# ----------------------------------------------------------------------------------
# Create dictonary where every number postion is safed for every label
digits = {}

for i in range(10):
    digits[i] = np.where(y_train == i)[0][:10]

# Plot a 10x10 image with 10 samples of each class
fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
for i in range(10):
    for j in range(10):
        plottable_image = np.reshape(x_train[digits[i][j]], (28, 28))
        ax[i, j].imshow(plottable_image, cmap='gray')
        ax[i, j].set_yticklabels([])
        ax[i, j].set_xticklabels([])

plt.savefig(experiment_dir / '10x10_plot')
plt.close('all')

# ----------------------------------------------------------------------------------
# Durchführen von der Hauptkomponentenanlyse
mu = x_train.mean(axis=0)
# Singulärwertzerlegung mit Numpy
# Return: U = Unitary array; s = Vector(s) with the singular values;
#         V = Unitary array
U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
z_pca = np.dot(x_train - mu, V.transpose())


# Plotting the Datadistribution
# Setting the classes
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
label_list = list(y_test)
classes = set(label_list)

fig = plt.figure(figsize=(15, 15))
colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
kwargs = {'alpha': 0.8, 'c': [colormap[i] for i in label_list]}
ax = plt.subplot(aspect='equal')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
handles = [mpatches.Circle((0, 0), label=class_, color=colormap[i])
           for i, class_ in enumerate(classes)]
ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
          fancybox=True, loc='center left')
plt.scatter(z_pca[:5000, 0], z_pca[:5000, 1], c=y_train[:5000], s=8, cmap='tab10')
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])
# plt.show()

plt.savefig(experiment_dir / 'mnist_pca')
plt.close('all')
