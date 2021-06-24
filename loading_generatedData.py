## Imports
# Matplot
from lib.DataHandler import MNIST
from typing import List

import numpy as np


def create_anomalie_dataset(anomaly_classes: List[int] = None):
    x_train_generated9 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_9/generated_Images.npy')
    y_train_generated9 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_9/generated_labels.npy')

    x_train_generated8 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_8/generated_Images.npy')
    y_train_generated8 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_8/generated_labels.npy')

    x_train_generated7 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_Images.npy')
    y_train_generated7 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_7/generated_labels.npy')

    x_train_generated6 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_6/generated_Images.npy')
    y_train_generated6 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_6/generated_labels.npy')

    x_train_generated5 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_5/generated_Images.npy')
    y_train_generated5 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_5/generated_labels.npy')

    x_train_generated4 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_4/generated_Images.npy')
    y_train_generated4 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_4/generated_labels.npy')

    x_train_generated3 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_3/generated_Images.npy')
    y_train_generated3 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_3/generated_labels.npy')

    x_train_generated2 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_2/generated_Images.npy')
    y_train_generated2 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_2/generated_labels.npy')

    x_train_generated1 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_1/generated_Images.npy')
    y_train_generated1 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_1/generated_labels.npy')

    x_train_generated0 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_0/generated_Images.npy')
    y_train_generated0 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/DataPoint_0/generated_labels.npy')

    x_train_generated = np.concatenate([x_train_generated0, x_train_generated1, x_train_generated2, x_train_generated3,
                                        x_train_generated4, x_train_generated5, x_train_generated6, x_train_generated7,
                                        x_train_generated8, x_train_generated9], axis=0)

    y_train_generated = np.concatenate([y_train_generated0, y_train_generated1, y_train_generated2, y_train_generated3,
                                        y_train_generated4, y_train_generated5, y_train_generated6, y_train_generated7,
                                        y_train_generated8, y_train_generated9], axis=0)

    x_generated = np.delete(x_train_generated, np.where(np.isin(y_train_generated, anomaly_classes, invert=True)), axis=0)
    y_generated = np.delete(y_train_generated, np.where(np.isin(y_train_generated, anomaly_classes, invert=True)), axis=0)
    print(x_generated.shape, y_generated.shape)

    np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_Images', x_generated)
    np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_labels', y_generated)

    x_generated = x_generated.reshape((-1, 28 * 28)) / 255.

    return x_generated, y_generated

a, b = create_anomalie_dataset([6, 7])
print(a.shape, b.shape)
print(b)

