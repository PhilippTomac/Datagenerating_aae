## Imports
# Matplot
from typing import List
import numpy as np

# TODO: Can be deleted... is included in the new DataHandler.py
'''
Function to generate a new mnist dataset with the the generated data from the aae
:input
anomaly_classes ->
'''

def create_anomalie_dataset(needed_classes: List[int] = None):
    # 1. Load all datasets from every number and the corresponding labels

    # Datapoint 9
    x_train_generated9 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint9_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated9 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint9_m00_stddev10'
        '/generated_labels.npy')

    # Datapoint 8
    x_train_generated8 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint8_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated8 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint8_m00_stddev10'
        '/generated_labels.npy')

    # Datapoint 7
    x_train_generated7 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint7_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated7 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint7_m00_stddev10'
        '/generated_labels.npy')

    # Datapoint 6
    x_train_generated6 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint6_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated6 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint6_m00_stddev10'
        '/generated_labels.npy')

    # # Datapoint 5
    # x_train_generated5 = np.load(
    #     '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint5_m00_stddev10'
    #     '/generated_Images.npy')
    # y_train_generated5 = np.load(
    #     '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint5_m00_stddev10'
    #     '/generated_labels.npy')
    #
    # # Datapoint 4
    # x_train_generated4 = np.load(
    #     '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint4_m00_stddev10'
    #     '/generated_Images.npy')
    # y_train_generated4 = np.load(
    #     '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint4_m00_stddev10'
    #     '/generated_labels.npy')

    # Datapoint 3
    x_train_generated3 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint3_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated3 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint3_m00_stddev10'
        '/generated_labels.npy')

    # Datapoint 2
    x_train_generated2 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint2_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated2 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint2_m00_stddev10'
        '/generated_labels.npy')

    # Datapoint 1
    x_train_generated1 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint1_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated1 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint1_m00_stddev10'
        '/generated_labels.npy')

    # Datapoint 0
    x_train_generated0 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint0_m00_stddev10'
        '/generated_Images.npy')
    y_train_generated0 = np.load(
        '/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/OneGeneratedClass/Datapoint0_m00_stddev10'
        '/generated_labels.npy')

    # 2. Concatenate all arrays into one big dataset
    #   Datapoints and Label separately
    x_train_generated = np.concatenate([x_train_generated0, x_train_generated1, x_train_generated2,
                                        x_train_generated3, x_train_generated6, x_train_generated7,
                                        x_train_generated8, x_train_generated9], axis=0)

    y_train_generated = np.concatenate([y_train_generated0, y_train_generated1, y_train_generated2,
                                        y_train_generated3, y_train_generated6, y_train_generated7,
                                        y_train_generated8, y_train_generated9], axis=0)

    # 3. Deleting the points that are not wanted in the Dataset
    #   all classes that are not in the list needed_classes will be deleted from the dataset
    if needed_classes:
        x_generated = np.delete(x_train_generated,
                                np.where(np.isin(y_train_generated, needed_classes, invert=True)), axis=0)
        y_generated = np.delete(y_train_generated,
                                np.where(np.isin(y_train_generated, needed_classes, invert=True)), axis=0)

    # 4. Save the created datasets
    np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_Images', x_generated)
    np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_labels', y_generated)

    # 5. Normalising the dataset
    # TODO: Maybe not needed becouse A3 loads the data and normalise itself. Must be checked!
    x_generated = x_generated.reshape((-1, 28 * 28)) / 255.

    # 6.  return the datasets
    return x_generated, y_generated

x, y = create_anomalie_dataset(needed_classes=[8, 9])
print(y)

