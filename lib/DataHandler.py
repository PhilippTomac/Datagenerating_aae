import abc
import numpy as np
import tensorflow as tf

from dataclasses import dataclass

from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

from typing import List, Union, Tuple

# from loading_generatedData import create_anomalie_dataset

'''
Class to split the data in train, test and validition.
Also to define what datapoints are included and what datapoints are normal/anomaly
'''


def create_anomalie_dataset(drop_classes: List[int] = None):
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

    x_train_generated = np.concatenate(
        [x_train_generated0, x_train_generated1, x_train_generated2, x_train_generated3,
         x_train_generated4, x_train_generated5, x_train_generated6, x_train_generated7,
         x_train_generated8, x_train_generated9], axis=0)

    y_train_generated = np.concatenate(
        [y_train_generated0, y_train_generated1, y_train_generated2, y_train_generated3,
         y_train_generated4, y_train_generated5, y_train_generated6, y_train_generated7,
         y_train_generated8, y_train_generated9], axis=0)

    x_generated = np.delete(x_train_generated, np.where(np.isin(y_train_generated, drop_classes, invert=True)),
                            axis=0)
    y_generated = np.delete(y_train_generated, np.where(np.isin(y_train_generated, drop_classes, invert=True)),
                            axis=0)
    print(x_generated.shape, y_generated.shape)

    np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_Images', x_generated)
    np.save('/home/fipsi/Documents/Code/Masterarbeit_GPU/Generated_Data/generated_labels', y_generated)

    x_generated = x_generated.reshape((-1, 28 * 28)) # / 255.


    return x_generated, y_generated


@dataclass
class DataLabels:
    """
    Class storing test/train data
    """
    # We'll put everything in the train data if no test data was given and split later
    x_train: np.ndarray  # Train data
    y_train: np.ndarray
    x_test: np.ndarray = None  # Test data
    y_test: np.ndarray = None
    x_val: np.ndarray = None  # Validation data
    y_val: np.ndarray = None

    # If needed: a scaler
    scaler: TransformerMixin = None

    # Configuration
    # Parameters to split the Data
    #   - training = 57000
    #   - test = 10000
    #   - validation = 3000
    test_split: float = .2  # Test data percentage
    val_split: float = .05  # Validation data percentage
    random_state: int = None  # Random seed

    # Metadata
    shape: tuple = None  # Shape of the data
    available_classes: Union[List[int], List[str]] = None  # all available classes

    ## Class methods
    def __repr__(self):
        return self.__class__.__name__


    def get_data_unsupervised(self, data_split: str,
                              drop_classes: Union[List[int], List[str]] = None,
                              include_classes: Union[List[int], List[str]] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for autoencoders
        :param data_split: get data of either "train", "val" or "test"
        :param drop_classes: which classes to drop, drop none if None
        :param include_classes: which classes to include (has priority over drop_classes)
        :return: features and labels -- ERROR: here are no labels, just the data
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split)

        # Drop the classes
        if include_classes:
            drop_classes = self.include_to_drop(include_classes)
        this_x = np.delete(this_data[0], np.where(np.isin(this_data[1], drop_classes)), axis=0)

        # For the autoencoder, we don't need much else than x
        return this_x, this_x

    def get_supervised_data(
            self, data_split: str,
            drop_classes: Union[List[int], List[str]] = None, include_classes: Union[List[int], List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for classifiers
        :param data_split: get data of either "train", "val" or "test"
        :param drop_classes: which classes to drop, drop none if None
        :param include_classes: which classes to include (has priority over drop_classes)
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split)

        # Drop the classes
        if include_classes:
            drop_classes = self.include_to_drop(include_classes)
        this_x = np.delete(this_data[0], np.where(np.isin(this_data[1], drop_classes)), axis=0)
        this_y = np.delete(this_data[1], np.where(np.isin(this_data[1], drop_classes)), axis=0)

        # Return the data
        return this_x, this_y

    # def get_semisupervised_data(
    #         self, data_split: str, anomaly_classes: Union[List[int], List[str]], drop_classes: List[int] = None,
    #         include_classes: List[int] = None,
    #         n_anomaly_samples: int = None
    # ) -> tuple[ndarray, ndarray, ndarray]:
    #     """
    #     Get the labels for the alarm network, i.e. with binary anomaly labels
    #     :param data_split: get data of either "train", "val" or "test"
    #     :param anomaly_classes: classes marked as anomaly
    #     :param drop_classes: which classes to drop (none if None)
    #     :param include_classes: which classes to include (has priority over drop_classes)
    #     :param n_anomaly_samples: reduce the number of anomaly samples
    #     :return: features and labels
    #     """
    #     # Get data
    #     this_data = self._get_data_set(data_split=data_split)
    #
    #     # Drop the classes
    #     if include_classes:
    #         drop_classes = self.include_to_drop(include_classes)
    #     this_x = np.delete(this_data[0], np.where(np.isin(this_data[1], drop_classes)), axis=0)
    #     this_y = np.delete(this_data[1], np.where(np.isin(this_data[1], drop_classes)), axis=0)
    #
    #     y_original = np.copy(this_y)
    #
    #     # Make labels binary
    #     this_y[np.where(~np.isin(this_y, anomaly_classes))] = -1
    #     this_y[np.where(np.isin(this_y, anomaly_classes))] = 0
    #     this_y += 1
    #     this_y = this_y.astype("uint8")
    #
    #     # If desired, reduce the number anomalous samples
    #     if n_anomaly_samples is not None:
    #         # IDs of all anomaly samples
    #         idx_anom = np.where(this_y == 1)[0]
    #         idx_original = np.where(y_original == anomaly_classes)[0]
    #
    #         # Select the indices to delete
    #         n_delete = len(idx_anom) - n_anomaly_samples
    #         idx_delete = np.random.choice(idx_anom, size=n_delete, replace=False)
    #
    #         n_delete_original = len(idx_original) - n_anomaly_samples
    #         original_delete = np.random.choice(idx_original, size=n_delete_original, replace=False)
    #
    #         # Delete indices
    #         this_x = np.delete(this_x, idx_delete, axis=0)
    #         this_y = np.delete(this_y, idx_delete, axis=0)
    #
    #         y_original = np.delete(y_original, original_delete, axis=0)
    #
    #         # Check if we really have the right amount of anomaly samples
    #         assert np.sum(this_y) == n_anomaly_samples
    #
    #     return this_x, this_y, y_original

    '''
    Function to prepare the Dataset in the way that just the normal data is labeled
    --> Delete Anormal labels
    '''

    def get_datasplit(
            self, data_split: str, anomaly_classes: Union[List[int], List[str]] = None, drop_classes: List[int] = None,
            include_classes: List[int] = None,
            delete_labels: List[int] = None,
            delete_x: List[int] = None,
            n_anomaly_samples: int = None
    ) -> tuple[ndarray, ndarray, ndarray]:
        """
        Get the labels for the alarm network, i.e. with binary anomaly labels
        :param data_split: get data of either "train", "val" or "test"
        :param anomaly_classes: classes marked as anomaly
        :param drop_classes: which classes to drop (none if None)
        :param include_classes: which classes to include (has priority over drop_classes)
        :param n_anomaly_samples: reduce the number of anomaly samples
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split)

        this_x = this_data[0]
        this_y = this_data[1]

        # TODO Add generated Data
        if data_split == 'train':
            this_x = np.delete(this_x, np.where(np.isin(this_y, anomaly_classes)), axis=0)
            this_y = np.delete(this_y, np.where(np.isin(this_y, anomaly_classes)), axis=0)
            gen_x, gen_y = create_anomalie_dataset(anomaly_classes)
            this_x = np.concatenate([this_x, gen_x], axis=0)
            this_y = np.concatenate([this_y, gen_y], axis=0)

        # Drop the classes
        if include_classes:
            drop_classes = self.include_to_drop(include_classes)
        this_x = np.delete(this_x, np.where(np.isin(this_y, drop_classes)), axis=0)
        this_y = np.delete(this_y, np.where(np.isin(this_y, drop_classes)), axis=0)
        y_original = np.copy(this_y)

        # Delete Labels of specific classes
        this_x = np.delete(this_x, np.where(np.isin(this_y, delete_x)), axis=0)
        this_y = np.delete(this_y, np.where(np.isin(this_y, delete_labels)), axis=0)

        # Make labels binary
        this_y[np.where(~np.isin(this_y, anomaly_classes))] = -1
        this_y[np.where(np.isin(this_y, anomaly_classes))] = 0
        this_y += 1
        this_y = this_y.astype("uint8")

        # If desired, reduce the number anomalous samples
        if n_anomaly_samples is not None:

            # IDs of all anomaly samples
            idx_anom = np.where(this_y == 1)[0]
            n_delete = len(idx_anom) - n_anomaly_samples
            idx_delete = np.random.choice(idx_anom, size=n_delete, replace=False)

            for i in range(len(anomaly_classes)):
                idx_original = np.where(y_original == anomaly_classes[i])[0]

                # Select the indices to delete
                n_delete_original = len(idx_original) - n_anomaly_samples
                original_delete = np.random.choice(idx_original, size=n_delete_original, replace=False)

                y_original = np.delete(y_original, original_delete, axis=0)

            # Delete indices
            this_x = np.delete(this_x, idx_delete, axis=0)
            this_y = np.delete(this_y, idx_delete, axis=0)

        return this_x, this_y, y_original



    ## Preprocessors
    @abc.abstractmethod
    def _preprocess(self):
        # Preprocessing steps, e.g. data normalisation
        raise NotImplementedError("Implement in subclass")

    def __post_init__(self):
        """
        Process the data
        :return:
        """

        # Fix randomness
        np.random.seed(seed=self.random_state)

        # Get all available classes
        # TODO: we're only looking at the training data so far
        self.available_classes = np.unique(self.y_train).tolist()

        # Split in test and train
        if self.x_test is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train, self.y_train, test_size=self.test_split, random_state=self.random_state
            )

        # Split in train and validation
        if self.x_val is None:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train, self.y_train, test_size=self.val_split, random_state=self.random_state
            )

        # Preprocess
        self._preprocess()

        # Note down the shape
        self.shape = self.x_train.shape[1:]

    ## Helpers
    def include_to_drop(self, include_data: Union[List[int], List[str]]) -> Union[List[int], List[str]]:
        """
        Convert a list of classes to include to a list of classes to drop
        :param include_data: classes to include
        :param all_classes: available classes
        :return: classes to drop
        """

        drop_classes = set(self.available_classes) - set(include_data)

        return list(drop_classes)

    def _get_data_set(self, data_split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the right data split
        :param data_split: train, val or test data?
        :return: the right data set
        """

        if data_split == "train":
            return self.x_train.copy(), self.y_train.copy()

        elif data_split == "test":
            return self.x_test.copy(), self.y_test.copy()

        elif data_split == "val":
            return self.x_val.copy(), self.y_val.copy()

        else:
            raise ValueError("The requested data must be of either train, val or test set.")


# ----------------------------------------------------------------------------------------------------------------------

class MNIST(DataLabels):
    def __init__(self, *args, **kwargs):
        """
        Load the MNIST data set
        """

        # Simply load the data with the kind help of Keras
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Flatten the dataset
        x_train = x_train.reshape((-1, 28 * 28))
        x_test = x_test.reshape((-1, 28 * 28))

        # # Add channel dimension to the data
        # x_train = np.expand_dims(x_train, -1)
        # x_test = np.expand_dims(x_test, -1)

        super(MNIST, self).__init__(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, *args, **kwargs
        )



    def _preprocess(self):
        """
        For MNIST, we can scale everything by just dividing by 255
        :return:
        """
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.
        self.x_val = self.x_val / 255.
