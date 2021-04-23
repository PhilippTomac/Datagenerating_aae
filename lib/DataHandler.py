import abc
import numpy as np
import tensorflow as tf

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

from typing import List, Union, Tuple


# TODO: Create DataHandler for MNIST
# Loading the MNIST Dataset

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
    test_split: float = .2  # Test data percentage
    val_split: float = .05  # Validation data percentage
    random_state: int = None  # Random seed

    # Metadata
    shape: tuple = None  # Shape of the data
    available_classes: Union[List[int], List[str]] = None  # all available classes

    ## Class methods
    def __repr__(self):
        return self.__class__.__name__

    ## Retrievers
    def get_target_autoencoder_data(
            self, data_split: str,
            drop_classes: Union[List[int], List[str]] = None, include_classes: Union[List[int], List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for autoencoders
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

        # For the autoencoder, we don't need much else than x
        return this_x, this_x

    def get_target_classifier_data(
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

    def get_alarm_data(
            self, data_split: str, anomaly_classes: Union[List[int], List[str]], drop_classes: List[int] = None,
            include_classes: List[int] = None,
            n_anomaly_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        # Drop the classes
        if include_classes:
            drop_classes = self.include_to_drop(include_classes)
        this_x = np.delete(this_data[0], np.where(np.isin(this_data[1], drop_classes)), axis=0)
        this_y = np.delete(this_data[1], np.where(np.isin(this_data[1], drop_classes)), axis=0)

        # Make labels binary
        this_y[np.where(~np.isin(this_y, anomaly_classes))] = -1
        this_y[np.where(np.isin(this_y, anomaly_classes))] = 0
        this_y += 1
        this_y = this_y.astype("uint8")

        # If desired, reduce the number anomalous samples
        if n_anomaly_samples is not None:
            # IDs of all anomaly samples
            idx_anom = np.where(this_y == 1)[0]

            # Select the indices to delete
            n_delete = len(idx_anom) - n_anomaly_samples
            idx_delete = np.random.choice(idx_anom, size=n_delete, replace=False)

            # Delete indices
            this_x = np.delete(this_x, idx_delete, axis=0)
            this_y = np.delete(this_y, idx_delete, axis=0)

            # Check if we really have the right amount of anomaly samples
            assert np.sum(this_y) == n_anomaly_samples

        return this_x, this_y

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


class MNIST(DataLabels):
    def __init__(self, *args, **kwargs):
        """
        Load the MNIST data set
        """

        # Simply load the data with the kind help of Keras
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Add channel dimension to the data
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

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


@dataclass
class ExperimentConfig:
    """
    Configuration which data is used in the respective experiment
    """
    data_set: DataLabels  # Data set to use
    train_normal: list  # Classes for normal samples
    train_anomaly: list  # Classes for known anomalies
    test_anomaly: list  # Classes for test anomalies


