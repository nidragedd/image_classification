"""
Created on 26/12/2018
@author: nidragedd

Utils package to perform some useful preprocessing operations
"""
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer


def preprocess_scale_row_pixel(data):
    """
    Scale the raw pixel intensities to the range [0, 1] (common preprocessing step)
    :param data: (array) image informations
    :return: (Numpy array) data transformed as np array
    """
    return np.array(data, dtype="float") / 255.0


def preprocess_labels(labels):
    """
    Just transform a simple array of labels to a numpy array
    :param labels: (array) labels classes as array
    :return: (Numpy array) labels transformed as np array
    """
    return np.array(labels)


def class_categorization(Y_train, Y_test):
    """
    Class labels are represented as strings ('dog' or 'cat')
    To be able to use Keras framework we have to transpose this information as vectors of integer (one-hot encoding)
    - For multi-class classification, we just have to use a LabelBinarizer object from sklearn package
    - For binary (i.e 2-class), the LabelBinarizer will not return a Vector so we have to use the to_categorical function
    :param Y_train: (Numpy array) the training dataset targets (i.e classes)
    :param Y_test: (Numpy array) the test dataset targets (i.e classes)
    :return: (tuple) label binarizer used (sklearn object), training and test dataset transformed as vectors of integers
    (Numpy arrays)
    """

    lb = LabelBinarizer()
    # 'fit_transform' finds all unique class labels in Y_train and transforms them into one-hot encoded labels
    Y_train = lb.fit_transform(Y_train)
    # 'transform' performs just the one-hot encoding step (unique class labels has already been found above
    Y_test = lb.transform(Y_test)

    if len(np.unique(Y_train)) == 2:
        # Handling the specific case of binary classification
        # use Keras to_categorical function as the scikit-learn's LabelBinarizer does not return a vector
        Y_train = to_categorical(Y_train, num_classes=2, dtype='int32')
        Y_test = to_categorical(Y_test, num_classes=2)

    return lb, Y_train, Y_test
