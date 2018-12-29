"""
Created on 25/12/2018
@author: nidragedd

Python package providing useful methods to load images dataset and make some manipulations
Note: download a big Kaggle cat/dogs dataset from here: https://www.microsoft.com/en-us/download/details.aspx?id=54765)
"""
import logging
import random
import os
import cv2

logger = logging.getLogger()


def load_training_dataset(training_dir, resize_to, do_flatten=False):
    """
    Load all images from the given directory and use the directory name as class
    :param training_dir: (string) path to training directory from where all images will be read (subfolders names are
    used as class labels)
    :param resize_to: (int) the size to resize the image (will be used for width and height)
    :param do_flatten: (boolean) set to True (False by default) to flatten the data. If using a Convolutional NN, do not
    flatten the data
    :return: (tuple) 1st element is an array of image flatten and resized, 2nd one is the according label for each one
    """
    logger.info("Loading training dataset")
    data = []
    labels = []

    # Grab all image paths and randomly shuffle them (with a given seed to be able to reproduce)
    images_path_list = load_images_path(training_dir)
    random.seed(123)
    random.shuffle(images_path_list)

    for image_path in images_path_list:
        data.append(read_image_data(image_path, resize_to, do_flatten))

        # From the image path and folder name, determine the class label, then add it to the labels list
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)

    logger.info("All images loaded")
    return data, labels


def load_validation_dataset(validation_dir, resize_to, do_flatten=False):
    """
    Quite the same as the training dataset loading except that we will keep a safe copy of the data (for displaying
    purpose)
    :param validation_dir: (string) path to directory from where we load images to classify
    :param resize_to: (int) the size to resize the image (will be used for width and height)
    :param do_flatten: (boolean) set to True (False by default) to flatten the data. If using a Convolutional NN, do not
    flatten the data
    :return: (tuple) 1st element is an array of original data loaded, 2nd one is the image resized and flatten (same
    process as for training)
    """
    logger.info("Loading validation dataset")
    data = []
    original_data = []

    # Grab all image paths
    images_path_list = load_images_path(validation_dir)
    for image_path in images_path_list:
        img_file_name = image_path.split(os.path.sep)[-1]
        img_file_name = img_file_name[:img_file_name.rfind(".")].lower()
        original_data.append((cv2.imread(image_path), img_file_name))
        data.append(read_image_data(image_path, resize_to, do_flatten))

    logger.info("All validation images loaded")
    return original_data, data


def load_images_path(image_folder):
    """
    From a given folder, build a sorted list of all images paths
    :param image_folder: (string) path to folder to iterate on
    :return: (list) a sorted list which contains all images paths
    """
    image_list = []
    for (dirpath, dirnames, filenames) in os.walk(image_folder):
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            image_list.append(image_path)
    return sorted(image_list)


def read_image_data(image_path, resize_to, do_flatten=False):
    """
    Use opencv library to load image and resize them to the given 'resize_to size'. For example: 32x32px (ignoring
    aspect ratio).
    If 'do_flatten' is specified, image is flattened. With our example it would be 32 x 32 x 3 = 3072px image
    :param image_path: (string) path to the image to read and transform
    :param resize_to: (int) the size to resize the image (will be used for width and height)
    :param do_flatten: (boolean) set to True (False by default) to flatten the data. If using a Convolutional NN, do not
    flatten the data
    :return: (numpy array) of shape (3072,) corresponding to the image data
    """
    data = cv2.resize(cv2.imread(image_path), (resize_to, resize_to))
    if do_flatten:
        data = data.flatten()
    return data
