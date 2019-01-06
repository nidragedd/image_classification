"""
Created on 25/12/2018
@author: nidragedd

This is the main file used to make a prediction by using an already trained model: either an homemade one, either one
of the already pretrained models available in keras:
* VGG (16 & 19) (for more information on those networks, please refer to: https://arxiv.org/abs/1409.1556)
* ResNet50 (for more information, please refer to: https://arxiv.org/abs/1512.03385)
"""
import cv2
import keras
import numpy as np
import logging
from keras_applications import imagenet_utils
from keras.applications import VGG16, VGG19, ResNet50
from keras_preprocessing.image import load_img, img_to_array
from dataloader.dataset_mgmt import load_validation_dataset, load_images_path
from models.train import trainer
from utils import save_and_restore, preprocess_utils
from utils.display_utils import draw_picture_with_label_and_score

PRETRAINED_MAPPING = {trainer.PRETRAINED_VGG16: VGG16,
                      trainer.PRETRAINED_VGG19: VGG19,
                      trainer.PRETRAINED_RESNET: ResNet50}

logger = logging.getLogger()


def do_magic(pgconf, model_name):
    """

    All images are then shown with an additional text displaying the prediction result.
    :param pgconf: (object) this program configuration handler
    :param model_name: (string) the model name (either 'simple' or 'convnet')
    """
    validation_dir = pgconf.get_validation_dir()
    load_dir = pgconf.get_output_dir()
    if model_name in PRETRAINED_MAPPING:
        use_pretrained_model(model_name, validation_dir)
    else:
        use_homemade_model(model_name, validation_dir, load_dir)


def use_homemade_model(model_name, validation_dir, load_dir):
    """
    Load an already (homemade) trained model from disk and use it against all images found in validation directory (see
    value in external config.json configuration file) in order to make predictions.
    Pictures of images and their predicted classification are shown at the end of the prediction phase.
    :param model_name: (string) the model name (either 'simple' or 'convnet')
    :param validation_dir: (string) full path to the directory that contains images to classify
    :param load_dir: (string) full path to the directory that contains data related to the homemade trained model to load
    """
    # Load validation dataset
    resize_to = 32
    flatten_images = True
    if model_name == trainer.CNN_NAME:
        resize_to = 64
        flatten_images = False
    original_data, data = load_validation_dataset(validation_dir, resize_to, flatten_images)

    # Preprocess steps (same as for training)
    data = preprocess_utils.preprocess_scale_row_pixel(data)

    # Load from disk the saved and trained keras model and its sklearn LabelBinarizer
    model, lb = save_and_restore.load_model_and_lb(load_dir, model_name)

    # make prediction on the given dataset
    predictions = model.predict(data)
    # argmax returns indice of the highest element in the np array
    indices_for_max = predictions.argmax(axis=1)

    # For each one, display the picture and the class determined by the model
    for i, prediction in enumerate(predictions):
        indice = indices_for_max[i]
        proba_class = lb.classes_[indice]
        proba_value = prediction[indice] * 100

        # Display the original image (window title is the file name so that we can check if the prediction is accurate)
        draw_picture_with_label_and_score(proba_class, proba_value, original_data[i])

    # Need to wait for user action otherwise all windows images are quickly closed
    cv2.waitKey(0)


def use_pretrained_model(model_name, validation_dir):
    """
    Load an existing pretrained model from keras library and use it against all images found in validation directory
    (see value in external config.json configuration file) in order to make predictions.
    We just have to find the right one, instantiate it and load the ImageNet weights from disk (if not present, it will
    be automatically downloaded and cached (so 1st call is 'quite' long, depending on network bandwidth)
    Pictures of images and their predicted classification are shown at the end of the prediction phase.
    :param model_name: (string) the model name (either 'simple' or 'convnet')
    :param validation_dir: (string) full path to the directory that contains images to classify
    """
    input_shape = (224, 224)  # Hardcoded because VGG16,19 or ResNet all accept this kind of input_shape

    # Based on the model name, instantiate the appropriate pretrained network
    pretrained_model_app = PRETRAINED_MAPPING[model_name]
    model = pretrained_model_app(weights="imagenet")
    print("backend: {}".format(keras.backend.image_data_format()))  # Used for debugging purpose (bug in 2.2.4 ?)

    # Load validation dataset, image per image
    images_path_list = load_images_path(validation_dir)
    for image_path in images_path_list:
        # Using keras utils methods to load an image into PIL format and convert it to a Numpy array
        img = load_img(image_path, target_size=input_shape)
        x = img_to_array(img)

        # At this point, image shape is (inputShape[0], inputShape[1], 3). To be able to give it to the network, we have
        # to expand the dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # This is mandatory as, in CNN, images are *often* classified in batches. Without expansion, the 'predict'
        # method will throw an error
        x = np.expand_dims(x, axis=0)

        print("backend: {}".format(keras.backend.image_data_format()))

        # Use the chosen preprocess method on the image
        # Seems there is this bug to solve in 2.2.4 (https://stackoverflow.com/questions/52717181/keras-create-mobilenet-v2-model-attributeerror)
        x = imagenet_utils.preprocess_input(x)

        # Do the magic !
        predictions = model.predict(x)
        # A list of lists of top class prediction tuples `(class_name, class_description, score)`
        top_preds = imagenet_utils.decode_predictions(predictions)

        # Display results with highest probabilities
        for (rank, (class_name, class_description, score)) in enumerate(top_preds[0]):
            logger.info("{}. {}: {:.2f}%".format(rank + 1, class_description, score * 100))

        # Display image with the most probable result on it
        class_description = top_preds[0][0][1]
        score = top_preds[0][0][2]
        draw_picture_with_label_and_score(class_description, score * 100, image_path)
        cv2.waitKey(0)

