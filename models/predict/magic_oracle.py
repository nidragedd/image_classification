"""
Created on 25/12/2018
@author: nidragedd

This is the main file used to make a prediction by using an already trained model
"""
import cv2
from dataloader.dataset_mgmt import load_validation_dataset
from models.train import trainer
from utils import save_and_restore, preprocess_utils


def do_magic(pgconf, model_name):
    """
    Load an already trained model from disk and use it against all images found in validation directory (see value in
    external config.json configuration file) in order to make predictions.
    All images are then shown with an additional text displaying the prediction result.
    :param pgconf: (object) this program configuration handler
    :param model_name: (string) the model name (either 'simple' or 'convnet')
    """
    validation_dir = pgconf.get_validation_dir()
    load_dir = pgconf.get_output_dir()

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

        original_image = original_data[i][0]
        original_filename = original_data[i][1]

        # Add determined class information on the image
        text = "{}: {:.2f}%".format(proba_class, proba_value)
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_color = (0, 0, 0)
        bottom_left_origin = (10, 30)
        cv2.putText(original_image, text, bottom_left_origin, font_face, 1, font_color, 2)

        # Display the original image (window title is the file name so that we can check if the prediction is accurate)
        cv2.imshow(original_filename, original_image)
    # Need to wait for user action otherwise all windows images are quickly closed
    cv2.waitKey(0)
