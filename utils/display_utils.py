"""
Created on 27/12/2018
@author: nidragedd

Utils package to perform some useful plotting/displaying operations
"""
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def save_plot_history(save_folder, display_range, fit_history, model_type):
    """
    Build and save a graph corresponding to the given fit_history
    :param save_folder: (string) path to output folder where we will save the generated graph image/data
    :param display_range: (int) number of epochs the model has been trained on
    :param fit_history: (keras object) the fitting history from where we will extract datas
    :param model_type: (string) model type (only used to be displayed in graph title and file output name)
    """
    # Set the matplotlib backend to a non-interactive one so figures can be saved in the background
    matplotlib.use("agg")
    np_range = np.arange(0, display_range)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np_range, fit_history.history["loss"], label="train_loss")
    plt.plot(np_range, fit_history.history["val_loss"], label="val_loss")
    plt.plot(np_range, fit_history.history["acc"], label="train_acc")
    plt.plot(np_range, fit_history.history["val_acc"], label="val_acc")
    plt.title("Training 'loss' and 'accuracy' metrics (model: {})".format(model_type))
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy value")
    plt.legend()

    save_file = os.path.join(save_folder, model_type + ".png")
    plt.savefig(save_file)


def draw_picture_with_label_and_score(proba_class, proba_value, image_path):
    """
    From a given image path, load it with opencv library and put a description on it with the label and score
    :param proba_class: (string) the label/class of the predicted value
    :param proba_value: (float) probability that this image is part of the class found
    :param image_path: (string) full path to the image on disk to load it
    """
    original_filename = image_path.split(os.path.sep)[-1]
    original_filename = original_filename[:original_filename.rfind(".")].lower()

    original_image = cv2.imread(image_path)

    # Add determined class information on the image
    text = "{}: {:.2f}%".format(proba_class, proba_value)
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0, 0, 0)
    bottom_left_origin = (10, 30)
    cv2.putText(original_image, text, bottom_left_origin, font_face, 1, font_color, 2)

    # Display the original image (window title is the file name so that we can check if the prediction is accurate)
    cv2.imshow(original_filename, original_image)
