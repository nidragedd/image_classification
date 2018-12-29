"""
Created on 27/12/2018
@author: nidragedd

Utils package to perform some useful plotting/displaying operations
"""
import os
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
