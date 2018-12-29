"""
Created on 27/12/2018
@author: nidragedd

Utils package to backup/load useful data related to models
"""
import pickle
import os
import logging

from keras.engine.saving import load_model

logger = logging.getLogger()


def save_model_and_lb(save_folder, model, lb, model_type):
    """
    Save model and LabelBinarizer to disk in order to reuse them later without having to train again the model
    :param save_folder: (string) path to output folder where we will save the informations
    :param model: (keras object) the model to persist on disk
    :param lb: (sklearn object) the Label Binarizer used to classify data
    :param model_type: (string) model type (only used to name files)
    """
    model_save_file = os.path.join(save_folder, model_type + ".hd5")
    lb_save_file = os.path.join(save_folder, model_type + "_lb.pickle")

    logger.info("Saving model and label binarizer to disk in {} folder".format(save_folder))
    model.save(model_save_file)
    f = open(lb_save_file, "wb")
    f.write(pickle.dumps(lb))
    f.close()


def load_model_and_lb(load_folder, model_type):
    """
    Load model and LabelBinarizer from disk in order to reuse them later without having to train again the model
    :param load_folder: (string) path to output folder from where we will load the informations
    :param model_type: (string) model type (only used to name files)
    :return: (tuple) 1st element is the keras model object, 2nd one is the sklearn used LabelBinarizer
    """
    model_save_file = os.path.join(load_folder, model_type + ".hd5")
    lb_save_file = os.path.join(load_folder, model_type + "_lb.pickle")

    logger.info("Loading model and label binarizer from {} folder on disk".format(load_folder))
    model = load_model(model_save_file)
    lb = pickle.loads(open(lb_save_file, "rb").read())
    return model, lb
