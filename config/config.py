"""
Created on 25/12/2018
@author: nidragedd
"""
import os
import json
import logging.config


class Configuration(object):
    """
    Class used to handle and maintain all parameters of this program
    """
    _config = None
    _working_dir = None

    def __init__(self, config_filepath):
        """
        Constructor
        :param config_filepath: (string) the configuration file to read (must be a JSON format file)
        """
        if os.path.exists(config_filepath):
            with open(config_filepath, 'rt') as f:
                self._config = json.load(f)
        else:
            raise Exception("Could not load config file " + config_filepath)

    def set_working_dir(self, working_dir):
        """
        Set the working directory
        :param working_dir: (string) directory path to the working directory from which all others directory will be
        computed
        """
        self._working_dir = working_dir

    def get_dir(self, dir_name):
        """
        Returns the output directory (according to what is set in config.json file)
        :return: (string) directory path to output (save) folder
        """
        if "main" in self._config and dir_name in self._config["main"]:
            return os.path.join(self._working_dir, self._config["main"][dir_name])
        else:
            raise Exception("Could not find {} folder data in configuration file !".format(dir_name))

    def get_output_dir(self):
        """
        Returns the output directory (according to what is set in config.json file)
        :return: (string) directory path to output (save) folder
        """
        return self.get_dir("output_folder")

    def get_training_dir(self):
        """
        Return the training folder (according to what is set in config.json file)
        :return: (string) directory path to training dataset folder
        """
        return self.get_dir("training_folder")

    def get_validation_dir(self):
        """
        Return the validation folder (according to what is set in config.json file)
        :return: (string) directory path to validation dataset folder
        """
        return self.get_dir("validation_test_folder")

    def get_hyperparameters(self, model):
        """
        Return hyperparameters written in configuration file as dict for model which has the given name
        :param model: (string) the model name (should be either 'simple' or 'cnn')
        :return: (dict) hyperparameters for this model
        """
        if model in self._config:
            return self._config[model]["hyperparameters"]
        else:
            raise Exception("Could not find model named {} in configuration file !".format(model))

    def get_test_size(self):
        """
        Return the value for testing size as configured in config.json file
        :return: value for testing size
        """
        if "main" in self._config and "test_size" in self._config["main"]:
            return self._config["main"]["test_size"]
        else:
            raise Exception("Could not find test size value in configuration file !")

    def get_nb_epochs(self, model):
        """
        Returns the number of epochs as configured in config.json file
        :param model: (string) the model name (should be either 'simple' or 'cnn')
        :return: (int) number of epochs
        """
        hp = self.get_hyperparameters(model)
        if "nb_epochs" in hp:
            return hp["nb_epochs"]
        else:
            raise Exception("Could not find number of epochs for model named {} in configuration file !".format(model))


def configure_logging(log_config_file):
    """
    Setup logging configuration
    :param log_config_file: (string) the logging configuration file to read (must be a JSON format file)
    """
    if os.path.exists(log_config_file):
        with open(log_config_file, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
