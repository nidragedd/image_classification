"""
Created on 29/12/2018
@author: nidragedd

Contains the whole logic to build and train a model
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import logging

from dataloader.dataset_mgmt import load_training_dataset
from models.train import simple, convnet
from utils import preprocess_utils, display_utils, save_and_restore

logger = logging.getLogger()

SIMPLE_NN_NAME = 'simple'
CNN_NAME = 'convnet'


def evaluate_training(lb, model, X_test, Y_test):
    """
    Allows us to see how performing is the model by making predictions on unseen data
    Using 'predict' function from keras so we can retrieve a Numpy array(s) of predictions (see documentation)
    Then, 'classification_report' from sklearn package (see documentation) allows us to print metrics: summary of the
    precision, recall, F1 score for each class.
    :param lb: (sklearn object) LabelBinarizer used to categorize classes
    :param model: (keras object) the model to use to perform predictions
    :param X_test: (Numpy array) test dataset to perform evaluation on
    :param Y_test: (Numpy array) target corresponding to the test dataset
    """
    logger.info("Evaluation results for this model training are:")
    predictions = model.predict(X_test)
    logger.info(classification_report(Y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))


def do_training(pgconf, model_name):
    training_folder = pgconf.get_training_dir()
    save_folder = pgconf.get_output_dir()
    test_size = pgconf.get_test_size()
    nb_epochs = pgconf.get_nb_epochs(model_name)

    resize_to = 32
    flatten_images = True
    if model_name == CNN_NAME:
        resize_to = 64
        flatten_images = False

    # Load training dataset and associated labels (i.e the class for each image)
    data, labels = load_training_dataset(training_folder, resize_to, flatten_images)

    # Preprocess steps
    data = preprocess_utils.preprocess_scale_row_pixel(data)
    labels = preprocess_utils.preprocess_labels(labels)

    # Split data into training and testing 'datasets' (using sklearn's package train_test_split function)
    # X_train, X_test relates to image data itself whereas Y_train, Y_test relates to the labels ('dog' or 'cat')
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=test_size, random_state=123)

    # Transform 'strings' labels into vectors of integers
    lb, Y_train, Y_test = preprocess_utils.class_categorization(Y_train, Y_test)

    # Build and compile the model, then train it
    fit_history, model = None, None
    if model_name == SIMPLE_NN_NAME:
        fit_history, model = simple.build_and_train_model(pgconf, len(lb.classes_), nb_epochs, X_train,
                                                          Y_train, X_test, Y_test)
    elif model_name == CNN_NAME:
        input_shape = (64, 64, 3)
        fit_history, model = convnet.build_and_train_model(pgconf, len(lb.classes_), input_shape, nb_epochs,
                                                           X_train, Y_train, X_test, Y_test)

    # Evaluate the network on our testing data
    evaluate_training(lb, model, X_test, Y_test)

    # Save some metrics for this training
    display_utils.save_plot_history(save_folder, nb_epochs, fit_history, model_name)

    # Save data resulting from training
    save_and_restore.save_model_and_lb(save_folder, model, lb, model_name)
