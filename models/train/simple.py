"""
Created on 23/12/2018
@author: nidragedd

Highly inspired from this blog post:
https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import logging

from dataloader.dataset_mgmt import load_training_dataset
from utils import preprocess_utils, display_utils, save_and_restore

logger = logging.getLogger()


def build_and_train_model(pgconf, nb_classes, nb_epochs, X_train, Y_train, X_test, Y_test):
    """
    Build one very simple Neural Network with 4 layers (3072-1024-512-x):
    - 3072 is input layer (3072 = 32x32x3 pixels = flattened images)
    - x as output is the number if possible class labels (2 if binary classification, n if multi-class classification)
    Activation function on each layers are common ones found over several tutorials
    :param pgconf: (object) the program configuration handler
    :param nb_classes: (int) the number of classes for the classification (if 2 => binary ;-))
    :param nb_epochs: (int) the number of epochs to train
    :param X_train: (Numpy array) training data
    :param Y_train: (Numpy array) training target (label) data
    :param X_test: (Numpy array) data used to evaluate the loss and any model metrics at the end of each epoch
    :param Y_test: (Numpy array) same purpose as X_test but on targets
    :return: (tuple)
            1st element is an `History` object (see keras documentation) which is a record of training loss values and
            metrics values at successive epochs, as well as validation loss values
            2nd element is the built model itself (keras Sequential object)
    """
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(nb_classes, activation="softmax"))

    # Hyperparameters settings
    hp = pgconf.get_hyperparameters('simple')
    learning_rate = hp["learning_rate"]
    optimizer = SGD(lr=learning_rate)  # SGD (Stochastic Gradient Descent)
    loss_function = hp["loss_function"]  # (use categorical_crossentropy if more than 2 classes)

    logger.info("Compiling model and train it")
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
    return model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=nb_epochs), model


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


def do_training(pgconf):
    model_name = 'simple'
    training_folder = pgconf.get_training_dir()
    save_folder = pgconf.get_output_dir()
    test_size = pgconf.get_test_size()
    nb_epochs = pgconf.get_nb_epochs(model_name)

    # Load training dataset and associated labels (i.e the class for each image)
    data, labels = load_training_dataset(training_folder)

    # Preprocess steps
    data = preprocess_utils.preprocess_scale_row_pixel(data)
    labels = preprocess_utils.preprocess_labels(labels)

    # Split data into training and testing 'datasets' (using sklearn's package train_test_split function)
    # X_train, X_test relates to image data itself whereas Y_train, Y_test relates to the labels ('dog' or 'cat')
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=test_size, random_state=123)

    # Transform 'strings' labels into vectors of integers
    lb, Y_train, Y_test = preprocess_utils.class_categorization(Y_train, Y_test)

    # Build and compile the model, then train it
    fit_history, model = build_and_train_model(pgconf, len(lb.classes_), nb_epochs, X_train, Y_train, X_test, Y_test)

    # Evaluate the network on our testing data
    evaluate_training(lb, model, X_test, Y_test)

    # Save some metrics for this training
    display_utils.save_plot_history(save_folder, nb_epochs, fit_history, model_name)

    # Save data resulting from training
    save_and_restore.save_model_and_lb(save_folder, model, lb, model_name)
