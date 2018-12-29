"""
Created on 23/12/2018
@author: nidragedd

Highly inspired from this blog post:
https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import logging
from models.train import trainer

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
    hp = pgconf.get_hyperparameters(trainer.SIMPLE_NN_NAME)
    learning_rate = hp["learning_rate"]
    optimizer = SGD(lr=learning_rate)  # SGD (Stochastic Gradient Descent)
    loss_function = hp["loss_function"]  # (use categorical_crossentropy if more than 2 classes)

    logger.info("Compiling model and train it")
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
    return model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=nb_epochs), model
