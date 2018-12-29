"""
Created on 29/12/2018
@author: nidragedd

Highly inspired from this blog post:
https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
"""
from keras import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from models.train import trainer
import logging

logger = logging.getLogger()


def build_and_train_model(pgconf, nb_classes, input_shape, nb_epochs, X_train, Y_train, X_test, Y_test):
    """
    Build a small version of a Convolutional Neural Network of VGG type.
    VGG architecture:
        - use only 3×3 convolutional layers stacked on top of each other in increasing depth.
        - reduce volume size by using max pooling.
        - in the end, 2 FC (fully-connected) layers followed by a softmax classifier
    INPUT -> ['CONV -> RELU -> POOL']
          -> [(CONV -> RELU) * 2 -> POOL]
          -> [(CONV -> RELU) * 3 -> POOL]
          -> [FC -> RELU]
          -> FC
          -> softmax classifier

    RELU (Rectified Linear Unit):
        the activation function used in this network architecture.
    Batch Normalization:
        used to normalize the activations of a given input volume before passing it to the next layer in the network.
        Proven to be very effective at reducing the number of epochs required to train a CNN as well as stabilizing
        training itself.
    POOL:
        used to progressively reducing the the input volume to a layer.
    Dropout:
        process of disconnecting random neurons between layers: reduce overfitting, increase accuracy, and allow our
        network to generalize better for unfamiliar images.
        Here, 25% of the node connections are randomly disconnected between layers during each training iteration.

    :param pgconf: (object) the program configuration handler
    :param nb_classes: (int) the number of classes for the classification (if 2 => binary ;-))
    :param input_shape: (tuple) represent width, height and depth of images to put in the network
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
    # With Tensorflow backend, depth is in last position ("channels_last")
    channels_dimensions = -1  # means the 'last dimension/value'
    filter_size = (3, 3)  # VGG architecture requirement
    dropout_value = 0.25
    max_pool_size = (2, 2)

    model = Sequential()
    # 1) 'CONV -> RELU -> POOL' layer
    model.add(Conv2D(32, filter_size, padding="same", input_shape=input_shape))  # 32 filters of size 3x3
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimensions))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_value))

    # 2) '(CONV -> RELU) * 2 -> POOL' layer
    model.add(Conv2D(64, filter_size, padding="same"))  # Twice more filters
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimensions))
    model.add(Conv2D(64, filter_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimensions))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_value))

    # 3) '(CONV -> RELU) * 3 -> POOL' layer
    model.add(Conv2D(128, filter_size, padding="same"))  # Again, twice more filters
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimensions))
    model.add(Conv2D(128, filter_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimensions))
    model.add(Conv2D(128, filter_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channels_dimensions))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_value))

    # 4) Fully-connected layer 'FC -> RELU'
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # todo: tester dropout x 2

    # The final layer is fully connected with 'nb_classes' outputs
    # 'softmax' returns the class probabilities for each class
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    # Hyperparameters settings
    hp = pgconf.get_hyperparameters(trainer.CNN_NAME)
    learning_rate = hp["learning_rate"]
    optimizer = SGD(lr=learning_rate)  # SGD (Stochastic Gradient Descent)
    loss_function = hp["loss_function"]  # (use categorical_crossentropy if more than 2 classes)

    logger.info("Compiling model and train it")
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
    return model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=nb_epochs), model
