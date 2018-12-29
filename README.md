# Disclaimer
* This project is only for personal challenge and educational purpose, no other pretention than those ones.
* My goal was not to (obviously) reivent the wheel, this project is highly inspired from several (good) readings (see 
some references below + do not forget that search engines are your friends)
* In the end, objective was also to discover, understand and improve my personal skills in scikit-learn and keras usage
when building neural networks.


# Useful and interesting reading before starting
* [pyimagesearch blog: How to get started with Keras, Deep Learning, and Python](https://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/).
You can also follow links inside the post, they all contains useful informations on VGGNet models for example.
*Almost everything in this project comes from this blog post, it is a very good starting point.*
* [pyimagesearch blog: Implement and use already trained networks](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/).
* [Kaggle kernel: Cats vs. Dogs](https://www.kaggle.com/stevenhurwitt/cats-vs-dogs-using-a-keras-convnet)
* [Keras Blog: Building powerful image classification models using little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
* [scikit-learn tutorial: face recognition with SVM](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py)

Grab also some informations on Convolutional Neural Networks to understand their purpose and (mainly) how it works. 
Wikipedia might help you ([FR](https://fr.wikipedia.org/wiki/R%C3%A9seau_neuronal_convolutif) or [EN](https://en.wikipedia.org/wiki/Convolutional_neural_network)).

# Module dependencies (requirements) and their usage
This project works with ***Python 3.6.x*** (not 3.7 as tensorflow backend is not *yet* supported for more than 3.6). If not
already installed, use pip to install those packages:
* [keras](https://keras.io/) (deep learning and neural networks made easy)
* [tensorflow](https://www.tensorflow.org/) (backend used by keras)
* [scikit-learn](https://scikit-learn.org/stable/) (machine learning)
* [numpy](http://www.numpy.org/)
* [opencv](https://opencv.org/) (image manipulation)

# How to use it and assumptions
Goal of this project is either:
* to build and train a neural network
* to use an already trained one
* to use a pre-trained network provided with Keras (which are able of to classify images among 1,000 different object 
categories, similar to objects we encounter in our day-to-day lives with high accuracy)
to be able, in the end, to classify a picture the model has never seen and says whether it is a dog or a cat (or actually
whatever else you would like to implement).

For that, the *main.py* takes 3 arguments (2 first are mandatory):
* -m (or "--model"), **mandatory**: shoud equals 'simple' or 'cvn', depending on which NN you would like to use (NB: so
far, only simple one is implemented)
* -o (or "--objective"), **mandatory**: should equals 'train' or 'predict', depending on what you would like to do
* -w (or "--working-dir"): use it to set another working directory than the one where this module is executed.

## Directory structure
Some folders are (configured) and required:
* ***training***: the folder for training image dataset (put images in subfolders named against categories). For example,
create 2 folders named 'dog' and 'cat' to classify between both those animals
* ***output***: folder where trained models are stored or loaded, depending on what is required
* ***validation_test***: folder where images are loaded to perform predictions

## External configuration file
Those parameters plus some model hyperparameters are set in the external *<working_directory>*/config/config.json file.
Change them to fit your needs.

## Simple NN
Goal here is to discover the libraries, it is written almost everywhere (for example 
[here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html))
that "The right tool for an image classification job is a convnet (convolutional network)".  
Our simple NN will not give wonderful results but might be able to predict most of the cases.
This simple Neural Network is built with 4 layers (3072-1024-512-x):
* 3072 as input layer (3072 = 32x32x3 pixels = flattened images)
* x as output is the number if possible class labels (2 if binary classification, n if multi-class classification)
Activation function (sigmoid) on each layers are common ones found over several tutorials

