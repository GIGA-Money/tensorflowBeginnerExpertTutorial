import tensorflow as tf
from tensorflow import keras as ker
import numpy as np
import matplotlib.pyplot as plt

data = ker.datasets.fashion_mnist

# Testing data and Training data.
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Label names.
class_name = ['t-shirt', 'pants', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'shoes', 'bag', 'boots']

train_images = train_images / 255
test_images = test_images / 255

# Here we define the layers.
# first we have an input layer, the keras flatten input layer, takes
# the shape will be of the size 28 x 28, so its passable to all the different layers.
# because as we get information from a array of n size,
# before we can pass the data we need to flatten it to a specific size.
#
# the second layer is a dense layer
# as each node is connected in the NN, this layer defines how.
# so 128 neurons, and the activation function will be rectify linear unit
# relu = https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
# we can pick different activation functions,
# making this selection almost arbitrary.
#
# following this initial layer of neurons will be a second layer of neurons.
# again this layer will be Dens as well with 10 neurons.
# this acts as the output layer, as we only have 10 types of images.
# this layer is a softmax layer more here:
# https://towardsdatascience.com/the-softmax-function-neural-net-outputs-as-probabilities-and-ensemble-classifiers-9bd94d75932
# this picks values for each neuron so that each value adds up to 1
# meaning we can look at the last layer and see the probability or what the network thinks per given class.
# we can find more activation layers here https://keras.io/activations/
#

# the output layer, the values must be at a minimum equal to the number items tested for.
# in this instance it would be 10 minimum, as we only have ten varieties of items.
model = ker.Sequential([
    ker.layers.Flatten(input_shape=(28, 28)),
    ker.layers.Dense(10, activation="relu"),
    ker.layers.Dense(9, activation="softmax")
])

#
# optimizer, adam is a standard solution. more here https://keras.io/optimizers/
# loss: https://keras.io/losses/
# metrics: https://keras.io/metrics/
#
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#
# to train we pass in the train images and labels.
# then set the episodes to run for.
# the order and number of consistent elements that come in influence the training process.
# working with the accuracy model.
#
model.fit(train_images, train_labels, epochs=1)

# .
# to see how this works we can set these values test loss and test accuracy.
# the print these vars.
#
# .
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("test accuracy:",  test_acc , '\ntest loss:', test_loss)