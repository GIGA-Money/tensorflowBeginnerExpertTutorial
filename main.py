import tensorflow as tf
from tensorflow import keras as ker
import numpy as np
import matplotlib.pyplot as plt


'''
this dataset is a CIFAR10 small image classification
what is returned are 2 tuples.

'''
data = ker.datasets.fashion_mnist

'''
here 2 tuples are loaded one for the test and one for the traing
the x images or train images, and the y labels or train labels
so (x-train, y-train) , (x-test, y-test) = data.load_data()
consider that the images are predictors, and then the labels are 
run through the algorithm, which is supervised
The algorithm produces a model used at run time. 
'''
# Testing data and Training data.
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Label names.
class_name = ['t-shirt', 'pants', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'shoes', 'bag', 'boots']
'''
bellow we ensure that the samples go from there integer representation 
to teh floating point representation. 
'''
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
    ker.layers.Dense(100, activation="relu"),
    ker.layers.Dropout(0.2),
    ker.layers.Dense(10, activation="softmax")
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

'''
to view the images, adjust the values between 0-9, as there are only 10 images, 
the value -1 in python indicates the last value in the array.  
plt.figure()
plt.imshow(train_images[-1])
plt.colorbar()
plt.grid(False)
plt.show()
'''
# .
# to see how this works we can set these values test loss and test accuracy.
# the print these vars.
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("test accuracy:",  test_acc , '\ntest loss:', test_loss)
#
#


'''
as we will continually train the model, we may want to save.
'''

# after testing we can use the model.
# The model.predict,
# takes in a list, what we need to do is put the array into a list.
# as test_images is already in a list by itself, we only give that one list,
# indicating that other list will need to become a list of list such as .predict([list1, list1])
prediction = model.predict(test_images)
'''
the output is only ten values, as we only have ten neurons.
representing how the model thinks of each different class. 
print(prediction)
'''
# if there was a need for a particular value within the prediction list,
# we can still use [] for random access.

'''
now if we wanted to select the max we can use the argmax from the numpy library. 
this this case we want the max from the last value in the list. 
this is the index so say it returns 5, then its indicating that sandal is what the model sees 
as being predicted. 
print(np.argmax(prediction[-1]))
'''
'''
to take this a step further with the list of names, this gives us back the name, 
now we can just past that index into the class names list, which is sandal. 
print(class_name[np.argmax(prediction[-1])])
'''

# here we'll dress up the solution to show the pictures and the prediction image.
for index in range(5):
    plt.grid(False)
    plt.imshow(test_images[index], cmap=plt.cm.binary)
    plt.xlabel("True image: " + class_name[test_labels[index]])
    plt.title("prediction image: " + class_name[np.argmax(prediction[index])])
    plt.ylabel("the predicted value: " + str(np.argmax(prediction[index])))
    plt.show()

