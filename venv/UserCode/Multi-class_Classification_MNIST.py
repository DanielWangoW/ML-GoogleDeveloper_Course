from __future__ import absolute_import, division, print_function, unicode_literals

#@title Import relevant modules.
import numpy as np
import pandas as pd 
import tensorflow as tf 
import pylab 
from tensorflow.keras import layers
from matplotlib import pyplot as plt 

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when outputting Numpy arrays.
np.set_printoptions(linewidth=200)

# Load both the training set and test set
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

#print(x_train[2917][10][16])
#plt.imshow(x_train[2917])
#pylab.show()

x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0
print(x_train_normalized[2900][12])

#@title Define the plotting function.
def plot_curve(epochs, list, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    
    plt.legend()
    plt.show()

print("Loading the plot_curve function.")


#@title Create a deep nerual net model.
def create_model(my_learning_rate):
    # Define the model to sequential.
    model = tf.keras.models.Sequential()
    # The features are stored in a two-dimensional 28x28 array.
    # Flatten that two-dimensional array into a one-dimensional 784-element array.
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=256,activation='relu'))
    # Define the second hidden layer.
    model.add(tf.keras.layers.Dense(units=128,activation='relu'))
    model.add(tf.keras.layers.Dense(units=128,activation='relu'))
    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate=0.2))
    # Define the output layer. The units parameter is set to 10 because
    # the model must choose among 10 possible output values(representing
    # the digits from 0 to 9, inclusive).
    #
    # Don't change this layer.s
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    # Construct the layers into a model that TensorFlow can execute.
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                                                    loss="sparse_categorical_crossentropy",
                                                    metrics=['accuracy'])
    return model

#@title Train a deep nerual net model
def train_model(model, train_features, train_label, epochs, batch_size=None, validation_split=0.1):
    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split)
    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch. 
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

# The following variables are the hyperparameters.
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2
# Establish the model's topography.
my_model = create_model(learning_rate)
# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train,
                           epochs, batch_size, validation_split)
# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)
# Evaluate against the test set.
print("\nEvaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)


