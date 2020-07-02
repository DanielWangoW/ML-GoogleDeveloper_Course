#@title Imoport relevant modules
import numpy as np
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras import layers
from matplotlib import pyplot as plt 
import seaborn as sns 

# The following lines adjust the granularity of reporter
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print ("Imported modules")


# Load the dataset
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
# Shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))


#@title Convert raw values to their Z-scores
# Calculate the Z-scores of each column in training set
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std

# Calculate the Z-scores of each column in test set
test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std

print("Normalized the value.")


# Create an empty list that will eventually hold all created feature columns
feature_columns = []

# We scaled all the columns, including latitude and longitude, into their
# Z scores. So, instead of picking a resolution in degrees, we're going
# to use resolution_in_Zs.  A resolution_in_Zs of 1 corresponds to 
# a full standard deviation. 

# 3/10 of a standard deviation
resolution_in_Zs = 0.3

# Creat a bucket feature column for latitude
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])),
                                     int(max(train_df_norm['latitude'])),
                                     resolution_in_Zs))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)

# Creat a bucket feature column for longitude
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])),
                                      int(max(train_df_norm['longitude'])),
                                      resolution_in_Zs))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries)

# Creat a feature cross of latitude and longitude
latitude_x_longitude = tf.feature_column.crossed_column([latitude,longitude],hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# Repersent population as a floating-point value
population = tf.feature_column.numeric_column("population")
feature_columns.append(population)

# Convert the list of feature columns into a layer that will later be fed into the model
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


#@title Define the plotting function.
def plot_the_loss_curve(epochs,mse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max()*1.03])
    plt.show

print("Defined the plot_the_loss_curve function.")


#@title Define functions to creat and train a linear regression model.
# def create_model(my_learning_rate, feature_layer):
#     # Most simple tf.keras models are sequential
#     model = tf.keras.models.Sequential()
#     # Add the layer containing the feature columns to the model
#     model.add(feature_layer)
#     # Add one linear layer to the model to yield a simple linear regressor.
#     model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
#     # Construct the layers into a model that TensorFlow can execute
#     model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
#                   loss="mean_squared_error",
#                   metrics=[tf.keras.metrics.MeanSquaredError()])
#     return model

# def train_model(model, dataset, epochs, batch_size, label_name):
#     # Split the dataset into features and label.
#     features = {name:np.array(value) for name, value in dataset.items()}
#     label = np.array(features.pop(label_name))
#     history = model.fit(x=features, y=label, batch_size=batch_size,
#                         epochs=epochs, shuffle=True)
#     # Get details that will be useful for plotting the loss curve.
#     epochs = history.epoch
#     hist = pd.DataFrame(history.history)
#     rmse = hist["mean_squared_error"]

#     return epochs, rmse

# print("Defined the creat_model and train_model functions.")


#@title Define a deep neural net model
def create_model(my_learning_rate, feature_layer):
    # The most simple tf.kreas model is sequential.
    model = tf.keras.models.Sequential()
    # Add the layer containing the feature columns to the model.
    model.add(my_feature_layer)
    # Describe the topography of the model by calling the tf.kreas.layer.Dense method once for each layer.
    # We've specified the following arguments:
    # *units specifies the number of nodes in this layer
    # *activation specifies the activation function (Rectified Linear Unit)
    # *name is just a string that can be useful when debugging.

    # Define the first hidden layer with 20 nodes
    model.add(tf.keras.layers.Dense(units=10,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.04),
                                    name='Hidden1'))
    # Define the second hidden layer with 12 nodes
    model.add(tf.keras.layers.Dense(units=6,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.04),
                                    name='Hidden2'))
    # Define the output layer
    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model
#@title Define a train function for deep neural net model
def train_model(model, dataset, epochs, label_name, batch_size=None):
    # Split the dataset into features and label
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)
    # The list of epochs is stored separately from the rest of history
    epochs = history.epoch
    # To track the progression of training, gather a snapshot of the model's mean squared error
    # at each epoch.
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse


# The following variabes are the hyperparameter
learning_rate = 0.07
epochs = 140
batch_size = 1000

# Specify the label
label_name = "median_house_value"

# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set.S
#epochs, mse = train_model(my_model, train_df_norm, epochs, batch_size, label_name)

# Train the model on the normalized training set. We're passing the entire
# normalized training set, but the model will only use the features 
# defined by the feature_layer.
epochs, mse = train_model(my_model, train_df_norm, epochs, label_name, batch_size)
plot_the_loss_curve(epochs,mse)

test_features = {name:np.array(value) for name, value in test_df_norm.items()}
# Isolate the label
test_label = np.array(test_features.pop(label_name))
print("\n Evaluate the linear regression model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)


