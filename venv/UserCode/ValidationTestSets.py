import numpy as np
import pandas as pd 
import tensorflow as tf 
from matplotlib import pyplot as plt 

pd.options.display.max_rows = 10 
pd.options.display.float_format = "{:.1f}".format

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

scale_factor = 1000.0

train_df["median_house_value"] /= scale_factor
test_df["median_house_value"] /= scale_factor

def build_model(my_learning_rate):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=1,input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model,df,feature,label,my_epochs,
                my_batch_size=None,my_validation_split=0.1):
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=my_batch_size,
                        epochs=my_epochs,
                        validation_split=my_validation_split)

    trained_weight = model.get_weights()[0] 
    trained_bias = model.get_weights()[1] 
    epochs = history.epoch

    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs,rmse,history.history

print("Defined the build_model and train_model functions")

def plot_the_loss_curve(epochs,mae_training,mae_validation):

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:],mae_training[1:],label="Training Loss")
    plt.plot(epochs[1:],mae_validation[1:],label="Validation Loss")
    plt.legend()

    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis,top_of_y_axis])
    plt.show()

print("Defined the plot_the_loss_curve funtion")

learning_rate = 0.08
epochs = 30
batch_size = 100

validation_split = 0.2

my_feature="median_income"
my_label="median_house_value"

my_model=None

shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

my_model = build_model(learning_rate)
epochs,rmse,history = train_model(my_model,shuffled_train_df,my_feature,
                                  my_label,epochs,batch_size,
                                  validation_split)
plot_the_loss_curve(epochs,history["root_mean_squared_error"],
                    history["val_root_mean_squared_error"])

x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x_test, y_test, batch_size=batch_size)