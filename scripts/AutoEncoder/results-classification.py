import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

###SCRIPT DESCRIPTION###
# This script emulates the process of anomaly detection implemented in the web application.
###SCRIPT INPUT###
# Unlike other results scripts, this script directly emulates the process of anomaly detection in the web application.
# As such, this script can be directly provided with the input data of data-generation.py, without first classifying it
# through the webapp.
###SCRIPT OUTPUT###
# This script provides textual output calculating precision and recall if the threshold for anomaly classification is
# set to the 95th-100th percentile classification error of the training set respectively.
###SCRIPT BEGIN####

input_file_train = input("Path to training file: ")
input_file_test = input("Path to test file: ")

#import training, test csv through pandas
train = pd.read_csv(input_file_train)
test = pd.read_csv(input_file_test)

#fit x, y
encoder = tf.keras.preprocessing.text.Tokenizer(char_level=True)
encoder.fit_on_texts(train["Date"])
x = encoder.texts_to_sequences(train["Date"])
max_length = max(map(len, x))
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
x = tf.keras.utils.normalize(x)
x = np.array(x)
#store the input dimension
input_dim = x.shape[1]

#create keras model equivalent to WebApp implementation
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])


#compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse',  metrics=['acc'])

#train the model equivalent to WebApp implementation
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.fit(x, x, epochs=500, batch_size=4, callbacks=[callback])

#fit x (only class 0 data)
train_predictions = model.predict(x)
mse_train = np.mean(np.power(x - train_predictions, 2), axis=1)
print("Max MSE:" + train.iloc[np.argmax(mse_train, axis=0)]["Date"])

# fit x, y from test dataset
x = encoder.texts_to_sequences(test["Date"])
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=input_dim)
x = tf.keras.utils.normalize(x)
x = np.array(x)
y = np.array(test["true_label"])

#run predictions and calculate mean square error
test_x_predictions = model.predict(x)
mse = np.mean(np.power(x - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y})

#see how the classification error changes for different thresholds
for i in range(95,101):
    msemax = np.percentile(mse_train, i, axis=0)
    error_df['Predicted_class'] = np.where(error_df['Reconstruction_error'] > msemax, 1, 0)
    recall = (error_df[(error_df['Predicted_class'] == 1) & (error_df['True_class'] == 1)].shape[0] /
                error_df[(error_df['True_class'] == 1)].shape[0])
    precision = (error_df[(error_df['Predicted_class'] == 1) & (error_df['True_class'] == 1)].shape[0] /
                error_df[(error_df['Predicted_class'] == 1)].shape[0])
    print("i = " + str(i))
    print("Class 1 recall: " + str(recall))
    print("Class 1 precision: " + str(precision))
    print()