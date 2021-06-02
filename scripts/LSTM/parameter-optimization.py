import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import talos
from talos.utils import hidden_layers

###SCRIPT DESCRIPTION###
# This script uses Talos to optimize parameters of an LSTM neural network.
###SCRIPT INPUT###
# This script should be provided with a test.csv and train.csv file containing training and test data for the LSTM
# The output of data-generation.py can be used as input for this script
###SCRIPT OUTPUT###
# This script will generate a single .csv file containing data about the executed experiments
###SCRIPT CONFIGURATION###
# this defines the tested hyperparameters
p = {
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.005, 0.0075, 0.01],
    'lstm_size': [8,16,32],
    'hidden_size': [2,4,8,16]
    }
# The fraction of permutations that is actually evaluated. Random search can find a similarly proficient model as grid search
permutation_fraction = 0.5
###SCRIPT BEGIN####

train_file = input("Path to training file: ")
test_file = input("Path to test file: ")
output_file = input("Path to output file: ")

#read train and tes files
data_test = pd.read_csv(test_file)
data_train = pd.read_csv(train_file)


#fit x and y of training and test
encoder = tf.keras.preprocessing.text.Tokenizer(char_level=True)
encoder.fit_on_texts(data_train["Date"])
x_train = encoder.texts_to_sequences(data_train["Date"])
max_length = max(map(len, x_train))
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_train = np.array(x_train)
x_test = encoder.texts_to_sequences(data_test["Date"])
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
x_test = np.array(x_test)

y_train = np.array(to_categorical(data_train["true_label"]))
y_int = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_int), y_int)
d_weights = dict(enumerate(class_weights))
y_test = np.array(to_categorical(data_test["true_label"]))

#create keras model
def tf_model(x, y, xval, yval, params):
  model = tf.keras.Sequential([
      #embedding layer which maps to a 32 bit onehot vector
      tf.keras.layers.Embedding
      (
          len(encoder.index_word) + 1,
          32
      ),
      #bidirectional LSTM layer
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['lstm_size'], dropout=params['dropout'])),
      #optional dense layer with dropout
      tf.keras.layers.Dropout(params['dropout']),
      tf.keras.layers.Dense(params['hidden_size']),
      #output layer
      tf.keras.layers.Dense(8, activation='softmax')
  ])

  #compile the model
  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                metrics=['acc'])

  #train the model against validation with an early stopper if no improvement is seen to validation accuracy for 10 iterations
  out = model.fit(x, y, epochs=100, batch_size=8, validation_data=(xval, yval), class_weight=d_weights, shuffle=True,
    callbacks=[talos.utils.early_stopper(100, monitor='val_loss', patience=5)])
  return out, model

#run a talos scan
m = talos.Scan(x_train, y_train, x_val=x_test, y_val=y_test, model=tf_model, params=p, experiment_name='LSTM_nohidden', random_method='quantum', fraction_limit=permutation_fraction)

#save the resulting dataframe to file
print(m.details)
m.data.to_csv(output_file)