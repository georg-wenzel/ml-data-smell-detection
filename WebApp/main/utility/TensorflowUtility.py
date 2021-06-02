import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Utility functions for TensorFlow

#return an in-memory version of the given .csv file (through pandas)
def get_csv(file):
    return pd.read_csv(file)

#return an encoder fitted on the given pandas x column
def get_encoder(x):
    #fit encoder to the x column
    encoder = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    encoder.fit_on_texts(x)
    return encoder

#encode a column of x data (by mapping to sequences and padding)
def encode_x(encoder, x):
    enc_x = encoder.texts_to_sequences(x)
    max_length = max(map(len, enc_x))
    enc_x = tf.keras.preprocessing.sequence.pad_sequences(enc_x, maxlen=max_length)
    enc_x = np.array(enc_x)
    return enc_x

#encode x to a fixed predetermined length. if none is given, encode normally and return the length
def encode_x_fixed_length(encoder, x, length=None):
    enc_x = encoder.texts_to_sequences(x)
    if not length: max_length = max(map(len, enc_x))
    else: max_length = length
    enc_x = tf.keras.preprocessing.sequence.pad_sequences(enc_x, maxlen=max_length)
    enc_x = tf.keras.utils.normalize(enc_x)
    enc_x = np.array(enc_x)
    return enc_x, max_length

#encode a column of y data (by converting to a one-hot representation in a numpy array)
def encode_y(y):
    return np.array(to_categorical(y))

def get_class_weights(encoded_y):
    y_int = np.argmax(encoded_y, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_int), y_int)
    d_weights = dict(enumerate(class_weights))
    return d_weights

#train a model on given encoded x and y, encoder, and class weights
def train_model(encoder, enc_x, enc_y, weights):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding
        (
            len(encoder.index_word) + 1,
            32
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, dropout=0.15)),
        tf.keras.layers.Dense(enc_y.shape[1], activation='softmax')
    ])

    #compile the model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        metrics=['accuracy'])

    #train the model with earlystopping when loss does not improve for 5 epochs
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.fit(enc_x, enc_y, epochs=100, batch_size=4, class_weight=weights, shuffle=True, callbacks=[callback])

    return model

#train an autoencoder on given x
def train_anomaly_model(enc_x, input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])

    #compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse',  metrics=['acc'])

    #train the model with early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)#train the model
    model.fit(enc_x, enc_x, epochs=700, batch_size=4, callbacks=[callback])

    return model