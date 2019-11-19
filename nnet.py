import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

X_train.shape

X_train[0, :]

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=32, input_shape=(13,), activation="relu"))
model.add(tf.keras.layers.Dense(units=32, activation="relu"))
model.add(tf.keras.layers.Dense(units=16, activation="relu"))
model.add(tf.keras.layers.Dense(units=8, activation="relu"))
model.add(tf.keras.layer.Dense(units=1, activation="linear"))

