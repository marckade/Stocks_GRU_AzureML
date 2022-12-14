import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def trainModel(X_train, y_train, X_valid, y_valid, X_test):

    model = keras.models.Sequential([
              keras.layers.GRU(50, return_sequences=True, input_shape=[None,1]), 
              keras.layers.Dropout(0.2),
              keras.layers.GRU(50, return_sequences=True),
              keras.layers.Dropout(0.2),
              keras.layers.GRU(50, return_sequences=True),
              keras.layers.Dropout(0.2),
              keras.layers.GRU(50),
              keras.layers.Dense(1)
    ])


    GRUModel = model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.fit(X_train, y_train, epochs=1, 
                    validation_data = (X_valid, y_valid), 
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)])

    predicted_stock_price = model.predict(X_test)
    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    return model, predicted_stock_price