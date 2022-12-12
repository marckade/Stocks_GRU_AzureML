import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def transformData():

    # load in dataset
    # looking at the stock history for AABA from 2006-2018
    # AABA is Altaba Inc. (formerly known as Yahoo! Inc.), and is an independent,
    # non-diversified, closed-end management investment company registered under the 1940 Act.
    df = pd.read_csv('/content/AABA_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
    training_set = df[:'2016'].iloc[:,1:2].values
    test_set = df['2017':].iloc[:,1:2].values

    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)

    # currently, takes 59 points and then the 60th point is predicted
    X_train = []
    y_train = []
    for i in range(60,2768):
        X_train.append(training_set_scaled[i-60:i,0])
        y_train.append(training_set_scaled[i,0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Add a dimension as keras expects 3D: (batch size, time steps, dimensionality)
    X_train = np.expand_dims(X_train, axis=-1)

    # Get train and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10)
    df_total = pd.concat((df["High"][:'2016'],df["High"]['2017':]),axis=0)

    # get values for the test set from df
    inputs = df_total[len(df_total)-len(test_set) - 60:].values
    inputs = inputs.reshape(-1,1)

    # use the same minmaxscalar to convert high values for the X_test set
    inputs  = sc.transform(inputs)

    # Preparing X_test 
    X_test = []
    for i in range(60,311):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, y_train, X_valid, y_valid, X_test