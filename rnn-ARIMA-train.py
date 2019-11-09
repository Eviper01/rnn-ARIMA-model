import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import requests
from lxml import html
import math
from sklearn.preprocessing import MinMaxScaler
sc = {}
df = (pd.read_csv("db.csv")).iloc[:,0:8]
print(df)
df = np.array(df)
for  (i,k) in enumerate(df.T):
    sc[i] = MinMaxScaler()

    k = np.reshape(k,(-1,1))

    k = sc[i].fit_transform(k)
    k = np.squeeze(k)
    print(k)
    df[:,i] = k

# fit transform the training data before it is sepreated
# df = sc.fit_transform(df)
print(df)
# print(df)
x_values = []
y_values = []
for i in range(int((math.floor(len(df)/60)*60)/60)):
    #input matrix
    x_values.append(df[60*i:60*i+50,2:7])
    #output matrix
    y_values.append(df[(i+1)*60 - 10:(i+1)*60,2])
print("db organised")
x_values = np.stack(x_values)

print("YEET")
y_values = np.squeeze(np.stack(y_values))
print(y_values)
# print(x_values)
X_train = x_values
y_train = y_values
print(X_train)
# print(y_train)
#X_train = np.reshape(X_train, (len(X_train), 1, 1))
# Importing the Keras libraries and packages
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
regressor = keras.models.load_model("reg.h5")
regressor.fit(X_train, y_train, batch_size = 3, epochs = 50)
regressor.save("reg.h5")
