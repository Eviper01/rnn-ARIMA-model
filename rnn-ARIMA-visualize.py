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
X_train = x_values
# print(y_train)
#X_train = np.reshape(X_train, (len(X_train), 1, 1))
# Importing the Keras libraries and packages
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
regressor = keras.models.load_model("reg-6-11-2019-(8LSTM-firstprototype).h5")
print(sc[2].inverse_transform(y_values).flatten())
out = np.reshape(regressor.predict(x_values).flatten(),(-1,1))
dummy_data = np.empty((10,7))
out = sc[2].inverse_transform(out)
print(out.flatten())
actual = sc[2].inverse_transform(y_values).flatten().tolist()
out = out[:,0].tolist()
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
# ax = plt.gca()
plt.plot(out, color = 'blue', label = 'Predicted BTC Price')
plt.plot(actual, color = 'red', label = 'Actual BTC Price')
plt.title('BTC Price Prediction', fontsize=40)
x=out.index
#labels = df_test['time']
#plt.xticks(x, labels, rotation = 'vertical')
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(18)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=20)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()
