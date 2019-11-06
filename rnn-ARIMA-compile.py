# First step, import libraries and then dataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import requests
from lxml import html
import math
from sklearn.preprocessing import MinMaxScaler
sc = {}
# def get_data():
#     print("Fetching Data")
#     url = "https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=USDT&e=BINANCE&aggregate=1&limit=2000&extraParams=PercepertronTrading&api_key=1a95b1abb57132503433dd9be659f9bc7c9c2f7a27f1cf78c0ceb89e59ee6e61"
#     page = requests.get(url)
#     tree = html.fromstring(page.content)
#     print("Got It")
#     return (json.loads(tree.text)["Data"])["Data"]
# df = pd.DataFrame(get_data())
# #print(df.keys())
# group = df.groupby('time')
# Real_Price = group['close'].mean()
# # # split data
# #number of days the model does not train on at the end?
# prediction_days = 100
# df_train= Real_Price[:len(Real_Price)-prediction_days]
# df_test= Real_Price[len(Real_Price)-prediction_days:]
# # # Data preprocess
# ## NEW PROCESSING GET 10 Last DATA POINTS IN A MATRIX --> Predict 10 next prices
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer --
regressor.add(LSTM(units = 8, activation = 'sigmoid', input_shape = (50, 5)))

# Adding the output layer
regressor.add(Dense(units = 10))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 3, epochs = 10)

# # Making the predictions
# test_set = df_test.values
# inputs = np.reshape(test_set, (len(test_set), 1))
# inputs = sc.transform(inputs)
# inputs = np.reshape(inputs, (len(inputs), 1, 1))
# predicted_BTC_price = regressor.predict(inputs)
# predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
# # Visualising the results
# plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
# ax = plt.gca()
# plt.plot(test_set, color = 'red', label = 'Real BTC Price')
# plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
# plt.title('BTC Price Prediction', fontsize=40)
# df_test = df_test.reset_index()
# x=df_test.index
# #labels = df_test['time']
# #plt.xticks(x, labels, rotation = 'vertical')
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(18)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(18)
# plt.xlabel('Time', fontsize=20)
# plt.ylabel('BTC Price(USD)', fontsize=40)
# plt.legend(loc=2, prop={'size': 25})
regressor.save("reg.h5")
# plt.show()
