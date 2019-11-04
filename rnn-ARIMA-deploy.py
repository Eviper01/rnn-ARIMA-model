import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ['KERAS_BACKEND'] = 'theano'
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df = (pd.read_csv("db.csv")).iloc[:,0:8]
df = sc.fit_transform(df)
import keras
x_values = np.array(df[len(df)-50:len(df),2:7])
regressor = keras.models.load_model("reg.h5")
x_values = np.reshape(x_values,(1,50,5))
out = np.reshape(regressor.predict(x_values).flatten(),(-1,1))
dummy_data = np.empty((10,7))
out = np.append(out,dummy_data,axis=1)
out = sc.inverse_transform(out)
print(out[:,0])
out = out[:,0].tolist()

plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
# ax = plt.gca()
plt.plot(out, color = 'blue', label = 'Predicted BTC Price')
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
