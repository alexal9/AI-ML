# Part 1 - Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset.iloc[:, 1:2].values # we slice the columns to get numpy array of shape rows x 1

# normalization x => [0, 1] is preferred for RNN
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append( training_set_scaled[i-60:i, 0] )
    y_train.append( training_set_scaled[i, 0] )
X_train, y_train = np.array(X_train), np.array(y_train)
    
# Reshaping
X_train = np.reshape(X_train, X_train.shape + (1,))


# Part 2 - Building the RNN

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

regressor = Sequential()

# 4 hidden layers and output layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)


# Part 3 - Making the predictions and visualizing the results

# actual values
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")

# getting predicted values by combining original datasets and extract the 
# last len(dataset_test) + 60 values to generate test inputs
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(dataset_test) + 60):
    X_test.append( inputs[i-60:i, 0] )
X_test = np.array(X_test)
X_test = np.reshape(X_test, X_test.shape + (1,))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show() 