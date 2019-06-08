# -*- coding: utf-8 -*-
"""
Created on Sun May 19 22:18:51 2019

@author: Sattwik
"""

"""
1.0 : Import data
"""

from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

"""
2.0 : Model Architecture
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

model = Sequential([
        Dense(16, activation = 'relu', input_shape = (13,)),
        Dense(16, activation='relu'),
        Dense(1, activation= 'sigmoid')
        ])

print(model.summary())

"""
3.0 : Compile Model
"""
model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ["mse", "mae", "mape"])
"""
4.0 : Train Model
"""

history = model.fit(x_train, y_train, epochs = 20, batch_size = 128)


"""
5.0 : Evaluate the model on test data
"""

eval = model.evaluate(x_test, y_test, batch_size = 128)

"""
6.0 : Model performance vizualization
"""

plt.plot(history.history["mean_squared_error"])
plt.plot(history.history["mean_absolute_error"])
plt.plot(history.history["mean_absolute_percentage_error"])
plt.show()







