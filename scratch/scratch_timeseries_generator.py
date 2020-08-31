"""
use keras TimeseriesGenerator to format input timeseries
for lstm forecasting
https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
"""

import keras
import numpy as np
import scratch_rnn as utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# generate synthetic data
data = utils.generate_test_arima(number_of_epochs=1000)
target = np.array(data)


n_z = 1
n_features = 4
batch_size = 8
data = data.reshape((len(data), n_z))


# split into train / test samples
split_frac = 0.7
split_idx = int(split_frac * len(data))
data_train = data[:split_idx]
data_test = data[split_idx:]

# normalise using min/max scaler
Xscaler = MinMaxScaler()
Xscaler.fit(data_train)
data_train_normed = Xscaler.transform(data_train)
data_test_normed = Xscaler.transform(data_test)

# define generator
generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data_train_normed, data_train_normed, length=n_features, batch_size=batch_size
)


# define model
model = keras.Sequential()
model.add(keras.layers.LSTM(100, activation="relu", input_shape=(n_features, n_z)))
model.add(keras.layers.Dense(1))
model.compile(optimizer="adam", loss="mse")
# fit model
model.fit_generator(generator, steps_per_epoch=1, epochs=500, verbose=True)

# get the 1-step predictions
test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data_test_normed, data_test_normed, length=n_features, batch_size=len(data_test)
)
X_test = test_generator[0][0]
yhat = model.predict(X_test, verbose=0)
print(yhat)


# evaluate predictions


# convert to keras feature / labeled required format
data_gen = keras.preprocessing.sequence.TimeseriesGenerator(
    data, target, length=4, sampling_rate=1, batch_size=len(target)
)
X, y = data_gen[0]
