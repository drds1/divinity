import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import statsmodels.tsa.arima_process as arima_process
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import keras

'''
scratch functions
'''
def generate_test_arima(number_of_epochs=250,
                        arparms=[0.75, -0.25],
                        maparms=[0.65, 0.35]):
    arp = np.array(arparms)
    map = np.array(maparms)
    ar = np.r_[1, -arp]  # add zero-lag and negate
    ma = np.r_[1, map]  # add zero-lag
    return arima_process.arma_generate_sample(ar, ma, number_of_epochs)

def generate_features_from_timeseries(y,nlags = 20):
    '''
    convert a 1D timeseries into a feature matrix of lags
    :param y:
    :param nlags:
    :return:
    '''
    yn = np.array(y)
    N = len(yn)
    X = np.zeros((N, nlags))
    for i in range(nlags):
        X[:, i] = np.roll(yn, i + 1)
    X = X[nlags:, :]
    yn = yn[nlags:]
    return X, yn


if __name__ == '__main__':
    np.random.seed(12345)
    nlags = 20
    y = generate_test_arima(number_of_epochs=1000)
    X, yn = generate_features_from_timeseries(y,nlags=nlags)

    #transform features
    Xscaler = MinMaxScaler()
    Xscaler.fit(X)
    Xt = Xscaler.transform(X)
    yscaler = MinMaxScaler()
    yscaler.fit(y.reshape(-1,1))
    yt = yscaler.transform(y.reshape(-1,1))

    #design rnn using keras
    model = keras.sequential()
    model.add(keras.layers.LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    #model.add(keras.layers.Dense(nlags))
    #model.add(keras.layers.RNN(return_sequences = True))
    #model.add(keras.layers.RNN(return_sequences=True))
    #model.add(keras.layers.RNN(return_sequences=True))
    #model.add(keras.layers.RNN(return_sequences=False))








