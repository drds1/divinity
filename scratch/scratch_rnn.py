import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import statsmodels.tsa.arima_process as arima_process
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import keras
from sklearn.model_selection import train_test_split

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

def fit_lstm(X, y, batch_size, nb_epoch, neurons):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = keras.Sequential()
    model.add(keras.layers.LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

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
    yscaler.fit(yn.reshape(-1,1))
    yt = yscaler.transform(yn.reshape(-1,1))[:,0]



    #model.add(keras.layers.Dense(nlags))
    #model.add(keras.layers.RNN(64, return_sequences = True))
    #model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.RNN(return_sequences=True))
    #model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.RNN(return_sequences=True))
    #model.add(keras.layers.RNN(return_sequences=False))
    #model.add(keras.layers.Dense(1,activation='relu'))
    #model.compile(loss='mean_squared_error', optimizer='adam')
    #model.summary()


    #split train test data

    Xtrain, Xtest, ytrain, ytest = train_test_split(Xt,
                                                    yt,
                                                    test_size=0.33,
                                                    random_state=42 )

    #reshape into format required for LSTM
    #https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    Xtrain_lstm = Xtrain[:,:,np.newaxis]
    Xtest_lstm = Xtest[:, :, np.newaxis]

    #design rnn using keras
    neurons = 4
    model = keras.Sequential()


    #model.predict(Xtrain)
    model.add(keras.layers.LSTM(64,input_shape=(20,1), activation='relu'))
    model.add(keras.layers.Dense(1,activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(Xtrain_lstm,ytrain)
    #



    '''
    batch_size = 1
    neurons = 4
    nb_epoch = 3000
    Xtrain1 = Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1])
    ytrain1 = np.array(ytrain)
    model = keras.Sequential()
    model.add(keras.layers.LSTM(neurons, batch_input_shape=(batch_size, Xtrain1.shape[1], Xtrain1.shape[2]), stateful=True))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(Xtrain1, ytrain1, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    #model = fit_lstm(Xtrain,ytrain, 1, 3000, 4)
    #model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    '''









