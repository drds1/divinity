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
import matplotlib.pylab as plt


class lstm_mod:
    def __init__(self, n_z=1, n_features=4, batch_size=8):
        self._n_z = n_z
        self._n_features = n_features
        self._batch_size = batch_size
        self.Xscaler = None
        pass

    def _transform_ts(self, X):
        """
        prepare timeseries by
        :return:
        """
        # normalise using min/max scaler
        if self.Xscaler is None:
            self.Xscaler = MinMaxScaler()
            self.Xscaler.fit(X)
        return self.Xscaler.transform(X)

    def _prep_model(self):
        """

        :return:
        """
        # define model
        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(
                100, activation="relu", input_shape=(self._n_features, self._n_z)
            )
        )
        model.add(keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X, steps_per_epoch=1, epochs=500, verbose=True):
        """
        use the timeseries generator to fit the lstm nn
        :return:
        """
        Xt = self._transform_ts(X)

        # define generator
        generator = keras.preprocessing.sequence.TimeseriesGenerator(
            Xt, Xt, length=self._n_features, batch_size=self._batch_size
        )

        # prepare the model
        self.model = self._prep_model()

        # fit model
        self.model.fit_generator(
            generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose
        )

    def predict(self, X, nsteps=1):
        """
        return a one step prediction
        :param X:
        :return:
        """
        # transform
        Xt = self.Xscaler.transform(X)

        # get the 1-step predictions
        test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
            Xt, Xt, length=self._n_features, batch_size=len(Xt)
        )
        X_test = test_generator[0][0]

        # evaluate the n-step predictions recursively
        all_predictions = np.zeros((len(X_test), nsteps))
        for idx in range(nsteps):
            pred = self.model.predict(X_test, verbose=0)
            X_test = np.roll(X_test, -1, axis=1)
            X_test[:, -1, :] = pred
            all_predictions[:, idx] = pred[:, 0]

        # return the inverse transform
        return self.Xscaler.inverse_transform(all_predictions)

    @staticmethod
    def correlation_plots(
        testdata,
        displaysteps=[1, 3, 5],
        file_correlation_report="correlation_tests.png",
    ):
        p7 = testdata
        fig = plt.figure()
        npred = len(p7)
        steptests = [i for i in displaysteps if i - 1 < np.shape(p7)[1]]
        true = data_test[-npred:, 0]
        idx = 1
        for step in steptests:
            pred = p7[:, step - 1]
            r2 = np.corrcoef(true, pred)[0, 1]
            var = np.sum((pred - true) ** 2) / npred
            rms = np.sqrt(var)
            ax1 = fig.add_subplot(2, len(steptests), idx)
            ax1.scatter(true, pred)
            ax1.set_xlabel("true")
            ax1.set_ylabel("predicted")
            xlim = list(ax1.get_xlim())
            ax1.plot(xlim, xlim, ls="--", color="k", label=None)
            ax1.set_xlim(xlim)
            ax1.set_ylim(xlim)
            ax1.set_title(str(step) + "-step: r2=" + str(np.round(r2, 2)))
            idx += 1
        ax1 = fig.add_subplot(2, 1, 2)
        ax1.set_xlabel("step")
        ax1.set_ylabel("r2")
        ax1.set_title("Correlation vs step-size")
        r2save = []
        stepsave = []
        for i in range(np.shape(p7)[1]):
            stepsave.append(i + 1)
            r2save.append(np.corrcoef(true, p7[:, i])[0, 1])
        ax1.plot(stepsave, r2save)
        plt.tight_layout()
        plt.savefig(file_correlation_report)


if __name__ == "__main__":
    """
    Manually generate synthetic data and build LSTM network to
    fit
    """

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
    model.fit_generator(generator, steps_per_epoch=1, epochs=50, verbose=True)

    # get the 1-step predictions
    test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
        data_test_normed, data_test_normed, length=n_features, batch_size=len(data_test)
    )
    X_test = test_generator[0][0]
    yhat = model.predict(X_test, verbose=0)
    print(yhat)

    # convert to keras feature / labeled required format
    data_gen = keras.preprocessing.sequence.TimeseriesGenerator(
        data, target, length=4, sampling_rate=1, batch_size=len(target)
    )
    X, y = data_gen[0]

    """
    Now test the lstm_mod class
    """
    lmod = lstm_mod()
    lmod.fit(data_train, steps_per_epoch=1, epochs=39, verbose=True)
    p1 = lmod.predict(data_test, nsteps=1)
    p7 = lmod.predict(data_test, nsteps=7)
    # generate correlation plots from test predictions
    lmod.correlation_plots(
        p7, displaysteps=[1, 5, 7], file_correlation_report="correlation_tests.png"
    )
