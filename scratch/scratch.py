import numpy as np
import pandas as pd
import divinity as dv
from scipy import signal
import matplotlib.pyplot as plt
import sklearn.linear_model
import statsmodels.tsa.arima_model as arima_model
from functools import wraps
import inspect
import warnings

def initializer(func):
    """
    Automatically assigns the class input parameters.
    """
    names, varargs, keywords, defaults, \
    kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

class divinity:
    @initializer
    def __init__(self,
                 forecast_length,
                 seasonal_periods = [7,14.,28.,30.,90.,120.,182.,365.],
                 trend_order = [0,1],
                 trend_seasonal_model = sklearn.linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0),
                                                                     fit_intercept=False,
                                                                     normalize=False,
                                                                     scoring=None,
                                                                     cv=None,
                                                                     gcv_mode=None,
                                                                     store_cv_values=False),
                 residual_model = arima_model.ARIMA,
                 residual_model_kwargs = {'order':(2,0,2),'exog':None},
                 residual_model_fit_kwargs = {'trend':'nc', 'disp':0}):
        self.features = None
        self._Ntot = None
        self._N = None
        #pass

    def _prep_features(self,N):
        '''
        prepare the feature matrix given the stated trend_order
        seasonal_periods arguments
        N should be the length of the training data plus the length of the
        required forecast
        :return:
        '''
        seasonality = dv.gen_season(N, periods=self.seasonal_periods, sine_amplitudes=None, cosine_amplitudes=None)
        trend = dv.gen_trend(N, coef=self.trend_order, amplitudes=None)
        self.features = pd.concat([trend['features'], seasonality['features']], axis=1)

    def _fit_trend_season(self, y):
        '''
        fit the trend and seasonal components
        :param X:
        :param y:
        :return:
        '''
        if self._N is None:
            self._N = len(y)
        if self._Ntot is None:
            self._Ntot = self._N + self.forecast_length
        if self.features is None:
            self._prep_features(self._Ntot)
        self.trend_seasonal_model.fit(self.features[:self._N],y)
        #compute the model residuals to pass on
        # to the residual model
        self._y_model = self.trend_seasonal_model.predict(self.features)
        self._y_res = y - self._y_model[:self._N]

    def _fit_residual_model(self):
        '''
        fit the arima residual model to mop up
        any predictive power missed by the trend/
        seasonality
        :return:
        '''
        self._live_res_model = self.residual_model(self._y_res, **self.residual_model_kwargs)
        try:
            self._live_res_model_fit = self._live_res_model.fit()
            forecast = self._live_res_model_fit.forecast(steps = self.forecast_length)
            #forecast['forecast'], forecast['stderr'], forecast['conf_int']
            self._yres_pred_forecast = forecast['forecast']
        except:
            warnings.warn("Residual ARIMA model failure... Using only trend and seasonal components.")
            self._yres_pred_forecast = np.zeros(self.forecast_length)

    def fit(self,y):
        '''
        combine the above steps into a general fit function
        :param X:
        :param y:
        :return:
        '''
        #fit the trend and seasonal components
        self._fit_trend_season(y)

        #fit the residual arima model
        self._fit_residual_model()

    def predict(self):
        '''
        return the foreward forecast for Nsteps specified by the
        forecast_length input argument
        :return:
        '''
        return(self._yres_pred_forecast + self._y_model[self._N:] )











if __name__ == '__main__':
    #test the synthetic data generation

    synthetic = dv.gen_fake(N=100, trend_order = 2,
             trend_amplitudes = [0.1,0.1,0.1],
             seasonal_periods = [10.,33.],
             seasonal_amplitudes_sine = [0.1,0.05],
             seasonal_amplitudes_cosine = [0.1, 0.2])

    features, y = synthetic['features'], synthetic['target']


    #generate periodic noisy signal
    t = np.arange(200)
    #t = np.linspace(0, 1, 500)
    #x = signal.sawtooth(2 * np.pi * 5 * t)

    #generate a repeating signal
    # note the repeat length is the lowest common multiple of the
    # periods
    sawtooth = dv.gen_season(N=len(t),periods=np.arange(5,25,5),
                  sine_amplitudes= np.random.randn(4),
                  cosine_amplitudes= np.random.randn(4))
    features = sawtooth['features']
    x = sawtooth['target']
    xs = features.sum(axis=1)
    plt.plot(t, x)


    #now test the class and the new auto assigner
    N = 120
    Ntest = 100
    t = np.arange(N)
    y_test = 0.1*t + np.sin(2*np.pi/20*t) + np.random.randn(N)*0.5
    dfc = divinity(forecast_length=N - Ntest, seasonal_periods=[20])#list(np.arange(2,50))
    dfc.fit(y_test[:Ntest])
    y_forecast = dfc.predict()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t, y_test,label='true')
    ax1.plot(t[Ntest:],y_forecast,label='forecast')
    ax1.set_ylabel('ROI timeseries')
    ax1.set_xlabel('day')
    plt.legend()
    plt.savefig('test_divinity_forecast.png')

