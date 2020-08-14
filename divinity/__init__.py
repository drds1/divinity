import numpy as np
import pandas as pd
import sklearn.linear_model
import statsmodels.tsa.arima_model as arima_model
from functools import wraps
import inspect
import warnings

def gen_season(N,periods = [10,20],sine_amplitudes = [1,1],cosine_amplitudes = [0,0]):
    '''
    generate sinusoidal features
    :param N:
    :param periods:
    :param sine_amplitudes:
    :param cosine_amplitudes:
    :return:
    '''
    features = {}
    t = np.arange(N)
    X = np.zeros(N)
    for idx in range(len(periods)):
        p = periods[idx]
        if p == 0:
            warnings.warn("Zero wavelength entered in 'gen_season' 'period' argument. Ignoring this.")
            continue
        s = np.sin(2*np.pi/p * t)
        c = np.cos(2 * np.pi / p * t)
        if sine_amplitudes is not None:
            sa = sine_amplitudes[idx]
            X += sa * s
        if cosine_amplitudes is not None:
            ca = cosine_amplitudes[idx]
            X += ca * c
        features['sin P=' + str(p)] = s
        features['cos P=' + str(p)] = c
    return {'features':pd.DataFrame(features),'target':X}

def gen_trend(N, coef = [0,1,2], amplitudes = [1,0.5,0.2]):
    '''
    generate trend features
    :param N:
    :param coef:
    :param amplitudes:
    :return:
    '''
    features = {}
    t = np.arange(N)
    X = np.zeros(N)
    for idx in range(len(coef)):
        order = coef[idx]
        f = np.array(t**order,dtype='float')
        if amplitudes is not None:
            amp = amplitudes[idx]
            X+= amp*f
        features['trend order '+str(order)] = f
    return {'features':pd.DataFrame(features),'target':X}



def gen_fake(N, trend_order,
             trend_amplitudes,
             seasonal_periods,
             seasonal_amplitudes_sine,
             seasonal_amplitudes_cosine):
    t = np.arange(N)
    features = {}

    #calculate trend features
    trend = gen_trend(N,coef = list(range(trend_order + 1)),
                      amplitudes=trend_amplitudes)
    features_trend, y_trend = trend['features'], trend['target']

    #calculate sine seasonal features
    season = gen_season(N, periods=seasonal_periods,
                        sine_amplitudes=seasonal_amplitudes_sine,
                        cosine_amplitudes=seasonal_amplitudes_cosine)
    features_season, y_season = season['features'], season['target']

    #combine and output features and target
    features_tot = pd.concat([features_trend, features_season], axis=1)
    y_tot = y_trend + y_season
    return{'features':features_tot,'target':y_tot}




def get_error(yres, conf_limit = 95):
    ordered_y_res = np.sort(yres)
    N = len(ordered_y_res)
    idx_lo = int((100. - conf_limit) / 2 / 100 * N)
    idx_hi = int(100 - (100. - conf_limit) / 2 / 100 * N)
    return ordered_y_res[idx_hi] - ordered_y_res[idx_lo]  # *np.std(self._trend_pred_res)


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
                 confidence_interval = 95.0,
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
        seasonality = gen_season(N, periods=self.seasonal_periods, sine_amplitudes=None, cosine_amplitudes=None)
        trend = gen_trend(N, coef=self.trend_order, amplitudes=None)
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
        self._trend_all = self.trend_seasonal_model.predict(self.features)
        self._trend_pred_res = y - self._trend_all[:self._N]
        sig = get_error(self._trend_pred_res, conf_limit=self.confidence_interval)
        self._trend_forecast_err = np.ones(self.forecast_length)*sig#*np.std(self._trend_pred_res)

    def _fit_residual_model(self):
        '''
        fit the arima residual model to mop up
        any predictive power missed by the trend/
        seasonality
        :return:
        '''
        self._live_res_model = self.residual_model(self._trend_pred_res, **self.residual_model_kwargs)
        try:
            self._live_res_model_fit = self._live_res_model.fit()
            forecast = self._live_res_model_fit.predict(start=0, end=self._Ntot-1)
            self._yres_all = forecast
            self._yres_forecast = forecast[self._N:]
            sig = get_error(forecast[:self._N] - self._trend_pred_res, conf_limit=self.confidence_interval)
            self._yres_forecast_err = sig*np.ones(self.forecast_length)
            #self._trend_forecast_err

            #forecast = self._live_res_model_fit.forecast(steps = self.forecast_length)
            #forecast['forecast'], forecast['stderr'], forecast['conf_int']
            #self._yres_forecast = forecast['forecast']
            #self._yres_pred_std = forecast['stderr']
        except:
            warnings.warn("Residual ARIMA model failure... Using only trend and seasonal components.")
            self._yres_forecast = np.zeros(self.forecast_length)
            self._yres_forecast_err = np.zeros(self.forecast_length)

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
        forecast_length input argument.
        :return:
        '''
        self.ypred = self._yres_forecast + self._trend_all[self._N:]
        self.ystd = np.sqrt(self._yres_forecast_err**2 + self._trend_forecast_err**2)/2
        return self.ypred


