import numpy as np
import pandas as pd
import sklearn.linear_model
import statsmodels.tsa.arima_model as arima_model
from functools import wraps
import inspect
import warnings
import matplotlib.pylab as plt

def cost_bic(ypred,ytrue,Ntrain, Nparms):
    return np.sum((ypred - ytrue)**2) + Ntrain*np.log(Nparms)


def group_seasonal_features(feature_columns):
    '''
    following the naming convention of seasonal features
    sin P=20, cos P=20....etc, group sin and cosine features together
    so these can be checked simultaneously by greedy fitting algorithms
    :param periods: list of input periods to group
    :return:
    '''
    #identify unique columns
    col1 = [f.replace('cos ','').replace('sin ','') for f in feature_columns]
    ucol1 = list(set(col1))
    return [['sin '+uc,'cos '+uc] for uc in ucol1]

class greedy_select:

    def __init__(self, X_train, y_train, X_test, y_test,
                 model = sklearn.linear_model.LinearRegression(fit_intercept=False),
                 feature_groups = None, features_compulsory= [],verbose = False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.features_compulsory = features_compulsory
        if feature_groups is None:
            self.unused_feature_groups = [[g] for g in X_train.columns]
        else:
            self.unused_feature_groups = feature_groups
        self.f_save = {'cumulative_features':[],'feature': [],
                       'test_cost': [], 'train_cost': []}
        self.used_features = []
        self.verbose = verbose


    def get_1iteration(self,used_features, unused_feature_groups):
        '''
        :return:
        '''
        train_cost_temp = []
        test_cost_temp = []
        temp_features_save = []
        Ntrain = len(self.y_train)
        Ntest = len(self.y_test)
        for i in range(len(unused_feature_groups)):
            feature_test = unused_feature_groups[i]
            temp_features = list(set(self.features_compulsory + used_features + feature_test))
            Nparms = len(temp_features)
            temp_features_save.append(temp_features)
            self.model.fit(self.X_train[temp_features].values, self.y_train)
            y_test_pred = self.model.predict(self.X_test[temp_features].values)
            y_train_pred = self.model.predict(self.X_train[temp_features].values)
            train_cost_temp.append(cost_bic(y_train_pred, self.y_train, Ntrain, Nparms))
            test_cost_temp.append(cost_bic(y_test_pred, self.y_test, Ntest, Nparms))
        idx_best = np.argmin(test_cost_temp)
        return {'best_feature_group':unused_feature_groups[idx_best],
                'best_train_cost':train_cost_temp[idx_best],
                'best_test_cost':test_cost_temp[idx_best],
                'best_idx':idx_best,
                'best_features_actual':temp_features_save[idx_best]}


    def fit(self):
        '''
        Iterate over all possible feature groups in the greedy algorithm
        :return:
        '''
        idx = 0
        while len(self.unused_feature_groups) > 0:
            results_1it = self.get_1iteration(self.used_features, self.unused_feature_groups)
            new_features = self.unused_feature_groups.pop(results_1it['best_idx'])
            self.used_features = self.used_features + new_features
            self.f_save['cumulative_features'].append(results_1it['best_features_actual'])
            self.f_save['feature'].append(new_features)
            self.f_save['test_cost'].append(results_1it['best_test_cost'])
            self.f_save['train_cost'].append(results_1it['best_train_cost'])
            idx += 1
            if self.verbose is True:
                plt.close()
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                t_all = np.arange(len(self.X_train) + len(self.X_test))
                ax1.plot(t_all,np.append(self.y_train,self.y_test))
                Ntrain = len(self.X_train)
                self.model.fit(self.X_train[results_1it['best_features_actual']].values,self.y_train)
                ax1.plot(t_all[:Ntrain],self.model.predict(self.X_train[results_1it['best_features_actual']]),label=','.join(results_1it['best_features_actual']))
                ax1.plot(t_all[Ntrain:], self.model.predict(self.X_test[results_1it['best_features_actual']]),
                         label='test')
                ax1.set_title('iteration '+str(idx))
                plt.legend()
                print(pd.DataFrame(self.f_save))
                plt.show()

        f_save = pd.DataFrame(self.f_save)
        chosen_features = list(f_save['cumulative_features'].values[:np.argmin(f_save['test_cost'].values) + 1])
        chosen_features = list(set([item for sublist in chosen_features for item in sublist]))
        return {'summary': f_save, 'chosen_features': chosen_features}


def gen_season(N,
               periods = [10,20],
               sine_amplitudes = [1,1],
               cosine_amplitudes = [0,0],
               remove_invalid_features = True):
    '''
    generate sinusoidal features
    :param N:
    :param periods:
    :param sine_amplitudes:
    :param cosine_amplitudes:
    :param remove_invalid_features: If the period is more than half the
    timeseries length, do not include the feature
    :return:
    '''
    features = {}
    t = np.arange(N)
    X = np.zeros(N)
    for idx in range(len(periods)):
        p = periods[idx]
        if remove_invalid_features is True:
            if p > N/2:
                warnings.warn("period "+str(p)+" too long to include for timeseries length. Ignoring this.")
                continue
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
             seasonal_amplitudes_cosine,
             normalize = '0to1'):
    '''
    generate synthetic timeseries and return input feature matrix
    If amplitude arguments supplied None, just calculate feature matrix
    and dont bother with the target
    :param N:
    :param trend_order:
    :param trend_amplitudes:
    :param seasonal_periods:
    :param seasonal_amplitudes_sine:
    :param seasonal_amplitudes_cosine:
    :param normalize:
    :return:
    '''
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

    #normalise features
    if normalize is not None:
        featuresstd = features_tot.std()
        colnorm = featuresstd[featuresstd > 0].index.tolist()
        if normalize == '0to1':
            features_tot[colnorm] = (features_tot[colnorm] - features_tot[colnorm].min())/features_tot[colnorm].max()
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
                 optimise_trend_season_features = True,
                 residual_model_kwargs = {'order':(2,0,2),'exog':None},
                 residual_model_fit_kwargs = {'trend':'nc', 'disp':0}):
        self.features = None
        self.input_features = None
        self._Ntot = None
        self._N = None
        self._y = None
        self._chosen_features = None
        self._greedy_results_trend_seasonal = None
        #pass

    def _prep_features(self,N):
        '''
        prepare the feature matrix given the stated trend_order
        seasonal_periods arguments
        N should be the length of the training data plus the length of the
        required forecast
        :return:
        '''
        #seasonality = gen_season(N, periods=self.seasonal_periods, sine_amplitudes=None, cosine_amplitudes=None)
        #trend = gen_trend(N, coef=self.trend_order, amplitudes=None)
        #self.features = pd.concat([trend['features'], seasonality['features']], axis=1)
        trend_seasonality = gen_fake(N, max(self.trend_order),
                 trend_amplitudes=None,
                 seasonal_periods = self.seasonal_periods,
                 seasonal_amplitudes_sine = None,
                 seasonal_amplitudes_cosine = None,
                 normalize='0to1')
        self.features = trend_seasonality['features']
        self.input_features = self.features.copy()

    def _auto_feature_select(self, features, y, training_fraction = 0.8, model = sklearn.linear_model.LinearRegression(fit_intercept=False)):
        '''
        use the above class to automate feature selection for trend and seasonal components
        :return:
        '''
        N = len(y)
        Ntest = int(training_fraction*N)
        trend_groups = [[f] for f in features.columns if 'trend order' in f]
        greedy_select_trend = greedy_select(features.iloc[:Ntest, :],
                                               y[:Ntest],
                                               features.iloc[Ntest:, :],
                                               y[Ntest:],
                                               model, feature_groups=trend_groups)
        greedy_results_trend = greedy_select_trend.fit()

        # select the best features to include in the model following trend feature selection
        seasonal_features = [f for f in features.columns if 'P=' in f]
        trend_seasonality_groups = group_seasonal_features(seasonal_features)
        greedy_select_trend_seasonal = greedy_select(features.iloc[:Ntest, :],
                                                        y[:Ntest],
                                                        features.iloc[Ntest:, :],
                                                        y[Ntest:],
                                                        model, feature_groups=trend_seasonality_groups,
                                                        features_compulsory=greedy_results_trend['chosen_features'])
        greedy_results_trend_seasonal = greedy_select_trend_seasonal.fit()
        all_chosen_features = greedy_results_trend_seasonal['chosen_features']
        self._chosen_features = all_chosen_features
        self._greedy_results_trend_seasonal = greedy_results_trend_seasonal

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
        if self.optimise_trend_season_features is True:
            self._auto_feature_select(self.features[:self._N],y)
            self.features = self.features[self._chosen_features]
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
        self._y = y
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

    def forecast(self, steps):
        '''
        extend the forecast (must already have used .fit)
        :return:
        '''
        self._Ntot = self._N + steps
        self.forecast_length = steps
        self.optimise_trend_season_features = False
        self._prep_features(self._N+steps)
        self.features = self.features[self._chosen_features]
        self.fit(self._y)
        return self.predict()




