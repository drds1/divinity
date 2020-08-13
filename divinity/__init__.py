import numpy as np
import pandas as pd
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


