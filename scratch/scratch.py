import numpy as np
import pandas as pd
import divinity as dv
import matplotlib.pyplot as plt


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
    dfc = dv.divinity(forecast_length=N - Ntest,
                      seasonal_periods=[20],
                      confidence_interval=70.)
    dfc.fit(y_test[:Ntest])
    y_forecast = dfc.predict()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t, y_test,label='True',color='k')
    ax1.plot(t[Ntest:],y_forecast,label='Forecast',color='b')
    ax1.fill_between(t[Ntest:], y_forecast - dfc.ystd,
                     y_forecast + dfc.ystd,
                     alpha = 0.3, color='b',label = None)
    ax1.set_ylabel('ROI timeseries')
    ax1.set_xlabel('Day')
    plt.grid(b=None, which='major', axis='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../Doccumentation/test_divinity_forecast.png')

