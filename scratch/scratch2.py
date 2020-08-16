import numpy as np
import pandas as pd
import divinity as dv
import matplotlib.pyplot as plt


if __name__ == '__main__':

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
    plt.show()
