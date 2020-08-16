import numpy as np
import pandas as pd
import divinity as dv
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    np.random.seed(123456)
    #now test the class and the new auto assigner
    N = 120
    Ntest = 100
    t = np.arange(N)
    y_test = 0.1*t + np.sin(2*np.pi/20*t) + np.random.randn(N)*0.5
    t1 = time.time()
    dfc = dv.divinity(forecast_length=N - Ntest,
                      seasonal_periods=list(np.arange(1,int(N/2))),
                      confidence_interval=70.)
    dfc.fit(y_test[:Ntest])
    y_forecast = dfc.predict()
    t2 = time.time()
    print('fit report')
    print('fit time...',t2 - t1)
    print('optimisation report...')
    print(str(len(dfc.input_features.columns))+' input features')
    print(str(len(dfc.features.columns))+' chosen features')
    print(dfc.features.columns)

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

    y_forecast_2 = dfc.forecast(steps= 50)
