"""
Here is a quick start guide to generating a timeseries forecast using divinity
"""

"""
Import the divinity library,
 numpy to generate a synthetic dataset and matplotlib to visualise the forecast
"""
import divinity as dv
import matplotlib.pylab as plt
import numpy as np


"""
Setup a simple synthetic dataset with a smooth increasing
trend and a 20-d periodic feature with gaussian random noise
"""
epochs = 100
forecast = 20
t = np.arange(epochs + forecast)
y_test = 0.1*t + np.sin(2*np.pi/20*t) + np.random.randn(epochs + forecast)*0.5

"""
generate a 20 step forecast and
calculate 95pc confidence limits
"""
dfc = dv.divinity(forecast_length=forecast,
                  confidence_interval=95.)
dfc.fit(y_test[:epochs])
y_forecast = dfc.predict()


"""
visualise the results in matplotlib
"""
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t, y_test,label='True',color='k')
ax1.plot(t[epochs:],y_forecast,label='Forecast',color='b')
ax1.fill_between(t[epochs:], y_forecast - dfc.ystd,
                 y_forecast + dfc.ystd,
                 alpha = 0.3, color='b',label = None)
ax1.set_ylabel('ROI timeseries')
ax1.set_xlabel('Day')
plt.grid(b=None, which='major', axis='both')
plt.legend()
plt.tight_layout()
plt.savefig('../notebooks/test_divinity_forecast2.png')