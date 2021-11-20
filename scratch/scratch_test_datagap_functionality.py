import numpy as np
import divinity as dv
import matplotlib.pylab as plt

# set seed for repeatability
np.random.seed(123456)

# now test the class and the new auto assigner
N = 120
Ntest = 100
t = np.arange(N)
y_test = 0.1 * t + np.sin(2 * np.pi / 20 * t) + np.random.randn(N) * 0.5
dfc = dv.divinity(
    forecast_length=N - Ntest,
    seasonal_periods=list(np.arange(1, int(N / 2))),
    confidence_interval=70.0,
)

# define train dataset (before forecast horizon)
ytrain = y_test[:Ntest]
ttrain = t[:Ntest]

# simulate data gaps
idxsample = np.random.choice(np.arange(Ntest), int(Ntest / 4), replace=False)
idxsample = np.unique(np.sort(np.append(idxsample, [0, Ntest - 1])))
dfc.fit(ytrain[idxsample], tinput=ttrain[idxsample])
y_forecast = dfc.predict()

# Visualise results
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(ttrain[idxsample], ytrain[idxsample], s=2, color="r", label="Data")
ax1.axvline(Ntest, label="Forecast Horizon", color="k", ls="--")
ax1.set_xlabel("time")
ax1.set_ylabel("y")
ax1.plot(t, dfc._yres_all + dfc._trend_all, label="Model / Forecast")
ax1.legend()
plt.savefig("divinity_forecast.png", dpi=100)
