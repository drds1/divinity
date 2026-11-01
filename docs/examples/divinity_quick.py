"""
Here is a quick start guide to generating a timeseries forecast using divinity
"""

"""
Import divinity and its synthetic test series. Each scenario has a training
history and the true future, so the forecast can be checked.
"""
import divinity as dv
from divinity import synthetic

case = synthetic.make("trend_seasonal")

"""
Fit the model and forecast 40 steps with a 95% prediction interval
"""
model = dv.Forecaster().fit(case.history)
result = model.predict(steps=40)
print(model.summary())
print(result.head())

"""
Plot the history, forecast and interval against the true future
"""
ax = model.plot(result, actual=case.future)
ax.figure.savefig("divinity_quick.png", dpi=110, bbox_inches="tight")
