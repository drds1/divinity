import numpy as np
import pandas as pd
import divinity as dv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # generate synthetic features
    synthetic = dv.gen_fake(
        N=120,
        trend_order=4,
        trend_amplitudes=None,
        seasonal_periods=list(np.arange(2, 50)),
        seasonal_amplitudes_sine=None,
        seasonal_amplitudes_cosine=None,
    )
    features = synthetic["features"]

    # now test the class and the new auto assigner
    N = 120
    Ntest = 100
    t = np.arange(N)
    y_test = 0.1 * t + np.sin(2 * np.pi / 20 * t) + np.random.randn(N) * 0.5
    dfc = dv.divinity(
        forecast_length=N - Ntest, seasonal_periods=[20], confidence_interval=70.0
    )
    dfc.fit(y_test[:Ntest])
    y_forecast = dfc.predict()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t, y_test, label="True", color="k")
    ax1.plot(t[Ntest:], y_forecast, label="Forecast", color="b")
    ax1.fill_between(
        t[Ntest:],
        y_forecast - dfc.ystd,
        y_forecast + dfc.ystd,
        alpha=0.3,
        color="b",
        label=None,
    )
    ax1.set_ylabel("ROI timeseries")
    ax1.set_xlabel("Day")
    plt.grid(b=None, which="major", axis="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../notebooks/test_divinity_forecast.png")

    import sklearn.linear_model

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)

    trend_groups = [[f] for f in features.columns if "trend order" in f]
    greedy_select_trend = dv.greedy_select(
        features.iloc[:Ntest, :],
        y_test[:Ntest],
        features.iloc[Ntest:, :],
        y_test[Ntest:],
        model,
        feature_groups=trend_groups,
    )
    greedy_results_trend = greedy_select_trend.fit()

    # select the best features to include in the model following trend feature selection
    seasonal_features = [f for f in features.columns if "P=" in f]
    trend_seasonality_groups = dv.group_seasonal_features(seasonal_features)
    greedy_select_trend_seasonal = dv.greedy_select(
        features.iloc[:Ntest, :],
        y_test[:Ntest],
        features.iloc[Ntest:, :],
        y_test[Ntest:],
        model,
        feature_groups=trend_seasonality_groups,
        features_compulsory=greedy_results_trend["chosen_features"],
    )
    greedy_results_trend_seasonal = greedy_select_trend_seasonal.fit()
    all_chosen_features = greedy_results_trend_seasonal["chosen_features"]
