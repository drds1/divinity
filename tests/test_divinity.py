import pathlib
import os
import sys

# need to append package to path for unit tests
filePath = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(filePath, ".."))
import unittest
import numpy as np
import divinity as dv
import time


class Test_Divinity(unittest.TestCase):
    def test_Divinity(self):
        """

        :return:
        """
        finished = False
        np.random.seed(123456)
        # now test the class and the new auto assigner
        N = 120
        Ntest = 100
        t = np.arange(N)
        y_test = 0.1 * t + np.sin(2 * np.pi / 20 * t) + np.random.randn(N) * 0.5
        t1 = time.time()
        dfc = dv.divinity(
            forecast_length=N - Ntest,
            seasonal_periods=list(np.arange(1, int(N / 2))),
            confidence_interval=70.0,
        )
        dfc.fit(y_test[:Ntest])
        y_forecast = dfc.predict()
        t2 = time.time()
        print("fit report")
        print("fit time...", t2 - t1)
        print("optimisation report...")
        print(str(len(dfc.input_features.columns)) + " input features")
        print(str(len(dfc.features.columns)) + " chosen features")
        print(dfc.features.columns)
        finished = True
        self.assertEqual(finished, True)


if __name__ == "__main__":
    unittest.main()
