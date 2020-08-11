import numpy as np
import pandas as pd
import divinity as dv
from scipy import signal
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
    t = np.arange(100)
    #t = np.linspace(0, 1, 500)
    #x = signal.sawtooth(2 * np.pi * 5 * t)

    #generate a sawtooth signal
    sawtooth = dv.gen_season(N=100,periods=np.arange(5,25,5),
                  sine_amplitudes= np.random.randn(4),
                  cosine_amplitudes= np.random.randn(4))
    features = sawtooth['features']
    x = sawtooth['target']

    plt.plot(t, x)
    plt.show()

