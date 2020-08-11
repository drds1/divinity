import numpy as np
import pandas as pd
import divinity as dv

if __name__ == '__main__':
    #test the synthetic data generation

    synthetic = dv.gen_fake(N=100, trend_order = 2,
             trend_amplitudes = [0.1,0.1,0.1],
             seasonal_periods = [10.,33.],
             seasonal_amplitudes_sine = [0.1,0.05],
             seasonal_amplitudes_cosine = [0.1, 0.2])

    features, y = synthetic['features'], synthetic['target']

