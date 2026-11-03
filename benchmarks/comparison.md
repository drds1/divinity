# Comparison on M4

100 series per frequency sampled with seed 0. Lower is better for MASE, sMAPE and MSIS, and coverage should be close to 95%. Best in each column in bold.

### Yearly (h=6, m=1), 100 series

| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |
|---|---:|---:|---:|---:|---:|---:|
| Divinity | 3.390 | 15.07 | 75.3% | 42.254 | 0.68 | 0 |
| Prophet | 4.091 | 17.11 | 38.5% | 86.739 | 0.40 | 0 |
| AutoARIMA | 3.589 | 14.29 | 67.7% | 52.680 | 0.27 | 0 |
| AutoETS | **2.953** | **12.19** | 82.5% | **32.410** | 0.01 | 0 |
| SeasonalNaive | 4.197 | 15.90 | 66.2% | 57.799 | 0.00 | 0 |
| Naive | 4.197 | 15.90 | 66.2% | 57.799 | 0.00 | 0 |

### Quarterly (h=8, m=4), 100 series

| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |
|---|---:|---:|---:|---:|---:|---:|
| Divinity | 1.260 | 11.79 | 86.2% | 13.310 | 0.65 | 0 |
| Prophet | 2.003 | 19.25 | 60.9% | 32.942 | 0.51 | 0 |
| AutoARIMA | 1.199 | **10.76** | 87.6% | 11.394 | 0.84 | 0 |
| AutoETS | **1.125** | 10.87 | 94.2% | **7.998** | 0.15 | 0 |
| SeasonalNaive | 1.514 | 13.22 | 90.4% | 12.089 | 0.00 | 0 |
| Naive | 1.314 | 12.65 | 87.2% | 11.107 | 0.00 | 0 |

### Monthly (h=18, m=12), 100 series

| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |
|---|---:|---:|---:|---:|---:|---:|
| Divinity | 1.026 | 14.93 | 89.9% | 9.005 | 1.49 | 0 |
| Prophet | 1.194 | 17.36 | 71.2% | 16.954 | 0.21 | 0 |
| AutoARIMA | **0.942** | 13.54 | 89.9% | 8.052 | 2.42 | 0 |
| AutoETS | 0.957 | **12.58** | 89.9% | **7.678** | 0.42 | 0 |
| SeasonalNaive | 1.347 | 15.85 | 90.8% | 11.772 | 0.00 | 0 |
| Naive | 1.227 | 14.05 | 89.4% | 14.132 | 0.00 | 0 |

### Weekly (h=13, m=1), 100 series

| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |
|---|---:|---:|---:|---:|---:|---:|
| Divinity | 2.379 | 9.26 | 95.1% | 16.317 | 2.13 | 0 |
| Prophet | 6.247 | 20.64 | 73.2% | 75.429 | 0.56 | 0 |
| AutoARIMA | **2.275** | 8.54 | 95.8% | **15.319** | 1.15 | 0 |
| AutoETS | 2.322 | **8.48** | 97.7% | 19.074 | 0.14 | 0 |
| SeasonalNaive | 2.489 | 9.18 | 96.1% | 18.504 | 0.02 | 0 |
| Naive | 2.489 | 9.18 | 96.1% | 18.504 | 0.02 | 0 |

### Daily (h=14, m=1), 100 series

| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |
|---|---:|---:|---:|---:|---:|---:|
| Divinity | **3.644** | **2.69** | 92.3% | **34.313** | 3.16 | 0 |
| Prophet | 14.117 | 11.00 | 58.9% | 198.205 | 1.07 | 0 |
| AutoARIMA | 3.739 | 2.75 | 91.4% | 37.437 | 0.69 | 0 |
| AutoETS | 3.752 | 2.71 | 94.6% | 56.934 | 0.20 | 0 |
| SeasonalNaive | 3.890 | 2.90 | 91.0% | 39.267 | 0.00 | 0 |
| Naive | 3.890 | 2.90 | 91.0% | 39.267 | 0.00 | 0 |

### Hourly (h=48, m=24), 100 series

| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |
|---|---:|---:|---:|---:|---:|---:|
| Divinity | **0.924** | 14.93 | 96.6% | **7.449** | 14.48 | 0 |
| Prophet | 1.672 | 18.55 | 81.1% | 16.940 | 0.31 | 0 |
| AutoARIMA | 0.951 | **13.88** | 95.1% | 7.790 | 97.68 | 0 |
| AutoETS | 1.643 | 17.47 | 81.8% | 18.700 | 1.85 | 0 |
| SeasonalNaive | 1.193 | 14.09 | 96.0% | 9.142 | 0.00 | 0 |
| Naive | 10.387 | 42.86 | 94.3% | 62.870 | 0.00 | 0 |

### All, 600 series

| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |
|---|---:|---:|---:|---:|---:|---:|
| Divinity | **2.104** | 11.44 | 89.2% | **20.441** | 1.39 | 0 |
| Prophet | 4.887 | 17.32 | 64.0% | 71.202 | 0.38 | 0 |
| AutoARIMA | 2.116 | **10.63** | 87.9% | 22.112 | 1.18 | 0 |
| AutoETS | 2.125 | 10.71 | 90.1% | 23.799 | 0.20 | 0 |
| SeasonalNaive | 2.438 | 11.86 | 88.4% | 24.762 | 0.00 | 0 |
| Naive | 3.918 | 16.26 | 87.4% | 33.946 | 0.00 | 0 |
