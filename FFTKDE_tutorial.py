from scipy.io import loadmat
import scipy.stats as Statz
import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale
import plotly.express as px
from scipy.stats import norm


from KDEpy import FFTKDE
from scipy.stats import norm


###------ 8. FFT-KDE  -------
### =========================
###

# Generate a distribution and draw 2**6 data points
dist = norm(loc = 0, scale = 1)
data = dist.rvs(2**6)

# Compute kernel density estimate on a grid using Silverman's rule for bw
x, y1 = FFTKDE(bw = "silverman").fit(data).evaluate(2**10)

# Compute a weighted estimate on the same grid, using verbose API
weights = np.arange(len(data)) + 1
estimator = FFTKDE(kernel = 'biweight', bw = 'silverman')
y2 = estimator.fit(data, weights = weights).evaluate(x)

plt.plot(x, y1, label = 'KDE estimate with defaults')
plt.plot(x, y2, label = 'KDE estimate with verbose API')
plt.plot(x, dist.pdf(x), label = 'True distribution')
plt.grid(True, ls = '--', zorder = -15); plt.legend()
