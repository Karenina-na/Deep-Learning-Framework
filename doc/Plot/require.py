# 导入依赖
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings(action='once')
plt.rcParams['figure.facecolor'] = 'white'

large = 22
medium = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': medium,
          'figure.figsize': (16, 10),
          'axes.labelsize': medium,
          'xtick.labelsize': medium,
          'ytick.labelsize': medium,
          'figure.titlesize': large}
plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')
sns.set_style("white")

print(np.__version__)
print(pd.__version__)
print(mpl.__version__)
print(sns.__version__)