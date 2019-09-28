import numpy as np
import pandas as pd
import collections
import warnings

from scipy import stats

def pearsonr_ci(x,y,alpha=0.05):
   r, p = stats.pearsonr(x, y)
   r_z = np.arctanh(r)
   se = 1 / np.sqrt(x.size - 3)
   z = stats.norm.ppf(1 - alpha / 2)
   lo_z, hi_z = r_z - z * se, r_z + z * se
   lo, hi = np.tanh((lo_z, hi_z))
   return r, p, lo, hi