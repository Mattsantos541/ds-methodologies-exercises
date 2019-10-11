##Evaluate
# prepare environment
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
import pydataset
from pydataset import data
#import warnings
#warnings.filterwarnings('ignore')


#from viz import evaluation_example1, evaluation_example2, evaluation_example3, evaluation_example4, evaluation_example5


#Load the tips dataset from either pydataset or seaborn.

tips= data('tips')
tips.head()
bill= tips[['total_bill', 'tip']]
bill.head()