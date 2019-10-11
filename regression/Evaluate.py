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
import pandas as pd


pip install viz
#import warnings
#warnings.filterwarnings('ignore')

import viz
from viz import evaluation_example1, evaluation_example2, evaluation_example3, evaluation_example4, evaluation_example5


#Load the tips dataset from either pydataset or seaborn.

tips= data('tips')
tips.head()

#Fit a linear regression model (ordinary least squares) and compute yhat, 
#predictions of tip using total_bill. You may follow these steps to do that:
bill= tips[['total_bill', 'tip']]
bill.head()

bill= pd.DataFrame(bill)
bill

x= bill['total_bill'])
y= bill['tip']

x= pd.DataFrame(x)
y= pd.DataFrame(y)
bill.head(3)
type(y)
len(x)
len(y)
type(x)
#import the method from statsmodels: from statsmodels.formula.api import ols
from statsmodels.formula.api import ols

ols_model = ols('y ~ x', data=bill).fit()

bill['yhat'] = ols_model.predict(x)


#fit the model to your data, where x = total_bill and y = tip: 
# regr = ols('y ~ x', data=df).fit()




regr = ols('y ~ x', data=bill).fit()

#bill['yhat'] = regr.predict(bill.x)
bill["yhat"] = ols_model.predict(pd.DataFrame(x))
#compute yhat, the predictions of tip using total_bill: df['yhat'] = regr.predict(df.x)

bill.head()

##Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, 
# and the dataframe as input and returns a residual plot. (hint: seaborn has an easy way to do this!)

bill['residual'] = bill['yhat'] - bill['y']
