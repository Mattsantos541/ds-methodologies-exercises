import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")



#Write a function, select_kbest_chisquared() that takes X_train, y_train and k as input (X_train and y_train should not be scaled!) and returns a list of the top k features.
from sklearn.feature_selection import SelectKBest
def select_kbest_chisquared():
    f_selector = SelectKBest(f_regression, k=2)
    f_selector.fit(X_train, y_train)
    f_support = f_selector.get_support()
f_feature = X_train.loc[:,f_support].columns.tolist()

print(str(len(f_feature)), 'selected features')
print(f_feature)



#Write a function, select_kbest_freg() that takes X_train, y_train (scaled) and k as input and returns a list of the top k features.


#Write a function, ols_backware_elimination() that takes X_train and y_train (scaled) as input and returns selected features based on the ols backwards elimination method
import statsmodels.api as sm
def ols_backware_elimination():
    ols_model = sm.OLS(y_train, X_train)
    fit = ols_model.fit()
    fit.summary()

#Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns the coefficients for each feature, along with a plot of the features and their weights.
from sklearn.linear_model import LassoCV

reg = LassoCV()
reg.fit(X_train, y_train)
#Write 3 functions, the first computes the number of optimum features (n) using rfe, the second takes n as input and returns the top n features, and the third takes the list of the top n features as input and returns a new X_train and X_test dataframe with those top features , recursive_feature_elimination() that computes the optimum number of features (n) and returns the top n features.

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def optimum_features(n):
    model = LinearRegression
    rfe = RFE(model, 3)
    x_rfe = rfe.fit_transform(x_train, y_train)
    model.fit(x_rfe, y_train)
    return x_rfe.ranking_, rfe.support_


def top_features(n):


def list_ top_features()