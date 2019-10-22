from env import host, user, password
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
url = f'mysql+pymysql://{user}:{password}@{host}/iris_db'
###Wrangle
def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def wrangle_iris():
    df= pd.read_sql("""SELECT * FROM measurements m JOIN species s on s.species_id = m.species_id;"""
    ,url)
    
    return df
    
    


import pandas as pd
from env import host, user, password

def get_db_url(username, hostname, password, db_name):
    return f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}'







def plot_residuals(x, y):
    '''
    Plots the residuals of a model that uses x to predict y. Note that we don't
    need to make any predictions ourselves here, seaborn will create the model
    and predictions for us under the hood with the `residplot` function.
    '''
    return sns.residplot(x, y)


#split Scale
def split_my_data(data, train_ratio=.80, seed=123):
    '''the function will take a dataframe and returns train and test dataframe split 
    where 80% is in train, and 20% in test. '''
    return train_test_split(data, train_size = train_ratio, random_state = seed)


def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled


def uniform_scaler(train, test, seed=123):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=seed, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

##Visual
def evaluation_example1(df, x, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='dimgray')



def plot_regression(x,y):
    res = sm.OLS(y, x).fit()
    prstd, iv_l, iv_u = wls_prediction_std(res)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'o', label="data")
    #ax.plot(x, y, 'b-', label="True")
    ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
    ax.plot(x, iv_u, 'g--',label='97.5% Confidence Level')
    ax.plot(x, iv_l, 'b--',label='2.5% Confidence Level')
    ax.legend(loc='best');
    plt.show()