import matplotlib.pyplot as plt
import seaborn as sns

#USE df.unique()<5 instead of this temp list
def sort(df):
    discretes = df.select_dtypes(include=['object','int64'])
    temp = []
    for column in discretes:
        columnSeriesObj = discretes[column]
        if len(columnSeriesObj.unique()) < 4:
            temp.append(columnSeriesObj.name)
    return temp

'''def plot_viable_categories(target, df):
    x = pick_viable_categories(df)
    _, ax = plt.subplots(nrows=1, ncols=len(x), figsize=(16,5))
    average_rate = df.target.mean()
    for i, feature in enumerate(x):
        sns.barplot(feature, target, data=df, ax=ax[i], alpha=.5)
        ax[i].set_ylabel('average_rate')
        ax[i].axhline(average_rate, ls='--', color='grey')'''

def barplot(df, features):
    ax = plt.subplots(nrows=1, ncols=3, figsize=(16,5))
    survival_rate = train_df.survived.mean()
    for i, feature in enumerate(features):
        sns.barplot(features, 'survived', data=df, ax=ax[i], alpha=.5)
        ax[i].set_ylabel('Survival Rate')
        ax[i].axhline(survival_rate, ls='--', color='grey')
        
def pick_viable_regressors(df):
    regressors = df.select_dtypes(include=['float64','int64'])
    temp = []
    for column in regressors:
        columnSeriesObj = regressors[column]
        temp.append(columnSeriesObj.name)
    return temp

'''dtypes= object
ndistinct (< 5)
box plot x= cat, y = target'''


'''dtypes = numeric
df= numeric +target
boxplot x=target, y = numeric'''
def plot_violin(features, target, df):
    for descrete in df[features].select_dtypes([object,int]).columns.tolist():
        if df[descrete].nunique() <= 5:
            for continous in df[features].select_dtypes(float).columns.tolist():
                sns.violinplot(descrete, continous, hue=target,
                data=df, split=True, palette=['blue','orange'])
                plt.title(continous + 'x' + descrete)
                plt.ylabel(continous)
                plt.show()



def plot_violins(target, df):
    for descrete in df.select_dtypes([object,int]).columns.tolist():
        if df[descrete].nunique() <= 5:
            for continous in df.select_dtypes(float).columns.tolist():
                sns.violinplot(descrete, continous, hue=target,
                data=df, split=True, palette=['blue','orange'])
                plt.title(continous + ' x ' + descrete)
                plt.ylabel(continous)
                plt.show()