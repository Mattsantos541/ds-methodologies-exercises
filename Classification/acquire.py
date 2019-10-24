import pandas as pd
import env
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers',get_connection('titanic_db'))

iris_sql="SELECT measurements.measurement_id,measurements.sepal_length,measurements.sepal_width,measurements.petal_length,measurements.petal_width,species.species_name,species.species_id FROM measurements JOIN species ON(species.species_id=measurements.species_id)"
def get_iris_data():
       return pd.read_sql(iris_sql,get_connection('iris_db'))


train = 'https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=341089357'    


def prep_titanic():
    df = acquire.get_titanic_data()
    df.embark_town.fillna('Other', inplace=True)
    df.embarked.fillna('Unknown', inplace=True)
    df.drop(columns=['deck'], inplace=True)
    
    encoder = LabelEncoder()
    df.embarked = encoder.fit_transform(df.embarked)
    
    scaler = MinMaxScaler()
    df.age = scaler.fit_transform(df[['age']])
    
    scaler = MinMaxScaler()
    df.fare = scaler.fit_transform(df[['fare']])
    
    return df

def prep_iris():
    df_iris = acquire.get_iris_data()
    df_iris = df_iris.drop(columns=['species_id'])
    df_iris = df_iris.drop(columns=['measurement_id'])
    df_iris = df_iris.rename(columns={'species_name':'species'})
    
    encoder = LabelEncoder()
    df_iris.species = encoder.fit_transform(df_iris.species)
    return df_iris
