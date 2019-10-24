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
    df = get_titanic_data()
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

#1)Use the function defined in `aquire.py` to load the iris data.


iris_df=get_iris_data()
#1a) Drop the `species_id` and `measurement_id` columns.
def drop_columns(df):
    return df.drop(columns=['species_id','measurement_id'])
#1b) Rename the `species_name` column to just `species`.
def rename_columns(df):
    df['species']=df['species_name']
    return df
#1c)Encode the species name using a sklearn encoder. Research the inverse_transform method
#of the label encoder.How might this be useful.
def encode_columns(df):
    encoder=LabelEncoder()
    encoder.fit(df.species)
    df.species=encoder.transform(df.species)
    return df,encoder
#create a function that accepts the untransformed iris
#data, and returns the data with the transformations above applied.
def prep_iris(df):
    df=df.pipe(drop_columns).pipe(rename_columns).pipe(encode_columns)
    return df 
# Titanic Data
# Use the function you defined in aquire.py to load the titanic data set    
df=get_titanic_data()
# 2a) Handle the missing values in the `embark_town` and `embarked`columns.
def titanic_missing_fill(df):
    df.embark_town.fillna('Other',inplace=True)
    df.embarked.fillna('Unknown',inplace=True)
    return df
# 2b) Remove the deck column.
def titanic_remove_columns(df):
    return df.drop(columns=['deck'])
# 2c) Use a label encoder to transform the `embarked` column
def encode_titanic(df):
    encoder_titanic=LabelEncoder()
    encoder_titanic.fit(titanic_df.embarked)
    titanic_df=encoder_titanic.transform(titanic_df.embarked)
    return titanic_df,encoder_titanic
# 2d) Scale the `age` and `fare` columns using a min/max scaler.
def scale_titanic(df):
    scaled=MinMaxScaler()
    scaled.fit(df[['age','fare']])
    df[['age','fare']]=scaled.transform(df[['age','fare']])
    return df,scaled
#Why might this be beneficial? When might this be beneficial? When might you not
#want to do this?
#Create a function named prep_titanic that accepts the untransformed titanic data,
#and returns the data with the transformations above applied
def prep_titanicss(df):
    df=df.pipe(titanic_missing_fill).pipe(titanic_remove_columns).pipe(encode_titanic).pipe(scale_titanic)
    return df