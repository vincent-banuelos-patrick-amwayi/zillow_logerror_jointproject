from datetime import datetime
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from env import user, password, host

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")

def get_zillow():
    '''
    Pulls data from codeup database and writes into a dataframe
    '''
    if os.path.isfile('zillow.csv'):
        return pd.read_csv('zillow.csv', index_col=0)
    
    else:

        url = f"mysql+pymysql://{user}:{password}@{host}/zillow"
        
        query = '''
                SELECT
                    prop.*,
                    predictions_2017.logerror,
                    predictions_2017.transactiondate,
                    air.airconditioningdesc,
                    arch.architecturalstyledesc,
                    build.buildingclassdesc,
                    heat.heatingorsystemdesc,
                    landuse.propertylandusedesc,
                    story.storydesc,
                    construct.typeconstructiondesc
                FROM properties_2017 prop
                JOIN (
                    SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                    FROM predictions_2017
                    GROUP BY parcelid
                ) pred USING(parcelid)
                JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                                    AND pred.max_transactiondate = predictions_2017.transactiondate
                LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
                LEFT JOIN storytype story USING (storytypeid)
                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                WHERE prop.latitude IS NOT NULL
                AND prop.longitude IS NOT NULL
                AND transactiondate <= '2017-12-31'
                AND propertylandusedesc = "Single Family Residential"
                ORDER BY transactiondate;
                '''

        df = pd.read_sql(query, url)
        # Ensuring no duplicates are in dataframe, keeping latest sale as query is ordered by transaction date
        df = df[~df.duplicated(subset=['parcelid'],keep='last')]

        df.to_csv('zillow.csv')

    return df



def prep_zillow(df):
    # create column with fips value converted from an integer to the county name string
    df['fips'] = df.fips.map({6037.0 : 'los_angeles', 6059.0 : 'orange', 6111.0 : 'ventura'})

    # Dropping columns of unnecessary columns that tell redundant information already told in other columns
    df = df.drop(columns=['rawcensustractandblock','roomcnt','finishedsquarefeet12','fireplaceflag','propertylandusetypeid'])
    # taking out nulls in tract column
    df = df[df['censustractandblock'].notna()]
    # filtering out oart of census tract to show only 
    df.censustractandblock = df.censustractandblock.astype(str).str[4:8]
    df.censustractandblock = df.censustractandblock.astype(int)

    df = df[df['lotsizesquarefeet'].notna()]

    df = df.drop(columns='regionidcity')

    # Changing bedrooms to a discrete variable
    df = df[df.bedroomcnt != 0.0]

    # Changing bedrooms to a discrete variable
    df = df[df.bathroomcnt != 0.0]

    # filtering for only transactions in 2017
    df = df[df.transactiondate < '2018-01-01']

    # convert lat and long to proper decimal format
    df['latitude'] = df['latitude'] / 1_000_000
    df['longitude'] = df['longitude'] / 1_000_000

    #Filling nulls for multiple columns as well as changing data types for certian columns
    df.parcelid = df.parcelid.astype(object)
    df.id = df.id.astype(object)
    df.buildingqualitytypeid = df.buildingqualitytypeid.fillna(0)
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(0).astype(object)
    df.fullbathcnt = df.fullbathcnt.fillna(0).astype(object)
    df.regionidzip = df.regionidzip.fillna(df.regionidzip.mode()).astype(object)
    df.yearbuilt = df.yearbuilt.fillna(df.yearbuilt.median())
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(df.structuretaxvaluedollarcnt.mean())
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(df.taxvaluedollarcnt.mean())
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(df.landtaxvaluedollarcnt.mean())
    df.taxamount = df.taxamount.fillna(df.taxamount.mean())
    df.censustractandblock = df.censustractandblock.fillna(df.censustractandblock.median()).astype(object)
    df.fireplacecnt = df.fireplacecnt.fillna(0).astype(object)
    df.garagecarcnt = df.garagecarcnt.fillna(0).astype(object)
    df.poolcnt = df.poolcnt.fillna(0).astype(object)
    df.garagetotalsqft = df.garagetotalsqft.fillna(0)
    df.hashottuborspa = df.hashottuborspa.fillna(0).astype(object)
    df.threequarterbathnbr = df.threequarterbathnbr.fillna(0).astype(object)
    df.regionidcounty = df.regionidcounty.astype(object)
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])

    #Making transaction month column
    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    df['transaction_month'] = pd.cut(df.transactiondate.dt.month, bins)
    df.transaction_month = df.transaction_month.astype(object)

    # Feature engineering new feature to show proportioins of house sqft to lot sqft and dropping rows where house sqft exceeds lot sqft.
    df['house_lotsize_ratio'] = (df.calculatedfinishedsquarefeet/df.lotsizesquarefeet) * 100
    df = df[df.house_lotsize_ratio < 100]
    
    # Making sure properties are not tax deliquent
    df = df[df.taxdelinquencyflag != 'Y']

    # drop rows with zip over 99999
    df= df[df.regionidzip <= 99999]

    # drop property use type that is no longer needed
    df.drop(columns=['propertylandusedesc'], inplace=True)

    # one-hot encode county
    dummies = pd.get_dummies(df['fips'],drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    # rename columns for clarity
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', \
        'calculatedfinishedsquarefeet':'sqft','taxvaluedollarcnt':'tax_value', \
        'lotsizesquarefeet':'lotsize','censustractandblock':'tract','regionidzip':'zip', 'fips':'county'})

    df = df.astype({'bathrooms':'object','bedrooms':'object','zip' :'object','buildingqualitytypeid' : 'object'})

    return df

def my_split(df):
    '''
    Splitting the dataframe into 3 seperate samples to prevent data leakage
    '''

    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test




def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    Removes observations that do not have above the user specified amount of data in each row and columns.
    '''
    
    prop_null_column = 1 - prop_required_column
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
            
    threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis=0, thresh=threshold, inplace=True)
    
    return df

def remove_outliers(df):
    ''' remove outliers from a list of columns in a dataframe
        and return that dataframe
    '''
    col_list = ['logerror','sqft','lotsize','tax_value','taxamount','bathrooms','bedrooms']

    for col in col_list:
        q1, q3 = df[col].quantile([.13, .87])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        upper_bound = q3 + 1.5 * iqr   # get upper bound
        lower_bound = q1 - 1.5 * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def wrangle_zillow():
    '''
    Combines data acquisition and preparation into one function and returns the 3 split samples.
    '''
    df = get_zillow()
    df = prep_zillow(df)
    df = handle_missing_values(df,.75,.75)
    df = remove_outliers(df)
    train, validate, test = my_split(df)

    return train, validate, test

#Write function to scale data for zillow data
def scale_data(train, validate, test, scalable_columns):
    """Scales the 3 data splits using MinMax Scaler.
    Takes in train, validate, and test data splits as well as a list of the features to scale.
    Returns dataframe with scaled counterparts on as columns"""
    # Make the thing to train data only
    scaler = MinMaxScaler()
    scaler.fit(train[scalable_columns])
    # Fit the thing with new column names with _scaled added on
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[scalable_columns])
    scaled_validate = scaler.transform(validate[scalable_columns])
    scaled_test = scaler.transform(test[scalable_columns])
    # Apply the scaled data to the original unscaled data
    train_scaled = pd.DataFrame(scaled_train,index=train.index, columns = scalable_columns)
    validate_scaled = pd.DataFrame(scaled_validate,index=validate.index, columns = scalable_columns)
    test_scaled = pd.DataFrame(scaled_test,index=test.index, columns = scalable_columns)

    return train_scaled, validate_scaled, test_scaled