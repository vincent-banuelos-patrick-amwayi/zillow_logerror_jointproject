import os
import pandas as pd
import numpy as np

import wrangle as wr
import explore as ex

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")



train, validate, test = wr.wrangle_zillow()

scalable_columns = ['house_lotsize_ratio', 'tax_value','los_angeles','orange','ventura','yearbuilt']

train_scaled, validate_scaled, test_scaled = wr.scale_data(train, validate, test, scalable_columns)

def scale_data_xy_split():
    X_train_scaled = train_scaled
    y_train = train['logerror']

    X_validate_scaled = validate_scaled
    y_validate = validate['logerror']

    X_test_scaled = test_scaled
    y_test = test['logerror']

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    return X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test


def run_da_stuff(X_train_scaled,y_train,X_validate_scaled,y_validate):
    '''
    returns results for different model types for train and validate dataset
    '''
    pred_mean = y_train.logerror.mean()
    y_train['pred_mean'] = pred_mean
    y_validate['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_mean, squared=False)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_mean, squared=False)

    # save the results
    metric_df = pd.DataFrame(data=[{
        'model': 'baseline_mean',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_mean),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.logerror, y_validate.pred_mean)
        }])

    #Linear Regression model
    # run the model
    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train.logerror)
    y_train['pred_lm'] = lm.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lm)**(1/2)
    y_validate['pred_lm'] = lm.predict(X_validate_scaled)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_lm)**(1/2)

    # save the results
    metric_df = metric_df.append({
        'model': 'Linear Regression',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_lm),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.logerror, y_validate.pred_lm)}, ignore_index=True)


    # LassoLars Model
    lars = LassoLars(alpha=1)
    lars.fit(X_train_scaled, y_train.logerror)
    y_train['pred_lars'] = lars.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lars, squared=False)
    y_validate['pred_lars'] = lars.predict(X_validate_scaled)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_lars, squared=False)

    # save the results
    metric_df = metric_df.append({
        'model': 'LarsLasso, alpha 1',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_lars),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.logerror, y_validate.pred_lars)}, ignore_index=True)

    # create the model object
    glm = TweedieRegressor(power=0, alpha=0)
    glm.fit(X_train_scaled, y_train.logerror)
    y_train['glm_pred'] = glm.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.logerror, y_train.glm_pred)**(1/2)
    y_validate['glm_pred'] = glm.predict(X_validate_scaled)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.glm_pred)**(1/2)


    # save the results
    metric_df = metric_df.append({
        'model': 'Tweedie Regressor',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.glm_pred),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.logerror, y_validate.glm_pred)}, ignore_index=True)

    # create the model object
    pf = PolynomialFeatures(degree=2)
    X_train_degree2 = pf.fit_transform(X_train_scaled)
    X_validate_degree2 = pf.transform(X_validate_scaled)
    lm = LinearRegression()
    lm.fit(X_train_degree2, y_train.logerror)
    y_train['logerror_pred_pf'] = lm.predict(X_train_degree2)
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_pf)**(1/2)
    y_validate['logerror_pred_pf'] = lm.predict(X_validate_degree2)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_pf)**(1/2)


    # save the results
    metric_df = metric_df.append({
        'model': 'Polynomial Features, D2',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.logerror_pred_pf),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.logerror, y_validate.logerror_pred_pf)}, ignore_index=True)
    
    
    # create the model object
    pf = PolynomialFeatures(degree=3)
    X_train_degree2 = pf.fit_transform(X_train_scaled)
    X_validate_degree2 = pf.transform(X_validate_scaled)
    lm = LinearRegression()
    lm.fit(X_train_degree2, y_train.logerror)
    y_train['logerror_pred_pf'] = lm.predict(X_train_degree2)
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_pf)**(1/2)
    y_validate['logerror_pred_pf'] = lm.predict(X_validate_degree2)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_pf)**(1/2)


    # save the results
    metric_df = metric_df.append({
        'model': 'Polynomial Features, D3',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.logerror_pred_pf),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.logerror, y_validate.logerror_pred_pf)}, ignore_index=True)
    
    
    return metric_df

def test_tester(X_train_scaled,y_train,X_validate_scaled,y_validate,X_test_scaled,y_test):
    ''' 
    This function takes in the X and y objects and then runs and returns a DataFrame of
    results for the Tweedie Regressor model 
    '''

    #baseline model
    pred_mean = y_train.logerror.mean()
    y_train['pred_mean'] = pred_mean
    y_validate['pred_mean'] = pred_mean
    base_rmse_train = mean_squared_error(y_train.logerror, y_train.pred_mean, squared=False)
    base_rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_mean, squared=False)

    # create the model object
    pf = PolynomialFeatures(degree=3)
    X_train_degree3 = pf.fit_transform(X_train_scaled)
    X_validate_degree3 = pf.transform(X_validate_scaled)
    X_test_degree3 = pf.transform(X_test_scaled)
    lm = LinearRegression()
    lm.fit(X_train_degree3, y_train.logerror)
    y_train['logerror_pred_pf'] = lm.predict(X_train_degree3)
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_pf)**(1/2)
    y_validate['logerror_pred_pf'] = lm.predict(X_validate_degree3)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_pf)**(1/2)
    y_test['logerror_pred_pf'] = lm.predict(X_test_degree3)
    rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_pf)**(1/2)

    test_metrics = pd.DataFrame({'baseline': 
                               {'rmse': base_rmse_train, 
                                'r2': explained_variance_score(y_train.logerror, y_train.logerror_pred_pf)},
        
                            'train': 
                               {'rmse': rmse_train, 
                                'r2': explained_variance_score(y_train.logerror, y_train.logerror_pred_pf)},
                           'validate': 
                               {'rmse': rmse_validate, 
                                'r2': explained_variance_score(y_validate.logerror, y_validate.logerror_pred_pf)},
                           'test': 
                               {'rmse': rmse_test, 
                                'r2': explained_variance_score(y_test.logerror, y_test.logerror_pred_pf)}
                                })

    return test_metrics.T