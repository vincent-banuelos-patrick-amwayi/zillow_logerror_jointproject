import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydataset
import seaborn as sns
sns.set()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from scipy import stats as stats


def countygraph(df):
    county_cluster = df[['los_angeles', 'orange', 'ventura']]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(county_cluster)

    df['county_cluster'] = kmeans.predict(county_cluster)

    sns.relplot(data=df.sample(1000), x='county', y='logerror', hue='county_cluster',height=8)
    plt.show()
    return

def ventura_test(df):
    ''' 
    Runs statitical testing for bed and bathroom comparison.
    '''
    long_le = df[df.county == 'ventura'].logerror
    lat_le = df[df.county != 'ventura'].logerror

    # Set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = stats.levene(long_le, lat_le)

    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(long_le, lat_le, equal_var=(pval >= alpha))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alpha and t > 0:
        print('''Reject the Null Hypothesis.
        
There is a difference in logerror between homes in ventura and homes not in ventura.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
No difference in logerror between homes in ventura and homes not in ventura.''')
    return


def tv_graph(df):
    tv_cluster = df[['tax_value','landtaxvaluedollarcnt','structuretaxvaluedollarcnt']]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(tv_cluster)


    df['tv_cluster'] = kmeans.predict(tv_cluster)
    sns.relplot(data=df, x='latitude', y='longitude', col='tv_cluster',height=10)
    plt.show()
    return

def tax_corr_test(df):
    
    #stats.spearmanr()
    r, p  = stats.spearmanr(df.tv_cluster, df.logerror)

    # Set alpha
    alpha = 0.05
    # Evaluate results based on the t-statistic and the p-value
    if p < alpha:
        print('''Reject the Null Hypothesis.
        
There IS a relationship between value cluster and logerror.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
There IS NOT a relationship between our value cluster and logerror.''')
    return


def htls_graph(df):
    hlsizeratio = df[['house_lotsize_ratio']]

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(hlsizeratio)


    df['hlsizeratio'] = kmeans.predict(hlsizeratio)
    sns.relplot(data=df, x='house_lotsize_ratio', y='logerror', hue='hlsizeratio',height=10)
    return

def sqft_lsize_ttest(df):
    ''' 
    Runs statistical test for yearbuilt column, seeing which houses are more expensive, pre-1972 or post-1972.
    '''
    below_50 = df.logerror[df.house_lotsize_ratio <= 60]
    above_50 = df.logerror[df.house_lotsize_ratio > 60]

    # Set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = stats.levene(below_50, above_50)

    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(below_50, above_50, equal_var=(pval >= alpha))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alpha and t > 0:
        print('''Reject the Null Hypothesis.
        
Homes with a less than 60 percent house to lot ratio DO have a lower logerror.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Homes with a less than 60 percent house to lot ratio DO NOT have a lower logerror.''')
    return

def yb_graph(df):
    yb_cluster = df[['yearbuilt']]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(yb_cluster)
    df['yb_cluster'] = kmeans.predict(yb_cluster)

    plt.figure(figsize=(14,8))
    sns.jointplot(x='yearbuilt',y='logerror', data=df, hue='yb_cluster',height=10)
    plt.show()
    plt.show()
    return


def yb_test(df):
    ''' 
    Runs statistical test for yearbuilt column, seeing which houses are more expensive, pre-1972 or post-1972.
    '''
    older_2k = df.logerror[df.yearbuilt < 2010]
    newer_2k = df.logerror[df.yearbuilt > 2010]

    # Set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = stats.levene(older_2k, newer_2k)

    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(older_2k, newer_2k, equal_var=(pval >= alpha))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alpha and t > 0:
        print('''Reject the Null Hypothesis.
        
Homes built before 2010 DO have a logerror closer to 0.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Homes built before 2010 DO NOT have a logerror cloeser to 0.''')
    return

