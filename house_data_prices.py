#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:20:50 2023

@author: aitorchagon
"""

import pandas as pd
import miceforest as mf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import numpy as np

train = pd.read_csv('test.csv')
test = pd.read_csv('train.csv')

house = pd.concat([train, test])

#we are going to delete ID column of House as it is of no interest to our study.
del house['Id']


#now we are going to classify our columns into nominal_categorical, ordinal_categorical and numerical
#this way, we will be ready to study statistical properties properly.

#Street,CentralAir, are binary valued, we need to take that into account.
nominal_categorical = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotConfig',
                       'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                       'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                       'GarageType', 'PavedDrive', 'MiscFeature', 'MoSold',
                       'SaleType', 'SaleCondition']
ordinal_categorical = ['LotShape', 'LandContour', 'Utilities', 'LandSlope',
                       'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                       'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical',
                       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish'
                       ,'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
numerical = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1',
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
             'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
             'Fireplaces', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'WoodDeckSF', 
             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
             'PoolArea', 'MiscVal', 'YrSold']
#As we can see, we are dealing with a regression problem.
target =  ['SalePrice']

#we are going to see whether we have any null value, at least declared as NaN.

null_values = house.isnull().sum()

'''We see that we have a really high value of NaNs declared at the following features:
    -Alley: 2721/2919 -> 93.21%
    -PoolQC: 2909/2919 -> 99.66%
    -Fence: 2348/2919 -> 80.43%
    -MiscFeature: 2814/2919 -> 96.4%
    -FireplaceQu: 1420/2919 -> 48.65%
As a result, we will proceed to delete those columns right away
We will deal with the rest of them using MICE forest algorithm'''

columns_to_delete = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
house = house.drop(columns_to_delete, axis = 1)
nominal_categorical.remove('Alley')
nominal_categorical.remove('MiscFeature')
ordinal_categorical.remove('PoolQC')
ordinal_categorical.remove('Fence')
ordinal_categorical.remove('FireplaceQu')

null_values_ = house.isnull().sum()

#Now we will proceed to codify the rest of variables; we will codify, temporarily, 
#nan values as a number; then, we will change this number for a np.nan and execute our
#mice forest model. In the case of nominal feature, we will not impute a value but remove them.
#As the amount of them is very low

ohe = OneHotEncoder(categories= 'auto', sparse_output = False)
ohe_df = pd.DataFrame()
#MSubClass has two categories quite similar 'Split or multi-level" and "Split"
#The same occurs with 1 and 1/2 finished and unfinished, 2 and 2 1/2
#let's merge them
house['MSSubClass'] = house['MSSubClass'].replace(85, 80)
house['MSSubClass'] = house['MSSubClass'].replace(45, 50)
house['MSSubClass'] = house['MSSubClass'].replace(150, 120)
house['MSSubClass'] = house['MSSubClass'].replace(40, 20)
house['MSSubClass'] = house['MSSubClass'].replace(75, 60)

df_number_categories = pd.DataFrame(columns = nominal_categorical)   
for elem in nominal_categorical:
    number = house[elem].value_counts(sort = False)
    df_number_categories = pd.concat([number, df_number_categories], axis = 1, join = 'outer')
    df_number_categories = df_number_categories.replace(np.nan, 0)

  




