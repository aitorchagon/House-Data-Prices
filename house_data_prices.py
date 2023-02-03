#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:20:50 2023

@author: aitorchagon
"""

import pandas as pd
import miceforest as mf

train = pd.read_csv('/home/aitorchagon/Desktop/Proyectos/House Data Prices/train.csv')
test = pd.read_csv('/home/aitorchagon/Desktop/Proyectos/House Data Prices/test.csv')

house = pd.concat([train, test])

#we are going to delete ID column of House as it is of no interest to our study.
del house['Id']


#now we are going to classify our columns into nominal_categorical, ordinal_categorical and numerical
#this way, we will be ready to study statistical properties properly.

#Street,CentralAir, are binary valued, we need to take that into account.
nominal_categorical = ['MSubClass', 'MSZoning', 'Street', 'Alley', 'LotConfig',
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
As a result, we will proceed to delete those columns right away
We will deal with the rest of them using MICE forest algorithm'''

columns_to_delete = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
house = house.drop(columns_to_delete, axis = 1)
nominal_categorical.remove('Alley')
nominal_categorical.remove('MiscFeature')
ordinal_categorical.remove('PoolQC')
ordinal_categorical.remove('Fence')


#Now we will deal with the rest of the imputations, as they are all under 50% of missing values.
#hay que pensar cómo hacer esto teniendo en cuenta que no queremos codificar las categorías
#(después va a ser difícil hacer onehotencoding o lo que sea en las nominales)
mice = mf.ImputationKernel(
  house,
  save_all_iterations=True,
  random_state=1999
)

mice.mice(2)

house = mice.complete_data()
null_values_ = house.isnull().sum()
#let's see whether we have cleaned our data.

#Let's see a statistical description of our data before turning to the plots.
description_data = house.describe()

