#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:13:43 2023

@author: aitorchagon
"""

for elem in nominal_categorical:
    if elem == 'MSSubClass':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(house['MSSubClass'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['1-STORY 1946', '2-STORY 1946 & NEWER', 
                               '1-STORY PUD - 1946 & NEWER', '2-STORY PUD - 1946 & NEWER'
                               , '1-STORY 1945 & OLDER', '1-1/2 STORY FINISHED ALL AGES',
                               'SPLIT OR MULTI-LEVEL', 'DUPLEX - ALL STYLES AND AGES', 
                               '2 FAMILY CONVERSION - ALL STYLES AND AGES', 
                               '2-STORY 1945 & OLDER', 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER']
        ohe_df = ohe_df_temp
        del ohe_df_temp
        del house['MSSubClass']
    elif elem == 'MSZoning':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(house['MSZoning'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C4-u', 'C4-y', 'C4-l']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C4']
    elif elem == 'C5':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(crx['C5'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C5-g', 'C5-p', 'C5-gg']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C5']
    elif elem == 'C6':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(crx['C6'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C6-c', 'C6-d', 'C6-cc', 'C6-i', 'C6-j', 'C6-k', 'C6-m', 'C6-r', 'C6-q',
                               'C6-w', 'C6-x', 'C6-e', 'C6-aa', 'C6-ff']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C6']
    elif elem == 'C7':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(crx['C7'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C7-v', 'C7-h', 'C7-bb', 'C7-j',
                               'C7-n', 'C7-z', 'C7-dd', 'C7-ff', 'C7-o']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C7']
    elif elem == 'C9':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(crx['C9'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C9-t', 'C9-f']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C9']
    elif elem == 'C10':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(crx['C10'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C10-t', 'C10-f']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C10']
    elif elem == 'C12':
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(crx['C12'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C12-t', 'C12-f']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C12']
    else:
        ohe_df_temp = pd.DataFrame(ohe.fit_transform(crx['C13'].values.reshape(-1, 1)))
        ohe_df_temp.columns = ['C13-g', 'C13-p', 'C13-s']
        ohe_df = ohe_df.join(ohe_df_temp)
        del ohe_df_temp
        del crx['C13']

crx = crx.join(ohe_df)


#We will deal with nan values .now
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