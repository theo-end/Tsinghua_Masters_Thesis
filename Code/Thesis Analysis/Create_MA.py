# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:12:22 2020

@author: theod
"""
import pandas as pd
import numpy as np
#We must create a funciton that takes the testing, death, and cases data (or any time series) and creates a new dataframe with a moving average instead or raw data

#creates a DataFrame of moving average instead of raw data and delete the resulting NaN columns
def norm_to_MA(df, MA_Len):
    #create a DataFrame with moving averages of window length MA_Len
    df1 = df.rolling(MA_Len, axis=1).mean()
    
    #delete the resulting NaN columns
    array = list(range(0,MA_Len-1))
    df1.drop(df1.columns[array], axis = 1, inplace = True)
    return df1

cases_MA = norm_to_MA(cases_File, 7)
deaths_MA = norm_to_MA(death_File, 7)
testing_MA = norm_to_MA(new_Testing, 7)