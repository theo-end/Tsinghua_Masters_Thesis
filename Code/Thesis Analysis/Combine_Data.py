# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:46:15 2020

@author: theod
"""
import numpy as np
import pandas as pd

#Both the Cases and Death files have data from some countries mutliple times (i.e. AUS appear 8 times), thus these need to be combined
cases_File = cases_File.groupby(['ISO'], as_index=False).sum()
death_File = death_File.groupby(['ISO'], as_index=False).sum()
population_Data = population_Data.groupby(['ISO'], as_index=False).sum()


#We need to clean up to testing data. Some countries only report cumulative cases, some only daily, and some only report very infreqently
#Cleanup needs to be done to get one, consistent, file so the data is all correct.

#Get the oldest and newest dates from the DataFrame
testing_File['Date'] = pd.to_datetime(testing_File['Date'])
oldest_Date = testing_File['Date'].min()
newest_Date = testing_File['Date'].max()

#Creates a list of all dates
date_List = pd.date_range(start=oldest_Date, end=newest_Date)

#Get the list of unique ISO codes
ISO_List = testing_File['ISO code'].unique()

#create new DataFrame for converted testing data and temporary DataFrames sto store the daily and cumulative testing of each country
daily_Testing = pd.DataFrame(data=None, columns = date_List, index=ISO_List)
cum_Testing = pd.DataFrame(data=None, columns = date_List, index=ISO_List)
new_Testing = pd.DataFrame(data=None, columns = date_List, index=ISO_List)


#Fill in the Daily and Cumulative testing dataframes with their data fromt he testing file
num_Rows = testing_File.shape[0]
for i in range(num_Rows):
    daily_Testing.at[testing_File.at[i, 'ISO code'], testing_File.at[i, 'Date']] = testing_File.at[i, 'Daily change in cumulative total']
    cum_Testing.at[testing_File.at[i, 'ISO code'], testing_File.at[i, 'Date']] = testing_File.at[i, 'Cumulative total']

#create a DataFrame where we can store whether or not the Daily or Cumualtive Testing data should be used    
daily_or_cum = pd.DataFrame(data=None, columns = {"Daily or Cum"}, index = ISO_List)

#based on the whether the daily or cumualtive testing data has more total cases, fill in hte daily_or_cum dataframe with the correct data to use
for i in ISO_List:
    if cum_Testing.loc[i, :].max() >= daily_Testing.loc[i,:].sum():
        daily_or_cum.loc[i, 'Daily or Cum'] = "Cum"
    else :
        daily_or_cum.loc[i, 'Daily or Cum'] = "Daily"

#Create the new_Testing data frame to be the cumulative testing data by default
new_Testing = cum_Testing
