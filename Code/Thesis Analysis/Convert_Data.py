# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:05:34 2020

@author: theod
"""
import numpy as np
import pandas as pd

#We must convert all of the case, death, and testing data into consistent and fair forms by making it noncumulative and by per million people

#Convert the population Data from thousands to millions
population_Data['Pop (1000s)'] = population_Data['Pop (1000s)'] / 1000

#Make the data in cases_File and death_File DataFrame Daily instead of cumulative
def cum_to_daily(DataFrame):
    #Create a reversed list of all of the column headers
    col_List = DataFrame.columns.values
    col_List = list(reversed(col_List))
    #Iterate through all the column headers, and make the current column = current column - previous column
    for i in range(len(col_List)-1):
        DataFrame[col_List[i]] = DataFrame[col_List[i]].sub(DataFrame[col_List[i+1]])
    return

#This cum_to_daily function works especially for the testing data, which needs special cosndierations due to it's placement of "nan" values
def cum_to_daily_Test(DataFrame):
    #Create a reversed list of all of the column headers
    col_List = DataFrame.columns.values
    col_List = list(reversed(col_List))
    #Create a list of the row indexes
    row_List = DataFrame.index.values
    #Loop through all of the column headers row by row and make the current cell = current cell - previous cell
    for i in row_List:
        #Create a blank list to store the indexes in the DataFrame that have values
        loop_List = []
        #Find which cells in the DataFrame's row i have data
        has_Values = pd.notna(DataFrame.loc[i])
        #If the cell has data, add it to the loop_List list
        for k in range(len(col_List)):
            if has_Values[k] == True:
                loop_List.append(k)
                
        #Loop through the cells that have data dn convert the data from cumulative to daily via subtraction
        loop_List = list(reversed(loop_List))
        for j in range(len(loop_List)-1):
            DataFrame.loc[i][loop_List[j]] = (DataFrame.loc[i][loop_List[j]] - DataFrame.loc[i][loop_List[j+1]])
            
    return

#Convert the case, death, and testing data from cumulative to daily
cum_to_daily(cases_File)
cum_to_daily(death_File)
cum_to_daily_Test(new_Testing)

#loop throuhg the daily_or_cum dataframe and, if the daily data should be used, replace the cumulative data with it in the new_Testing dataframe
for i in ISO_List:
    if daily_or_cum.loc[i, 'Daily or Cum'] == "Daily":
        new_Testing.loc[i, :] = daily_Testing.loc[i, :]


#Make the data in the cases, deaths, and testing files per million instead of in total
def total_to_per_Mil(df, pop_df):
    #By using the .isin function, we only divide the pop_df value in it's only column by the row in the df DataFrame if it's index matches an index in the df DataFrame
    df = df.div(pop_df[pop_df.index.isin(df.index)]['Pop (1000s)'], axis=0)
    df.fillna(0, inplace = True)
    return df

#convert the case, death, and testing data into incidents per million
cases_File = total_to_per_Mil(cases_File, population_Data)
death_File = total_to_per_Mil(death_File, population_Data)
new_Testing = total_to_per_Mil(new_Testing, population_Data)
    