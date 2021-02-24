# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:25:49 2020

@author: theod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#This function reads in a dataframe, ISO, and Date and returns the corrected value and excess value for the outlier based on a 7 day average before and after the outlier
def new_Value_and_Excess(df, ISO, Date):
    #gets the iloc of the feed in ISO and Date
    row = df.index.get_loc(ISO)
    col = df.columns.get_loc(Date)
    
    #Sets the before and after avarages to zero
    seven_MA_after = 0
    seven_MA_before = 0
    
    if col < 8:
        #Accounts for the situation when the outlier is too close to the start of the data
        seven_MA_after = df.iloc[row, (col+1):(col+8)].mean()
        seven_MA_before = df.iloc[row, 0:(col)].mean()
    elif (len(df.columns.values) - col) == 1:
        #accounts for the situation where the oultier is on the late date in the data
        seven_MA_after = 0
        seven_MA_before = df.iloc[row, (col-14):(col)].mean()
    elif (len(df.columns.values) - col) < 8:
        #accounts for the situation where the outlier is too close to the end of the data
        seven_MA_after = df.iloc[row, (col+1):(len(df.columns.values))].mean()
        seven_MA_before = df.iloc[row, (col-7):(col)].mean()
    else:
        #accounts for the normal case where the outlier is not too close to the start or end of the data
        seven_MA_after = df.iloc[row, (col+1):(col+8)].mean()
        seven_MA_before = df.iloc[row, (col-7):(col)].mean()
    
    #calcualtes teh new value and the excess to be redistubuted
    new_Value = (seven_MA_after + seven_MA_before) / 2
    
    #accounts for the situation the outlier is negative, thus the excess amount must not also be negative
    if df.iloc[row,col] > 0:
        Excess = df.iloc[row, col] - new_Value
    else:
        Excess = abs(df.iloc[row,col]) + new_Value

    return [new_Value, Excess] 

#This function finds all instances of negative data and organizes them into a datframe format so they can be corrected by the function remove_Outlier
def find_Negatives(df, dataset_Name):
    #Create DataFrame to hold instances of negative values
    negatives = (df<0)
    #Create a filtered dataframe that only includes rows and columns with at least one negative value
    filtered_data = df[negatives].dropna(axis = 0, how = 'all')
    filtered_data.dropna(axis = 1, how = 'all', inplace = True)  
    
    #create Dataframe to store locations of negative values
    negative_Locations = pd.DataFrame(columns = ['ISO', 'filler', 'Date', 'Dataset'])
    
    #Loop throuhg all elements in the filtered_data DataFrame and, if the value is negative, append the ISO, Date, and Dataset_Name to the negative_Locations DataFrame
    for i in range(filtered_data.shape[0]):
        for j in range(filtered_data.shape[1]):
            if filtered_data.iloc[i, j] < 0:
                negative_Locations = negative_Locations.append({'ISO': filtered_data.index[i], 'Date' : filtered_data.columns[j], 'Dataset' : dataset_Name}, ignore_index=True)
    
    #Sort the negative_Locations by date from newest to oldest for later processing
    negative_Locations.sort_values(by='Date', ascending = False, inplace=True)    

    #Reindex negative_Locations so the ISO is the index for consistency reasons
    negative_Locations.set_index('ISO', inplace = True)     

    #Print negative_Lcoations dataframe to an excel file for record keeping
    negative_Locations.to_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + dataset_Name + '_Negative_Instances.csv')
                
    return negative_Locations

#This function takes in the dataframe of all outliers, cases, and deaths and corrects the outliers
def remove_Outlier(outlier_df, cases_df, deaths_df):
    #Loops through every row in the outlier dataframe
    for i in range(len(outlier_df)):
        #Checks to see if the dataframe that needs to be corrected it Cases or Deaths
        if outlier_df.iloc[i, 2] == "Cases":
            #Calculates teh new value of the outlier's index and column location the the excess that needs to be redistrubuted
            new_Value, Excess = new_Value_and_Excess(cases_df, outlier_df.index[i], outlier_df.iloc[i, 1])
            #Finds the row and column location of the outlying data point in the cases dataframe
            row = cases_df.index.get_loc(outlier_df.index[i])
            col = cases_df.columns.get_loc(outlier_df.iloc[i, 1])
            #updates the value of the outlying data point
            cases_df.iloc[row, col] = new_Value
            #Updates the previous 30 days of values based on the excess amount proportionally while considering edge cases
            if col > 30:
                range_Total = cases_df.iloc[row, (col-30):col].sum()
                range_Array = cases_df.iloc[row, (col-30):col].div(range_Total)
                range_Array.mul(Excess)
                cases_df.iloc[row, (col-30):col].sub(range_Array)
            else:
                range_Total = cases_df.iloc[row, 0:col].sum()
                range_Array = cases_df.iloc[row, 0:col].div(range_Total)
                range_Array.mul(Excess)
                cases_df.iloc[row, 0:col].sub(range_Array)
        else:
            #Calculates teh new value of the outlier's index and column location the the excess that needs to be redistrubuted
            new_Value, Excess = new_Value_and_Excess(deaths_df, outlier_df.index[i], outlier_df.iloc[i, 1])
            #Finds the row and column location of the outlying data point in the cases dataframe
            row = deaths_df.index.get_loc(outlier_df.index[i])
            col = deaths_df.columns.get_loc(outlier_df.iloc[i, 1])
            #updates the value of the outlying data point
            deaths_df.iloc[row, col] = new_Value
            #Updates the previous 30 days of values based on the excess amount proportionally while considering edge cases
            if col > 30:
                range_Total = deaths_df.iloc[row, (col-30):col].sum()
                range_Array = deaths_df.iloc[row, (col-30):col].div(range_Total)
                range_Array.mul(Excess)
                deaths_df.iloc[row, (col-30):col].sub(range_Array)
            else:
                range_Total = deaths_df.iloc[row, 0:col].sum()
                range_Array = deaths_df.iloc[row, 0:col].div(range_Total)
                range_Array.mul(Excess)
                deaths_df.iloc[row, 0:col].sub(range_Array)
    return

cases_Negatives = find_Negatives(cases_File, "Cases")
deaths_Negatives = find_Negatives(death_File, "Deaths")

# =============================================================================
# fig = plt.figure(constrained_layout = True)
# gs = fig.add_gridspec(1,1)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.plot(cases_File.columns.values, cases_File.loc['FRA', :], marker = '.', ls="", color = 'red', label = 'Outliers')
# ax1.set_title("FRA: Data with Outliers and without Outliers")
# ax1.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
# ax1.xaxis.set_major_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
# =============================================================================



remove_Outlier(Known_Outliers, cases_File, death_File)
remove_Outlier(Manual_Outliers, cases_File, death_File)
remove_Outlier(cases_Negatives, cases_File, death_File)
remove_Outlier(deaths_Negatives, cases_File, death_File)

# =============================================================================
# ax1.plot(cases_File.columns.values, cases_File.loc['FRA', :], marker = '.', ls="", color = 'blue', label = 'Non-Outliers')
# ax1.legend()
# ax1.grid()
# fig.savefig("France Outliers.png")
# =============================================================================
