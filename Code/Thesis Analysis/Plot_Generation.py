# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:52:15 2020

@author: theod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import scipy.interpolate as interp
import scipy.signal
from scipy.signal import savgol_filter as svg


#Disable figure limit warning
plt.rcParams.update({'figure.max_open_warning': 0})

#turn off autoamtic plot generation in spyder
plt.ioff()

#fucntion to create the basic plots of COVID cases, deaths, tests, and stringency
def make_Basic_Plots(case_df, death_df, testing_df, case_MA_df, death_MA_df, testing_MA_df, cases_Smoothed_df, deaths_Smoothed_df, government_Response_df, stringency_df, containment_Health_df, economic_Support_df):
    #get the list of DataFrames passed as arguements into the funciton
    arg_List = vars()

    #create a list of all ISOs in all DataFrames soreted in alphabetical order
    ISO_List = set()
    for i, j in arg_List.items():
        ISO_List.update(j.index.values)

    ISO_List = sorted(list(ISO_List))
    
    #Loop throug all possible and make the graphs that are applicable
    #Create Array indicating which lists have the ISO
    df_has_ISO = []
    for i in ISO_List:
        tracker_array = [i]
        for j, k in arg_List.items():
            tracker_array.append(k.index.isin([i]).any())
        df_has_ISO.append(tracker_array)  

    #Create the fole for graphs to be printed to
    graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/data_Graphs.pdf')
    
    #loop through every ISO
    for i in range(len(df_has_ISO)):
        #Keeps track of the graph making process in the console
        percentage = "{:.2%}".format(i/len(df_has_ISO))
        print(df_has_ISO[i][0], " ", percentage)
        
        #Make the figure and set up some formatting for the case, death, testing, and stringency plots
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize = (8.5, 12))
        fig.tight_layout(rect = [0,0,1,.95])
        fig.suptitle(df_has_ISO[i][0] + " COVID-19 Data")
        
        #create different subplots based on data availability
        if df_has_ISO[i][1] == True:
            ax[0].plot(case_df.columns.values, case_df.loc[df_has_ISO[i][0], :], '.', label = "Cases")
            ax[0].plot(case_MA_df.columns.values, case_MA_df.loc[df_has_ISO[i][0], :], label = "7-Day MA")
            ax[0].plot(case_MA_df.columns.values, cases_Smoothed_df.loc[df_has_ISO[i][0], :], label = "Smoothed")
            ax[0].legend()
            ax[0].set(xlabel = "Date (Month)", ylabel = "Cases per Million")
            ax[0].xaxis.set_major_locator(mdates.MonthLocator())
            ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
            ax[0].grid()
        else:
            ax[0].plot()
            ax[0].text(0, 0, "No Data Available", fontsize = 18, horizontalalignment='center')
            
        if df_has_ISO[i][2] == True:
            ax[1].plot(death_df.columns.values, death_df.loc[df_has_ISO[i][0], :], '.', label = "Deaths")
            ax[1].plot(death_MA_df.columns.values, death_MA_df.loc[df_has_ISO[i][0], :], label = '7-Day MA')
            ax[1].plot(death_MA_df.columns.values, deaths_Smoothed_df.loc[df_has_ISO[i][0], :], label = "Smoothed")
            ax[1].legend()
            ax[1].set(xlabel = "Date (Month)", ylabel = "Deaths per Million")
            ax[1].xaxis.set_major_locator(mdates.MonthLocator())
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
            ax[1].grid()
        else:
            ax[1].plot()
            ax[1].text(0, 0, "No Data Available", fontsize = 18, horizontalalignment='center')
            
        if df_has_ISO[i][3] == True:
            ax[2].plot(testing_df.columns.values, testing_df.loc[df_has_ISO[i][0], :],'.', label = "Tests")
            ax[2].plot(testing_MA_df.columns.values, testing_MA_df.loc[df_has_ISO[i][0], :], label = "7-Day MA")
            ax[2].legend()
            ax[2].set(xlabel = "Date (Month)", ylabel = "Tests per Million")
            ax[2].xaxis.set_major_locator(mdates.MonthLocator())
            ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
            ax[2].grid()
        else:
            ax[2].plot()
            ax[2].text(0, 0, "No Data Available", fontsize = 18, horizontalalignment='center')
            
        if df_has_ISO[i][9] == True:
            ax[3].plot(government_Response_df.columns.values, government_Response_df.loc[df_has_ISO[i][0], :], label = "Government Response")
            ax[3].plot(stringency_df.columns.values, stringency_df.loc[df_has_ISO[i][0], :], label = "Stringency")
            ax[3].plot(containment_Health_df.columns.values, containment_Health_df.loc[df_has_ISO[i][0], :], label = "Containment Health")
            ax[3].plot(economic_Support_df.columns.values, economic_Support_df.loc[df_has_ISO[i][0], :], label = "Economic Support")
            ax[3].legend()
            ax[3].set(xlabel = "Date (Month)", ylabel = "Index Score")
            ax[3].xaxis.set_major_locator(mdates.MonthLocator())
            ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
            ax[3].grid()
        else:
            ax[3].plot()
            ax[3].text(0, 0, "No Data Available", fontsize = 18, horizontalalignment='center')
        
        #save the finished figure to file
        graphs_File.savefig(fig)

    #close the file with all the graphs
    graphs_File.close()    
    
    return   

#This function reads in a moving average file and keypoints file, and itnerpolation file and create plots of the interpolation methods and their first derivatives
#PChip Interpolation is the one that was chosen. this function will reveal the other intepolation methods in the event they are desired.
#If this function is used, don't forget to uncomment out the other interpolation methods in the interpolation_Function file
def make_Interpolation_Plots(MA_df, key_points_df, interpolation_df, data_Type):
    #Create the fole for graphs to be printed to
    graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + data_Type + '_interpolation_Graphs.pdf')
    
    #loop through every country in the Moving Average data
    count = 0
    for i in MA_df.index.values:
        #Track the plot generation process
        percentage = "{:.2%}".format(count/len(MA_df.index.values))
        print(i, " ", percentage)
        
        #Make the figure and set up some formatting for the plots
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize = (16, 12))
        fig.tight_layout(rect = [0,0,1,.95])
        fig.suptitle(i + " " + data_Type + " Interpolation Data")
        
        #Create the interpolation plots
        
        x = np.arange(len(MA_df.columns.values))        
        ax[0,0].scatter(x, MA_df.loc[i, :], color = 'orange', marker = '.', label = "7-Day MA")
        ax[0,0].scatter(key_points_df.loc[i, 'rel max'][0], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'rel max'][0]], color = 'blue', marker = 'o', label = "Rel. Max.")
        ax[0,0].scatter(key_points_df.loc[i, 'rel min'][0], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'rel min'][0]], color = 'red', marker = 'o', label = "Rel. Min.")
        ax[0,0].scatter(key_points_df.loc[i, 'First Case'], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'First Case']], color = 'green', marker = 'o', label = "1st Case")
        ax[0,0].plot(x, interpolation_df.loc[i, 'PChip'](x), color = 'blue', label = 'PChip')
        ax[0,0].legend()
        ax[0,0].set(xlabel = "Time", ylabel = "Cases per Million")
        ax[0,0].grid()
        
        ax[0,1].plot(x, interpolation_df.loc[i, 'PChip'](x, 1), color = 'purple', label = 'PChip 1st Derv.')
        ax[0,1].legend()
        ax[0,1].set(xlabel = "Time", ylabel = "Cases per Million per Day")
        ax[0,1].grid()
        
        ax[1,0].scatter(x, MA_df.loc[i, :], color = 'orange', marker = '.', label = "7-Day MA")
        ax[1,0].scatter(key_points_df.loc[i, 'rel max'][0], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'rel max'][0]], color = 'blue', marker = 'o', label = "Rel. Max.")
        ax[1,0].scatter(key_points_df.loc[i, 'rel min'][0], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'rel min'][0]], color = 'red', marker = 'o', label = "Rel. Min.")
        ax[1,0].scatter(key_points_df.loc[i, 'First Case'], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'First Case']], color = 'green', marker = 'o', label = "1st Case")
        ax[1,0].plot(x, interpolation_df.loc[i, 'Akima'](x), color = 'blue', label = 'Akima')
        ax[1,0].legend()
        ax[1,0].set(xlabel = "Time", ylabel = "Cases per Million")
        ax[1,0].grid()
        
        ax[1,1].plot(x, interpolation_df.loc[i, 'Akima'](x, 1), color = 'purple', label = 'Akima 1st Derv.')
        ax[1,1].legend()
        ax[1,1].set(xlabel = "Time", ylabel = "Cases per Million per Day")
        ax[1,1].grid()
        
        ax[2,0].scatter(x, MA_df.loc[i, :], color = 'orange', marker = '.', label = "7-Day MA")
        ax[2,0].scatter(key_points_df.loc[i, 'rel max'][0], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'rel max'][0]], color = 'blue', marker = 'o', label = "Rel. Max.")
        ax[2,0].scatter(key_points_df.loc[i, 'rel min'][0], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'rel min'][0]], color = 'red', marker = 'o', label = "Rel. Min.")
        ax[2,0].scatter(key_points_df.loc[i, 'First Case'], MA_df.loc[i, :].iloc[key_points_df.loc[i, 'First Case']], color = 'green', marker = 'o', label = "1st Case")
        ax[2,0].plot(x, interpolation_df.loc[i, 'Cubic Spline'](x), color = 'blue', label = 'Cubic Spline')
        ax[2,0].legend()
        ax[2,0].set(xlabel = "Time", ylabel = "Cases per Million")
        ax[2,0].grid()
        
        ax[2,1].plot(x, interpolation_df.loc[i, 'Cubic Spline'](x, 1), color = 'purple', label = 'Cubic Spline 1st Derv.')
        ax[2,1].legend()
        ax[2,1].set(xlabel = "Time", ylabel = "Cases per Million per Day")
        ax[2,1].grid()
        
        #save the finished figure to file
        graphs_File.savefig(fig)
        
        #update counter
        count = count + 1
        
    #close the file with all the graphs
    graphs_File.close()
    
    return

#This function makes the PChip interpolation graphs
def make_PChip_Plots(cases_MA, cases_Smoothed, cases_Keypoints, cases_Interp, deaths_MA, deaths_Smoothed, deaths_Keypoints, deaths_Interp, cases_smoothed, deaths_smoothed):
    #Create the fole for graphs to be printed to
    graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'PChip_interpolation_Graphs.pdf')
    
    #loop through every country in the Moving Average data
    count = 0
    for i in cases_MA.index.values:
        #Track the plot generation process
        percentage = "{:.2%}".format(count/len(cases_MA.index.values))
        print(i, " ", percentage)
        
        #Make the figure and set up some formatting for the plots
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (10, 8))
        fig.tight_layout(rect = [0,0,1,.95])
        fig.suptitle(i + " " + " PChip Interpolation Data")
        
        #Create the interpolation plots
        y1 = cases_Smoothed.loc[i, :].values
        y2 = deaths_Smoothed.loc[i, :].values
        x = np.arange(len(cases_MA.columns.values))        
        ax[0,0].plot(x, cases_MA.loc[i, :], label = "7-Day MA")
        ax[0,0].plot(x, y1, label = "Smoothed")
        ax[0,0].scatter(cases_Keypoints.loc[i, 'Maxes Final'], y1[cases_Keypoints.loc[i, 'Maxes Final']], marker = 'x', label = "Peaks")
        ax[0,0].scatter(cases_Keypoints.loc[i, 'Mins Final'], y1[cases_Keypoints.loc[i, 'Mins Final']], marker = 'x', label = "Valleys")
        ax[0,0].plot(x, cases_Interp.loc[i, 'PChip'](x), label = 'PChip')
        
        ax[0,0].legend()
        ax[0,0].set(xlabel = "Time", ylabel = "Cases per Million per Day")
        ax[0,0].grid()
        
        ax[1,0].plot(x, cases_Interp.loc[i, 'PChip'](x, 1), color = 'purple', label = 'PChip 1st Derv.')
        ax[1,0].legend()
        ax[1,0].set(xlabel = "Time", ylabel = "Cases per Million per Day")
        ax[1,0].grid()
        
        ax[0,1].plot(x, deaths_MA.loc[i, :], label = "7-Day MA")
        ax[0,1].plot(x, y2, label = "Smoothed")
        ax[0,1].scatter(deaths_Keypoints.loc[i, 'Maxes Final'], y2[deaths_Keypoints.loc[i, 'Maxes Final']], marker = 'x', label = "Peaks")
        ax[0,1].scatter(deaths_Keypoints.loc[i, 'Mins Final'], y2[deaths_Keypoints.loc[i, 'Mins Final']], marker = 'x', label = "Valleys")
        ax[0,1].plot(x, deaths_Interp.loc[i, 'PChip'](x), label = 'PChip')
        
        ax[0,1].legend()
        ax[0,1].set(xlabel = "Time", ylabel = "Cases per Million per Day")
        ax[0,1].grid()
        
        ax[1,1].plot(x, deaths_Interp.loc[i, 'PChip'](x, 1), color = 'purple', label = 'PChip 1st Derv.')
        ax[1,1].legend()
        ax[1,1].set(xlabel = "Time", ylabel = "Cases per Million per Day")
        ax[1,1].grid()
        
        #save the finished figure to file
        graphs_File.savefig(fig)
        
        #update counter
        count = count + 1
        
    #close the file with all the graphs
    graphs_File.close()    
    
    return

def make_Extrema_Plots(cases_smoothed_df, deaths_smoothed_df, cases_extrema_df, deaths_extrema_df):
    graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Extrema_Testing_Final.pdf')
    for i in cases_extrema_df.index.values:
        x = range(0, len(cases_smoothed_df.columns.values))
        y = cases_smoothed_df.loc[i, :].values
        y1 = deaths_smoothed_df.loc[i, :].values
    
        fig, ax = plt.subplots(nrows=2, ncols=1)
        fig.suptitle(i)
    
        ax[0].plot(x, y, zorder = 0, label = "SVG 15,3")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Min'], y[cases_extrema_df.loc[i, 'Rel Min']], marker = '|', color = 'red', zorder = 5, label = "Rel Min")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Max'], y[cases_extrema_df.loc[i, 'Rel Max']], marker = '|', color = 'green', zorder = 5, label = "Rel Max")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Max 2'], y[cases_extrema_df.loc[i, 'Rel Max 2']], marker = 'x', color = 'black', zorder = 10, label = "Rel Max 2")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Min 2'], y[cases_extrema_df.loc[i, 'Rel Min 2']], marker = 'x', color = 'gray', zorder = 10, label = "Rel Min 2")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Max 3'], y[cases_extrema_df.loc[i, 'Rel Max 3']], marker = '+', color = 'orange', zorder = 15, label = "Rel Max 3")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Min 3'], y[cases_extrema_df.loc[i, 'Rel Min 3']], marker = '+', color = 'indigo', zorder = 15, label = "Rel Min 3")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Max 4'], y[cases_extrema_df.loc[i, 'Rel Max 4']], marker = 'o', color = 'cyan', zorder = 20, label = "Rel Max 4")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Min 4'], y[cases_extrema_df.loc[i, 'Rel Min 4']], marker = 'o', color = 'magenta', zorder = 20, label = "Rel Min 4")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Max 5'], y[cases_extrema_df.loc[i, 'Rel Max 5']], marker = 'o', color = 'lime', zorder = 25, label = "Rel Max 5")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Min 5'], y[cases_extrema_df.loc[i, 'Rel Min 5']], marker = 'o', color = 'darkgoldenrod', zorder = 25, label = "Rel Min 5")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Max 6'], y[cases_extrema_df.loc[i, 'Rel Max 6']], marker = 'X', color = 'black', zorder = 30, label = "Rel Max 6")
        ax[0].scatter(cases_extrema_df.loc[i, 'Rel Min 6'], y[cases_extrema_df.loc[i, 'Rel Min 6']], marker = 'X', color = 'dimgrey', zorder = 30, label = "Rel Min 6")
        ax[0].plot(np.sort(np.append(cases_extrema_df.loc[i, 'Rel Max F'],cases_extrema_df.loc[i, 'Rel Min F'])), y[np.sort(np.append(cases_extrema_df.loc[i, 'Rel Max F'],cases_extrema_df.loc[i, 'Rel Min F']))], zorder = 35, label = "Rel Max Final")
    
        #ax[0].legend()
        
        ax[1].plot(x, y1, zorder = 0, label = "SVG 15,3")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Min'], y1[deaths_extrema_df.loc[i, 'Rel Min']], marker = '|', color = 'red', zorder = 5, label = "Rel Min")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Max'], y1[deaths_extrema_df.loc[i, 'Rel Max']], marker = '|', color = 'green', zorder = 5, label = "Rel Max")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Max 2'], y1[deaths_extrema_df.loc[i, 'Rel Max 2']], marker = 'x', color = 'black', zorder = 10, label = "Rel Max 2")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Min 2'], y1[deaths_extrema_df.loc[i, 'Rel Min 2']], marker = 'x', color = 'gray', zorder = 10, label = "Rel Min 2")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Max 3'], y1[deaths_extrema_df.loc[i, 'Rel Max 3']], marker = '+', color = 'orange', zorder = 15, label = "Rel Max 3")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Min 3'], y1[deaths_extrema_df.loc[i, 'Rel Min 3']], marker = '+', color = 'indigo', zorder = 15, label = "Rel Min 3")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Max 4'], y1[deaths_extrema_df.loc[i, 'Rel Max 4']], marker = 'o', color = 'cyan', zorder = 20, label = "Rel Max 4")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Min 4'], y1[deaths_extrema_df.loc[i, 'Rel Min 4']], marker = 'o', color = 'magenta', zorder = 20, label = "Rel Min 4")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Max 5'], y1[deaths_extrema_df.loc[i, 'Rel Max 5']], marker = 'o', color = 'lime', zorder = 25, label = "Rel Max 5")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Min 5'], y1[deaths_extrema_df.loc[i, 'Rel Min 5']], marker = 'o', color = 'darkgoldenrod', zorder = 25, label = "Rel Min 5")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Max 6'], y1[deaths_extrema_df.loc[i, 'Rel Max 6']], marker = 'X', color = 'black', zorder = 30, label = "Rel Max 6")
        ax[1].scatter(deaths_extrema_df.loc[i, 'Rel Min 6'], y1[deaths_extrema_df.loc[i, 'Rel Min 6']], marker = 'X', color = 'dimgrey', zorder = 30, label = "Rel Min 6")
        ax[1].plot(np.sort(np.append(deaths_extrema_df.loc[i, 'Rel Max F'],deaths_extrema_df.loc[i, 'Rel Min F'])), y1[np.sort(np.append(deaths_extrema_df.loc[i, 'Rel Max F'],deaths_extrema_df.loc[i, 'Rel Min F']))], zorder = 35, label = "Rel Max Final")
    
        #ax[1].legend()
    
        graphs_File.savefig(fig)
    
    graphs_File.close() 
        
    return

#make_Basic_Plots(cases_File, death_File, new_Testing, cases_MA, deaths_MA, testing_MA, cases_smoothed_15_3, deaths_smoothed_15_3, government_Response_Index, stringency_Index, containment_Health_Index, economic_Support_Index)
#make_PChip_Plots(cases_MA, cases_smoothed_15_3, cases_Extrema, case_Interpolation, deaths_MA, deaths_smoothed_15_3, deaths_Extrema, death_Interpolation, cases_smoothed_15_3, deaths_smoothed_15_3)
#make_Extrema_Plots(cases_smoothed_15_3, deaths_smoothed_15_3, cases_Extrema, deaths_Extrema)

#These are old functions that plot the comparrison between canidate iterpolation methods
# =============================================================================
# make_Interpolation_Plots(cases_MA, cases_key_points, case_Interpolation, "Cases")
# make_Interpolation_Plots(deaths_MA, deaths_key_points, death_Interpolation, "Deaths")
# =============================================================================
