# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:56:27 2020

@author: theod
"""
import pandas as pd
import numpy as np
import scipy.interpolate as interp
import scipy.signal
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import scipy.interpolate as interp

#This function takes in a Dataframe of the moving average data and returns a data frame of Savitzky-Golay smoothed data
def smoothing(df, window_len, poly_order):
    #Creates DataFrame for the smoothed data to stored in
    smoothed_data_df = pd.DataFrame(data = None, columns = df.columns.values, index = df.index.values)
    
    #loops through every row of the input dataframe and smooths the data
    ISO_List = df.index.values
    for i in ISO_List:
        y = df.loc[i, :]
        #Apply filter
        smoothed_y = scipy.signal.savgol_filter(y, window_len, poly_order)
        #Remove zeros
        smoothed_y = np.clip(smoothed_y, a_min = 0, a_max = None)
        smoothed_data_df.loc[i, :] = smoothed_y
        
    return smoothed_data_df

cases_smoothed_15_3 = smoothing(cases_MA, 15, 3)
deaths_smoothed_15_3 = smoothing(deaths_MA, 15, 3)


# =============================================================================
# fig = plt.figure(constrained_layout = True)
# gs = fig.add_gridspec(1,1)
# ax1 = fig.add_subplot(gs[0,0])
# ax1.plot(cases_File.columns.values, cases_File.loc['BEN', :], marker = '.', ls="", label = "Cases")
# ax1.plot(cases_MA.columns.values, cases_MA.loc['BEN', :], label = "7-Day MA")
# ax1.plot(cases_smoothed_15_3.columns.values, cases_smoothed_15_3.loc['BEN', :], label = "S-G Smoothed")
# ax1.set_title("BEN: Smoothed COVID-19 Data")
# ax1.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
# ax1.xaxis.set_major_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
# ax1.grid()
# ax1.legend()
# fig.savefig("BEN Smoothed COVID-19 Data.png")
# =============================================================================


#Make the basic data graphs and tables if it is turned on. This is done at this step becuase all of the data preprocessing has been completed.
Basic_Data_Graphs = True

#Disable figure limit warning
plt.rcParams.update({'figure.max_open_warning': 0})
    
#turn off autoamtic plot generation in spyder
plt.ioff()

if Basic_Data_Graphs == True:
    #Make the Graph_File
    graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Basic_Data_Graphs' + '.pdf')
    
    #Create the master ISO List
    ISO_List = set()
    ISO_List.update(cases_File.index.values)
    ISO_List.update(death_File.index.values)
    ISO_List.update(new_Testing.index.values)
    ISO_List.update(stringency_Index.index.values)
    ISO_List = sorted(list(ISO_List))
    
    for i in ISO_List:
        print(i)
        #Make Figure
        fig = plt.figure(constrained_layout = True, figsize=(8.5, 11))
        gs = fig.add_gridspec(4,1)
        
        #Initialize Axes
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, :])
        ax4 = fig.add_subplot(gs[3, :])
        
        #Make the Cases Graph if there is data for it
        if (i in cases_File.index.values) == True:
            ax1.plot(cases_File.columns.values, cases_File.loc[i, :], marker = '.', ls = "", label = "Cases")
            ax1.plot(cases_MA.columns.values, cases_MA.loc[i, :], label = "7-Day MA")
            ax1.plot(cases_smoothed_15_3.columns.values, cases_smoothed_15_3.loc[i, :], label = "S-G Smoothed")
            ax1.legend()
            ax1.set_title(i + ": Cases vs. Time")
            ax1.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        else:
            ax1.plot()
            ax1.set_title(i + ": Cases vs. Time")
            ax1.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
            ax1.text(0, 0, "No Case Data Available", fontsize = 18, horizontalalignment='center')
        
        #Make the Deaths Graph if there is data for it
        if (i in cases_File.index.values) == True:
            ax2.plot(death_File.columns.values, death_File.loc[i, :], marker = '.', ls = "", label = "Deaths")
            ax2.plot(deaths_MA.columns.values, deaths_MA.loc[i, :], label = "7-Day MA")
            ax2.plot(deaths_smoothed_15_3.columns.values, deaths_smoothed_15_3.loc[i, :], label = "S-G Smoothed")
            ax2.legend()
            ax2.set_title(i + ": Deaths vs. Time")
            ax2.set(xlabel = "Date (Month)", ylabel = "Deaths per Million")
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        else:
            ax2.plot()
            ax2.set_title(i + ": Deaths vs. Time")
            ax2.set(xlabel = "Date (Month)", ylabel = "Deaths per Million")
            ax2.text(0, 0, "No Death Data Available", fontsize = 18, horizontalalignment='center')
        
        #Make the Tests Graph is there is data for it
        if (i in new_Testing.index.values) == True:
            ax3.plot(new_Testing.columns.values, new_Testing.loc[i, :], marker = '.', ls = "", label = "Tests")
            ax3.plot(testing_MA.columns.values, testing_MA.loc[i, :], label = "7-Day MA")
            ax3.legend()
            ax3.set_title(i + ": Tests vs. Time")
            ax3.set(xlabel = "Date (Month)", ylabel = "Tests per Million")
            ax3.xaxis.set_major_locator(mdates.MonthLocator())
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        else:
            ax3.plot()
            ax3.set_title(i + ": Tests vs. Time")
            ax3.set(xlabel = "Date (Month)", ylabel = "Tests per Million")
            ax3.text(0, 0, "No Test Data Available", fontsize = 18, horizontalalignment='center')
        
        #Make the Oxford Index Graph if there is data for it
        if (i in stringency_Index.index.values) == True:
            ax4.plot(stringency_Index.columns.values, stringency_Index.loc[i, :], label = "Stringency")
            ax4.plot(economic_Support_Index.columns.values, economic_Support_Index.loc[i, :], label = "Economic Support")
            ax4.plot(government_Response_Index.columns.values, government_Response_Index.loc[i, :], label = "Government Response")
            ax4.plot(containment_Health_Index.columns.values, containment_Health_Index.loc[i, :], label = "Containment Health")
            ax4.legend()
            ax4.set_title(i + ": Oxford Indices vs. Time")
            ax4.set(xlabel = "Date (Month)", ylabel = "Index Score")
            ax4.xaxis.set_major_locator(mdates.MonthLocator())
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        else: 
            ax4.plot()
            ax4.set_title(i + ": Oxford Indices vs. Time")
            ax4.set(xlabel = "Date (Month)", ylabel = "Index Score")
            ax4.text(0, 0, "No Oxford Index Data Available", fontsize = 18, horizontalalignment='center')
        
        #Save the Figure
        graphs_File.savefig(fig)
    
    #Save all the graphs to the PDF
    graphs_File.close() 