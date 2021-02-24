# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:37:58 2020

@author: theod
"""
import pandas as pd
import numpy as np
import scipy.interpolate as interp
from scipy.signal import find_peaks, peak_widths, peak_prominences, find_peaks_cwt
import scipy.signal
from scipy import signal
import itertools

#This file create the splines and other interpolation methods that will be used to evaluate the COVID-19 data and derive metrics

#This function finds additional keypoints that should be added to the mins and maxes to imporve the interpolation accuracy
def extra_points_Poly_Reg(keypoints, y_values):
    #crete array to hold additional x-points to add
    x_extra = []
    
    #loop though every kepoint (i, i+1)
    for i in range(0, len(keypoints)-1):
        #determine if the values go up or down
        direction = 0
        if y_values[keypoints[i]] < y_values[keypoints[i+1]]:
            direction = 1
        elif y_values[keypoints[i]] > y_values[keypoints[i+1]]:
            direction = -1
            
        #only perform the analysis if both endpoints are not 0
        if y_values[keypoints[i]] > 0 or y_values[keypoints[i+1]] > 0:
            #create the range of x-values and y-values for the polynomial fitting
            x = np.array(np.arange(keypoints[i], keypoints[i+1]+1), int)
            y = np.array(y_values[x], float)
            #perform the polynomial (cubic) fitting
            model = np.poly1d(np.polyfit(x,y,3))
            #determine the points where the fitted polynomial and the actual data intersect
            idx = np.argwhere(np.diff(np.sign(y-model(x)))).flatten()
            #update the points so they reflect their positon in the real data
            x_loc = idx + keypoints[i]
            
            #create arrays to store the x-points that need to be removed and updata theanalysis range to incllude the endpoints
            x_remove = []
            x_analysis = np.sort(np.append(x_loc, [keypoints[i], keypoints[i+1]]))
            
            #loop though every potential newpoint pair (j, j+1)
            for j in range(0, len(x_analysis)-1):
                #if the point shsould be moving up, remove all points that are smallet than a preceding points
                if direction == 1:
                    x_remove = np.append(x_remove, x_analysis[np.argwhere(y_values[x_analysis[j+1:-1]] <= y_values[x_analysis[j]])+j+1])
                #if the points are supposed to be moving down, remove all points that are larger than a preceding point
                elif direction == -1:
                    x_remove = np.append(x_remove, x_analysis[np.argwhere(y_values[x_analysis[j+1:-1]] >= y_values[x_analysis[j]])+j+1])
            
            #add the new points that should be cosnidered to the x_extra array
            x_extra = np.append(x_extra, np.setdiff1d(x_analysis, x_remove))
    
    #make sure the origional keypoints are also added to the new points            
    x_final = np.array(np.sort(np.unique(np.append(keypoints, x_extra))), int)
    
    return x_final


#This function takes in the moving average data and the keypoints data as input and create the interpolation
def interpolation(smoothed_df, keypoints_df, min_Col, max_Col):
    #Create a dataframe to store the interpolation functions
    interpolation_df = pd.DataFrame(data = None, columns = ['PChip', 'Extra Keypoints', 'Akima', 'Cubic Spline'], index = smoothed_df.index.values)
    
    #For every country, create the interpolate the data based on the keypoints and store the result in the interpolation_df DataFrame
    for i in smoothed_df.index.values:
        #print(i)
        #Determine th x-values for the interpolations
        peaks = keypoints_df.loc[i, max_Col]
        valleys = keypoints_df.loc[i, min_Col]
        
        #organize the x-values by eliminating duplicates and sorting
        x = np.sort(np.unique(np.append(peaks, valleys)))
        y = smoothed_df.loc[i, :].values
        
        #add the first and last point if there are not mins/maxes (i.e. no data)
        if x.size == 0:
            x = np.append(0, len(y)-1)
            interpolation_df.loc[i, 'Extra Keypoints'] = ["None"]
        else:
            x = extra_points_Poly_Reg(x, y)
            interpolation_df.loc[i, 'Extra Keypoints'] = x
            
            
        #Determine the y-values for the interpolations
        y1 = y[x]
        
        #Calculate the intepolations for all methods and all countries
        #PChip
        PcI = interp.PchipInterpolator(x, y1, extrapolate=False)
        
        #Store interpolations in the interpolation_df DataFrame
        interpolation_df.loc[i, 'PChip'] = PcI
        
    return interpolation_df

case_Interpolation = interpolation(cases_smoothed_15_3, cases_Extrema, 'Maxes Final', 'Mins Final')
death_Interpolation = interpolation(deaths_smoothed_15_3, deaths_Extrema, 'Maxes Final', 'Mins Final')

#Make graphs of a steps of the peak finding algorithm if graphing is turned on
Interpolation_Graphing = False

#turn off autoamtic plot generation in spyder
plt.ioff()

#Disable figure limit warning
plt.rcParams.update({'figure.max_open_warning': 0})

if Interpolation_Graphing == True:
    #Make the Graph_File
    graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Interpolation_Graphs' + '.pdf')
    
    for i in cases_smoothed_15_3.index.values:
        print(i)
        #Make Figure
        fig = plt.figure(constrained_layout = True, figsize=(8.5, 11))
        gs = fig.add_gridspec(3,2)
        
        #Initialize Axes
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[0, 1])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[2, 1])
        
        #Perform the extra case interpolation on only the Final Peaks and Valleys
        case_Peaks = cases_Extrema.loc[i, 'Maxes Final']
        case_Valleys = cases_Extrema.loc[i, 'Mins Final']
        x = np.sort(np.unique(np.append(case_Peaks, case_Valleys)))
        y = cases_smoothed_15_3.loc[i, :]
        if x.size == 0:
            x = np.append(0, len(y)-1)
        extra_PCI = interp.PchipInterpolator(x, y[x], extrapolate=False)
        x_range = np.arange(len(cases_smoothed_15_3.columns.values))   
        
        ax1.plot(x_range, extra_PCI(x_range), zorder = 10, label = "PChip w/o Keypoints")
        ax1.plot(x, y[x], marker = '.', ls="", zorder = 15, label = 'Peaks and Valleys')
        ax1.plot(x_range, y, zorder = 5, label = "Smoothed")
        ax1.legend()
        ax1.set_title(i + ": PChip Interpolaton w/o Extra Keypoints")
        ax1.set(xlabel="Cases per Million", ylabel="Time")
        
        ax2.plot(x_range, case_Interpolation.loc[i, 'PChip'](x_range), zorder = 10, label = "PChip w/ Keypoints")
        ax2.plot(case_Interpolation.loc[i, 'Extra Keypoints'], y[case_Interpolation.loc[i, 'Extra Keypoints']], marker = '.', ls='', zorder = 15, label = 'Extra Keypoints')
        ax2.plot(x_range, y, zorder = 5, label = "Smoothed")
        ax2.legend()
        ax2.set_title(i + ": PChip Interpolaton w/ Extra Keypoints")
        ax2.set(ylabel="Cases per Million",xlabel="Time")
        
        ax3.plot(x_range, case_Interpolation.loc[i, 'PChip'](x_range, 1), label = "PChip 1st Derivative")
        ax3.set_title(i + ": PChip 1st Derivative")
        ax3.set(ylabel="Cases per Million per Day",xlabel="Time")
        
        #Perform the extra death interpolation on only the Final Peaks and Valleys
        death_Peaks = deaths_Extrema.loc[i, 'Maxes Final']
        death_Valleys = deaths_Extrema.loc[i, 'Mins Final']
        x1 = np.sort(np.unique(np.append(death_Peaks, death_Valleys)))
        y1 = deaths_smoothed_15_3.loc[i, :]
        if x1.size == 0:
            x1 = np.append(0, len(y)-1)
        extra_PCI = interp.PchipInterpolator(x1, y1[x1], extrapolate=False)
        x1_range = np.arange(len(cases_smoothed_15_3.columns.values)) 
        
        ax4.plot(x1_range, extra_PCI(x1_range), zorder = 10, label = "PChip w/o Keypoints")
        ax4.plot(x1, y1[x1], marker = '.', ls="", zorder = 15, label = 'Peaks and Valleys')
        ax4.plot(x1_range, y1, zorder = 5, label = "Smoothed")
        ax4.legend()
        ax4.set_title(i + ": PChip Interpolaton w/o Extra Keypoints")
        ax4.set(ylabel="Deaths per Million", xlabel="Time")
        
        ax5.plot(x1_range, death_Interpolation.loc[i, 'PChip'](x1_range), zorder = 10, label = "PChip w/ Keypoints")
        if death_Interpolation.loc[i, 'Extra Keypoints'][0] == 'None':
            x_Extra = [0, len(deaths_smoothed_15_3.columns.values)-1]
            ax5.plot(x_Extra, y1[x_Extra], marker = '.', ls='', zorder = 15, label = 'Extra Keypoints')
        else:
            ax5.plot(death_Interpolation.loc[i, 'Extra Keypoints'], y1[death_Interpolation.loc[i, 'Extra Keypoints']], marker = '.', ls='', label = 'Extra Keypoints')
        ax5.plot(x1_range, y1, zorder = 5, label = "Smoothed")
        ax5.legend()
        ax5.set_title(i + ": PChip Interpolaton w/o Extra Keypoints")
        ax5.set(ylabel="Deaths per Million",xlabel="Time")
        
        ax6.plot(x1_range, death_Interpolation.loc[i, 'PChip'](x1_range, 1), label = "PChip 1st Derivative")
        ax6.set_title(i + ": PChip 1st Derivative")
        ax6.set(ylabel="Deaths per Million per Day", xlabel="Time")
        
        #Save the Figure
        graphs_File.savefig(fig)
    
    #Save all the graphs to the PDF
    graphs_File.close()
    

# =============================================================================
# fig = plt.figure(constrained_layout = True)
# gs = fig.add_gridspec(1,1)
# ax1 = fig.add_subplot(gs[0,0])
# ax1.set_title("AFG: Interpolation with and without Extra Keypoints")
# x = np.arange(len(cases_smoothed_15_3.columns.values))
# y = cases_smoothed_15_3.loc['AFG', :]
# case_Peaks = cases_Extrema.loc['AFG', 'Maxes Final']
# case_Valleys = cases_Extrema.loc['AFG', 'Mins Final']
# x_int = np.sort(np.unique(np.append(case_Peaks, case_Valleys)))
# extra_PCI = interp.PchipInterpolator(x_int, y[x_int], extrapolate=False)
# ax1.plot(x, y, label = "Smoothed")
# ax1.plot(x, extra_PCI(x), zorder = 10, label = "PChip w/o Keypoints")
# ax1.plot(x_int, y[x_int], marker = '.', ls="", zorder = 15, label = 'Peaks and Valleys')
# ax1.plot(x, case_Interpolation.loc['AFG', 'PChip'](x), zorder = 10, label = "PChip w/ Keypoints")
# ax1.plot(case_Interpolation.loc['AFG', 'Extra Keypoints'], y[case_Interpolation.loc['AFG', 'Extra Keypoints']], marker = '.', ls='', zorder = 15, label = 'Extra Keypoints')
#         
# 
# ax1.legend()
# ax1.set(xlabel = "Days", ylabel = "Cases per Million")
# ax1.grid()
# ax1.legend()
# fig.savefig("AFG Interpolation.png")
# =============================================================================
