# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:29:48 2020

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
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences
import warnings

#Disable figure limit warning
plt.rcParams.update({'figure.max_open_warning': 0})

#turn off autoamtic plot generation in spyder
plt.ioff()
warnings.filterwarnings("ignore")

#Create the file for the graphs to be saved to
graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/smoothing_Testing.pdf')

#creates a dataframe to store the keypoint data
peak_info = pd.DataFrame(data=None, index = cases_MA.index.values, columns = ['Case Peaks', 'Case Width', 'Death Peaks', 'Death Width'])

#This function takes the list of peaks, widths, and prominence and removes instances based on peak overlapping, peak width, and peak prominence conditions and dependencies
def remove_overlapping_peaks(peaks, peak_widths, peak_prom):
    #This takes the data from the peaks and peak_widths array and turns them into on large matrix
    data = np.append([peaks], [peak_widths[0]], axis=0)
    data = np.append(data, [peak_widths[1]], axis = 0)
    data = np.append(data, [peak_widths[2]], axis = 0)
    data = np.append(data, [peak_widths[3]], axis = 0)
    data = np.append(data, [peak_prom], axis = 0)
    
    #creates an amepty array to store instaces of peaks that are overlapped
    overlapped_data = []
    #loop through every combination of peaks and remove the ones that are overlapped by a larger peak AND that overlapped peak's width or prominence is less than 30% of the larger peaks width and prominence
    for i in range(0, len(data[0])):
        for j in range(0, len(data[0])):
            if (data[3][i] < data[3][j] and data[4][i] > data[4][j]):
                #Only remove if the width of the overlapped peak is less than 30% of the overlapping peak
                if (data[1][i] * .301) > data[1][j]:
                    overlapped_data = np.append(overlapped_data, data[0][j])
                elif ((data[5][i] * .301) > data[5][j]):
                    overlapped_data = np.append(overlapped_data, data[0][j])
                
    #reduce the overlapped array into it's unique parts
    data_to_delete = np.unique(overlapped_data)  
    #create a positonal array to track where overlapped peaks are
    overlap_array = []
    #loop through every the overlapped data and create an array of postions
    for k in range(0, len(data_to_delete)):
        for l in range(0, len(data[0])):
            if data_to_delete[k] == data[0][l]:
                overlap_array = np.append(overlap_array, l)
    
    #determine the width of all peaks and the promince of all peaks
    total_width = np.sum(data[1])
    total_prom = np.sum(data[5])
    
    #create and empty array for removed peaks to go
    too_small_width = []
    #remove peaks that don't account for over 10% of the total peak width AND whose prominence is less than 30% of total prominence
    for m in range(0, len(data[0])):
        if ((data[1][m] / total_width) < .1) and ((data[5][m] / total_prom) < .3):
            too_small_width = np.append(too_small_width, m)
    
    #an arrray that contian peaks that are removed due to overlapping or width length considerations
    position_array = np.unique(np.append(overlap_array, too_small_width))
    
    #return the psotions of peaks that need to be deleted
    return position_array

#Thsi function takes in peak promince and peak width and outputs an array of peaks to be removed due to being too small
def remove_small_peaks(peak_prom, peak_widths):
    total_prom = np.sum(peak_prom)
    total_width = np.sum(peak_widths)
    #print(np.around(peak_prom,3), " ",np.around(peak_prom / total_prom, 2), " ", np.around(peak_widths, 2), " ", np.around(peak_widths / total_width, 2))
    
    position_array = []
    for i in range(0, len(peak_prom)):
        if ((peak_prom[i] / total_prom) < .05 and (peak_widths[i] / total_width) < .2):
            position_array = np.append(position_array, i)
        elif (peak_prom[i] <= 0.005):
            position_array = np.append(position_array, i)
    
    return position_array

#create the graphs of the case and death moving averages, their smoothed equations, and their peask + widths + prominences
for i in cases_MA.index.values:
    print(i)
    #create the graphs
    fig, ax = plt.subplots(2, 1, figsize=(11.5,8))
    fig.suptitle(i + " COVID-19 Data")
    
    #find the initsl peaks and peak widths for the case data
    x = range(0, len(cases_MA.columns.values))
    y = cases_smoothed_15_3.loc[i, 'Smoothed Data']
    case_peaks, _ = find_peaks(y, distance = 45)
    case_peaks_widths = peak_widths(y, case_peaks, rel_height=1)

    #find the peaks that overlap
    case_remove_list = remove_overlapping_peaks(case_peaks, case_peaks_widths, peak_prominences(y, case_peaks)[0])
    
    #delete the peaks that overlap in the peaks and peaks width data
    case_peaks = np.delete(case_peaks, case_remove_list)
    case_peaks_widths = np.delete(case_peaks_widths, case_remove_list, axis = 1)
    
    #calculate peak prominence form the remaining peaks
    case_prom = peak_prominences(y, case_peaks)[0]
    case_contour_heights = y[case_peaks] - case_prom

    #remove the peaks that are too small (incomplete function)    
    case_remove_list2 = remove_small_peaks(case_prom, case_peaks_widths[0])
    
    #Delete the peaks that don't meet the peak prominence requirements
    case_peaks = np.delete(case_peaks, case_remove_list2)
    case_peaks_widths = np.delete(case_peaks_widths, case_remove_list2, axis = 1)
    case_prom = np.delete(case_prom, case_remove_list2)
    case_contour_heights = np.delete(case_contour_heights, case_remove_list2)
    
    #find the intial peaks and peak widths for the death data
    y1 = deaths_smoothed_15_3.loc[i, 'Smoothed Data']
    death_peaks, _ = find_peaks(y1, distance = 45)
    death_peaks_widths = peak_widths(y1, death_peaks, rel_height=1)

    #find the peaks that overlap
    death_remove_list = remove_overlapping_peaks(death_peaks, death_peaks_widths, peak_prominences(y1, death_peaks)[0])
    
    #delte the peaks that are overlaped in the peaks and peak width data
    death_peaks = np.delete(death_peaks, death_remove_list)
    death_peaks_widths = np.delete(death_peaks_widths, death_remove_list, axis = 1)

    #calcualte the death peak priminace of the remining peaks
    death_prom = peak_prominences(y1, death_peaks)[0]
    death_contour_heights = y1[death_peaks] - death_prom
    
    #remove the peaks that are too small (incomplete function)
    death_remove_list2 = remove_small_peaks(death_prom, death_peaks_widths[0])

    #Delete the peaks that don't meet the peak prominence requirements
    death_peaks = np.delete(death_peaks, death_remove_list2)
    death_peaks_widths = np.delete(death_peaks_widths, death_remove_list2, axis = 1)
    death_prom = np.delete(death_prom, death_remove_list2)
    death_contour_heights = np.delete(death_contour_heights, death_remove_list2)

    #Calculate the occurance of the first case
    first_case = 0
    count_case = 0
    while (first_case == 0 and count_case < len(y)):
        first_case = y[count_case]
        count_case = count_case + 1

    #adjust first_case for scenarios where there was no data or the first case occurs on the first date
    if count_case == 1:
        count_case = [0]
    elif count_case == len(y):
        count_case = [0, len(y-1)]
    else:
        count_case = [count_case - 2]
        
    #Calculate the occurance of the first death
    first_death = 0
    count_death = 0
    while (first_death == 0 and count_death < len(y)):
        first_death = y[count_death]
        count_death = count_death + 1

    #adjust first_death for scenarios where there was no data or the first death occurs on the first date
    if count_death == 1:
        count_death = [0]
    elif count_death == len(y1):
        count_death = [0, len(y1-1)]
    else:
        count_death = [count_death - 2]
        
    #Calculate the global maximum for cases and deaths
    case_global_max = np.argmax(y)
    death_global_max = np.argmax(y1)
        
    #store the peak data in the peak_info data frame
    peak_info.loc[i, 'Case Peaks'] = case_peaks
    peak_info.loc[i, 'Case Width'] = case_peaks_widths
    peak_info.loc[i, 'Death Peaks'] = death_peaks
    peak_info.loc[i, 'Death Width'] = death_peaks_widths
    
    #plote the data, peaks, peak prominece, peak width, first case, and global maximum
    ax[0].plot(x, cases_MA.loc[i, :], label = 'MA')
    ax[0].plot(x, cases_smoothed_15_3.loc[i, 'Smoothed Data'], label = "15_3")
    ax[0].plot(case_peaks, y[case_peaks], "x", color = 'black', label = "Keypoint")
    ax[0].hlines(*case_peaks_widths[1:], color = "black", label = "Keypoint Width")
    ax[0].vlines(x=case_peaks, ymin = case_contour_heights, ymax = y[case_peaks], color = 'black', label = "Keypoint Prominence")
    ax[0].plot(count_case, y[count_case], "x", color = 'g', label = "Start")
    ax[0].plot(case_global_max, y[case_global_max], "x", color = 'm', label = "Global Max")
    ax[0].legend()
    ax[0].grid()
    
    ax[1].plot(x, deaths_MA.loc[i, :], label = 'MA')
    ax[1].plot(x, deaths_smoothed_15_3.loc[i, 'Smoothed Data'], label = "15_3")
    ax[1].plot(death_peaks, y1[death_peaks], "x", color = 'black', label = "Keypoint")
    ax[1].hlines(*death_peaks_widths[1:], color = "black", label = "Keypoint Width")
    ax[1].vlines(x=death_peaks, ymin = death_contour_heights, ymax = y1[death_peaks], color = 'black', label = "Keypoint Prominence")
    ax[1].plot(count_death, y1[count_death], "x", color = 'g', label = "Start")
    ax[1].plot(death_global_max, y1[death_global_max], "x", color = 'm', label = "Global Max")
    ax[1].legend()
    ax[1].grid()
    
    graphs_File.savefig(fig)
    
graphs_File.close()