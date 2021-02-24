# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:46:13 2020

@author: theod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

#Disable figure limit warning
plt.rcParams.update({'figure.max_open_warning': 0})

#turn off autoamtic plot generation in spyder
plt.ioff()

#This function calculates the inital peaks by finding points that completely dominate over their range
def initial_Extrema_Detection(smoothed_df, extrema_df, win_Length, max_Col):
    if win_Length % 2 != 0:
        print("Window Length Must Be Even")
        return
    
    for i in extrema_df.index.values:
        #select the y-data that needs to be used for the initial detection algoritm
        y = smoothed_df.loc[i, :].values
        
        win_Side_Len = int(win_Length / 2)
        
        #create arrays to store the calculated rel maxes
        rel_max = []
        for j in range(0, len(y)):
            #intialize the max_value variable
            max_value = 0
            #if the left-hand window extends beyond the first point, cut the left-hand window at the first point
            if j < win_Side_Len:
                max_value = np.amax(y[0:j + win_Side_Len])
            #if the right-hand window extends beyond the last point, cut the right-hand window at the last point
            elif len(y) - j < win_Side_Len:
                max_value = np.amax(y[j - win_Side_Len:len(y)])
            #look at the points on both sides of j in accorandence with the window length
            else:
                max_value = np.amax(y[j - win_Side_Len:j + win_Side_Len])
                    
            #if point j represents a maximum in its window, add it to the list
            if y[j] == max_value and max_value > 0.00001:
                rel_max = np.append(rel_max, j)

        #Handle plateaus by only keeping the middle point
        plateau_points = []
        for k in range(0, len(rel_max)-1):
            #if two points in the rel_max array are consecutive...
            if rel_max[k] == rel_max[k+1] - 1:
                #if we are at the second to last point, the last two points are plateau points
                if k+2 == len(rel_max):
                    n = 1
                    plateau_points = np.append(plateau_points, rel_max[k:k+n])
                #otherwise...
                else:
                    n=0
                    #keep looping until you hit the end or the plateau ends
                    while k+n < len(rel_max) and rel_max[k+n] == rel_max[k+n+1] - 1:
                        plateau_points = np.append(plateau_points, rel_max[k+n])
                        n = n + 1
            #if two points are too close to eachother and the same value, keep the first one
            elif rel_max[k+1] - rel_max[k] < 15 and y[int(rel_max[k])] == y[int(rel_max[k+1])]:
                plateau_points = np.append(plateau_points, rel_max[k+1])
                        
        #remove the plateaus from the rel_maxes
        rel_max = np.setdiff1d(rel_max, plateau_points)
                
        #add these points (all possible mins and maxes) the the dataframe           
        extrema_df.loc[i, max_Col] = np.sort(np.unique(np.array(rel_max).astype(int)))
    
    return

#this function calculates the mins by taking the calulated maxes and finding the absolute minimums between them
def Min_Calculation(smoothed_df, extrema_df, max_Col, min_Col):
    for i in extrema_df.index.values:
        y = smoothed_df.loc[i, :].values
        x = extrema_df.loc[i, max_Col]
        
        min_points = []
        for j in range(0, len(x)):
            if j == 0 and x[j] == 0:
                y_slice = y[x[j]:x[j+1]]
                y_min_point = np.argmin(y_slice)
                min_points = np.append(min_points, y_min_point)
            elif j == 0 and x[j] != 0 and len(x) > 1:
                y_slice_behind = y[0:x[j]]
                y_reverse = y_slice_behind[::-1]
                y_min_point_behind = len(y_reverse) - np.argmin(y_reverse) - 1
                y_slice_ahead = y[x[j]:x[j+1]]
                y_min_point_ahead = np.argmin(y_slice_ahead) + x[j]
                min_points = np.append(min_points, [y_min_point_behind, y_min_point_ahead])
            elif j == len(x) -1 and x[j] == len(y) - 1 and len(x) > 1:
                y_slice = y[x[j-1]:x[j]]
                y_reverse = y_slice[::-1]
                y_min_point = len(y_reverse) - np.argmin(y_reverse) - 1 + x[j-1]
                min_points = np.append(min_points, y_min_point)
            elif j == len(x) -1 and x[j] != len(y) - 1 and len(x) > 1:
                y_slice_behind = y[x[j-1]:x[j]]
                y_reverse = y_slice_behind[::-1]
                y_min_point_behind = len(y_reverse) - np.argmin(y_reverse) - 1 + x[j-1]
                y_slice_ahead = y[x[j]:len(y)-1]
                y_min_point_ahead = np.argmin(y_slice_ahead) + x[j] + 1
                min_points = np.append(min_points, [y_min_point_behind, y_min_point_ahead])
            elif len(x) == 1 and x[j] != len(y)-1:
                y_slice_behind = y[0:x[j]]
                y_reverse = y_slice_behind[::-1]
                y_min_point_behind = len(y_reverse) - np.argmin(y_reverse) - 1
                y_slice_ahead = y[x[j]:len(y)-1]
                y_min_point_ahead = np.argmin(y_slice_ahead) + x[j] + 1
                min_points = np.append(min_points, [y_min_point_behind, y_min_point_ahead])
            elif len(x) == 1 and x[j] == len(y)-1:
                y_slice = y[0:x[j]]
                y_reverse = y_slice[::-1]
                y_min_point = len(y_reverse) - np.argmin(y_reverse) - 1
                min_points = np.append(min_points, y_min_point)
            else:
                y_slice_behind = y[x[j-1]:x[j]]
                y_reverse = y_slice_behind[::-1]
                y_min_point_behind = len(y_reverse) - np.argmin(y_reverse) + x[j-1] - 1
                y_slice_ahead = y[x[j]:x[j+1]]
                y_min_point_ahead = np.argmin(y_slice_ahead) + x[j]
                min_points = np.append(min_points, [y_min_point_behind, y_min_point_ahead])
        
        extrema_df.loc[i, min_Col] = np.sort(np.unique(np.array(min_points).astype(int)))
    
    return

#this function calculates all maxes height, width, relaitve height, and distnace from mins for use in the pruning algorithm
def pruning_data_calculation(i, smoothed_df, extrema_df, min_Col, max_Col):
    #Create a 2d array of all rel_min labeled as mins
    rel_mins = extrema_df.loc[i, min_Col]
    rel_mins = np.vstack((rel_mins, [0] * len(rel_mins)))
    #create a 2d array of all rel_max labeld as max
    rel_maxes = extrema_df.loc[i, max_Col]
    rel_maxes = np.vstack((rel_maxes, [1] * len(rel_maxes)))
    #Combine the rel min and rel max arrays and sort them by x-point
    all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
    all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
    all_extrema = all_extrema.astype(int)    
    
    y = smoothed_df.loc[i, :].values
    data = np.zeros((9, rel_maxes.shape[1]))
    
    #loop through all the extrema and catalog the data for the rel_maxes
    n=0
    for j in range(0, all_extrema.shape[1]):
        #calculate if the first extrema is a max
        if j == 0 and all_extrema[1][j] == 1:
            width = (all_extrema[0][j+1] - all_extrema[0][j]) * 2
            height = y[all_extrema[0][j]]
            ht_above_min = y[all_extrema[0][j]] - y[all_extrema[0][j+1]] 
            dist_from_min = all_extrema[0][j+1] - all_extrema[0][j]
            data[0][n] = all_extrema[0][j]
            data[1][n] = width
            data[3][n] = height
            data[5][n] = ht_above_min
            data[7][n] = dist_from_min
            n=n+1
        #calculate if the last extrema is a max
        elif j == all_extrema.shape[1] - 1 and all_extrema[1][j] == 1:
            width = (all_extrema[0][j] - all_extrema[0][j-1]) * 2
            height = y[all_extrema[0][j]]
            ht_above_min = y[all_extrema[0][j]] - y[all_extrema[0][j-1]] 
            dist_from_min = all_extrema[0][j] - all_extrema[0][j-1]
            data[0][n] = all_extrema[0][j]
            data[1][n] = width
            data[3][n] = height
            data[5][n] = ht_above_min
            data[7][n] = dist_from_min
            n=n+1
        #calculate if the extrema is a max and not an endpoint
        elif all_extrema[1][j] == 1:
            width = (all_extrema[0][j] - all_extrema[0][j-1]) + (all_extrema[0][j+1] - all_extrema[0][j])
            height = y[all_extrema[0][j]]
            ht_above_min = (y[all_extrema[0][j]] - y[all_extrema[0][j+1]] + y[all_extrema[0][j]] - y[all_extrema[0][j-1]]) / 2
            dist_from_min = min(all_extrema[0][j+1] - all_extrema[0][j], all_extrema[0][j] - all_extrema[0][j-1])
            data[0][n] = all_extrema[0][j]
            data[1][n] = width
            data[3][n] = height
            data[5][n] = ht_above_min
            data[7][n] = dist_from_min
            n=n+1
    
    #Add calculated metrics (%) to the matrix
    total_width = np.sum(data[1])
    total_height = np.sum(data[3])
    total_ht_above_min = np.sum(data[5])
    total_dist_from_min = np.sum(data[7])
    for j in range(0, data.shape[1]):
        data[2][j] = data[1][j] / total_width
        data[4][j] = data[3][j] / total_height
        data[6][j] = data[5][j] / total_ht_above_min
        data[8][j] = data[7][j] / total_dist_from_min
        
    return data

#this funciton prunes peaks base don their widht, height, and realtive height
def rel_max_pruning(smoothed_df, extrema_df, min_Col, max_Col, new_max_Col, too_thin_width, too_thin_width_per, too_thin_ht_per, too_short_per, too_short_width_per, ht_above_min_too_short, ht_above_min_too_short_width_per, first_point_max_ht_per):
    for i in extrema_df.index.values:
        max_data = pruning_data_calculation(i, smoothed_df, extrema_df, min_Col, max_Col)
        
        maxes_to_prune = []
        for j in range(0, max_data.shape[1]):
            #Prune if the width is too thin...
            if max_data[1][j] < too_thin_width and max_data[2][j] < too_thin_width_per:
                #AND the height is too short ANd it's not the first rel max
                if max_data[4][j] < too_thin_ht_per and j != 0:
                    maxes_to_prune = np.append(maxes_to_prune, max_data[0][j])
            #Prune if the width is too short...
            if max_data[4][j] < too_short_per:
                #AND the width is too thin
                if max_data[2][j] < too_short_width_per:
                    maxes_to_prune = np.append(maxes_to_prune, max_data[0][j])
            #Prune if minimum height above min is too short...
            if max_data[6][j] < ht_above_min_too_short:
                #AND the width to too thin AND it is not the first rel_max
                if max_data[2][j] < ht_above_min_too_short_width_per and j != 0:
                    maxes_to_prune = np.append(maxes_to_prune, max_data[0][j])
            #Prune if the very first point is a rel_max and the height is too short
            if max_data[0][j] == 0 and max_data[4][j] < first_point_max_ht_per:
                maxes_to_prune = np.append(maxes_to_prune, max_data[0][j])
                
        maxes_to_prune = np.sort(np.unique(maxes_to_prune))

        extrema_df.loc[i, new_max_Col] = np.setdiff1d(extrema_df.loc[i, max_Col], maxes_to_prune)
    
    return

#this function remove the last mins and maxes. it finds maxes that don't dominate over their range and remove them and neighboring mins accordingly
def final_Pruning(smoothed_df, extrema_df, min_Col, max_Col, new_Min_Col, new_Max_Col, threshold, firstPt_threshold):
    #loop though all countires
    for i in extrema_df.index.values:
        #Create a 2d array of all rel_min labeled as mins
        rel_mins = extrema_df.loc[i, min_Col]
        rel_mins = np.vstack((rel_mins, [0] * len(rel_mins)))
        #create a 2d array of all rel_max labeld as max
        rel_maxes = extrema_df.loc[i, max_Col]
        rel_maxes = np.vstack((rel_maxes, [1] * len(rel_maxes)))
        #Combine the rel min and rel max arrays and sort them by x-point
        all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
        all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
        all_extrema = all_extrema.astype(int) 
        
        #initalize the y-data
        y = smoothed_df.loc[i, :].values
        
        #create a list of maxes that don't dominate and the points that do dominate over the maxes range
        max_to_remove = []
        for j in range(0, all_extrema.shape[1]):
            #if the first point in all the data is a max...
            if all_extrema[1][j] == 1 and j == 0 and all_extrema[0][j] == 0:
                #find the maximum accross the range
                range_max = np.amax(y[0:all_extrema[0][j+1]])
                #if the max doesn't dominate and falls below a threshold, remove it
                if y[all_extrema[0][j]] < range_max and range_max / y[all_extrema[0][j]] > threshold:
                    #check if the point is the first rel max
                    if all_extrema[0][j] == rel_maxes[0][0] and range_max / y[all_extrema[0][j]] > firstPt_threshold:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [np.argmax(y[0:all_extrema[0][j+1]])]])
                    elif all_extrema[0][j] != rel_maxes[0][0]:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [np.argmax(y[0:all_extrema[0][j+1]])]])
            #if the extrema is a maximum...
            elif all_extrema[1][j] == 1 and j == 0:
                #find the maximum accross the range
                range_max = np.amax(y[all_extrema[0][j]:all_extrema[0][j+1]])
                #if the max doesn't dominate and falls below a threshold, remove it
                if y[all_extrema[0][j]] < range_max and range_max / y[all_extrema[0][j]] > threshold:
                    #check if the point is the first rel max
                    if all_extrema[0][j] == rel_maxes[0][0] and range_max / y[all_extrema[0][j]] > firstPt_threshold:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j] + np.argmax(y[all_extrema[0][j]:all_extrema[0][j+1]])]])
                    elif all_extrema[0][j] != rel_maxes[0][0]:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j] + np.argmax(y[all_extrema[0][j]:all_extrema[0][j+1]])]])
            #if the last point in all the data is a max...
            elif all_extrema[1][j] == 1 and j == all_extrema.shape[1] - 1 and all_extrema[0][j] == len(y)-1:
                #find the maximum accross the range
                range_max = np.amax(y[all_extrema[0][j-1]:len(y)-1])
                #if the max doesn't dominate and falls below a threshold, remove it
                if y[all_extrema[0][j]] < range_max and range_max / y[all_extrema[0][j]] > threshold:
                    #check if the point is the first rel max
                    if all_extrema[0][j] == rel_maxes[0][0] and range_max / y[all_extrema[0][j]] > firstPt_threshold:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j-1] + np.argmax(y[all_extrema[0][j-1]:len(y)-1])]])
                    elif all_extrema[0][j] != rel_maxes[0][0]:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j-1] + np.argmax(y[all_extrema[0][j-1]:len(y)-1])]])
            #if the last extrema is a max...
            elif all_extrema[1][j] == 1 and j == all_extrema.shape[1] - 1:
                #find the maximum accross the range
                range_max = np.amax(y[all_extrema[0][j-1]:len(y)-1])
                #if the max doesn't dominate and falls below a threshold, remove it
                if y[all_extrema[0][j]] < range_max and range_max / y[all_extrema[0][j]] > threshold:
                    #check if the point is the first rel max
                    if all_extrema[0][j] == rel_maxes[0][0] and range_max / y[all_extrema[0][j]] > firstPt_threshold:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j-1] + np.argmax(y[all_extrema[0][j-1]:len(y)-1])]])
                    elif all_extrema[0][j] != rel_maxes[0][0]:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j-1] + np.argmax(y[all_extrema[0][j-1]:len(y)-1])]])
            #if a point is a max...
            elif all_extrema[1][j] == 1:
                #find the maximum accross the range
                range_max = np.amax(y[all_extrema[0][j-1]:all_extrema[0][j+1]])
                #if the max doesn't dominate and falls below a threshold, remove it
                if y[all_extrema[0][j]] < range_max and range_max / y[all_extrema[0][j]] > threshold:
                    #check if the point is the first rel max
                    if all_extrema[0][j] == rel_maxes[0][0] and range_max / y[all_extrema[0][j]] > firstPt_threshold:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j-1] + np.argmax(y[all_extrema[0][j-1]:all_extrema[0][j+1]])]])
                    elif all_extrema[0][j] != rel_maxes[0][0]:
                        max_to_remove = np.append(max_to_remove, [[all_extrema[0][j]], [all_extrema[0][j-1] + np.argmax(y[all_extrema[0][j-1]:all_extrema[0][j+1]])]])
        
        #Determine how the rel_maxes and mrel_mins need to be removed
        mins_to_remove = []
        maxes_to_remove = []
        if len(max_to_remove) > 0:
            for k in range(0, len(max_to_remove), 2):
                #always remove the max
                maxes_to_remove = np.append(maxes_to_remove, max_to_remove[k])
                #find the location ins all_extrema where the max_to_remove is located
                n = np.where(all_extrema[0] == max_to_remove[k])[0]
                #If the max to remove is smaller than the next max to remove...
                if max_to_remove[k] < max_to_remove[k+1]:
                    #if we are out of bounds, do nothing
                    if k+n+2 >= len(maxes_to_remove):
                        pass
                    #If the nex rel max is 0, add the nest two mins for removal
                    elif all_extrema[1][k+n+2] == 0:
                        mins_to_remove = np.append(mins_to_remove, [all_extrema[0][k+n+1], all_extrema[0][k+n+2]])
                    #otherwise, just remove the next min
                    else:
                        mins_to_remove = np.append(mins_to_remove, all_extrema[0][k+n+1])
                #If the next max is less than the current max...
                elif max_to_remove[k] > max_to_remove[k+1]:
                    #if the prior two rel mins are zero, remove both rel mins
                    if all_extrema[1][k+n-2] == 0:
                        mins_to_remove = np.append(mins_to_remove, [all_extrema[0][k+n-1], all_extrema[0][k+n-2]])
                    else:
                    #othersie jsut remove the rpviosu rel min
                        mins_to_remove = np.append(mins_to_remove, all_extrema[0][k+n-1])
        
        extrema_df.loc[i, new_Min_Col] = np.setdiff1d(extrema_df.loc[i, min_Col], np.sort(np.unique(np.array(mins_to_remove).astype(int))))
        extrema_df.loc[i, new_Max_Col] = np.setdiff1d(extrema_df.loc[i, max_Col], np.sort(np.unique(np.array(maxes_to_remove).astype(int))))

    return

cases_Extrema = pd.DataFrame(data = None, index = cases_smoothed_15_3.index.values, columns = ['All Rel Maxes', 'Initial Mins', 'Rel Maxes 2', 'Mins 2', 'Maxes Final', 'Mins Final'])
deaths_Extrema = pd.DataFrame(data = None, index = deaths_smoothed_15_3.index.values, columns = ['All Rel Maxes', 'Initial Mins', 'Rel Maxes 2', 'Mins 2', 'Maxes Final', 'Mins Final'])

initial_Extrema_Detection(cases_smoothed_15_3, cases_Extrema, 50, 'All Rel Maxes')
initial_Extrema_Detection(deaths_smoothed_15_3, deaths_Extrema, 50, 'All Rel Maxes')

Min_Calculation(cases_smoothed_15_3, cases_Extrema, 'All Rel Maxes', 'Initial Mins')
Min_Calculation(deaths_smoothed_15_3, deaths_Extrema, 'All Rel Maxes', 'Initial Mins')

rel_max_pruning(cases_smoothed_15_3, cases_Extrema, 'Initial Mins', 'All Rel Maxes', 'Rel Maxes 2', 30, .25, .1, .15, .1, .05, .2, .05)
Min_Calculation(cases_smoothed_15_3, cases_Extrema, 'Rel Maxes 2', 'Mins 2')

rel_max_pruning(deaths_smoothed_15_3, deaths_Extrema, 'Initial Mins', 'All Rel Maxes', 'Rel Maxes 2', 30, .25, .1, .15, .1, .05, .2, .05)
Min_Calculation(deaths_smoothed_15_3, deaths_Extrema, 'Rel Maxes 2', 'Mins 2')

final_Pruning(cases_smoothed_15_3, cases_Extrema, 'Mins 2', 'Rel Maxes 2', 'Mins Final', 'Maxes Final', 1.4, 2)
final_Pruning(deaths_smoothed_15_3, deaths_Extrema, 'Mins 2', 'Rel Maxes 2', 'Mins Final', 'Maxes Final', 1.2, 2)

#Make graphs of a steps of the peak finding algorithm if graphing is turned on
Peak_Finding_Graphing = False

#turn off autoamtic plot generation in spyder
plt.ioff()

#Disable figure limit warning
plt.rcParams.update({'figure.max_open_warning': 0})

if Peak_Finding_Graphing == True:
    #Make the Graph_File
    graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Peak_Finding_Graphs' + '.pdf')
    
    for i in cases_Extrema.index.values:
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
        
        #set the inital x and y variables
        x = cases_smoothed_15_3.columns.values
        y = cases_smoothed_15_3.loc[i, :]
        x1 = deaths_smoothed_15_3.columns.values
        y1 = deaths_smoothed_15_3.loc[i, :]
        
        #Make the Initial Peaks Graph (Cases)
        ax1.plot(x, y, label = "Smoothed")
        ax1.plot(x[cases_Extrema.loc[i, 'All Rel Maxes']], y[cases_Extrema.loc[i, 'All Rel Maxes']], marker = 'o', color = "red", ls = "", label = "Initial Peaks")
        ax1.plot(x[cases_Extrema.loc[i, 'Initial Mins']], y[cases_Extrema.loc[i, 'Initial Mins']], marker = 'o', color = 'lightcoral', ls = "", label = "Initial Valleys")
        ax1.legend()
        ax1.set_title(i + ": Initial Peak Finding Algorithm")
        ax1.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        
        #Make the Pruned Peaks Graph (Cases)
        ax2.plot(x, y, label = "Smoothed")
        ax2.plot(x[cases_Extrema.loc[i, 'Rel Maxes 2']], y[cases_Extrema.loc[i, 'Rel Maxes 2']], marker = 'o', color = 'green', ls = "", label = "Pruned Peaks")
        ax2.plot(x[cases_Extrema.loc[i, 'Mins 2']], y[cases_Extrema.loc[i, 'Mins 2']], marker = 'o', color = 'lightgreen', ls = "", label = "Pruned Valleys")
        ax2.legend()
        ax2.set_title(i + ": Peak Pruning Algorithm")
        ax2.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        
        #Make the Final Peaks Graph (Cases)
        ax3.plot(x, y, label = "Smoothed")
        ax3.plot(x[cases_Extrema.loc[i, 'Maxes Final']], y[cases_Extrema.loc[i, 'Maxes Final']], marker = 'o', color = 'orange', ls = "", label = "Final Peaks")
        ax3.plot(x[cases_Extrema.loc[i, 'Mins Final']], y[cases_Extrema.loc[i, 'Mins Final']], marker = 'o', color = 'navajowhite', ls = "", label = "Final Valleys")
        ax3.legend()
        ax3.set_title(i + ": Final Peak Finding Algorithm")
        ax3.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        
        #Make the Initial Peaks Graph (Deaths)
        ax4.plot(x1, y1, label = "Smoothed")
        ax4.plot(x1[deaths_Extrema.loc[i, 'All Rel Maxes']], y1[deaths_Extrema.loc[i, 'All Rel Maxes']], marker = 'o', color = "red", ls = "", label = "Initial Peaks")
        ax4.plot(x1[deaths_Extrema.loc[i, 'Initial Mins']], y1[deaths_Extrema.loc[i, 'Initial Mins']], marker = 'o', color = 'lightcoral', ls = "", label = "Initial Valleys")
        ax4.legend()
        ax4.set_title(i + ": Initial Peak Finding Algorithm")
        ax4.set(xlabel = "Date (Month)", ylabel = "Deaths per Million")
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        
        #Make the Pruned Peaks Graph (Deaths)
        ax5.plot(x1, y1, label = "Smoothed")
        ax5.plot(x1[deaths_Extrema.loc[i, 'Rel Maxes 2']], y1[deaths_Extrema.loc[i, 'Rel Maxes 2']], marker = 'o', color = 'green', ls = "", label = "Pruned Peaks")
        ax5.plot(x1[deaths_Extrema.loc[i, 'Mins 2']], y1[deaths_Extrema.loc[i, 'Mins 2']], marker = 'o', color = 'lightgreen', ls = "", label = "Pruned Valleys")
        ax5.legend()
        ax5.set_title(i + ": Peak Pruning Algorithm")
        ax5.set(xlabel = "Date (Month)", ylabel = "Deaths per Million")
        ax5.xaxis.set_major_locator(mdates.MonthLocator())
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        
        #Make the Final Peaks Graph (Deaths)
        ax6.plot(x1, y1, label = "Smoothed")
        ax6.plot(x1[deaths_Extrema.loc[i, 'Maxes Final']], y1[deaths_Extrema.loc[i, 'Maxes Final']], marker = 'o', color = 'orange', ls = "", label = "Final Peaks")
        ax6.plot(x1[deaths_Extrema.loc[i, 'Mins Final']], y1[deaths_Extrema.loc[i, 'Mins Final']], marker = 'o', color = 'navajowhite', ls = "", label = "Final Valleys")
        ax6.legend()
        ax6.set_title(i + ": Final Peak Finding Algorithm")
        ax6.set(xlabel = "Date (Month)", ylabel = "Deaths per Million")
        ax6.xaxis.set_major_locator(mdates.MonthLocator())
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        
        #Save the Figure
        graphs_File.savefig(fig)
    
    #Save all the graphs to the PDF
    graphs_File.close() 
    

# =============================================================================
# fig = plt.figure(constrained_layout = True)
# gs = fig.add_gridspec(1,1)
# ax1 = fig.add_subplot(gs[0,0])
# x = cases_smoothed_15_3.columns.values
# y = cases_smoothed_15_3.loc['CMR', :]
# ax1.set_title("CMR: Final Pruning")
# ax1.plot(x, y, label = "Smoothed")
# ax1.plot(x[cases_Extrema.loc['CMR', 'All Rel Maxes']], y[cases_Extrema.loc['CMR', 'All Rel Maxes']], marker = 'o', color = "dimgrey", ls = "", label = "Initial Peaks")
# ax1.plot(x[cases_Extrema.loc['CMR', 'Initial Mins']], y[cases_Extrema.loc['CMR', 'Initial Mins']], marker = 'o', color = 'darkgrey', ls = "", label = "Initial Valleys")
# ax1.plot(x[cases_Extrema.loc['CMR', 'Rel Maxes 2']], y[cases_Extrema.loc['CMR', 'Rel Maxes 2']], marker = 'o', color = "grey", ls = "", label = "Pruned Peaks")
# ax1.plot(x[cases_Extrema.loc['CMR', 'Mins 2']], y[cases_Extrema.loc['CMR', 'Mins 2']], marker = 'o', color = 'lightgrey', ls = "", label = "Pruned Valleys")
# ax1.plot(x[cases_Extrema.loc['CMR', 'Maxes Final']], y[cases_Extrema.loc['CMR', 'Maxes Final']], marker = 'o', color = 'red', ls = "", label = "Final Peaks")
# ax1.plot(x[cases_Extrema.loc['CMR', 'Mins Final']], y[cases_Extrema.loc['CMR', 'Mins Final']], marker = 'o', color = 'lightcoral', ls = "", label = "Final Valleys")
# ax1.legend()
# ax1.set(xlabel = "Date (Month)", ylabel = "Cases per Million")
# ax1.xaxis.set_major_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
# ax1.grid()
# ax1.legend()
# fig.savefig("CMR Final Pruning.png")
# =============================================================================


    
# =============================================================================
# #Graphs the points
# graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Extrema_Testing_2.pdf')
# for i in cases_Extrema.index.values:
# 
#     x = range(0, len(cases_MA.columns.values))
#     y1 = cases_smoothed_15_3.loc[i, :].values
#     y2 = deaths_smoothed_15_3.loc[i, :].values
#     
#     fig, ax = plt.subplots(nrows=2, ncols=1)
#     fig.suptitle(i)
#     
#     ax[0].plot(x, y1, zorder = 0, label = "Smoothed")
#     ax[0].scatter(cases_Extrema.loc[i, 'All Rel Maxes'], y1[cases_Extrema.loc[i, 'All Rel Maxes']], zorder = 5, color = 'black', marker = '|', label = 'All Rel Maxes')
#     ax[0].scatter(cases_Extrema.loc[i, 'Initial Mins'], y1[cases_Extrema.loc[i, 'Initial Mins']], zorder = 5, color = 'grey', marker = '|', label = 'Initial Mins')
#     ax[0].scatter(cases_Extrema.loc[i, 'Rel Maxes 2'], y1[cases_Extrema.loc[i, 'Rel Maxes 2']], zorder = 10, color = 'red', marker = '|', label = 'Rel Maxes 2')
#     ax[0].scatter(cases_Extrema.loc[i, 'Mins 2'], y1[cases_Extrema.loc[i, 'Mins 2']], zorder = 10, color = 'orange', marker = '|', label = 'Mins 2')
#     ax[0].scatter(cases_Extrema.loc[i, 'Maxes Final'], y1[cases_Extrema.loc[i, 'Maxes Final']], zorder = 15, color = 'purple', marker = '|', label = 'Maxes Final')
#     ax[0].scatter(cases_Extrema.loc[i, 'Mins Final'], y1[cases_Extrema.loc[i, 'Mins Final']], zorder = 15, color = 'green', marker = '|', label = 'Mins Final')
#     
#     ax[1].plot(x, y2, zorder = 0)
#     ax[1].scatter(deaths_Extrema.loc[i, 'All Rel Maxes'], y2[deaths_Extrema.loc[i, 'All Rel Maxes']], zorder = 5, color = 'black', marker = '|')
#     ax[1].scatter(deaths_Extrema.loc[i, 'Initial Mins'], y2[deaths_Extrema.loc[i, 'Initial Mins']], zorder = 5, color = 'grey', marker = '|')
#     ax[1].scatter(deaths_Extrema.loc[i, 'Rel Maxes 2'], y2[deaths_Extrema.loc[i, 'Rel Maxes 2']], zorder = 10, color = 'red', marker = '|')
#     ax[1].scatter(deaths_Extrema.loc[i, 'Mins 2'], y2[deaths_Extrema.loc[i, 'Mins 2']], zorder = 10, color = 'orange', marker = '|')
#     ax[1].scatter(deaths_Extrema.loc[i, 'Maxes Final'], y2[deaths_Extrema.loc[i, 'Maxes Final']], zorder = 15, color = 'purple', marker = '|')
#     ax[1].scatter(deaths_Extrema.loc[i, 'Mins Final'], y2[deaths_Extrema.loc[i, 'Mins Final']], zorder = 15, color = 'green', marker = '|')
#     
#     fig.legend()
#     
#     graphs_File.savefig(fig)
#     
# graphs_File.close() 
# =============================================================================

# =============================================================================
# workbook = xlsxwriter.Workbook('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Extrema_Testing_2.xlsx')
# worksheet = workbook.add_worksheet()
# number_format = workbook.add_format()
# number_format.set_num_format('0.000')
# percent_format = workbook.add_format()
# percent_format.set_num_format(10)
# 
# row = 0
# for i in cases_Extrema.index.values:
#     print(i)
#     #Create a 2d array of all rel_min labeled as mins
#     cases_rel_mins = cases_Extrema.loc[i, 'Initial Mins']
#     cases_rel_mins = np.vstack((cases_rel_mins, [0] * len(cases_rel_mins)))
#     #create a 2d array of all rel_max_4 labeld as max
#     cases_rel_maxes = cases_Extrema.loc[i, "All Rel Maxes"]
#     cases_rel_maxes = np.vstack((cases_rel_maxes, [1] * len(cases_rel_maxes)))
#     #Combine the rel min and rel max arrays and sort them by x-point
#     cases_all_extrema = np.concatenate((cases_rel_mins, cases_rel_maxes), axis = 1)
#     cases_all_extrema = cases_all_extrema[:, cases_all_extrema[0].astype(int).argsort()]
#     cases_all_extrema = cases_all_extrema.astype(int)
#     
#     #Create a 2d array of all rel_min labeled as mins
#     deaths_rel_mins = deaths_Extrema.loc[i, 'Initial Mins']
#     deaths_rel_mins = np.vstack((deaths_rel_mins, [0] * len(deaths_rel_mins)))
#     #create a 2d array of all rel_max_4 labeld as max
#     deaths_rel_maxes = deaths_Extrema.loc[i, "All Rel Maxes"]
#     deaths_rel_maxes = np.vstack((deaths_rel_maxes, [1] * len(deaths_rel_maxes)))
#     #Combine the rel min and rel max arrays and sort them by x-point
#     deaths_all_extrema = np.concatenate((deaths_rel_mins, deaths_rel_maxes), axis = 1)
#     deaths_all_extrema = deaths_all_extrema[:, deaths_all_extrema[0].astype(int).argsort()]
#     deaths_all_extrema = deaths_all_extrema.astype(int)
#     
#     y1 = cases_smoothed_15_3.loc[i, :].values
#     cases_data = np.zeros((5, cases_rel_maxes.shape[1]))
#     
#     n=0
#     for j in range(0, cases_all_extrema.shape[1]):
#         if j == 0 and cases_all_extrema[1][j] == 1:
#             width = (cases_all_extrema[0][j+1] - cases_all_extrema[0][j]) * 2
#             height = y1[cases_all_extrema[0][j]]
#             ht_above_min = y1[cases_all_extrema[0][j]] - y1[cases_all_extrema[0][j+1]] 
#             dist_from_min = cases_all_extrema[0][j+1] - cases_all_extrema[0][j]
#             cases_data[0][n] = cases_all_extrema[0][j]
#             cases_data[1][n] = width
#             cases_data[2][n] = height
#             cases_data[3][n] = ht_above_min
#             cases_data[4][n] = dist_from_min
#             n=n+1
#         elif j == cases_all_extrema.shape[1] - 1 and cases_all_extrema[1][j] == 1:
#             width = (cases_all_extrema[0][j] - cases_all_extrema[0][j-1]) * 2
#             height = y1[cases_all_extrema[0][j]]
#             ht_above_min = y1[cases_all_extrema[0][j]] - y1[cases_all_extrema[0][j-1]] 
#             dist_from_min = cases_all_extrema[0][j] - cases_all_extrema[0][j-1]
#             cases_data[0][n] = cases_all_extrema[0][j]
#             cases_data[1][n] = width
#             cases_data[2][n] = height
#             cases_data[3][n] = ht_above_min
#             cases_data[4][n] = dist_from_min
#             n=n+1
#         elif cases_all_extrema[1][j] == 1:
#             width = (cases_all_extrema[0][j] - cases_all_extrema[0][j-1]) + (cases_all_extrema[0][j+1] - cases_all_extrema[0][j])
#             height = y1[cases_all_extrema[0][j]]
#             ht_above_min = min(y1[cases_all_extrema[0][j]] - y1[cases_all_extrema[0][j+1]], y1[cases_all_extrema[0][j]] - y1[cases_all_extrema[0][j-1]]) 
#             dist_from_min = min(cases_all_extrema[0][j+1] - cases_all_extrema[0][j], cases_all_extrema[0][j] - cases_all_extrema[0][j-1])
#             cases_data[0][n] = cases_all_extrema[0][j]
#             cases_data[1][n] = width
#             cases_data[2][n] = height
#             cases_data[3][n] = ht_above_min
#             cases_data[4][n] = dist_from_min
#             n=n+1
#             
#     y2 = deaths_smoothed_15_3.loc[i, :].values
#     deaths_data = np.zeros((5, deaths_rel_maxes.shape[1]))
#     
#     n=0
#     for j in range(0, deaths_all_extrema.shape[1]):
#         if j == 0 and deaths_all_extrema[1][j] == 1:
#             width = (deaths_all_extrema[0][j+1] - deaths_all_extrema[0][j]) * 2
#             height = y2[deaths_all_extrema[0][j]]
#             ht_above_min = y2[deaths_all_extrema[0][j]] - y2[deaths_all_extrema[0][j+1]] 
#             dist_from_min = deaths_all_extrema[0][j+1] - deaths_all_extrema[0][j]
#             deaths_data[0][n] = deaths_all_extrema[0][j]
#             deaths_data[1][n] = width
#             deaths_data[2][n] = height
#             deaths_data[3][n] = ht_above_min
#             deaths_data[4][n] = dist_from_min
#             n=n+1
#         elif j == deaths_all_extrema.shape[1] - 1 and deaths_all_extrema[1][j] == 1:
#             width = (deaths_all_extrema[0][j] - deaths_all_extrema[0][j-1]) * 2
#             height = y2[deaths_all_extrema[0][j]]
#             ht_above_min = y2[deaths_all_extrema[0][j]] - y2[deaths_all_extrema[0][j-1]] 
#             dist_from_min = deaths_all_extrema[0][j] - deaths_all_extrema[0][j-1]
#             deaths_data[0][n] = deaths_all_extrema[0][j]
#             deaths_data[1][n] = width
#             deaths_data[2][n] = height
#             deaths_data[3][n] = ht_above_min
#             deaths_data[4][n] = dist_from_min
#             n=n+1
#         elif deaths_all_extrema[1][j] == 1:
#             width = (deaths_all_extrema[0][j] - deaths_all_extrema[0][j-1]) + (deaths_all_extrema[0][j+1] - deaths_all_extrema[0][j])
#             height = y2[deaths_all_extrema[0][j]]
#             ht_above_min = min(y2[deaths_all_extrema[0][j]] - y2[deaths_all_extrema[0][j+1]], y2[deaths_all_extrema[0][j]] - y2[deaths_all_extrema[0][j-1]]) 
#             dist_from_min = min(deaths_all_extrema[0][j+1] - deaths_all_extrema[0][j], deaths_all_extrema[0][j] - deaths_all_extrema[0][j-1])
#             deaths_data[0][n] = deaths_all_extrema[0][j]
#             deaths_data[1][n] = width
#             deaths_data[2][n] = height
#             deaths_data[3][n] = ht_above_min
#             deaths_data[4][n] = dist_from_min
#             n=n+1
#             
#     worksheet.write(row, 0, i)
#     worksheet.write(row+1, 0, np.sum(cases_data[1]))
#     worksheet.write(row+3, 0, np.sum(cases_data[2]), number_format)
#     worksheet.write(row+5, 0, np.sum(cases_data[3]), number_format)
#     worksheet.write(row+7, 0, np.sum(cases_data[4]))
#     
#     worksheet.write_row(row, 1, cases_data[0])
#     worksheet.write_row(row+1, 1, cases_data[1])
#     worksheet.write_row(row+3, 1, cases_data[2], number_format)
#     worksheet.write_row(row+5, 1, cases_data[3], number_format)
#     worksheet.write_row(row+7, 1, cases_data[4])
#     
#     for j in range(0, cases_data.shape[1]):
#         worksheet.write(row+2, 1+j, '=%s/%s' %(xl_rowcol_to_cell(row+1, 1+j), xl_rowcol_to_cell(row+1, 0)), percent_format)
#         worksheet.write(row+4, 1+j, '=%s/%s' %(xl_rowcol_to_cell(row+3, 1+j), xl_rowcol_to_cell(row+3, 0)), percent_format)
#         worksheet.write(row+6, 1+j, '=%s/%s' %(xl_rowcol_to_cell(row+5, 1+j), xl_rowcol_to_cell(row+5, 0)), percent_format)
#         worksheet.write(row+8, 1+j, '=%s/%s' %(xl_rowcol_to_cell(row+7, 1+j), xl_rowcol_to_cell(row+7, 0)), percent_format)
#     
#     worksheet.write(row, 10, i)
#     worksheet.write(row+1, 10, np.sum(deaths_data[1]))
#     worksheet.write(row+3, 10, np.sum(deaths_data[2]), number_format)
#     worksheet.write(row+5, 10, np.sum(deaths_data[3]), number_format)
#     worksheet.write(row+7, 10, np.sum(deaths_data[4]))
#     
#     worksheet.write_row(row, 11, deaths_data[0])
#     worksheet.write_row(row+1, 11, deaths_data[1])
#     worksheet.write_row(row+3, 11, deaths_data[2], number_format)
#     worksheet.write_row(row+5, 11, deaths_data[3], number_format)
#     worksheet.write_row(row+7, 11, deaths_data[4])
#     
#     for j in range(0, deaths_data.shape[1]):
#         worksheet.write(row+2, 11+j, '=%s/%s' %(xl_rowcol_to_cell(row+1, 11+j), xl_rowcol_to_cell(row+1, 10)), percent_format)
#         worksheet.write(row+4, 11+j, '=%s/%s' %(xl_rowcol_to_cell(row+3, 11+j), xl_rowcol_to_cell(row+3, 10)), percent_format)
#         worksheet.write(row+6, 11+j, '=%s/%s' %(xl_rowcol_to_cell(row+5, 11+j), xl_rowcol_to_cell(row+5, 10)), percent_format)
#         worksheet.write(row+8, 11+j, '=%s/%s' %(xl_rowcol_to_cell(row+7, 11+j), xl_rowcol_to_cell(row+7, 10)), percent_format)
#         
#     
#     row = row + 10
#     
#     
# workbook.close()
# 
# =============================================================================

