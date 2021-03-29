# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:34:19 2020

@author: theod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter as svg
import xlwt
from xlwt import Workbook 
#This file contains a different algorithm for selecting relative mins, maxes, and other keypoints from the Moving Average (once it has been smoothed) for use by the interpolation

#Disable figure limit warning
plt.rcParams.update({'figure.max_open_warning': 0})

#turn off autoamtic plot generation in spyder
plt.ioff()

#Create a dataframe the store the relative mins and maxes calcualted
#'Rel Min and Rel Max both contian all possible relative mins and maxes, the 'Rel Min 2 and Rel Max 2' contain a pruned set of relative mins and maxes
Extrema_Testing = pd.DataFrame(data = None, index = cases_MA.index.values, columns = ['Rel Max', 'Rel Min', 'Rel Max 2', 'Rel Min 2', 'Rel Max 3', 'Rel Min 3', 'Rel Max 4', 'Rel Min 4', 'Rel Max 5', 'Rel Min 5', 'Rel Max 6', 'Rel Min 6', 'Rel Max F', 'Rel Min F'])
#loop through every country that has data to find all possible mins and maxes
for i in Extrema_Testing.index.values:
    #smooth the moving average data and replace all negative values with zero
    y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)
    #create arrays to store the calculated rel mins and maxes
    rel_max = []
    rel_min = []
    
    #loop through every data point in the smoothed moving average data
    #this check for special edges cases (i.e. at teh start or end of the data, the values are 0)
    for j in range(1, len(y)):
        #if the point prior to j is 0 and j is > 0 then add j-1 as a rel min
        if y[j - 1] == 0 and y[j] > 0:
            rel_min = np.append(rel_min, j-1)
        #if j-1 is > 0 and j = 0 then add j as a rel min
        elif y[j-1] > 0 and y[j] == 0:
            rel_min = np.append(rel_min, j)
        #if j-1 is the first point, and it is less than j, add j-1 as a rel min
        elif j-1 == 0 and y[j-1] < y[j]:
            rel_min = np.append(rel_min, j-1)
        #if j-1 is the first point and j < j - 1, add j-1 as a rel max
        elif j-1 == 0 and y[j-1] > y[j]:
            rel_max = np.append(rel_max, j-1)
        #if we are at the end and j > j-1, add j-1 as a rel max
        elif j == len(y) - 1 and y[j-1] < y[j]:
            rel_max = np.append(rel_max, j)
        #i we are at the end and j-1 > j, add j as a rel min
        elif j == len(y) - 1 and y[j-1] > y[j]:
            rel_min = np.append(rel_min, j)
    
    #loop through every data point int he smoothed moving average
    #this test non-edge cases 
    for j in range(1, len(y) - 1):
        #if j is greater than j-1 and j+1, add it as a rel max
        if y[j] > y[j-1] and y[j] > y[j+1]:
            rel_max = np.append(rel_max, j)
        #if j is less than j-1 and j+1, add it as a rel min
        elif y[j] < y[j-1] and y[j] < y[j+1]:
            rel_min = np.append(rel_min, j)
        #if j is greater than j-1, but equal to j+1 (i.e. we climb onto a plateau)
        elif y[j-1] < y[j] and y[j] == y[j+1]:
            n=1
            #start loop thing through following points
            while y[j] == y[j+n] and j+n != len(y) - 1:
                n = n + 1
                #if we hit then end, add j and the last point as rel max
                if j+n == len(y) - 1:
                    rel_max = np.append(rel_max, [j, j+n-1])
                #if we dropp of the plateau, add j and the last point on the plateau as rel max
                elif y[j] > y[j+n]:
                    rel_max = np.append(rel_max, [j, j+n-1])
        #if j is less than j-1 and equal to j+1 (i.e. we fall onto a plateau)
        elif y[j-1] > y[j] and y[j] == y[j+1]:
            n = 1
            #loop trhough follwoing points
            while y[j] == y[j+n] and j+n != len(y) - 1:
                n = n + 1
                #if we hit the last point, add it and j as rel min
                if j+n == len(y) - 1:
                    rel_min = np.append(rel_min, [j, j+n-1])
                #if we begin to climnb up form teh palteu, add j and the last point ont he palteu as rel min
                elif y[j] < y[j+n]:
                    rel_min = np.append(rel_min, [j, j+n-1])
    #add these points (all possible mins and maxes) the the dataframe           
    Extrema_Testing.loc[i, 'Rel Max'] = np.sort(np.unique(np.array(rel_max).astype(int)))
    Extrema_Testing.loc[i, 'Rel Min'] = np.sort(np.unique(np.array(rel_min).astype(int)))

#loop through the extrema poitns dataframe to prune of rel mins and rel maxes that are not rel mins or rel maxes of compared to all rel mins and rel maxes
for i in Extrema_Testing.index.values:
    #grab the rel-min and max data dn ensure it is sorted
    rel_max = np.sort(Extrema_Testing.loc[i, 'Rel Max'])
    rel_min = np.sort(Extrema_Testing.loc[i, 'Rel Min'])
    #grab the smoothed moving average data
    y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)
    
    #loop through to the relative maxes and pruen those that are nto rel maxes of the rel maxes
    rel_max_2 = []
    rel_min_2 = []
    for j in range(1, len(rel_max) - 1):
        #if j is grager than j-1 and j+1, add it as a rel max 2
        if y[rel_max[j]] > y[rel_max[j-1]] and y[rel_max[j]] > y[rel_max[j+1]]:
            rel_max_2 = np.append(rel_max_2, rel_max[j])
        #if the first point is greater than the second point, add the first point as a rel max 2
        elif j-1 == 0 and y[rel_max[j-1]] > y[rel_max[j]]:
            rel_max_2 = np.append(rel_max_2, rel_max[j-1])
        #if the last rel max is grater than the second to last rel max, add it as a rel max 2
        elif j+1 == len(rel_max) - 1:
            if y[rel_max[j+1]] > y[rel_max[j]]:
                rel_max_2 = np.append(rel_max_2, rel_max[j+1])
        #if we climb onto a platau
        elif y[rel_max[j-1]] < y[rel_max[j]] and y[rel_max[j]] == y[rel_max[j+1]]:
            n=0
            #loop through all following points
            while y[rel_max[j]] >= y[rel_max[j+n]] and j+n != len(rel_max) - 1:
                n = n + 1
                #if we hit the end, include j and the last point as rel max 2
                if j+n == len(rel_max):
                    rel_max_2 = np.append(rel_max_2, [rel_max[j], rel_max[j+n]])
                #if we fall off the palteau, include j and the last point on the plateau as rel max 2
                elif y[rel_max[j]] > y[rel_max[j+n]]:
                    rel_max_2 = np.append(rel_max_2, [rel_max[j], rel_max[j+n-1]])
        #if two rel_maxes (j and j+1) are far apart (60 days)...
        if rel_max[j+1] - rel_max[j] > 60:
            #Find the positions of relative mins between then
            mins_between = np.logical_and(rel_min > rel_max[j], rel_min < rel_max[j+1])
            #find the number of rel mins between then
            num_mins_between = np.count_nonzero(mins_between)
            #if there is only one rel min between them...
            if num_mins_between == 1:
                #determine the location of the rel min
                min_point = rel_min[np.where(mins_between == True)]
                #if the rel min is at least 15 days away from the rel max before and ahead of it, add the rel max behind and head it and the rel min itself
                if min_point - rel_max[j] > 15 and rel_max[j+1] - min_point > 15:
                    rel_max_2 = np.append(rel_max_2, [rel_max[j], rel_max[j+1]])
                    rel_min_2 = np.append(rel_min_2, min_point)
        #if two rel_maxes combine to make a plateau...
        if y[rel_max[j]] == y[rel_max[j+1]]:
            #find the location of the min before j and after j+1
            min_before = np.count_nonzero(rel_min < rel_max[j]) - 1
            min_after = np.count_nonzero(rel_min < rel_max[j+1]) 
            #if the values at min_before and min_after are both zero and the peaks are tall, add rel_max j and j+1
            if y[rel_min[min_before]] == 0 and y[rel_min[min_after]] == 0 and y[rel_max[j]] > 5:
                rel_max_2 = np.append(rel_max_2, [rel_max[j], rel_max[j+1]])
                rel_min_2 = np.append(rel_min_2, [rel_min[min_before], rel_min[min_after]])
            
    #in the event there are less than three rel_maxes to to analyze, pick the largest one
    if len(rel_max) < 3:
        if y[rel_max[0]] > y[rel_max[1]]:
            rel_max_2 = np.append(rel_max_2, rel_max[0])
        else:
            rel_max_2 = np.append(rel_max_2, rel_max[1])
                    
    #loop through all rel mins to prune those that are not rel min of rel mins
    for j in range(1, len(rel_min) - 1):
        #if j is less than j-1 and j+2, add it to rel min 2
        if y[rel_min[j]] < y[rel_min[j-1]] and y[rel_min[j]] < y[rel_min[j+1]]:
            rel_min_2 = np.append(rel_min_2, rel_min[j])
        #if teh first point is less than the seocnd point, add the first point as a rel min 2
        elif j-1 == 0 and y[rel_min[j-1]] < y[rel_min[j]]:
            rel_min_2 = np.append(rel_min_2, rel_min[j-1])
        #if the first rel_min is 0, then include it
        elif j-1 == 0 and y[rel_min[j-1]] == 0:
            rel_min_2 = np.append(rel_min_2, rel_min[j-1])
        #if we are at the last rel min
        elif j+1 == len(rel_min) - 1:
            #if the last point is less than the rel min before it, add the last point to the rel min 2
            if y[rel_min[j+1]] < y[rel_min[j]]:
                rel_min_2 = np.append(rel_min_2, rel_min[(j+1)])
            #if both the last point and the prior rel min are both 0, add both as rel min 2
            elif y[rel_min[j]] == 0 and y[rel_min[j+1]] == 0:
                rel_min_2 = np.append(rel_min_2, [rel_min[j], rel_min[(j+1)]])
        #if we fall onto a plateau
        elif y[rel_min[j-1]] > y[rel_min[j]] and y[rel_min[j]] == y[rel_min[j+1]]:
            #id that pleaueu is zero, add both j and j+1 as rel min of mins
            if y[rel_min[j]] == 0:
                rel_min_2 = np.append(rel_min_2, [rel_min[j], rel_min[j+1]])
            #otherwise...
            else:
                n=1
                #loop through all follwoing rel mins
                while y[rel_min[j]] < y[rel_min[j+n]] and j+n != len(rel_min) - 1:
                    n = n + 1
                    #if we hit the end, and j and the last rel min as rel min 2
                    if j+n == len(rel_min):
                        rel_min_2 = np.append(rel_min_2, [rel_min[j], rel_min[j+n]])
                    #if we climb up form the plateau, add j and the last rel min on the plateuau as rel min 2
                    elif y[rel_min[j]] < y[rel_min[j+n]]:
                        rel_min_2 = np.append(rel_min_2, [rel_min[j], rel_min[j+n-1]])
        #If the first rel_min is followed by the rel max, include it
        if j-1 == 0:
            if rel_max_2[0] == rel_max[0] and rel_min[j-1] < rel_max[0]:
                rel_min_2 = np.append(rel_min_2, rel_min[j-1])
        #if two rel_mins are far apart (60 days)...
        if rel_min[j+1] - rel_min[j] > 60:
            #Find the positions of the relative maxes between them
            maxes_between = np.logical_and(rel_max > rel_min[j], rel_max < rel_min[j+1])
            #find the number of rel maxes between them
            num_maxes_between = np.count_nonzero(maxes_between)
            #if there is only one rel max between them...
            if num_maxes_between == 1:
                max_point = rel_max[np.where(maxes_between == True)]
                #if the rel max is at least 15 days away from the rel min before and ahead of it, add the rel min behind and ahead it and the rel max itself
                if max_point - rel_min[j] > 15 and rel_min[j+1] - max_point > 15:
                    rel_max_2 = np.append(rel_max_2, max_point)
                    rel_min_2 = np.append(rel_min_2, [rel_min[j], rel_min[j+1]])
    
    #Check if the last point should be a rel_min, rel_max, or nothing
    last_point = np.sort(np.append(rel_max, rel_min))[-1]
    last_rel_min = np.sort(rel_min)[-1]
    last_rel_max = np.sort(rel_max)[-1]
    last_extrema_2 = int(np.sort(np.append(rel_max_2, rel_min_2))[-1])
    #if the last rel exrema is the global max and the drop to the next rel min is < 10% of itm include it as a rel min 2
    if y[last_extrema_2] == np.amax(y) and last_rel_min > last_extrema_2 and ((y[last_rel_min] - np.amax(y)) / np.amax(y)) < -.10:
        rel_min_2 = np.append(rel_min_2, last_rel_min)
    #if the drop from the last extrema to the the last point is greater than 30% and this differnece is also more than 20% of the global max, add it as a rel_min_2 or rel_max_2 
    elif abs((y[last_extrema_2] - y[last_point]) / y[last_extrema_2]) > .3 and abs((np.amax(y) - y[last_point]) / y[last_point]) > .2:
        #if the point it a rel min, add it as a rel_min 2
        if y[last_extrema_2] > y[last_point] and last_extrema_2 < last_rel_min:
            rel_min_2 = np.append(rel_min_2, last_rel_min)
        #if the point is a rel max, add it as a rel_max 2
        elif y[last_extrema_2] < y[last_point] and last_extrema_2 < last_rel_max:
            rel_max_2 = np.append(rel_max_2, last_rel_max)
    
    #If two rel_max_2s are very far apart (150 days), but no rel_min_2 exists betwwen them, find the smallest rel_min_1 between them and add it as a rel_min_2
    for k in range(0, len(rel_max_2) - 1):
        #if two rel_max 2 are super far apart (140 days)
        if rel_max_2[k+1] - rel_max_2[k] > 140:
            #find the rel_min_1 between them
            mins_2_between = np.logical_and(rel_min_2 > rel_max_2[k], rel_min_2 < rel_max_2[k+1])
            #find the number of rel_min_2 between them
            num_mins_2_between = np.count_nonzero(mins_2_between)
            #if there are not rel_min_2 between...
            if num_mins_2_between == 0:
                #find the rel_min_1 between
                mins_1_between = np.logical_and(rel_min > rel_max_2[k], rel_min < rel_max_2[k+1])
                min_1_points = rel_min[np.where(mins_1_between == True)]
                #determine the smallest min
                min_1_point = min_1_points[np.argmin(y[min_1_points])]
                #add it to the rel_min_2
                rel_min_2 = np.append(rel_min_2, min_1_point)
        #if two rel maxes are afrish apart (80 days) and are of substantial height and a single dip or substantial amgnitude exists between them, add a rel_min betwen them
        if rel_max_2[k+1] - rel_max_2[k] > 80:
            if y[int(rel_max_2[k])] / np.amax(y) > .5:
                min_value = np.amin(y[int(rel_max_2[k]):int(rel_max_2[k+1])])
                min_value_point = np.argmin(y[int(rel_max_2[k]):int(rel_max_2[k+1])]) + rel_max_2[k]
                first_point = y[int(rel_max_2[k])]
                second_point = y[int(rel_max_2[k+1])]
                if min_value / first_point > .2 and min_value / second_point > .2:
                    if first_point / second_point > .75 and first_point / second_point < 1.25:
                        rel_min_2 = np.append(rel_min_2, min_value_point)
                        rel_max_2 = np.append(rel_max_2, [rel_max_2[k], rel_max_2[k+1]])
                
    
    
    #add the non-pruned rel min and rel amxes to the dataframe
    Extrema_Testing.loc[i, 'Rel Max 2'] = np.sort(np.unique(np.array(rel_max_2).astype(int)))
    Extrema_Testing.loc[i, 'Rel Min 2'] = np.sort(np.unique(np.array(rel_min_2).astype(int)))
    
#loop throguh the Extrema Testing dataframe and enforce alternating order upon the (2nd set of) relative mins and maxes 
for i in Extrema_Testing.index.values:
    #Create a 2d array of all rel_min_2 labeled as mins
    rel_mins = Extrema_Testing.loc[i, 'Rel Min 2']
    rel_mins = np.vstack((rel_mins, ["min"] * len(rel_mins)))
    #create a 2d array of all rel_max_2 labeld as max
    rel_maxes = Extrema_Testing.loc[i, 'Rel Max 2']
    rel_maxes = np.vstack((rel_maxes, ["max"] * len(rel_maxes)))
    #Combine the rel min and rel max arrays and sort them by x-point
    all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
    all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
    
    #loop through this array and create a new array of start and endpoints of errors in the array
    errors = []
    for j in range(0, all_extrema.shape[1] - 1):
        #If j and j+1 are both "min" or "max"....
        if all_extrema[1, j] == all_extrema[1, j+1]:
            n=0
            #Loop thtough the following vlaues until you hit the end or find the next point has the opposite "min/max" value
            while (j+n < (all_extrema.shape[1] - 1) and all_extrema[1, j] == all_extrema[1, j+n]):
                n = n + 1
                #If the next point has the oppostie "min/max" value, add j and j+n-1 to the errors array
                if all_extrema[1, j] != all_extrema[1, j+n]:
                    errors = np.append(errors, j)
                    errors = np.append(errors, j+n-1)
                #If we hit the end and the point's "min/max" values remaind the same, add j and the last value to the errors array
                elif all_extrema[1, j] == all_extrema[1, j+n] and j+n == all_extrema.shape[1] - 1:
                    errors = np.append(errors, j)
                    errors = np.append(errors, j+n)
    
    #reshape the errors data frame into a list of two points
    errors = np.reshape(errors, (-1,2))
    
    #loop through the errors dataframe and delete rows whose start points are less than the prior rows endpoints
    final_error_ranges = []
    for k in range(0, errors.shape[0]):
        #always append the first row (pair of points) to the final_error_ranges array
        if k == 0:
            final_error_ranges = np.append(final_error_ranges, [errors[0, :]])
        #Before adding subsequent rows (paits of points), check to make sure the starting point is greater than the last point in the final_errors_range array
        elif errors[k, 0] > final_error_ranges.flat[-1]:
            final_error_ranges = np.append(final_error_ranges, [errors[k, :]])
    
    #rehape the error range into list of two points
    final_error_ranges = np.reshape(final_error_ranges, (-1,2)).astype(int)
    
    #grab the smoothed moving average data
    y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)    
    
    #loop through the list of ordered pairs of final point ranges and mark points for removal
    mins_to_remove = []
    maxes_to_remove = []
    
    #loop through the final_error_ranges array to cataloge the points that should be removed
    for p in range(0, final_error_ranges.shape[0]):
        #Define the range of x-values we will compare before pruning
        x = all_extrema[0][range(final_error_ranges[p][0], final_error_ranges[p][1] + 1)].astype(int)
        #if the range of values were are comparing are all "mins"...
        if all_extrema[1][final_error_ranges[p, 0]] == "min":
            #if the first and last point in the range are both zero...
            if y[x[0]] == 0 and y[x[-1]] == 0:
                #mark all but the first and last point for removal
                if len(x) > 2:
                    mins_to_remove = np.append(mins_to_remove, np.setdiff1d(x, np.append(x[0], x[-1])))
            #otherwise, all therel_min points we are companing are not zero, so we remove all but the smallest value 
            else:
                min_point = np.argmin(y[x])
                mins_to_remove = np.append(mins_to_remove, np.setdiff1d(x, x[min_point]))
        #otherwise if the range of values we are comparing ar all "maxes", remove all but the largest point
        elif all_extrema[1][final_error_ranges[p, 0]] == "max":
            max_point = np.argmax(y[x])
            maxes_to_remove = np.append(maxes_to_remove, np.setdiff1d(x, x[max_point]))

    #Add the new prunned set o rel_min and rel_maxes to the Extrema Dataframe
    rel_max_2 = Extrema_Testing.loc[i, 'Rel Max 2']
    rel_min_2 = Extrema_Testing.loc[i, 'Rel Min 2']
    
    Extrema_Testing.loc[i, 'Rel Max 3'] = np.setdiff1d(rel_max_2, maxes_to_remove)
    Extrema_Testing.loc[i, 'Rel Min 3'] = np.setdiff1d(rel_min_2, mins_to_remove)
    
#Loop through the rel_min_3 and rel_max_3 and determine whether or not another rel_min or rel_min / rel_max or rel_max loops needs to be performed
for i in Extrema_Testing.index.values:
    #Declare y as the smoothed MA data
    y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)    
    
    #Create a 2d array of all rel_min_3 labeled as mins
    rel_mins = Extrema_Testing.loc[i, 'Rel Min 3']
    rel_mins_labels = np.vstack((rel_mins, ["min"] * len(rel_mins)))
    #create a 2d array of all rel_max_2 labeld as max
    rel_maxes = Extrema_Testing.loc[i, 'Rel Max 3']
    rel_maxes_labels = np.vstack((rel_maxes, ["max"] * len(rel_maxes)))
    #Combine the rel min and rel max arrays and sort them by x-point
    all_extrema = np.concatenate((rel_mins_labels, rel_maxes_labels), axis = 1)
    all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
    
    #Loop through the all_extrema array and count the number of maxes and the number of changes
    num_maxes = 0
    num_changes = 1
    for j in range(0, all_extrema.shape[1]):
        if all_extrema[1][j] == "max":
            num_maxes = num_maxes + 1
        if j+1 == all_extrema.shape[1]:
            k=0
        else:
            if all_extrema[1][j] != all_extrema[1][j+1]:
                num_changes = num_changes + 1
    
    rel_max_4 = []
    rel_min_4 = []
    
    #If num_changes >= 10, then perform another rel_min or rel_min / rel_max or rel_max loop
    if num_changes >= 10:
        #perform the rel_max of rel_max loop
        for k in range(1, len(rel_maxes) - 1):
            #if k is greater than k-1 and k+1, add it as a rel max 4
            if y[rel_maxes[k]] > y[rel_maxes[k-1]] and y[rel_maxes[k]] > y[rel_maxes[k+1]]:
                rel_max_4 = np.append(rel_max_4, rel_maxes[k])
            #if the first point is greater than the second point, add the first point as a rel max 4
            elif k-1 == 0 and y[rel_maxes[k-1]] > y[rel_maxes[k]]:
                rel_max_4 = np.append(rel_max_4, rel_maxes[k-1])
            #if the last rel max is greater than the second to last rel max, add it as a rel max 4
            elif k+1 == len(rel_maxes) - 1:
                if y[rel_maxes[k+1]] > y[rel_maxes[k]]:
                    rel_max_4 = np.append(rel_max_4, rel_maxes[k+1])
            #if the distances between rel maxes i very large (60 days)...
            if rel_maxes[k+1] - rel_maxes[k] > 60:
                #find the postions of the rel mins between them
                mins_between = np.logical_and(rel_mins > rel_maxes[k], rel_mins < rel_maxes[k+1])
                #find the number of mins between them
                num_mins_between = np.count_nonzero(mins_between)
                #if there is only one min between them...
                if num_mins_between == 1:
                    #determine the point of the rel min
                    min_point = rel_mins[np.where(mins_between == True)]
                    #if the rel_min is at least 15 days away from the rel max befor eand after it, add the rel max before and behind it and the rel min
                    if min_point - rel_maxes[k] > 15 and rel_maxes[k+1] - min_point > 15:
                        rel_max_4 = np.append(rel_max_4, [rel_maxes[k], rel_maxes[k+1]])
                        rel_min_4 = np.append(rel_min_4, min_point)
            #if we are at the first point and the distance between the first and second point is > 60...
            if k-1 == 0 and rel_maxes[k] - rel_maxes[k-1] > 60:
                #find the postions of the rel mins between them
                mins_between = np.logical_and(rel_mins > rel_maxes[k-1], rel_mins < rel_maxes[k])
                #find the number of mins between them
                num_mins_between = np.count_nonzero(mins_between)
                if num_mins_between == 1:
                    #determine the point ot the rel min
                    min_point = rel_mins[np.where(mins_between == True)]
                    #if the rel min is at least 15 days away from k-1 and k, add both k and k-1 and the rel min to rel_max_4 and rel_min_4
                    if rel_maxes[k] - min_point > 15 and min_point - rel_maxes[k-1]:
                        rel_min_4 = np.append(rel_min_4, min_point)
                        rel_max_4 = np.append(rel_max_4, [rel_maxes[k-1], rel_maxes[k]])
                #if there are two mins between the rel maxes....
                elif num_mins_between == 2:
                    #BUT they are both == 0
                    if y[rel_mins[mins_between]][0] == 0 and y[rel_mins[mins_between]][1] == 0: 
                        #as long as the first min is more than 15 spaces from the first point (and same for the second min and second rel_max) add them all as rel_max_4/rel_min_4
                        if rel_mins[mins_between][0] - rel_maxes[k-1] > 15 and rel_maxes[k] - rel_mins[mins_between][1] > 15:
                            rel_min_4 = np.append(rel_min_4, rel_mins[mins_between])
                            rel_max_4 = np.append(rel_max_4, [rel_maxes[k-1], rel_maxes[k]])
        
        #perform the rel_min of rel_min loop
        for k in range(1, len(rel_mins) - 1):
            #if k is less than k-1 and k+1, add it to rel_min_4
            if y[rel_mins[k]] < y[rel_mins[k-1]] and y[rel_mins[k]] < y[rel_mins[k+1]]:
                rel_min_4 = np.append(rel_min_4, rel_mins[k])
            #if the first point it less than the second point, add the first point as a rel min 4
            elif k-1 == 0 and y[rel_mins[k-1]] < y[rel_mins[k]]:
                rel_min_4 = np.append(rel_min_4, rel_mins[k-1])
            #if the first rel min is 0, include it
            elif k-1 ==0 and y[rel_mins[k-1]] == 0:
                rel_min_4 = np.append(rel_min_4, rel_mins[k-1])
            #if we are at the last rel min
            elif k+1 == len(rel_mins) - 1:
                #if the last point is less than the rel min before it, add the last point to the rel min 2
                if y[rel_mins[k+1]] < y[rel_mins[k]]:
                    rel_min_4 = np.append(rel_min_4, rel_mins[(k+1)])
                #if both the last point and the prior rel min are both 0, add both as rel min 2
                elif y[rel_mins[k]] == 0 and y[rel_mins[k+1]] == 0:
                    rel_min_4 = np.append(rel_min_4, [rel_mins[k], rel_mins[k+1]])
            # if we fall onto a plateau, add k and k+1 as rel mins 4
            elif y[rel_mins[k-1]] > y[rel_mins[k]] and y[rel_mins[k]] == y[rel_mins[k+1]]:
                if y[rel_mins[k]] == 0:
                    rel_min_4 = np.append(rel_min_4, [rel_mins[k], rel_mins[k+1]])
            #If a rel_min_3 exists before the first Rel_max_4, include it as a rel_min_4
            if k-1 == 0 and Extrema_Testing.loc[i, 'Rel Min 3'][0] < rel_max_4[0]:
                rel_min_4 = np.append(rel_min_4, Extrema_Testing.loc[i, 'Rel Min 3'][0])
            #if the distances between rel mins is very large (60 days)...
            if rel_mins[k+1] - rel_mins[k] > 60:
                #find the postions of the rel maxes between them
                maxes_between = np.logical_and(rel_maxes > rel_mins[k], rel_maxes < rel_mins[k+1])
                #find the number of mins between them
                num_maxes_between = np.count_nonzero(maxes_between)
                #if there is only one max between them...
                if num_maxes_between == 1:
                    #determine the point of the rel max
                    max_point = rel_maxes[np.where(maxes_between == True)]
                    #if the rel_max is at least 15 days away from the rel min befor eand after it, add the rel min before and behind it and the rel max
                    if max_point - rel_mins[k] > 15 and rel_mins[k+1] - max_point > 15:
                        rel_min_4 = np.append(rel_min_4, [rel_mins[k], rel_mins[k+1]])
                        rel_max_4 = np.append(rel_max_4, max_point)
            #if we are at the first point and the distance between the first and second point is > 60...
            if k-1 == 0 and rel_mins[k] - rel_mins[k-1] > 60:
                #find the postions of the rel maxe between them
                maxes_between = np.logical_and(rel_maxes > rel_mins[k-1], rel_maxes < rel_mins[k])
                #find the number of maxes between them
                num_maxes_between = np.count_nonzero(maxes_between)
                if num_maxes_between == 1:
                    #determine the point ot the rel min
                    max_point = rel_maxes[np.where(maxes_between == True)]
                    #if the rel max is at least 15 days away from k-1 and k, add both k and k-1 and the rel min to rel_max_4 and rel_min_4
                    if rel_mins[k] - max_point > 15 and max_point - rel_mins[k-1]:
                        rel_max_4 = np.append(rel_max_4, max_point)
                        rel_min_4 = np.append(rel_min_4, [rel_mins[k-1], rel_mins[k]])
        
        #Check if the last point should be a rel_min, rel_max, or nothing
        last_point = all_extrema[0][-1].astype(int)
        last_rel_min = np.sort(rel_mins)[-1]
        last_rel_max = np.sort(rel_maxes)[-1]
        last_extrema_4 = int(np.sort(np.append(rel_max_4, rel_min_4))[-1])
        #if the last rel exrema is the global max and the drop to the next rel min is < 10% of itm include it as a rel min 4
        if y[last_extrema_4] == np.amax(y) and last_rel_min > last_extrema_2 and ((y[last_rel_min] - np.amax(y)) / np.amax(y)) < -.10:
            rel_min_4 = np.append(rel_min_4, last_rel_min)
        #if the drop from the last extrema to the the last point is greater than 30% and this differnece is also more than 20% of the global max, add it as a rel_min_4 or rel_max_4 
        elif abs((y[last_extrema_4] - y[last_point]) / y[last_extrema_4]) > .3 and abs((np.amax(y) - y[last_point]) / y[last_point]) > .2:
            #if the point it a rel min, add it as a rel_min 4
            if y[last_extrema_4] > y[last_point] and last_extrema_4 < last_rel_min:
                rel_min_4 = np.append(rel_min_4, last_rel_min)
            #if the point is a rel max, add it as a rel_max 4
            elif y[last_extrema_4] < y[last_point] and last_extrema_4 < last_rel_max:
                rel_max_4 = np.append(rel_max_4, last_rel_max)
                
        #Add the rel_max_4 and rel_min 4 to the Extrema Dataframe
        Extrema_Testing.loc[i, 'Rel Max 4'] = np.sort(np.unique(np.array(rel_max_4).astype(int)))
        Extrema_Testing.loc[i, 'Rel Min 4'] = np.sort(np.unique(np.array(rel_min_4).astype(int)))
    
    #otherwise, leave the point sunchanged
    else:
        m = 0

#Loop through the Extrema Testing Dataframe and enforce alternating order upon the (4th set og) relative mins and maxes (where applicable)
for i in Extrema_Testing.index.values:
    #Check to make sure there are rel_min_4 and rel_min_4 to prune
    if np.any(np.isnan(Extrema_Testing.loc[i, 'Rel Min 4'])):
        m = 0
    else:
        #Create a 2d array of all rel_min_4 labeled as mins
        rel_mins = Extrema_Testing.loc[i, 'Rel Min 4']
        rel_mins = np.vstack((rel_mins, ["min"] * len(rel_mins)))
        #create a 2d array of all rel_max_4 labeld as max
        rel_maxes = Extrema_Testing.loc[i, 'Rel Max 4']
        rel_maxes = np.vstack((rel_maxes, ["max"] * len(rel_maxes)))
        #Combine the rel min and rel max arrays and sort them by x-point
        all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
        all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
        
        #loop through this array and create a new array of start and endpoints of errors in the array
        errors = []
        for j in range(0, all_extrema.shape[1] - 1):
            #If j and j+1 are both "min" or "max"....
            if all_extrema[1, j] == all_extrema[1, j+1]:
                n=0
                #Loop thtough the following vlaues until you hit the end or find the next point has the opposite "min/max" value
                while (j+n < (all_extrema.shape[1] - 1) and all_extrema[1, j] == all_extrema[1, j+n]):
                    n = n + 1
                    #If the next point has the oppostie "min/max" value, add j and j+n-1 to the errors array
                    if all_extrema[1, j] != all_extrema[1, j+n]:
                        errors = np.append(errors, j)
                        errors = np.append(errors, j+n-1)
                    #If we hit the end and the point's "min/max" values remaind the same, add j and the last value to the errors array
                    elif all_extrema[1, j] == all_extrema[1, j+n] and j+n == all_extrema.shape[1] - 1:
                        errors = np.append(errors, j)
                        errors = np.append(errors, j+n)
    
        #reshape the errors data frame into a list of two points
        errors = np.reshape(errors, (-1,2))
        
        #loop through the errors dataframe and delete rows whose start points are less than the prior rows endpoints
        final_error_ranges = []
        for k in range(0, errors.shape[0]):
            #always append the first row (pair of points) to the final_error_ranges array
            if k == 0:
                final_error_ranges = np.append(final_error_ranges, [errors[0, :]])
            #Before adding subsequent rows (paits of points), check to make sure the starting point is greater than the last point in the final_errors_range array
            elif errors[k, 0] > final_error_ranges.flat[-1]:
                final_error_ranges = np.append(final_error_ranges, [errors[k, :]])
    
        #rehape the error range into list of two points
        final_error_ranges = np.reshape(final_error_ranges, (-1,2)).astype(int)
        
        #grab the smoothed moving average data
        y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)    
    
        #loop through the list of ordered pairs of final point ranges and mark points for removal
        mins_to_remove = []
        maxes_to_remove = []
    
        #loop through the final_error_ranges array to cataloge the points that should be removed
        for p in range(0, final_error_ranges.shape[0]):
            #Define the range of x-values we will compare before pruning
            x = all_extrema[0][range(final_error_ranges[p][0], final_error_ranges[p][1] + 1)].astype(int)
            #if the range of values were are comparing are all "mins"...
            if all_extrema[1][final_error_ranges[p, 0]] == "min":
                #if the first and last point in the range are both zero...
                if y[x[0]] == 0 and y[x[-1]] == 0:
                    #mark all but the first and last point for removal
                    if len(x) > 2:
                        mins_to_remove = np.append(mins_to_remove, np.setdiff1d(x, np.append(x[0], x[-1])))
                #otherwise, all therel_min points we are companing are not zero, so we remove all but the smallest value 
                else:
                    min_point = np.argmin(y[x])
                    mins_to_remove = np.append(mins_to_remove, np.setdiff1d(x, x[min_point]))
            #otherwise if the range of values we are comparing ar all "maxes", remove all but the largest point
            elif all_extrema[1][final_error_ranges[p, 0]] == "max":
                max_point = np.argmax(y[x])
                maxes_to_remove = np.append(maxes_to_remove, np.setdiff1d(x, x[max_point]))
        
        #Add the new prunned set o rel_min and rel_maxes to the Extrema Dataframe
        rel_max_4 = Extrema_Testing.loc[i, 'Rel Max 4']
        rel_min_4 = Extrema_Testing.loc[i, 'Rel Min 4']
        
        Extrema_Testing.loc[i, 'Rel Max 5'] = np.setdiff1d(rel_max_4, maxes_to_remove)
        Extrema_Testing.loc[i, 'Rel Min 5'] = np.setdiff1d(rel_min_4, mins_to_remove)

#Loop through the extrema testing dataframe do the final rpuning of small peaks via their height or their widths

wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1') 
excel_row = 0
for i in Extrema_Testing.index.values:
    #print(i)
    
    #grab the smoothed moving average data
    y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)   
    
    #check if we need to run the analysis on rel_min/max3 or rel_min/max 5
    if np.any(np.isnan(Extrema_Testing.loc[i, 'Rel Min 5'])):
        #Create a 2d array of all rel_min_4 labeled as mins
        rel_mins = Extrema_Testing.loc[i, 'Rel Min 3'].astype(int)
        rel_mins = np.vstack((rel_mins, ["min"] * len(rel_mins)))
        #create a 2d array of all rel_max_4 labeld as max
        rel_maxes = Extrema_Testing.loc[i, 'Rel Max 3'].astype(int)
        rel_maxes = np.vstack((rel_maxes, ["max"] * len(rel_maxes)))
        #Combine the rel min and rel max arrays and sort them by x-point
        all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
        all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
        all_extrema_index = all_extrema[0].astype(int)
        all_extrema_min_max = all_extrema[1]
        
        #Loop through the extrema to determine the width, overall height, average height over surroudning mins, and minimum height over surroudning mins
        rel_maxes_data = np.zeros((6, rel_maxes.shape[1]))
        rel_maxes_data[0] = rel_maxes[0].astype(int)
        count = 0
        for j in range(0, all_extrema.shape[1]):
            if j == 0 and all_extrema_min_max[j] == 'max':
                if i == 'IMN':
                    print('Here1')
                rel_maxes_data[1][count] = (all_extrema_index[j+1] - all_extrema_index[j])*2 #Rel Max width
                rel_maxes_data[2][count] = y[all_extrema_index[j]] #Rel Max height
                rel_maxes_data[3][count] = y[all_extrema_index[j]] - y[all_extrema_index[j+1]] #Average Height above mins
                rel_maxes_data[4][count] = y[all_extrema_index[j]] - y[all_extrema_index[j+1]] #smallest height above mins
                rel_maxes_data[5][count] = all_extrema_index[j+1] - all_extrema_index[j] #smallest distance from mins
                count = count + 1
            elif j == all_extrema.shape[1] - 1 and all_extrema_min_max[j] == 'max':
                rel_maxes_data[1][count] = (all_extrema_index[j] - all_extrema_index[j-1])*2
                rel_maxes_data[2][count] = y[all_extrema_index[j]]
                rel_maxes_data[3][count] = y[all_extrema_index[j]] - y[all_extrema_index[j-1]]
                rel_maxes_data[4][count] = y[all_extrema_index[j]] - y[all_extrema_index[j-1]]
                rel_maxes_data[5][count] = all_extrema_index[j] - all_extrema_index[j-1]
            elif all_extrema_min_max[j] == 'max':
                rel_maxes_data[1][count] = all_extrema_index[j+1] - all_extrema_index[j-1]
                rel_maxes_data[2][count] = y[all_extrema_index[j]]
                rel_maxes_data[3][count] = ((y[all_extrema_index[j]] - y[all_extrema_index[j-1]]) + (y[all_extrema_index[j]] - y[all_extrema_index[j+1]])) / 2
                rel_maxes_data[4][count] = min(y[all_extrema_index[j]] - y[all_extrema_index[j-1]], y[all_extrema_index[j]] - y[all_extrema_index[j+1]])
                rel_maxes_data[5][count] = min(all_extrema_index[j] - all_extrema_index[j-1], all_extrema_index[j+1] - all_extrema_index[j])
                count = count + 1
        
        maxes_to_remove = []
        #Loop through the rel_maxes_data array and mark which points should be removed
        for k in range(0, rel_maxes_data.shape[1]):
            total_width = np.sum(rel_maxes_data[1])
            total_height = np.sum(rel_maxes_data[2])
            total_avg_height_over_min = np.sum(rel_maxes_data[3])
            total_min_height_over_min = np.sum(rel_maxes_data[4])
            
            #if a rel max is shorter than a neighboring rel_min, mark it for removal
            if rel_maxes_data[4][k] < 0:
                print(i, rel_maxes_data[0][k], "Min > Max")
                maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if a rel_max is too thin, mark it for removal as long as...
            if rel_maxes_data[1][k] < 30 and rel_maxes_data[1][k] < .25 * total_width:
                #The total height of the max is less than 10% of the total
                if rel_maxes_data[2][k] < total_height * .1:
                    print(i, rel_maxes_data[0][k], "Width too Thin")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if a rel max is too short, remove it as long as ...
            if rel_maxes_data[2][k] < total_height * .15:
                #The width of the assocaited peak is less than 20% of the total
                if rel_maxes_data[1][k] < total_width * .10:
                    print(i, rel_maxes_data[0][k], "Height too Short")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if the height, avg. height above mins, and smallest height above mins are less than 1% of total, mark it for removal
            if rel_maxes_data[2][k] < total_height * .01 and rel_maxes_data[3][k] < total_avg_height_over_min * .01 and rel_maxes_data[4][k] < total_min_height_over_min * .01:
                #as long as the width isn't too large
                if rel_maxes_data[1][k] < total_width * .15:
                    print(i, rel_maxes_data[0][k], "3 Things too small")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if the max to too close to a rel min and it's height over that rel min is too small, remove it
            if rel_maxes_data[5][k] < 10:
                if rel_maxes_data[4][k] < total_min_height_over_min * .05  and rel_maxes_data[1][k] < total_width * .23:
                    print(i, rel_maxes_data[0][k], "Too Close to Min and Short")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
        
        #Add the new prunned set o rel_min and rel_maxes to the Extrema Dataframe
        rel_max_3 = Extrema_Testing.loc[i, 'Rel Max 3']
        
        Extrema_Testing.loc[i, 'Rel Max 6'] = np.setdiff1d(rel_max_3, maxes_to_remove)
        Extrema_Testing.loc[i, 'Rel Min 6'] = Extrema_Testing.loc[i, 'Rel Min 3']
                
                
    else:       
        #Create a 2d array of all rel_min_5 labeled as mins
        rel_mins = Extrema_Testing.loc[i, 'Rel Min 5'].astype(int)
        rel_mins = np.vstack((rel_mins, ["min"] * len(rel_mins)))
        #create a 2d array of all rel_max_4 labeld as max
        rel_maxes = Extrema_Testing.loc[i, 'Rel Max 5'].astype(int)
        rel_maxes = np.vstack((rel_maxes, ["max"] * len(rel_maxes)))
        #Combine the rel min and rel max arrays and sort them by x-point
        all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
        all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
        all_extrema_index = all_extrema[0].astype(int)
        all_extrema_min_max = all_extrema[1]
        
        #Loop through the extrema to determine the width, overall height, average height over surroudning mins, and minimum height over surroudning mins
        rel_maxes_data = np.zeros((6, rel_maxes.shape[1]))
        rel_maxes_data[0] = rel_maxes[0].astype(int)        
        count = 0
        for j in range(0, all_extrema.shape[1]):
            if j == 0 and all_extrema_min_max[j] == 'max':
                rel_maxes_data[1][count] = (all_extrema_index[j+1] - all_extrema_index[j])*2 #Rel Max width
                rel_maxes_data[2][count] = y[all_extrema_index[j]] #Rel Max height
                rel_maxes_data[3][count] = y[all_extrema_index[j]] - y[all_extrema_index[j+1]] #Average Height above mins
                rel_maxes_data[4][count] = y[all_extrema_index[j]] - y[all_extrema_index[j+1]] #smallest height above mins
                rel_maxes_data[5][count] = all_extrema_index[j+1] - all_extrema_index[j] #smallest distance from mins
                count = count + 1
            elif j == all_extrema.shape[1] - 1 and all_extrema_min_max[j] == 'max':
                rel_maxes_data[1][count] = (all_extrema_index[j] - all_extrema_index[j-1])*2
                rel_maxes_data[2][count] = y[all_extrema_index[j]]
                rel_maxes_data[3][count] = y[all_extrema_index[j]] - y[all_extrema_index[j-1]]
                rel_maxes_data[4][count] = y[all_extrema_index[j]] - y[all_extrema_index[j-1]]
                rel_maxes_data[5][count] = all_extrema_index[j] - all_extrema_index[j-1]
            elif all_extrema_min_max[j] == 'max':
                rel_maxes_data[1][count] = all_extrema_index[j+1] - all_extrema_index[j-1]
                rel_maxes_data[2][count] = y[all_extrema_index[j]]
                rel_maxes_data[3][count] = ((y[all_extrema_index[j]] - y[all_extrema_index[j-1]]) + (y[all_extrema_index[j]] - y[all_extrema_index[j+1]])) / 2
                rel_maxes_data[4][count] = min(y[all_extrema_index[j]] - y[all_extrema_index[j-1]], y[all_extrema_index[j]] - y[all_extrema_index[j+1]])
                rel_maxes_data[5][count] = min(all_extrema_index[j] - all_extrema_index[j-1], all_extrema_index[j+1] - all_extrema_index[j])
                count = count + 1
        
        maxes_to_remove = []
        #Loop through the rel_maxes_data array and mark which points should be removed
        for k in range(0, rel_maxes_data.shape[1]):
            total_width = np.sum(rel_maxes_data[1])
            total_height = np.sum(rel_maxes_data[2])
            total_avg_height_over_min = np.sum(rel_maxes_data[3])
            total_min_height_over_min = np.sum(rel_maxes_data[4])
            
            #if a rel max is shorter than a neighboring rel_min, mark it for removal
            if rel_maxes_data[4][k] < 0:
                print(i, rel_maxes_data[0][k], "Min > Max")
                maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if a rel_max is too thin, mark it for removal as long as...
            if rel_maxes_data[1][k] < 45  and rel_maxes_data[1][k] < .25 * total_width:
                #The total height of the max is less than 10% of the total
                if rel_maxes_data[2][k] < total_height * .1:
                    print(i, rel_maxes_data[0][k], "Width too Thin")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if a rel max is too short, remove it as long as ...
            if rel_maxes_data[2][k] < total_height * .15:
                #The width of the assocaited peak is less than 20% of the total
                if rel_maxes_data[1][k] < total_width * .10:
                    print(i, rel_maxes_data[0][k], "Height too Short")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if the height, avg. height above mins, and smallest height above mins are less than 1% of total, mark it for removal
            if rel_maxes_data[2][k] < total_height * .01 and rel_maxes_data[3][k] < total_avg_height_over_min * .01 and rel_maxes_data[4][k] < total_min_height_over_min * .01:
                #as long as the width isn't too large
                if rel_maxes_data[1][k] < total_width * .15:
                    print(i, rel_maxes_data[0][k], "3 Things too small")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])
            #if the max to too close to a rel min and it's height over that rel min is too small, remove it
            if rel_maxes_data[5][k] < 10:
                if rel_maxes_data[4][k] < total_min_height_over_min * .05 and rel_maxes_data[1][k] < total_width * .22:
                    print(i, rel_maxes_data[0][k], "Too Close to Min and Short")
                    maxes_to_remove = np.append(maxes_to_remove, rel_maxes_data[0][k])

        #Add the new prunned set o rel_min and rel_maxes to the Extrema Dataframe
        rel_max_5 = Extrema_Testing.loc[i, 'Rel Max 5']

        Extrema_Testing.loc[i, 'Rel Max 6'] = np.setdiff1d(rel_max_5, np.sort(np.unique(maxes_to_remove)))
        Extrema_Testing.loc[i, 'Rel Min 6'] = Extrema_Testing.loc[i, 'Rel Min 5']
                
# =============================================================================
#     sheet1.write(excel_row, 0, i)
#     for k in range(0, rel_maxes_data.shape[1]):
#         sheet1.write(excel_row + 1, k, rel_maxes_data[0][k])
#         sheet1.write(excel_row + 2, k, rel_maxes_data[1][k])
#         sheet1.write(excel_row + 3, k, rel_maxes_data[2][k])
#         sheet1.write(excel_row + 4, k, rel_maxes_data[3][k])
#         sheet1.write(excel_row + 5, k, rel_maxes_data[4][k])
#         sheet1.write(excel_row + 6, k, rel_maxes_data[5][k])
#     excel_row = excel_row + 7
#            
# wb.save('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Small_Peak_Removal.xls')
# =============================================================================

#Loop through the Extrema Testing Dataframe and enforce alternating order upon the (6th set og) relative mins and maxes (where applicable)
for i in Extrema_Testing.index.values:
    #Create a 2d array of all rel_min_4 labeled as mins
    rel_mins = Extrema_Testing.loc[i, 'Rel Min 6']
    rel_mins = np.vstack((rel_mins, ["min"] * len(rel_mins)))
    #create a 2d array of all rel_max_4 labeld as max
    rel_maxes = Extrema_Testing.loc[i, 'Rel Max 6']
    rel_maxes = np.vstack((rel_maxes, ["max"] * len(rel_maxes)))
    #Combine the rel min and rel max arrays and sort them by x-point
    all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
    all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
        
    #loop through this array and create a new array of start and endpoints of errors in the array
    errors = []
    for j in range(0, all_extrema.shape[1] - 1):
        #If j and j+1 are both "min" or "max"....
        if all_extrema[1, j] == all_extrema[1, j+1]:
            n=0
            #Loop thtough the following vlaues until you hit the end or find the next point has the opposite "min/max" value
            while (j+n < (all_extrema.shape[1] - 1) and all_extrema[1, j] == all_extrema[1, j+n]):
                n = n + 1
                #If the next point has the oppostie "min/max" value, add j and j+n-1 to the errors array
                if all_extrema[1, j] != all_extrema[1, j+n]:
                    errors = np.append(errors, j)
                    errors = np.append(errors, j+n-1)
                #If we hit the end and the point's "min/max" values remaind the same, add j and the last value to the errors array
                elif all_extrema[1, j] == all_extrema[1, j+n] and j+n == all_extrema.shape[1] - 1:
                    errors = np.append(errors, j)
                    errors = np.append(errors, j+n)
    
    #reshape the errors data frame into a list of two points
    errors = np.reshape(errors, (-1,2))
        
    #loop through the errors dataframe and delete rows whose start points are less than the prior rows endpoints
    final_error_ranges = []
    for k in range(0, errors.shape[0]):
        #always append the first row (pair of points) to the final_error_ranges array
        if k == 0:
            final_error_ranges = np.append(final_error_ranges, [errors[0, :]])
        #Before adding subsequent rows (paits of points), check to make sure the starting point is greater than the last point in the final_errors_range array
        elif errors[k, 0] > final_error_ranges.flat[-1]:
            final_error_ranges = np.append(final_error_ranges, [errors[k, :]])
    
    #rehape the error range into list of two points
    final_error_ranges = np.reshape(final_error_ranges, (-1,2)).astype(int)
        
    #grab the smoothed moving average data
    y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)    
    
    #loop through the list of ordered pairs of final point ranges and mark points for removal
    mins_to_remove = []
    maxes_to_remove = []
    
    #loop through the final_error_ranges array to cataloge the points that should be removed
    for p in range(0, final_error_ranges.shape[0]):
        #Define the range of x-values we will compare before pruning
        x = all_extrema[0][range(final_error_ranges[p][0], final_error_ranges[p][1] + 1)].astype(int)
        #if the range of values were are comparing are all "mins"...
        if all_extrema[1][final_error_ranges[p, 0]] == "min":
            #if the first and last point in the range are both zero...
            if y[x[0]] == 0 and y[x[-1]] == 0:
                #mark all but the first and last point for removal
                if len(x) > 2:
                    mins_to_remove = np.append(mins_to_remove, np.setdiff1d(x, np.append(x[0], x[-1])))
            #otherwise, all therel_min points we are companing are not zero, so we remove all but the smallest value 
            else:
                min_point = np.argmin(y[x])
                mins_to_remove = np.append(mins_to_remove, np.setdiff1d(x, x[min_point]))
        #otherwise if the range of values we are comparing ar all "maxes", remove all but the largest point
        elif all_extrema[1][final_error_ranges[p, 0]] == "max":
            max_point = np.argmax(y[x])
            maxes_to_remove = np.append(maxes_to_remove, np.setdiff1d(x, x[max_point]))
    
    #Add the new prunned set o rel_min and rel_maxes to the Extrema Dataframe
    rel_max_6 = Extrema_Testing.loc[i, 'Rel Max 6']
    rel_min_6 = Extrema_Testing.loc[i, 'Rel Min 6']
        
    Extrema_Testing.loc[i, 'Rel Max F'] = np.setdiff1d(rel_max_6, maxes_to_remove)
    Extrema_Testing.loc[i, 'Rel Min F'] = np.setdiff1d(rel_min_6, mins_to_remove)


#graphs the smoothed MA data, all rel ins and rel amxes, and the non-pruned rel mins and rel maxes    
graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Extrema_Testing.pdf')
for i in Extrema_Testing.index.values:
    x = range(0, len(cases_MA.columns.values))
    y = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(i)
    
    ax.plot(x, y, zorder = 0, label = "SVG 15,3")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Min'], y[Extrema_Testing.loc[i, 'Rel Min']], marker = '|', color = 'red', zorder = 5, label = "Rel Min")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Max'], y[Extrema_Testing.loc[i, 'Rel Max']], marker = '|', color = 'green', zorder = 5, label = "Rel Max")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Max 2'], y[Extrema_Testing.loc[i, 'Rel Max 2']], marker = 'x', color = 'black', zorder = 10, label = "Rel Max 2")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Min 2'], y[Extrema_Testing.loc[i, 'Rel Min 2']], marker = 'x', color = 'gray', zorder = 10, label = "Rel Min 2")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Max 3'], y[Extrema_Testing.loc[i, 'Rel Max 3']], marker = '+', color = 'orange', zorder = 15, label = "Rel Max 3")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Min 3'], y[Extrema_Testing.loc[i, 'Rel Min 3']], marker = '+', color = 'indigo', zorder = 15, label = "Rel Min 3")
    if np.any(np.isnan(Extrema_Testing.loc[i, 'Rel Max 4'])):
        l=0
    else:
        ax.scatter(Extrema_Testing.loc[i, 'Rel Max 4'], y[Extrema_Testing.loc[i, 'Rel Max 4']], marker = 'o', color = 'cyan', zorder = 20, label = "Rel Max 4")
        ax.scatter(Extrema_Testing.loc[i, 'Rel Min 4'], y[Extrema_Testing.loc[i, 'Rel Min 4']], marker = 'o', color = 'magenta', zorder = 20, label = "Rel Min 4")
        ax.scatter(Extrema_Testing.loc[i, 'Rel Max 5'], y[Extrema_Testing.loc[i, 'Rel Max 5']], marker = 'o', color = 'lime', zorder = 25, label = "Rel Max 5")
        ax.scatter(Extrema_Testing.loc[i, 'Rel Min 5'], y[Extrema_Testing.loc[i, 'Rel Min 5']], marker = 'o', color = 'darkgoldenrod', zorder = 25, label = "Rel Min 5")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Max 6'], y[Extrema_Testing.loc[i, 'Rel Max 6']], marker = 'X', color = 'black', zorder = 30, label = "Rel Max 6")
    ax.scatter(Extrema_Testing.loc[i, 'Rel Min 6'], y[Extrema_Testing.loc[i, 'Rel Min 6']], marker = 'X', color = 'dimgrey', zorder = 30, label = "Rel Min 6")
    ax.plot(np.sort(np.append(Extrema_Testing.loc[i, 'Rel Max F'],Extrema_Testing.loc[i, 'Rel Min F'])), y[np.sort(np.append(Extrema_Testing.loc[i, 'Rel Max F'],Extrema_Testing.loc[i, 'Rel Min F']))], zorder = 35, label = "Rel Max Final")
    
    ax.legend()
    
    graphs_File.savefig(fig)
    
graphs_File.close() 
# =============================================================================
# 
# for i in Extrema_Testing.index.values:
#     a = len(Extrema_Testing.loc[i, 'Rel Min'])
#     b = len(Extrema_Testing.loc[i, 'Rel Max'])
#     c = a + b
#     d = len(Extrema_Testing.loc[i, 'Rel Min 2'])
#     e = len(Extrema_Testing.loc[i, 'Rel Max 2'])
#     f = d + e
#     g = len(Extrema_Testing.loc[i, 'Rel Min 3'])
#     h = len(Extrema_Testing.loc[i, 'Rel Max 3'])
#     j = g + h
#     print(i, a, b, c, d, e, f, g, h, j)
# =============================================================================

# =============================================================================
# SVG_Cases_Smoothed = pd.DataFrame(data = None, index = Extrema_Testing.index.values, columns = range(0, len(cases_MA.columns.values) - 1))
# for i in Extrema_Testing.index.values:
#     x = np.clip(svg(cases_MA.loc[i, :].values, 15, 3), a_min = 0, a_max = None)
#     for j in range(0, len(x)):
#         SVG_Cases_Smoothed.loc[i, j] = x[j]
# 
# SVG_Cases_Smoothed.to_excel('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/Extrema_Testing_Cases_Smoothed.xlsx')
# =============================================================================
