# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:24:14 2020

@author: theod
"""
import pandas as pd
import numpy as np
import scipy.interpolate as interp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks

#This file contains the metric generations functions (and performs the metric generation) on the COVID-19 Data
#All functions add columns to the dataframe that stores all of the derived metrics

#Create the dataframe that will store the generated metrics
#Determine the unique list of all ISO that can be found in all the datasets
All_ISO = np.append(cases_File.index.values, death_File.index.values)
All_ISO = np.append(All_ISO, new_Testing.index.values)
All_ISO = np.append(All_ISO, economic_IMF_File.index.values)
All_ISO = np.append(All_ISO, government_Response_Index.index.values)
All_ISO = np.append(All_ISO, healthcare_WHO_File.index.values)
All_ISO = np.append(All_ISO, pandemic_prepardness_JH_File.index.values)
All_ISO = np.append(All_ISO, press_freedom_File.index.values)
All_ISO = np.unique(All_ISO)

#Create the dataframe with the indexes of all the countries
Country_Metrics = pd.DataFrame(data = None, index = All_ISO)

#This function reads in a dataframe, the metric dataframe, and a column name and calculated the sums of the values of the dataframe for every country in it.
def summation_Func(data_df, metric_df, Name):
    #Sum the data in the data dataframe
    summation = data_df.sum(axis = 1)
    
    #add the new column to the metric storage dataframe
    metric_df[Name] = np.nan
    
    #Add the new values to the metric dataframe
    for i in summation.index.values:
        metric_df.loc[i, Name] = summation.loc[i]

    return

#This function reads in the metric dataframe and the names of two columns, it then adds the comparision between the second and first column as a ratio
#For example, the fatality ratio of would be the "Total Cases per Million" to "Total Deaths per Million"
def percent_Comparison(metric_df, Col1, Col2, Name):
    #Create a list of ISOs that are in both Col1 and Col2
    #Get all values that aren't null from column 1
    col1_values = metric_df.loc[:, Col1]
    col1_values = col1_values.dropna()
    #Get all values tht are not null from column 2
    col2_values = metric_df.loc[:, Col2]
    col2_values = col2_values.dropna()
    #Combine the lists where we only keep indexes that exist in both lists
    ISO_List = set(col1_values.index.values).intersection(col2_values.index.values)
    
    #add the new column to the metric storage dataframe
    metric_df[Name] = np.nan
    
    #Add the new values to the  metric dataframe
    for i in ISO_List:
        #Have an if statement to do nothing if either values is zero
        if (metric_df.loc[i, Col1] == 0 or metric_df.loc[i, Col2] == 0):
            pass
        else:
            metric_df.loc[i, Name] = metric_df.loc[i, Col2] / metric_df.loc[i, Col1]
    
    return

#This function reads in the metric dataframe and the names of two columns, it then adds the comparision between the first and second column as an absolute |%|
def abs_percent_Comparison(metric_df, Col1, Col2, Name):
    #Create a list of ISOs that are in both Col1 and Col2
    #Get all values that aren't null from column 1
    col1_values = metric_df.loc[:, Col1]
    col1_values = col1_values.dropna()
    #Get all values tht are not null from column 2
    col2_values = metric_df.loc[:, Col2]
    col2_values = col2_values.dropna()
    #Combine the lists where we only keep indexes that exist in both lists
    ISO_List = set(col1_values.index.values).intersection(col2_values.index.values)
    
    #add the new column to the metric storage dataframe
    metric_df[Name] = np.nan
    
    #Add the new values to the  metric dataframe
    for i in ISO_List:
        #Have an if statement to do nothing if either values is zero
        if (metric_df.loc[i, Col1] == 0 or metric_df.loc[i, Col2] == 0):
            k=0
        else:
            metric_df.loc[i, Name] = abs(metric_df.loc[i, Col1]) / abs(metric_df.loc[i, Col2])
    
    return

#This function reads in the metric dataframe and the target dataframe and returns the maximum value for eveyr row of the target dataframe in the metric dataframe
def max_value_Func(metric_df, target_df, Name):
    #Find the maximum values from teh target dataframe
    maxes = target_df.max(axis = 1)
    
    #add the new column to the metric storage dataframe
    metric_df[Name] = np.nan
    
    #Add the new values to the metric dataframe
    for i in maxes.index.values:
        metric_df.loc[i, Name] = maxes.loc[i]
        
    return

#This function reads in the metric dataframe and an interpolation data frame and, based on the first derivatie of the interpolation, calculates the number of peaks
#the Moving average dataframe is used for ange setting 
def num_Peaks(metric_df, extrema_df, extrema_df_col, Name):
    #Add the new column to the metric dataframe
    metric_df[Name] = np.nan
    
    #Add the values to the metric dataframe
    for i in extrema_df.index.values:
        metric_df.loc[i, Name] = len(extrema_df.loc[i, extrema_df_col])
    
    return

#This function reads in the mertic and interpolation dataframe and then, based on the first derivative, returns the avgerage and maximum rate of growht for the first peak and all subsequent peaks
def rate_of_Growth(metric_df, interp_df, extrema_df, dataset, Outbreak_Day, Avg_Growth, Max_Growth, Growth_Len, Avg_Submission, Max_Submission, Submission_Len, Total_Len, Peak_Pt, Valley_Pt):
    #Add the new columns to the metric dataframe ()make sure the data columns are formatted to be able to hold arrays 
    metric_df[dataset + Outbreak_Day] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Avg_Growth] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Max_Growth] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Growth_Len] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Avg_Submission] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Max_Submission] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Submission_Len] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Total_Len] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Peak_Pt] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Valley_Pt] = np.empty((len(metric_df), 0)).tolist()
    metric_df[dataset + Outbreak_Day + "(First Outbreak)"] = np.nan
    metric_df[dataset + Avg_Growth + "(First Outbreak)"] = np.nan
    metric_df[dataset + Max_Growth + "(First Outbreak)"] = np.nan
    metric_df[dataset + Growth_Len + "(First Outbreak)"] = np.nan
    metric_df[dataset + Avg_Submission + "(First Outbreak)"] = np.nan
    metric_df[dataset + Max_Submission + "(First Outbreak)"] = np.nan
    metric_df[dataset + Submission_Len + "(First Outbreak)"] = np.nan
    metric_df[dataset + Total_Len + "(First Outbreak)"] = np.nan
    metric_df[dataset + Peak_Pt + "(First Outbreak)"] = np.nan
    metric_df[dataset + Valley_Pt + "(First Outbreak)"] = np.nan
    
    #Loop through the extrema data frame and, using the interpolation data, fill in the lists
    for i in extrema_df.index.values:
        #Initialize all of the arrays that will be used to store data
        Outbreak_Day_List = []
        Avg_Growth_List = []
        Max_Growth_List = []
        Growth_Len_List = []
        Avg_Submission_List = []
        Max_Submission_List = []
        Submission_Len_List = []
        Total_Len_List = []
        Peak_Pt_List = []
        Valley_Pt_List = []
        
        #Make a list of all the mins and maxes in order
        #Create a 2d array of all rel_min labeled as mins
        rel_mins = extrema_df.loc[i, "Mins Final"]
        rel_mins = np.vstack((rel_mins, [0] * len(rel_mins)))
        #create a 2d array of all rel_max labeld as max
        rel_maxes = extrema_df.loc[i, "Maxes Final"]
        rel_maxes = np.vstack((rel_maxes, [1] * len(rel_maxes)))
        #Combine the rel min and rel max arrays and sort them by x-point
        all_extrema = np.concatenate((rel_mins, rel_maxes), axis = 1)
        all_extrema = all_extrema[:, all_extrema[0].astype(int).argsort()]
        all_extrema = all_extrema.astype(int) 
        
        #Is there is data the can be analyzed
        if all_extrema.shape[1] > 0:
            #Get the interpolation values and their derivative from the interpolaiton dataframe
            x = np.arange(0, all_extrema[0][-1]+1)
            y = interp_df.loc[i, 'PChip'](x)
            dy = interp_df.loc[i, 'PChip'](x,1)
        
            #Loop through the all_extrema data to fill in the rest of the data
            for j in range(0, all_extrema.shape[1]):
                if j == 0:
                    if all_extrema[1][j] == 1:
                        Outbreak_Day_List = np.append(Outbreak_Day_List, all_extrema[0][j])
                        Avg_Growth_List = np.append(Avg_Growth_List, None)
                        Max_Growth_List = np.append(Max_Growth_List, None)
                        Growth_Len_List = np.append(Growth_Len_List, None)
                        Avg_Submission_List = np.append(Avg_Submission_List, np.mean(dy[all_extrema[0][j]:all_extrema[0][j+1]]))
                        Max_Submission_List = np.append(Max_Submission_List, np.amin(dy[all_extrema[0][j]:all_extrema[0][j+1]]))
                        Submission_Len_List = np.append(Submission_Len_List, all_extrema[0][j+1] - all_extrema[0][j])
                        Total_Len_List = np.append(Total_Len_List, None)
                        Peak_Pt_List = np.append(Peak_Pt_List, y[all_extrema[0][j]])
                        Valley_Pt_List = np.append(Valley_Pt_List, y[all_extrema[0][j+1]])
                elif j == all_extrema.shape[1] - 1:
                    if all_extrema[1][j] == 1:
                        Outbreak_Day_List = np.append(Outbreak_Day_List, all_extrema[0][j])
                        Avg_Growth_List = np.append(Avg_Growth_List, np.mean(dy[all_extrema[0][j-1]:all_extrema[0][j]]))
                        Max_Growth_List = np.append(Max_Growth_List, np.amax(dy[all_extrema[0][j-1]:all_extrema[0][j]]))
                        Growth_Len_List = np.append(Growth_Len_List, all_extrema[0][j] - all_extrema[0][j-1])
                        Avg_Submission_List = np.append(Avg_Submission_List, None)
                        Max_Submission_List = np.append(Max_Submission_List, None)
                        Submission_Len_List = np.append(Submission_Len_List, None)
                        Total_Len_List = np.append(Total_Len_List, None)
                        Peak_Pt_List = np.append(Peak_Pt_List, y[all_extrema[0][j]])
                        Valley_Pt_List = np.append(Valley_Pt_List, None)
                elif all_extrema[1][j] == 1:
                    Outbreak_Day_List = np.append(Outbreak_Day_List, all_extrema[0][j])
                    Avg_Growth_List = np.append(Avg_Growth_List, np.mean(dy[all_extrema[0][j-1]:all_extrema[0][j]]))
                    Max_Growth_List = np.append(Max_Growth_List, np.amax(dy[all_extrema[0][j-1]:all_extrema[0][j]]))
                    Growth_Len_List = np.append(Growth_Len_List, all_extrema[0][j] - all_extrema[0][j-1])
                    Avg_Submission_List = np.append(Avg_Submission_List, np.mean(dy[all_extrema[0][j]:all_extrema[0][j+1]]))
                    Max_Submission_List = np.append(Max_Submission_List, np.amin(dy[all_extrema[0][j]:all_extrema[0][j+1]]))
                    Submission_Len_List = np.append(Submission_Len_List, all_extrema[0][j+1] - all_extrema[0][j])
                    Total_Len_List = np.append(Total_Len_List, all_extrema[0][j+1] - all_extrema[0][j-1])
                    Peak_Pt_List = np.append(Peak_Pt_List, y[all_extrema[0][j]])
                    Valley_Pt_List = np.append(Valley_Pt_List, y[all_extrema[0][j+1]])
                        
            #Fill in the other columns with their data from calculated in the for loop above
            metric_df.at[i, dataset + Outbreak_Day] = Outbreak_Day_List
            metric_df.at[i, dataset + Avg_Growth] = Avg_Growth_List
            metric_df.at[i, dataset + Max_Growth] = Max_Growth_List
            metric_df.at[i, dataset + Growth_Len] = Growth_Len_List
            metric_df.at[i, dataset + Avg_Submission] = Avg_Submission_List
            metric_df.at[i, dataset + Max_Submission] = Max_Submission_List
            metric_df.at[i, dataset + Submission_Len] = Submission_Len_List
            metric_df.at[i, dataset + Total_Len] = Total_Len_List
            metric_df.at[i, dataset + Peak_Pt] = Peak_Pt_List
            metric_df.at[i, dataset + Valley_Pt] = Valley_Pt_List
            if Outbreak_Day_List.size == 0 or Outbreak_Day_List.all() == [None]:
                pass   
            else:
                metric_df.at[i, dataset + Outbreak_Day + "(First Outbreak)"] = Outbreak_Day_List[0]
            if Avg_Growth_List.size == 0 or Avg_Growth_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Avg_Growth + "(First Outbreak)"] = Avg_Growth_List[0]
            if Max_Growth_List.size == 0 or Max_Growth_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Max_Growth + "(First Outbreak)"] = Max_Growth_List[0]
            if Growth_Len_List.size == 0 or Growth_Len_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Growth_Len + "(First Outbreak)"] = Growth_Len_List[0]
            if Avg_Submission_List.size == 0 or Avg_Submission_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Avg_Submission + "(First Outbreak)"] = Avg_Submission_List[0]
            if Max_Submission_List.size == 0 or Max_Submission_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Max_Submission + "(First Outbreak)"] = Max_Submission_List[0]
            if Submission_Len_List.size == 0 or Submission_Len_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Submission_Len + "(First Outbreak)"] = Submission_Len_List[0]
            if Total_Len_List.size == 0 or Total_Len_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Total_Len + "(First Outbreak)"] = Total_Len_List[0]
            if Peak_Pt_List.size == 0 or Peak_Pt_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Peak_Pt + "(First Outbreak)"] = Peak_Pt_List[0]
            if Valley_Pt_List.size == 0 or Valley_Pt_List.all() == [None]:
                pass
            else:
                metric_df.at[i, dataset + Valley_Pt + "(First Outbreak)"] = Valley_Pt_List[0]
            
    return

#This function utilizes K-Means clustering in order to pair up the case and death peaks in terms of their peak date, and start/end date
def peak_Pairing(metric_df, cases_Extrema_df, deaths_Extrema_df, Case_Death_Pairs, case_smoothed_df, death_smoothed_df, Graph):
    #Create the new column in the metric df for data storage
    metric_df[Case_Death_Pairs] = np.empty((len(metric_df), 0)).tolist()
    
    if Graph == True:
        graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Peak_Pairing_Graphs' + '.pdf')
    
    #loop through all the data (cases and death data will have the same indexes)
    for i in cases_Extrema_df.index.values:
        #Create a stoage list to store a peak's origin dataset, date, start date, and end date 
        points = []
        
        #Make a list of all the cases mins and maxes in order
        #Create a 2d array of all rel_min labeled as mins
        cases_rel_mins = cases_Extrema_df.loc[i, "Mins Final"]
        cases_rel_mins = np.vstack((cases_rel_mins, [0] * len(cases_rel_mins)))
        #create a 2d array of all rel_max labeld as max
        cases_rel_maxes = cases_Extrema_df.loc[i, "Maxes Final"]
        cases_rel_maxes = np.vstack((cases_rel_maxes, [1] * len(cases_rel_maxes)))
        #Combine the rel min and rel max arrays and sort them by x-point
        cases_extrema = np.concatenate((cases_rel_mins, cases_rel_maxes), axis = 1)
        cases_extrema = cases_extrema[:, cases_extrema[0].astype(int).argsort()]
        cases_extrema = cases_extrema.astype(int) 
        
        #Loop through the cases_extrema array and use the information to fill in the points_df dataframe
        for j in range(0, cases_extrema.shape[1]):
            if j == 0 and cases_extrema[1][j] == 1:
                points.append(['Case', cases_extrema[0][j], cases_extrema[0][j], cases_extrema[0][j+1]])
            elif j == cases_extrema.shape[1] - 1 and cases_extrema[1][j] == 1:
                points.append(['Case', cases_extrema[0][j], cases_extrema[0][j-1], cases_extrema[0][j]])
            elif cases_extrema[1][j] == 1:
                points.append(['Case', cases_extrema[0][j], cases_extrema[0][j-1], cases_extrema[0][j+1]])
        
        #Make a list of all the deaths mins and maxes in order
        #Create a 2d array of all rel_min labeled as mins
        deaths_rel_mins = deaths_Extrema_df.loc[i, "Mins Final"]
        deaths_rel_mins = np.vstack((deaths_rel_mins, [0] * len(deaths_rel_mins)))
        #create a 2d array of all rel_max labeld as max
        deaths_rel_maxes = deaths_Extrema_df.loc[i, "Maxes Final"]
        deaths_rel_maxes = np.vstack((deaths_rel_maxes, [1] * len(deaths_rel_maxes)))
        #Combine the rel min and rel max arrays and sort them by x-point
        deaths_extrema = np.concatenate((deaths_rel_mins, deaths_rel_maxes), axis = 1)
        deaths_extrema = deaths_extrema[:, deaths_extrema[0].astype(int).argsort()]
        deaths_extrema = deaths_extrema.astype(int) 
        
        #Loop through the deaths_extrema array and use the information to fill in the points_df dataframe
        for j in range(0, deaths_extrema.shape[1]):
            if j == 0 and deaths_extrema[1][j] == 1:
                points.append(['Death', deaths_extrema[0][j], deaths_extrema[0][j], deaths_extrema[0][j+1]])
            elif j == deaths_extrema.shape[1] - 1 and deaths_extrema[1][j] == 1:
                points.append(['Death', deaths_extrema[0][j], deaths_extrema[0][j-1], deaths_extrema[0][j]])
            elif deaths_extrema[1][j] == 1:
                points.append(['Death', deaths_extrema[0][j], deaths_extrema[0][j-1], deaths_extrema[0][j+1]])
        
        #Convert the points list into a dataframe
        points_df = pd.DataFrame(points, columns = ['Dataset', 'Peak Date', 'Start Date', 'End Date'])
        
        #Determine the number of case, death, min, and max clusters
        case_Clusters = (points_df['Dataset']=='Case').sum()
        death_Clusters = (points_df['Dataset']=='Death').sum()
        min_Clusters = min(case_Clusters, death_Clusters)
        max_Clusters = max(case_Clusters, death_Clusters)
        
        #create an array of silhouette coeficents that will determine the number of cluster to be performed
        sil = []
        opt_K_Found = 0
        #as long as the minimum number of clusters is greater than 1
        if min_Clusters > 1:
            #Update the optimal k found indicator
            opt_K_Found = 1
            #Loop through all possible cluster sizes and record the shilouette values
            for k in range(min_Clusters, max_Clusters+1):
                kmeans = KMeans(n_clusters = k).fit(points_df[['Peak Date', 'Start Date', 'End Date']])
                labels = kmeans.labels_
                sil.append(silhouette_score(points_df[['Peak Date', 'Start Date', 'End Date']], labels, metric = 'euclidean'))
       
        #Update the number of cluster values according to the silhouette values and whether or no the shilutette values could be found
        num_Clusters = None
        #If the min_Clusters is > 1 and the optimal k value could be calucalted, use it a num_Clusters
        if min_Clusters > 1 and opt_K_Found == 1:
            num_Clusters = np.argmax(sil) + min_Clusters
        #Otherwise, make num_Clusters = max_Clusters
        elif opt_K_Found == 0:
            num_Clusters = max_Clusters
        
        #Perform the KMeans clustering analysis
        #If there are only one case and death peak...
        if case_Clusters == 1 and death_Clusters == 1:
            #Initalize the Clusterign Parameters
            km = KMeans(n_clusters = 1)
            #Perfrom the CLsutering
            cluster_predictions = km.fit_predict(points_df[['Peak Date', 'Start Date', 'End Date']])
            #Add the clusting data to the points_df dataframe
            points_df['Cluster'] = cluster_predictions
        #Othersiese
        elif min_Clusters > 0:
            #Initalize the Clusterign Parameters
            km = KMeans(n_clusters = num_Clusters)
            #Perfrom the CLsutering
            cluster_predictions = km.fit_predict(points_df[['Peak Date', 'Start Date', 'End Date']])
            #Add the clusting data to the points_df dataframe
            points_df['Cluster'] = cluster_predictions
        
        #Create the list for the metric dataframe
        case_death_Peak_Pairs = []
        cluster_list = []
        #If the clustering was able to be performed
        if min_Clusters > 0:
            #Loop through all rows in the points_df dataframe
            for n in range(0, len(points_df.index.values)):
                #If the cluster group has not been added yet...
                if (points_df.loc[n, 'Cluster'] in cluster_list) == False:
                    #Add the cluster points ot the tracking list
                    cluster_list.append(points_df.loc[n, 'Cluster'])
                    case_points = []
                    death_points = []
                    #Loop trhough all following rows
                    for m in range(n, len(points_df.index.values)):
                        #if the point belongs to the clusting group... 
                        if points_df.loc[m, 'Cluster'] == points_df.loc[n, 'Cluster']:
                            #And is is a case, add it to the case group
                            if points_df.loc[m, 'Dataset'] == "Case":
                                case_points.append(points_df.loc[m, 'Peak Date'])
                            #And is is a death, add it to the death group
                            elif points_df.loc[m, 'Dataset'] == "Death":
                                death_points.append(points_df.loc[m, 'Peak Date'])
                    #Add the final case-death pairs for the clsuter to the overall list of case-death pairs
                    case_death_Peak_Pairs.append([case_points, death_points])
        
        #Add the final case-death-peak pairs to the metric dataframe
        metric_df.at[i, Case_Death_Pairs] = case_death_Peak_Pairs
        
        #If graphing is truned onn, graph the clusters
        if Graph == True:
            print(i)
            #make the figure
            fig = plt.figure(constrained_layout = True)
            gs = fig.add_gridspec(2,1)
            
            #Initialize the axes
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title(i + ": Cases vs. Time")
            ax1.set(xlabel = "Time", ylabel = "Cases per Million")
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set(xlabel = "Time", ylabel = "Deaths per Million")
            ax2.set_title(i + ": Deaths vs. Time")
            
            #make the case graph
            y = case_smoothed_df.loc[i, :]
            y1 = death_smoothed_df.loc[i, :]
            
            #Plot the case and death data in their entirity in the background
            ax1.plot(np.arange(0, len(y)), y, zorder = 5, color = 'grey')
            ax2.plot(np.arange(0, len(y1)), y1, zorder = 5, color = 'grey')
            
            for k in range(0, len(points_df.index.values)):
                if len(points_df.columns.values) == 4:
                    if points_df.loc[k, 'Dataset'] == "Case":
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax1.plot(x, y[x], zorder = 10, color = 'black')
                    elif points_df.loc[k, 'Dataset'] == "Death":
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax2.plot(x, y1[x], zorder = 10, color = 'black')
                elif points_df.loc[k, 'Dataset'] == "Case":
                    if points_df.loc[k, 'Cluster'] == 0:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax1.plot(x, y[x], zorder = 10, color = 'blue')
                    elif points_df.loc[k,'Cluster'] == 1:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax1.plot(x, y[x], zorder = 10, color = 'red')
                    elif points_df.loc[k,'Cluster'] == 2:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax1.plot(x, y[x], zorder = 10, color = 'green')
                    elif points_df.loc[k,'Cluster'] == 3:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax1.plot(x, y[x], zorder = 10, color = 'orange')
                    elif points_df.loc[k,'Cluster'] == 4:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax1.plot(x, y[x], zorder = 10, color = 'purple')
                    elif points_df.loc[k,'Cluster'] == 5:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax1.plot(x, y[x], zorder = 10, color = 'pink')
                    else:
                        print('ERROR')
                elif points_df.loc[k, 'Dataset'] == "Death":
                    if points_df.loc[k, 'Cluster'] == 0:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax2.plot(x, y1[x], zorder = 10, color = 'blue')
                    elif points_df.loc[k,'Cluster'] == 1:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax2.plot(x, y1[x], zorder = 10, color = 'red')
                    elif points_df.loc[k,'Cluster'] == 2:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax2.plot(x, y1[x], zorder = 10, color = 'green')
                    elif points_df.loc[k,'Cluster'] == 3:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax2.plot(x, y1[x], zorder = 10, color = 'orange')
                    elif points_df.loc[k,'Cluster'] == 4:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax2.plot(x, y1[x], zorder = 10, color = 'purple')
                    elif points_df.loc[k,'Cluster'] == 5:
                        x = np.arange(points_df.loc[k, 'Start Date'], points_df.loc[k, 'End Date'])
                        ax2.plot(x, y1[x], zorder = 10, color = 'pink')
                    else:
                        print('ERROR')
                
            #Save the Figure
            if Graph == True:
                graphs_File.savefig(fig)
    
    #Save all the graphs to the PDF
    if Graph == True:
        graphs_File.close()
                
        
#THIS IS LEFTOVER CODE THAT CAN PLOT THE PAIRED PEAKS ON THE SAME GRAPH FOR VALIDATION AND ERROR CHECKING
# =============================================================================
#         fig, ax1 = plt.subplots()
#         ax2 = ax1.twinx()
#         fig.suptitle(i + ", " + num_Clusters.astype(str) + ", " + case_Clusters.astype(str) + ", " + death_Clusters.astype(str))
#         
#         #Perform the clustering analysis
#         if case_Clusters == 1 and death_Clusters == 1:
#             km = KMeans(n_clusters = 1)
#             cluster_predictions = km.fit_predict(points_df[['Peak Date', 'Start Date', 'End Date']])
#             points_df['Cluster'] = cluster_predictions
#             
#             case_x = np.arange(points_df.loc[0, 'Start Date'], points_df.loc[0, 'End Date'])
#             death_x = np.arange(points_df.loc[1, 'Start Date'], points_df.loc[1, 'End Date'])
#             case_y = case_smoothed_df.loc[i][case_x]
#             death_y = death_smoothed_df.loc[i][death_x]
#             
#             ax1.plot(case_x, case_y, color = 'blue', ls = '-')
#             ax2.plot(death_x, death_y, color = 'blue', ls = '--')
#             graphs_File.savefig(fig)
#         elif min_Clusters > 0:
#             km = KMeans(n_clusters = num_Clusters)
#             cluster_predictions = km.fit_predict(points_df[['Peak Date', 'Start Date', 'End Date']])
#             points_df['Cluster'] = cluster_predictions
#         
#             for j in range(0, len(points_df.index.values)):
#                 x = np.arange(points_df.loc[j, 'Start Date'], points_df.loc[j, 'End Date'])
#                 case_y = case_smoothed_df.loc[i][x]
#                 death_y = death_smoothed_df.loc[i][x]
#                 if points_df.loc[j, 'Dataset'] == "Case":
#                     if points_df.loc[j, 'Cluster'] == 0:
#                         ax1.plot(x, case_y, color = 'blue', ls = '-')
#                     elif points_df.loc[j, 'Cluster'] == 1:
#                         ax1.plot(x, case_y, color = 'orange', ls = '-')
#                     elif points_df.loc[j, 'Cluster'] == 2:
#                         ax1.plot(x, case_y, color = 'green', ls = '-')
#                     elif points_df.loc[j, 'Cluster'] == 3:
#                         ax1.plot(x, case_y, color = 'red', ls = '-')
#                     elif points_df.loc[j, 'Cluster'] == 4:
#                         ax1.plot(x, case_y, color = 'purple', ls = '-')
#                     elif points_df.loc[j, 'Cluster'] == 5:
#                         ax1.plot(x, case_y, color = 'black', ls = '-')
#                 elif points_df.loc[j, 'Dataset'] == "Death":
#                     if points_df.loc[j, 'Cluster'] == 0:
#                         ax2.plot(x, death_y, color = 'blue', ls = '--')
#                     elif points_df.loc[j, 'Cluster'] == 1:
#                         ax2.plot(x, death_y, color = 'orange', ls = '--')
#                     elif points_df.loc[j, 'Cluster'] == 2:
#                         ax2.plot(x, death_y, color = 'green', ls = '--')
#                     elif points_df.loc[j, 'Cluster'] == 3:
#                         ax2.plot(x, death_y, color = 'red', ls = '--')
#                     elif points_df.loc[j, 'Cluster'] == 4:
#                         ax2.plot(x, death_y, color = 'purple', ls = '--')
#                     elif points_df.loc[j, 'Cluster'] == 5:
#                         ax2.plot(x, death_y, color = 'black', ls = '--')
#             graphs_File.savefig(fig)
#         else:
#             x = np.arange(0, len(case_smoothed_df.columns.values))
#             case_y = case_smoothed_df.loc[i][x]
#             death_y = death_smoothed_df.loc[i][x]
#             ax1.plot(x, case_y, color = 'c')
#             ax2.plot(x, death_y, color = 'm')
#             graphs_File.savefig(fig)    
#         
#         
#     graphs_File.close() 
# =============================================================================
        
    return

#This function finds some of the metric data for the case-death peak pairs that have been prviosly identified
def rate_of_Growth_Peak_Pairs(metric_df, case_smoothed_df, death_smoothed_df, new_Col):
    #Create the new columns in the metric dataframe
    metric_df['Number of Case-Death Pairs'] = np.nan
    metric_df[new_Col] = np.empty((len(metric_df), 0)).tolist()
    
    #loop though all the countries
    for i in case_smoothed_df.index.values:
        #determine the number of case-death pairs
        case_death_pairs = metric_df.loc[i, 'Case-Death Pairs']
        #define the cases and death data
        cases = case_smoothed_df.loc[i, :]
        deaths = death_smoothed_df.loc[i, :]
        #make a list ot record the case-death pair ratios
        case_death_ratio_list = []
        count = 0
        
        #for every case-death pair
        for j in range(0, len(case_death_pairs)):
            #If a case-death pair exists
            if case_death_pairs[j][0] != [] and case_death_pairs[j][1] != []:
                #increase the counter
                count = count + 1
                #loop though the cases and deaths (incase there are more than one case/death for a case-death pair) and record the maximum case/death peak
                case_peak = 0
                death_peak = 0
                for n in range(0, len(case_death_pairs[j][0])):
                    if case_death_pairs[j][0][n] > case_peak:
                        case_peak = cases[case_death_pairs[j][0][n]]
                for m in range(0, len(case_death_pairs[j][1])):
                    if case_death_pairs[j][1][m] > death_peak:
                        death_peak = deaths[case_death_pairs[j][1][m]]
                
                #calculaate the case_death pair ratio and add it to the list 
                case_death_ratio = case_peak / death_peak
                case_death_ratio_list.append(case_death_ratio)
                
        #Add the list of case-death pair ratios and the count to the metric dataframe
        metric_df.loc[i, 'Number of Case-Death Pairs'] = count 
        metric_df.at[i, new_Col] = case_death_ratio_list
    return

#This function calculates the total number of days that a country was at or above a certain level for a certain stringency index sub metric
def num_Days_Above_Level(metric_df, sub_index_df, level, level_Name):
    #Create the new column in the metric dataframe
    metric_df[level_Name] = np.nan
    
    for i in sub_index_df.index.values:
        #define the time-series data for country i's submetric
        country_sub_index_data = sub_index_df.loc[i, :]
        
        #count the number of days where the value is greater than the level
        count = sum(j >= level for j in country_sub_index_data)
        
        #add the number (count) to the metric dataframe
        metric_df.loc[i, level_Name] = count
        
    return

#This function reads in a subindex dataset and return the number of instances, the avg rate of grwoth and the max rate of growth of the COVID data beore the level of stringency was reached
def total_and_rate_Before_Max_Subindex(metric_df, sub_index_df, level, smoothed_df, interp_df, incidents_Col, Avg_Rate_Col, Max_Rate_Col):
    #Create a list of countries that exist in both the sub_index_df and smoothed_df
    country_list = set(sub_index_df.index.values).intersection(smoothed_df.index.values)
    
    #Create the columns in the metric dataframe
    metric_df[incidents_Col] = np.nan
    metric_df[Avg_Rate_Col] = np.nan
    metric_df[Max_Rate_Col] = np.nan
    
    #loop through all the countries in the country_list
    for i in country_list:
        #declare the sub_index_df, smoothed_df, and interp_df 1st derivative data
        sub_index_data = sub_index_df.loc[i, :]
        smoothed_data = smoothed_df.loc[i, :]
        x = np.arange(0, len(smoothed_df.columns.values))
        dy = interp_df.loc[i, 'PChip'](x,1)
        
        #calculate the offset between the smoothed data and the sub_metric data
        smoothed_offset = np.where(sub_index_df.columns.values == smoothed_df.columns.values[0])[0]
        
        #find the points where the sub_index data jumps to exceed the threshold
        jump_points = []
        for j in range(len(sub_index_data)):
            if j == 0:
                if sub_index_data[j] >= level:
                    jump_points.append(j)
            elif sub_index_data[j-1] < level and sub_index_data[j] >= level:
                jump_points.append(j)
        
        #Calculate the number of incidents, max_rate, and avg_rate that was observed before the first instance where the sub_index exceeded the level
        incidents = np.nan
        avg_rate = np.nan
        max_rate = np.nan
        #if the country met or exceeded the level...
        if len(jump_points) > 0:
            #if they started at the level before data exists or cases were recorded, make incidents, avg_rate, and max_rate as 0
            if jump_points[0] < int(smoothed_offset):
                incidents = 0
                avg_rate = 0
                max_rate = 0
            #Otherwise..
            else:
                #calculate instances
                incidents = sum(smoothed_data[0:jump_points[0]-int(smoothed_offset)])
                #determine the indeces where the first derviative of the interpolation exists
                valid_dy = ~np.isnan(dy[0:jump_points[0]-int(smoothed_offset)])
                #if the first derivative doesn't exist in the range, make the avg_rate and max_rate 0
                if sum(valid_dy) == 0:
                    avg_rate = 0
                    max_rate = 0
                #otherwise...
                else:
                    #determine the values that should be calculaated in the avergae (i.e. the values that exist)
                    nan_Locations = np.isnan(dy)
                    dy[nan_Locations] = 0
                    #calculate the avg_rate and max_rate
                    avg_rate = np.average(dy[0:jump_points[0]-int(smoothed_offset)], weights = valid_dy)
                    max_rate = np.amax(dy[0:jump_points[0]-int(smoothed_offset)])
        
        #Add the instances, avg_rate, and max_rate to the metric dataframe
        metric_df.loc[i, incidents_Col] = incidents
        metric_df.loc[i, Avg_Rate_Col] = avg_rate
        metric_df.loc[i, Max_Rate_Col] = max_rate
        
        
    return

#This function reads in a Index and return the total instacnes and rate of growth in the COVID data defore the first peak was reached.
def total_and_rate_Before_First_Max_Index_Peak(metric_df, index_df, smoothed_df, interp_df, incidents_Col, Avg_Rate_Col, Max_Rate_Col, Ratio_Col):
    #Create a list of countries that exist in both the index_df and smoothed_df
    country_list = set(index_df.index.values).intersection(smoothed_df.index.values)
    
    #Create the columns in the metric dataframe
    metric_df[incidents_Col] = np.nan
    metric_df[Avg_Rate_Col] = np.nan
    metric_df[Max_Rate_Col] = np.nan
    metric_df[Ratio_Col] = np.nan
    
    for i in country_list:
        #declare the index_df, smoothed_df, and interp_df 1st derivative data
        index_data = index_df.loc[i, :]
        smoothed_data = smoothed_df.loc[i, :]
        x = np.arange(0, len(smoothed_df.columns.values))
        dy = interp_df.loc[i, 'PChip'](x,1)
        
        #Declare the data that will be output
        instances = np.nan
        Avg_Rate = np.nan
        Max_Rate = np.nan
        index_instances_ratio = np.nan
        
        #calculate the offset between the smoothed data and the sub_metric data
        smoothed_offset = np.where(index_df.columns.values == smoothed_df.columns.values[0])[0]
        
        #Add a try-except statement to catch instances where all teh data is nans
        all_nans = False
        try:
            np.nanargmax(index_data.values)
        except ValueError:
            all_nans = True
        
        #Determine the date at which the max point was reached in the index, (as long as the there is data to analyze)
        if all_nans == False:
            #Use the find_peaks algorithm to determine the first peak in the index data
            peaks, _ = find_peaks(index_data)
            #Then ensure the first point in the peak is chosen (to account for instances where the peaks are plateaus)
            if len(peaks) == 0:
                max_index = np.nanargmax(index_data)
            else:
                max_index = np.nanargmax(index_data[0:peaks[0]+1])
            
            #Calulate the number of instacnes, Max Rate og Growth, Avg Rate of Growht, and Instances to Max Index Ratio before the first peak
            #if the peak was reached before data was recorded, make the variables zero
            if max_index < smoothed_offset:
                incidents = 0
                avg_rate = 0
                max_rate = 0
                index_instances_ratio = 0
            #Otherwise..
            else:
                #calculate instances and instacnes_index ratio
                instances = sum(smoothed_data[0:max_index-int(smoothed_offset)])
                index_instances_ratio = instances / index_data[max_index]
                #determine the indeces where the first derviative of the interpolation exists
                valid_dy = ~np.isnan(dy[0:max_index-int(smoothed_offset)])
                #if the first derivative doesn't exist in the range, make the avg_rate and max_rate 0
                if sum(valid_dy) == 0:
                    avg_rate = 0
                    max_rate = 0
                #otherwise...
                else:
                    #determine the values that should be calculaated in the avergae (i.e. the values that exist)
                    nan_Locations = np.isnan(dy)
                    dy[nan_Locations] = 0
                    #calculate the avg_rate and max_rate
                    avg_rate = np.average(dy[0:max_index-int(smoothed_offset)], weights = valid_dy)
                    max_rate = np.amax(dy[0:max_index-int(smoothed_offset)]) 
        
        #Add the new metrics to the metric_dataframe
        metric_df.loc[i, incidents_Col] = instances
        metric_df.loc[i, Avg_Rate_Col] = avg_rate
        metric_df.loc[i, Max_Rate_Col] = max_rate
        metric_df.loc[i, Ratio_Col] = index_instances_ratio
        
    return

#This funciton reads in two indexes and determine their ratio to one another (i.e. how they compare to one another)
def index_to_index_Ratio(metric_df, index1_df, index2_df, Col_Name):
    #Create the columns in the metric dataframe
    metric_df[Col_Name] = np.nan
    
    #Declare the ratio variable
    ratio = np.nan
    
    #Loop through all the countrues
    for i in index1_df.index.values:
        #Declare the two index time series data
        index1_data = index1_df.loc[i, :].values
        index2_data = index2_df.loc[i, :].values
        
        #Add a try-except statement to catch instances where all teh data is nans
        all_nans = False
        try:
            np.nanargmax(index1_data)
            np.nanargmax(index2_data)
        except ValueError:
            all_nans = True
        
        if all_nans == False:
            #Determine the points that are nan (and non-nan) inboth indeces
            index1_nan = np.isnan(index1_data)
            index2_nan = np.isnan(index2_data)
            index1_nonnan = ~np.isnan(index1_data)
            index2_nonnan = ~np.isnan(index2_data)
        
            #make the nan locations 0
            index1_data[index1_nan] = 0
            index2_data[index2_nan] = 0
        
            #Determine the average value of the indeces (negecting nan values)
            index1_avg = np.average(index1_data, weights = index1_nonnan)
            index2_avg = np.average(index2_data, weights = index2_nonnan)
        
            #Determine the final ratio
            ratio = index1_avg / index2_avg
        
        #Add the ratio to the metric_df dataframe
        metric_df.loc[i, Col_Name] = ratio
        
    return

summation_Func(cases_File, Country_Metrics, "Total Cases per Million")
summation_Func(death_File, Country_Metrics, "Total Deaths per Million")
summation_Func(new_Testing, Country_Metrics, "Total Tests per Million")

percent_Comparison(Country_Metrics, "Total Deaths per Million", "Total Cases per Million", "Case-Death Ratio")
percent_Comparison(Country_Metrics, "Total Cases per Million", "Total Tests per Million", "Test-Case Ratio")

max_value_Func(Country_Metrics, stringency_Index, "Max Stringency")
max_value_Func(Country_Metrics, government_Response_Index, "Max Government Response")
max_value_Func(Country_Metrics, containment_Health_Index, "Max Containment Health")
max_value_Func(Country_Metrics, economic_Support_Index, "Max Economic Support")

num_Peaks(Country_Metrics, cases_Extrema, "Maxes Final", "Number of Cases Peaks")
num_Peaks(Country_Metrics, deaths_Extrema, "Maxes Final", "Number of Deaths Peaks")

rate_of_Growth(Country_Metrics, case_Interpolation, cases_Extrema, "Cases ", "Outbreak Day", "Avg. Growth Rate", "Max. Growth Rate", "Growth Length", "Avg. Submission Rate", "Max. Submission Rate", "Submission Length", "Total Length", "Peak Value", "Valley Value")
rate_of_Growth(Country_Metrics, death_Interpolation, deaths_Extrema, "Deaths ", "Outbreak Day", "Avg. Growth Rate", "Max. Growth Rate", "Growth Length", "Avg. Submission Rate", "Max. Submission Rate", "Submission Length", "Total Length", "Peak Value", "Valley Value")

peak_Pairing(Country_Metrics, cases_Extrema, deaths_Extrema, "Case-Death Pairs", cases_smoothed_15_3, deaths_smoothed_15_3, False)

rate_of_Growth_Peak_Pairs(Country_Metrics, cases_smoothed_15_3, deaths_smoothed_15_3, "Case-Death Pair Peak Ratio")

num_Days_Above_Level(Country_Metrics, h1_publicinfocampaign, 2, "Num Days Public Information Campaign")
num_Days_Above_Level(Country_Metrics, h6_facialcoverings, 2, "Num Days Facial Coverings in Some Public Places")
num_Days_Above_Level(Country_Metrics, h6_facialcoverings, 3, "Num Days Facial Coverings in All Public Places")
num_Days_Above_Level(Country_Metrics, h6_facialcoverings, 4, "Num Days Facial Coverings Mandate")
num_Days_Above_Level(Country_Metrics, c1_schoolclosing, 2, "Num Days Some Schools Closed")
num_Days_Above_Level(Country_Metrics, c1_schoolclosing, 3, "Num Days All Schools Closed")
num_Days_Above_Level(Country_Metrics, c3_cancelpublicevents, 2, "Num Days Public Events Cancelled")
num_Days_Above_Level(Country_Metrics, c4_restrictionsongatherings, 2, "Num Days Restricting Gatherings (<1000)")
num_Days_Above_Level(Country_Metrics, c4_restrictionsongatherings, 3, "Num Days Restricting Gatherings (<100)")
num_Days_Above_Level(Country_Metrics, c4_restrictionsongatherings, 4, "Num Days Restricting Gatherings (<10)")

num_Days_Above_Level(Country_Metrics, c2_workplaceclosing, 2, "Num Days Some Sectors Closed")
num_Days_Above_Level(Country_Metrics, c2_workplaceclosing, 3, "Num Days All Non-Essential Sectors Closed")
num_Days_Above_Level(Country_Metrics, c5_closepublictransport, 2, "Num Days Public Transport Closed")
num_Days_Above_Level(Country_Metrics, c6_stayathomerequirements, 2, "Num Days Stay-at-Home except for Essential Trips")
num_Days_Above_Level(Country_Metrics, c6_stayathomerequirements, 3, "Num Days Stay-at-Home Total Lockdown")
num_Days_Above_Level(Country_Metrics, c7_movementrestrictions, 2, "Num Days Internal Movement Restricted")
num_Days_Above_Level(Country_Metrics, c8_internationaltravel, 3, "Num Days Int'l Bans for some Countries")
num_Days_Above_Level(Country_Metrics, c8_internationaltravel, 4, "Num Days Int'l Bans Total Border Closure")

total_and_rate_Before_Max_Subindex(Country_Metrics, h1_publicinfocampaign, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Public Information Campaign", "Avg Case Rate Before Public Information Campaign", "Max Case Rate Before Public Information Campaign")
total_and_rate_Before_Max_Subindex(Country_Metrics, h1_publicinfocampaign, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Public Information Campaign", "Avg Death Rate Before Public Information Campaign", "Max Death Rate Before Public Information Campaign")
total_and_rate_Before_Max_Subindex(Country_Metrics, h6_facialcoverings, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Facial Coverings in Some Public Places", "Avg Case Rate Before Facial Coverings in Some Public Places", "Max Case Rate Before Facial Coverings in Some Public Places")
total_and_rate_Before_Max_Subindex(Country_Metrics, h6_facialcoverings, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Facial Coverings in Some Public Places", "Avg Death Rate Before Facial Coverings in Some Public Places", "Max Death Rate Before Facial Coverings in Some Public Places")
total_and_rate_Before_Max_Subindex(Country_Metrics, h6_facialcoverings, 3, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Facial Coverings in All Public Places", "Avg Case Rate Before Facial Coverings in All Public Places", "Max Case Rate Before Facial Coverings in All Public Places")
total_and_rate_Before_Max_Subindex(Country_Metrics, h6_facialcoverings, 3, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Facial Coverings in All Public Places", "Avg Death Rate Before Facial Coverings in All Public Places", "Max Death Rate Before Facial Coverings in All Public Places")
total_and_rate_Before_Max_Subindex(Country_Metrics, h6_facialcoverings, 4, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Facial Coverings Mandate", "Avg Case Rate Before Facial Coverings Mandate", "Max Case Rate Before Facial Coverings Mandate")
total_and_rate_Before_Max_Subindex(Country_Metrics, h6_facialcoverings, 4, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Facial Coverings Mandate", "Avg Death Rate Before Facial Coverings Mandate", "Max Death Rate Before Facial Coverings Mandate")

total_and_rate_Before_Max_Subindex(Country_Metrics, c1_schoolclosing, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Some Schools Closed", "Avg Case Rate Before Some Schools Closed", "Max Case Rate Before Some Schools Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c1_schoolclosing, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Some Schools Closed", "Avg Death Rate Before Some Schools Closed", "Max Death Rate Before Some Schools Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c1_schoolclosing, 3, cases_smoothed_15_3, case_Interpolation, "Num Cases Before All Schools Closed", "Avg Case Rate Before All Schools Closed", "Max Case Rate Before All Schools Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c1_schoolclosing, 3, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before All Schools Closed", "Avg Death Rate Before All Schools Closed", "Max Death Rate Before All Schools Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c3_cancelpublicevents, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Public Events Cancelled", "Avg Case Rate Before Public Events Cancelled", "Max Case Rate Before Public Events Cancelled")
total_and_rate_Before_Max_Subindex(Country_Metrics, c3_cancelpublicevents, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Public Events Cancelled", "Avg Death Rate Before Public Events Cancelled", "Max Death Rate Before Public Events Cancelled")
total_and_rate_Before_Max_Subindex(Country_Metrics, c4_restrictionsongatherings, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Restricting Gatherings (<1000)", "Avg Case Rate Before Restricting Gatherings (<1000)", "Max Case Rate Before Restricting Gatherings (<1000)")
total_and_rate_Before_Max_Subindex(Country_Metrics, c4_restrictionsongatherings, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Restricting Gatherings (<1000)", "Avg Death Rate Before Restricting Gatherings (<1000)", "Max Death Rate Before Restricting Gatherings (<1000)")
total_and_rate_Before_Max_Subindex(Country_Metrics, c4_restrictionsongatherings, 3, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Restricting Gatherings (<100)", "Avg Case Rate Before Restricting Gatherings (<100)", "Max Case Rate Before Restricting Gatherings (<100)")
total_and_rate_Before_Max_Subindex(Country_Metrics, c4_restrictionsongatherings, 3, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Restricting Gatherings (<100)", "Avg Death Rate Before Restricting Gatherings (<100)", "Max Death Rate Before Restricting Gatherings (<100)")
total_and_rate_Before_Max_Subindex(Country_Metrics, c4_restrictionsongatherings, 4, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Restricting Gatherings (<10)", "Avg Case Rate Before Restricting Gatherings (<10)", "Max Case Rate Before Restricting Gatherings (<10)")
total_and_rate_Before_Max_Subindex(Country_Metrics, c4_restrictionsongatherings, 4, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Restricting Gatherings (<10)", "Avg Death Rate Before Restricting Gatherings (<10)", "Max Death Rate Before Restricting Gatherings (<10)")

total_and_rate_Before_Max_Subindex(Country_Metrics, c2_workplaceclosing, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Some Sectors Closed", "Avg Case Rate Before Some Sectors Closed", "Max Case Rate Before Some Sectors Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c2_workplaceclosing, 3, cases_smoothed_15_3, case_Interpolation, "Num Cases Before All Non-Essential Sectors Closed", "Avg Case Rate Before All Non-Essential Sectors Closed", "Max Case Rate Before All Non-Essential Sectors Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c2_workplaceclosing, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Some Sectors Closed", "Avg Death Rate Before Some Sectors Closed", "Max Death Rate Before Some Sectors Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c2_workplaceclosing, 3, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before All Non-Essential Sectors Closed", "Avg Death Rate Before All Non-Essential Sectors Closed", "Max Death Rate Before All Non-Essential Sectors Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c5_closepublictransport, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Public Transport Closed", "Avg Cases Rate Before Public Transport Closed", "Max Case Rate Before Public Transport Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c5_closepublictransport, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Public Transport Closed", "Avg Deaths Rate Before Public Transport Closed", "Max Death Rate Before Public Transport Closed")
total_and_rate_Before_Max_Subindex(Country_Metrics, c6_stayathomerequirements, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Stay-at-Home except for Essential Trips", "Avg Cases Rate Before Stay-at-Home except for Essential Trips", "Max Case Rate Before Stay-at-Home except for Essential Trips")
total_and_rate_Before_Max_Subindex(Country_Metrics, c6_stayathomerequirements, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Stay-at-Home except for Essential Trips", "Avg Deaths Rate Before Stay-at-Home except for Essential Trips", "Max Death Rate Before Stay-at-Home except for Essential Trips")
total_and_rate_Before_Max_Subindex(Country_Metrics, c6_stayathomerequirements, 3, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Stay-at-Home Total Lockdown", "Avg Cases Rate Before Stay-at-Home Total Lockdown", "Max Case Rate Before Stay-at-Home Total Lockdown")
total_and_rate_Before_Max_Subindex(Country_Metrics, c6_stayathomerequirements, 3, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Stay-at-Home Total Lockdown", "Avg Deaths Rate Before Stay-at-Home Total Lockdown", "Max Death Rate Before Stay-at-Home Total Lockdown")
total_and_rate_Before_Max_Subindex(Country_Metrics, c7_movementrestrictions, 2, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Internal Movement Restricted", "Avg Cases Rate Before Internal Movement Restricted", "Max Case Rate Before Internal Movement Restricted")
total_and_rate_Before_Max_Subindex(Country_Metrics, c7_movementrestrictions, 2, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Internal Movement Restricted", "Avg Deaths Rate Before Internal Movement Restricted", "Max Death Rate Before Internal Movement Restricted")
total_and_rate_Before_Max_Subindex(Country_Metrics, c8_internationaltravel, 3, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Int'l Bans for some Countries", "Avg Cases Rate Before Int'l Bans for some Countries", "Max Case Rate Before Int'l Bans for some Countries")
total_and_rate_Before_Max_Subindex(Country_Metrics, c8_internationaltravel, 3, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Int'l Bans for some Countries", "Avg Deaths Rate Before Int'l Bans for some Countries", "Max Death Rate Before Int'l Bans for some Countries")
total_and_rate_Before_Max_Subindex(Country_Metrics, c8_internationaltravel, 4, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Int'l Bans Total Border Closure", "Avg Cases Rate Before Int'l Bans Total Border Closure", "Max Case Rate Before Int'l Bans Total Border Closure")
total_and_rate_Before_Max_Subindex(Country_Metrics, c8_internationaltravel, 4, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Int'l Bans Total Border Closure", "Avg Deaths Rate Before Int'l Bans Total Border Closure", "Max Death Rate Before Int'l Bans Total Border Closure")

total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, stringency_Index, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Peak Stringency", "Avg Case Growth Before Peak Stringency", "Max Case Growth Before Peak Stringency", "First Max Stringency to Preceding Cases Ratio")
total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, stringency_Index, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Peak Stringency", "Avg Death Growth Before Peak Stringency", "Max Death Growth Before Peak Stringency", "First Max Stringency to Preceding Deaths Ratio")
total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, containment_Health_Index, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Peak Containment Health", "Avg Case Growth Before Peak Containment Health", "Max Case Growth Before Peak Containment Health", "First Max Containment Health to Preceding Cases Ratio")
total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, containment_Health_Index, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Peak Containment Health", "Avg Death Growth Before Peak Containment Health", "Max Death Growth Before Peak Containment Health", "First Max Containment Health to Preceding Deaths Ratio")
total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, economic_Support_Index, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Peak Economic Support", "Avg Case Growth Before Peak Economic Support", "Max Case Growth Before Peak Economic Support", "First Max Economic Support to Preceding Cases Ratio")
total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, economic_Support_Index, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Peak Economic Support", "Avg Death Growth Before Peak Economic Support", "Max Death Growth Before Peak Economic Support", "First Max Economic Support to Preceding Deaths Ratio")
total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, government_Response_Index, cases_smoothed_15_3, case_Interpolation, "Num Cases Before Peak Government Response", "Avg Case Growth Before Peak Government Response", "Max Case Growth Before Peak Government Response", "First Max Government Response to Preceding Cases Ratio")
total_and_rate_Before_First_Max_Index_Peak(Country_Metrics, government_Response_Index, deaths_smoothed_15_3, death_Interpolation, "Num Deaths Before Peak Government Response", "Avg Death Growth Before Peak Government Response", "Max Death Growth Before Peak Government Response", "First Max Government Response to Preceding Deaths Ratio")

index_to_index_Ratio(Country_Metrics, economic_Support_Index, containment_Health_Index, "Economic Support to Containment Health Ratio")
index_to_index_Ratio(Country_Metrics, economic_Support_Index, stringency_Index, "Economic Support to Stringency Ratio")
index_to_index_Ratio(Country_Metrics, economic_Support_Index, government_Response_Index, "Economic Support to Government Response Ratio")

#Write the Country_Metrics DataFrame to an excel file
Country_Metrics.to_excel('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/Country_Metrics.xlsx')