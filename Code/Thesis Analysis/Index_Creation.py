# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:04:41 2021

@author: theod
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

#Create the Dataframe that will store all of the normalized index values:
Normalized_Metrics = Country_Metrics.drop(columns = ['Case-Death Pairs', 'Cases Outbreak Day', 'Cases Outbreak Day(First Outbreak)', 'Deaths Outbreak Day', 'Deaths Outbreak Day(First Outbreak)', 'Number of Case-Death Pairs'])

#loop through the columns of Normalized_Metrics to identify columns that contain arrays
for j in Normalized_Metrics.columns.values:
    #if the column contains arrays...
    if Normalized_Metrics.dtypes[j] == object:
        #loop through all rows
        for i in Normalized_Metrics.index.values:
            array = Normalized_Metrics.loc[i,j]
            #create a new array where "None" labels are removed
            new_array = np.delete(array, np.where(array == None))
            #Replace the array at the cell with it's average
            Normalized_Metrics.loc[i,j] = np.mean(new_array, dtype=float) 


#Find outliers that need to be removed from the datset before scaling
outliers_list = []
for j in Normalized_Metrics.columns.values:
    #Set the data for analysis
    data = Normalized_Metrics.loc[:,j]
    data = data[~pd.isnull(data)]
    data = np.absolute(data)
    x = data.index.values
    #Find the IQR
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    IQR = q75 - q25
    #set the bounds
    k = 3
    UB = q75 + k*IQR
    LB = q25 - k*IQR
    #determine the outliers
    low_outliers = x[np.where(data < LB)] 
    high_outliers = x[np.where(data > UB)]
    #add the outliers to the outlier list
    outliers_list.append([j, list(low_outliers), list(high_outliers), LB, UB])

#Alter the outliers from the list so that scaling can be performed
#make the Normalized Metrics Dataframe all (+) values
Normalized_Metrics = Normalized_Metrics.abs()
#remove outliers
for i in range(0,len(outliers_list)):
    #extract the column name
    x = outliers_list[i][0]
    for j in range(0, len(outliers_list[i][1])):
        #extract the row name
        y1=outliers_list[i][1][j]
        #extract the Lower Bound
        LB=outliers_list[i][3]
        #Update the outlying value
        Normalized_Metrics.loc[y1,x] = LB
    for k in range(0, len(outliers_list[i][2])):
        #extract the row nmae
        y2=outliers_list[i][2][k]
        #extract the upper bound value
        UB=outliers_list[i][4]
        #update the outlying value
        Normalized_Metrics.loc[y2, x] = UB
    

#Scale the Normalizaed Data with new values from 0 to 1
scaler = MinMaxScaler()
Metrics = Normalized_Metrics.values
New_Metrics = scaler.fit_transform(Metrics)
Scaled_Metrics = pd.DataFrame(data = New_Metrics, index = Normalized_Metrics.index.values, columns = Normalized_Metrics.columns.values)

#Add the list of metrics and whether or not high or low values are desirable
High_Low_List = [["Cases Avg. Growth Rate(First Outbreak)", "Low"], 
["Cases Max. Growth Rate(First Outbreak)", "Low"], 
["Cases Growth Length(First Outbreak)", "Low"], 
["Cases Avg. Submission Rate(First Outbreak)", "High"], 
["Cases Max. Submission Rate(First Outbreak)", "High"], 
["Cases Submission Length(First Outbreak)", "Low"], 
["Cases Total Length(First Outbreak)", "Low"], 
["Cases Peak Value(First Outbreak)", "Low"], 
["Cases Valley Value(First Outbreak)", "Low"], 
["Deaths Avg. Growth Rate(First Outbreak)", "Low"], 
["Deaths Max. Growth Rate(First Outbreak)", "Low"], 
["Deaths Growth Length(First Outbreak)", "Low"], 
["Deaths Avg. Submission Rate(First Outbreak)", "High"], 
["Deaths Max. Submission Rate(First Outbreak)", "High"], 
["Deaths Submission Length(First Outbreak)", "Low"], 
["Deaths Total Length(First Outbreak)", "Low"], 
["Deaths Peak Value(First Outbreak)", "Low"], 
["Deaths Valley Value(First Outbreak)", "Low"], 
["Total Cases per Million", "Low"], 
["Total Deaths per Million", "Low"], 
["Total Tests per Million", "High"], 
["Case-Death Ratio", "High"], 
["Test-Case Ratio", "High"], 
["Case-Death Pair Peak Ratio", "High"], 
["Number of Cases Peaks", "Low"], 
["Cases Avg. Growth Rate", "Low"], 
["Cases Max. Growth Rate", "Low"], 
["Cases Growth Length", "Low"], 
["Cases Avg. Submission Rate", "High"], 
["Cases Max. Submission Rate", "High"], 
["Cases Submission Length", "Low"], 
["Cases Total Length", "Low"], 
["Cases Peak Value", "Low"], 
["Cases Valley Value", "Low"], 
["Number of Deaths Peaks", "Low"], 
["Deaths Avg. Growth Rate", "Low"], 
["Deaths Max. Growth Rate", "Low"], 
["Deaths Growth Length", "Low"], 
["Deaths Avg. Submission Rate", "High"], 
["Deaths Max. Submission Rate", "High"], 
["Deaths Submission Length", "Low"], 
["Deaths Total Length", "Low"], 
["Deaths Peak Value", "Low"], 
["Deaths Valley Value", "Low"], 
["Max Stringency", "High"], 
["Max Government Response", "High"], 
["Max Containment Health", "High"], 
["Max Economic Support", "High"], 
["Economic Support to Containment Health Ratio", "Low"], 
["Economic Support to Stringency Ratio", "Low"], 
["Economic Support to Government Response Ratio", "Low"], 
["Num Days Some Sectors Closed", "High"], 
["Num Days All Non-Essential Sectors Closed", "High"], 
["Num Days Public Transport Closed", "High"], 
["Num Days Stay-at-Home except for Essential Trips", "High"], 
["Num Days Stay-at-Home Total Lockdown", "High"], 
["Num Days Internal Movement Restricted", "High"], 
["Num Days Int'l Bans for some Countries", "High"], 
["Num Days Int'l Bans Total Border Closure", "High"], 
["Num Days Public Information Campaign", "High"], 
["Num Days Facial Coverings in Some Public Places", "High"], 
["Num Days Facial Coverings in All Public Places", "High"], 
["Num Days Facial Coverings Mandate", "High"], 
["Num Days Some Schools Closed", "High"], 
["Num Days All Schools Closed", "High"], 
["Num Days Public Events Cancelled", "High"], 
["Num Days Restricting Gatherings (<1000)", "High"], 
["Num Days Restricting Gatherings (<100)", "High"], 
["Num Days Restricting Gatherings (<10)", "High"], 
["Num Cases Before Some Sectors Closed", "Low"], 
["Avg Case Rate Before Some Sectors Closed", "Low"], 
["Max Case Rate Before Some Sectors Closed", "Low"], 
["Num Cases Before Stay-at-Home except for Essential Trips", "Low"], 
["Avg Cases Rate Before Stay-at-Home except for Essential Trips", "Low"], 
["Max Case Rate Before Stay-at-Home except for Essential Trips", "Low"], 
["Num Cases Before Int'l Bans for some Countries", "Low"], 
["Avg Cases Rate Before Int'l Bans for some Countries", "Low"], 
["Max Case Rate Before Int'l Bans for some Countries", "Low"], 
["Num Cases Before Public Information Campaign", "Low"], 
["Avg Case Rate Before Public Information Campaign", "Low"], 
["Max Case Rate Before Public Information Campaign", "Low"], 
["Num Cases Before Facial Coverings in Some Public Places", "Low"], 
["Avg Case Rate Before Facial Coverings in Some Public Places", "Low"], 
["Max Case Rate Before Facial Coverings in Some Public Places", "Low"], 
["Num Cases Before Some Schools Closed", "Low"], 
["Avg Case Rate Before Some Schools Closed", "Low"], 
["Max Case Rate Before Some Schools Closed", "Low"], 
["Num Cases Before Restricting Gatherings (<1000)", "Low"], 
["Avg Case Rate Before Restricting Gatherings (<1000)", "Low"], 
["Max Case Rate Before Restricting Gatherings (<1000)", "Low"], 
["Num Deaths Before Some Sectors Closed", "Low"], 
["Avg Death Rate Before Some Sectors Closed", "Low"], 
["Max Death Rate Before Some Sectors Closed", "Low"], 
["Num Deaths Before Stay-at-Home except for Essential Trips", "Low"], 
["Avg Deaths Rate Before Stay-at-Home except for Essential Trips", "Low"], 
["Max Death Rate Before Stay-at-Home except for Essential Trips", "Low"], 
["Num Deaths Before Int'l Bans for some Countries", "Low"], 
["Avg Deaths Rate Before Int'l Bans for some Countries", "Low"], 
["Max Death Rate Before Int'l Bans for some Countries", "Low"], 
["Num Deaths Before Public Information Campaign", "Low"], 
["Avg Death Rate Before Public Information Campaign", "Low"], 
["Max Death Rate Before Public Information Campaign", "Low"], 
["Num Deaths Before Facial Coverings in Some Public Places", "Low"], 
["Avg Death Rate Before Facial Coverings in Some Public Places", "Low"], 
["Max Death Rate Before Facial Coverings in Some Public Places", "Low"], 
["Num Deaths Before Some Schools Closed", "Low"], 
["Avg Death Rate Before Some Schools Closed", "Low"], 
["Max Death Rate Before Some Schools Closed", "Low"], 
["Num Deaths Before Restricting Gatherings (<1000)", "Low"], 
["Avg Death Rate Before Restricting Gatherings (<1000)", "Low"], 
["Max Death Rate Before Restricting Gatherings (<1000)", "Low"], 
["Num Cases Before All Non-Essential Sectors Closed", "Low"], 
["Avg Case Rate Before All Non-Essential Sectors Closed", "Low"], 
["Max Case Rate Before All Non-Essential Sectors Closed", "Low"], 
["Num Cases Before Public Transport Closed", "Low"], 
["Avg Cases Rate Before Public Transport Closed", "Low"], 
["Max Case Rate Before Public Transport Closed", "Low"], 
["Num Cases Before Stay-at-Home Total Lockdown", "Low"], 
["Avg Cases Rate Before Stay-at-Home Total Lockdown", "Low"], 
["Max Case Rate Before Stay-at-Home Total Lockdown", "Low"], 
["Num Cases Before Internal Movement Restricted", "Low"], 
["Avg Cases Rate Before Internal Movement Restricted", "Low"], 
["Max Case Rate Before Internal Movement Restricted", "Low"], 
["Num Cases Before Int'l Bans Total Border Closure", "Low"], 
["Avg Cases Rate Before Int'l Bans Total Border Closure", "Low"], 
["Max Case Rate Before Int'l Bans Total Border Closure", "Low"], 
["Num Cases Before Facial Coverings in All Public Places", "Low"], 
["Avg Case Rate Before Facial Coverings in All Public Places", "Low"], 
["Max Case Rate Before Facial Coverings in All Public Places", "Low"], 
["Num Cases Before Facial Coverings Mandate", "Low"], 
["Avg Case Rate Before Facial Coverings Mandate", "Low"], 
["Max Case Rate Before Facial Coverings Mandate", "Low"], 
["Num Cases Before All Schools Closed", "Low"], 
["Avg Case Rate Before All Schools Closed", "Low"], 
["Max Case Rate Before All Schools Closed", "Low"], 
["Num Cases Before Public Events Cancelled", "Low"], 
["Avg Case Rate Before Public Events Cancelled", "Low"], 
["Max Case Rate Before Public Events Cancelled", "Low"], 
["Num Cases Before Restricting Gatherings (<100)", "Low"], 
["Avg Case Rate Before Restricting Gatherings (<100)", "Low"], 
["Max Case Rate Before Restricting Gatherings (<100)", "Low"], 
["Num Cases Before Restricting Gatherings (<10)", "Low"], 
["Avg Case Rate Before Restricting Gatherings (<10)", "Low"], 
["Max Case Rate Before Restricting Gatherings (<10)", "Low"], 
["Num Deaths Before All Non-Essential Sectors Closed", "Low"], 
["Avg Death Rate Before All Non-Essential Sectors Closed", "Low"], 
["Max Death Rate Before All Non-Essential Sectors Closed", "Low"], 
["Num Deaths Before Public Transport Closed", "Low"], 
["Avg Deaths Rate Before Public Transport Closed", "Low"], 
["Max Death Rate Before Public Transport Closed", "Low"], 
["Num Deaths Before Stay-at-Home Total Lockdown", "Low"], 
["Avg Deaths Rate Before Stay-at-Home Total Lockdown", "Low"], 
["Max Death Rate Before Stay-at-Home Total Lockdown", "Low"], 
["Num Deaths Before Internal Movement Restricted", "Low"], 
["Avg Deaths Rate Before Internal Movement Restricted", "Low"], 
["Max Death Rate Before Internal Movement Restricted", "Low"], 
["Num Deaths Before Int'l Bans Total Border Closure", "Low"], 
["Avg Deaths Rate Before Int'l Bans Total Border Closure", "Low"], 
["Max Death Rate Before Int'l Bans Total Border Closure", "Low"], 
["Num Deaths Before Facial Coverings in All Public Places", "Low"], 
["Avg Death Rate Before Facial Coverings in All Public Places", "Low"], 
["Max Death Rate Before Facial Coverings in All Public Places", "Low"], 
["Num Deaths Before Facial Coverings Mandate", "Low"], 
["Avg Death Rate Before Facial Coverings Mandate", "Low"], 
["Max Death Rate Before Facial Coverings Mandate", "Low"], 
["Num Deaths Before All Schools Closed", "Low"], 
["Avg Death Rate Before All Schools Closed", "Low"], 
["Max Death Rate Before All Schools Closed", "Low"], 
["Num Deaths Before Public Events Cancelled", "Low"], 
["Avg Death Rate Before Public Events Cancelled", "Low"], 
["Max Death Rate Before Public Events Cancelled", "Low"], 
["Num Deaths Before Restricting Gatherings (<100)", "Low"], 
["Avg Death Rate Before Restricting Gatherings (<100)", "Low"], 
["Max Death Rate Before Restricting Gatherings (<100)", "Low"], 
["Num Deaths Before Restricting Gatherings (<10)", "Low"], 
["Avg Death Rate Before Restricting Gatherings (<10)", "Low"], 
["Max Death Rate Before Restricting Gatherings (<10)", "Low"], 
["Num Cases Before Peak Stringency", "Low"], 
["Avg Case Growth Before Peak Stringency", "Low"], 
["Max Case Growth Before Peak Stringency", "Low"], 
["First Max Stringency to Preceding Cases Ratio", "Low"], 
["Num Deaths Before Peak Stringency", "Low"], 
["Avg Death Growth Before Peak Stringency", "Low"], 
["Max Death Growth Before Peak Stringency", "Low"], 
["First Max Stringency to Preceding Deaths Ratio", "Low"], 
["Num Cases Before Peak Containment Health", "Low"], 
["Avg Case Growth Before Peak Containment Health", "Low"], 
["Max Case Growth Before Peak Containment Health", "Low"], 
["First Max Containment Health to Preceding Cases Ratio", "Low"], 
["Num Deaths Before Peak Containment Health", "Low"], 
["Avg Death Growth Before Peak Containment Health", "Low"], 
["Max Death Growth Before Peak Containment Health", "Low"], 
["First Max Containment Health to Preceding Deaths Ratio", "Low"], 
["Num Cases Before Peak Economic Support", "Low"], 
["Avg Case Growth Before Peak Economic Support", "Low"], 
["Max Case Growth Before Peak Economic Support", "Low"], 
["First Max Economic Support to Preceding Cases Ratio", "Low"], 
["Num Deaths Before Peak Economic Support", "Low"], 
["Avg Death Growth Before Peak Economic Support", "Low"], 
["Max Death Growth Before Peak Economic Support", "Low"], 
["First Max Economic Support to Preceding Deaths Ratio", "Low"], 
["Num Cases Before Peak Government Response", "Low"], 
["Avg Case Growth Before Peak Government Response", "Low"], 
["Max Case Growth Before Peak Government Response", "Low"], 
["First Max Government Response to Preceding Cases Ratio", "Low"], 
["Num Deaths Before Peak Government Response", "Low"], 
["Avg Death Growth Before Peak Government Response", "Low"], 
["Max Death Growth Before Peak Government Response", "Low"], 
["First Max Government Response to Preceding Deaths Ratio", "Low"]]

#Adjust the scales in order to reflect whether high or low values are good
for i in range(0, len(High_Low_List)):
    x = High_Low_List[i][0]
    if High_Low_List[i][1] == "Low":
        Scaled_Metrics.loc[:,x] = 1 - Scaled_Metrics.loc[:,x]

#Add list of columns in Scaled_Metrics whose Null values need to be accounted for
Null_Col_Adj = ["Num Cases Before Some Sectors Closed", 
"Avg Case Rate Before Some Sectors Closed", 
"Max Case Rate Before Some Sectors Closed", 
"Num Cases Before Stay-at-Home except for Essential Trips", 
"Avg Cases Rate Before Stay-at-Home except for Essential Trips", 
"Max Case Rate Before Stay-at-Home except for Essential Trips", 
"Num Cases Before Int'l Bans for some Countries", 
"Avg Cases Rate Before Int'l Bans for some Countries", 
"Max Case Rate Before Int'l Bans for some Countries", 
"Num Cases Before Public Information Campaign", 
"Avg Case Rate Before Public Information Campaign", 
"Max Case Rate Before Public Information Campaign", 
"Num Cases Before Facial Coverings in Some Public Places", 
"Avg Case Rate Before Facial Coverings in Some Public Places", 
"Max Case Rate Before Facial Coverings in Some Public Places", 
"Num Cases Before Some Schools Closed", 
"Avg Case Rate Before Some Schools Closed", 
"Max Case Rate Before Some Schools Closed", 
"Num Cases Before Restricting Gatherings (<1000)", 
"Avg Case Rate Before Restricting Gatherings (<1000)", 
"Max Case Rate Before Restricting Gatherings (<1000)", 
"Num Deaths Before Some Sectors Closed", 
"Avg Death Rate Before Some Sectors Closed", 
"Max Death Rate Before Some Sectors Closed", 
"Num Deaths Before Stay-at-Home except for Essential Trips", 
"Avg Deaths Rate Before Stay-at-Home except for Essential Trips", 
"Max Death Rate Before Stay-at-Home except for Essential Trips", 
"Num Deaths Before Int'l Bans for some Countries", 
"Avg Deaths Rate Before Int'l Bans for some Countries", 
"Max Death Rate Before Int'l Bans for some Countries", 
"Num Deaths Before Public Information Campaign", 
"Avg Death Rate Before Public Information Campaign", 
"Max Death Rate Before Public Information Campaign", 
"Num Deaths Before Facial Coverings in Some Public Places", 
"Avg Death Rate Before Facial Coverings in Some Public Places", 
"Max Death Rate Before Facial Coverings in Some Public Places", 
"Num Deaths Before Some Schools Closed", 
"Avg Death Rate Before Some Schools Closed", 
"Max Death Rate Before Some Schools Closed", 
"Num Deaths Before Restricting Gatherings (<1000)", 
"Avg Death Rate Before Restricting Gatherings (<1000)", 
"Max Death Rate Before Restricting Gatherings (<1000)", 
"Num Cases Before All Non-Essential Sectors Closed", 
"Avg Case Rate Before All Non-Essential Sectors Closed", 
"Max Case Rate Before All Non-Essential Sectors Closed", 
"Num Cases Before Public Transport Closed", 
"Avg Cases Rate Before Public Transport Closed", 
"Max Case Rate Before Public Transport Closed", 
"Num Cases Before Stay-at-Home Total Lockdown", 
"Avg Cases Rate Before Stay-at-Home Total Lockdown", 
"Max Case Rate Before Stay-at-Home Total Lockdown", 
"Num Cases Before Internal Movement Restricted", 
"Avg Cases Rate Before Internal Movement Restricted", 
"Max Case Rate Before Internal Movement Restricted", 
"Num Cases Before Int'l Bans Total Border Closure", 
"Avg Cases Rate Before Int'l Bans Total Border Closure", 
"Max Case Rate Before Int'l Bans Total Border Closure", 
"Num Cases Before Facial Coverings in All Public Places", 
"Avg Case Rate Before Facial Coverings in All Public Places", 
"Max Case Rate Before Facial Coverings in All Public Places", 
"Num Cases Before Facial Coverings Mandate", 
"Avg Case Rate Before Facial Coverings Mandate", 
"Max Case Rate Before Facial Coverings Mandate", 
"Num Cases Before All Schools Closed", 
"Avg Case Rate Before All Schools Closed", 
"Max Case Rate Before All Schools Closed", 
"Num Cases Before Public Events Cancelled", 
"Avg Case Rate Before Public Events Cancelled", 
"Max Case Rate Before Public Events Cancelled", 
"Num Cases Before Restricting Gatherings (<100)", 
"Avg Case Rate Before Restricting Gatherings (<100)", 
"Max Case Rate Before Restricting Gatherings (<100)", 
"Num Cases Before Restricting Gatherings (<10)", 
"Avg Case Rate Before Restricting Gatherings (<10)", 
"Max Case Rate Before Restricting Gatherings (<10)", 
"Num Deaths Before All Non-Essential Sectors Closed", 
"Avg Death Rate Before All Non-Essential Sectors Closed", 
"Max Death Rate Before All Non-Essential Sectors Closed", 
"Num Deaths Before Public Transport Closed", 
"Avg Deaths Rate Before Public Transport Closed", 
"Max Death Rate Before Public Transport Closed", 
"Num Deaths Before Stay-at-Home Total Lockdown", 
"Avg Deaths Rate Before Stay-at-Home Total Lockdown", 
"Max Death Rate Before Stay-at-Home Total Lockdown", 
"Num Deaths Before Internal Movement Restricted", 
"Avg Deaths Rate Before Internal Movement Restricted", 
"Max Death Rate Before Internal Movement Restricted", 
"Num Deaths Before Int'l Bans Total Border Closure", 
"Avg Deaths Rate Before Int'l Bans Total Border Closure", 
"Max Death Rate Before Int'l Bans Total Border Closure", 
"Num Deaths Before Facial Coverings in All Public Places", 
"Avg Death Rate Before Facial Coverings in All Public Places", 
"Max Death Rate Before Facial Coverings in All Public Places", 
"Num Deaths Before Facial Coverings Mandate", 
"Avg Death Rate Before Facial Coverings Mandate", 
"Max Death Rate Before Facial Coverings Mandate", 
"Num Deaths Before All Schools Closed", 
"Avg Death Rate Before All Schools Closed", 
"Max Death Rate Before All Schools Closed", 
"Num Deaths Before Public Events Cancelled", 
"Avg Death Rate Before Public Events Cancelled", 
"Max Death Rate Before Public Events Cancelled", 
"Num Deaths Before Restricting Gatherings (<100)", 
"Avg Death Rate Before Restricting Gatherings (<100)", 
"Max Death Rate Before Restricting Gatherings (<100)", 
"Num Deaths Before Restricting Gatherings (<10)", 
"Avg Death Rate Before Restricting Gatherings (<10)", 
"Max Death Rate Before Restricting Gatherings (<10)"
]

#Adjust all instances of Null values in the Scaled_Metrics dataframe accordingly
for j in Null_Col_Adj:
    #Find all of the indexes in column j that have Null values
    Nan_Indexes = Scaled_Metrics.index.values[np.where(np.isnan(Scaled_Metrics.loc[:,j]))]
    for i in Nan_Indexes:
        #If the countrry exists in the cases and stringency file, but is NaN, it means the stringency condition was never reached (i.e. the coutry eanr the best value: 1)
        if i in cases_File.index.values and i in stringency_Index.index.values:
            Scaled_Metrics.loc[i,j] = 1

#Make the DataFrame to hold the results of the composite index calculation:
valid_index = set(stringency_Index.index.values).intersection(cases_File.index.values)
valid_index.remove('COM')
Composite_Index = pd.DataFrame(index = sorted(valid_index), columns = ["Composite Index", "Outbreak Severity", "Response Severity", "Risk Tolerance", "Initial Outbreak",  "Initial Outbreak: Cases", "Initial Outbreak: Deaths", "Total Outbreak", "Total Outbreak: Total Handeling", "Total Outbreak: Case Handeling", "Total Outbreak: Death Handeling", "Response Severity: Max Severity", "Response Severity: Relative Economic Support", "Response Severity: Length of Response", "Risk Tolerance: Outbreak Severity before Actions", "Risk Tolerance: Strict Measures", "Risk Tolerance: Cases before Strict Measures", "Risk Tolerance: Cases before Most Strict Measures", "Risk Tolerance: Most Strict Measures", "Risk Tolerance: Deaths before Strict Measures", "Risk Tolerance: Deaths before Most Strict Measures", "Risk Tolerance: Outbreak Severity before Peak Action"])

#This function calcuates the score for a su-index given an input of sub-index column names and groupings
def sub_index_calc(Comp_Index, Sub_Index, Scaled_df, Name):
    #loop through every country
    for i in Comp_Index.index.values:
        #create an array to store the averaged values for every sub-index grouping
        sub_index_data = []
        #loop over very group
        for j in np.unique(Sub_Index[:,1]):
            #grab the name of eery column in the group
            cols = Sub_Index[:,0][np.where(Sub_Index[:,1]==j)]
            #find the average fo the columns in the group
            avg = Scaled_df.loc[i, cols].mean()
            #add the average to the storage array
            sub_index_data = np.append(sub_index_data, avg)
        #compute the sub-index score (via averaging the components) and store the value in the composit index dataframe
        Comp_Index.loc[i, Name] = np.nanmean(sub_index_data)
    return

#Add arrays to hold the information on the contents fo the subindexes
I_O_Cases = np.array([["Cases Avg. Growth Rate(First Outbreak)", 1], ["Cases Max. Growth Rate(First Outbreak)", 1], ["Cases Growth Length(First Outbreak)", 2], ["Cases Avg. Submission Rate(First Outbreak)", 3], ["Cases Max. Submission Rate(First Outbreak)", 3], ["Cases Submission Length(First Outbreak)", 2], ["Cases Total Length(First Outbreak)", 2], ["Cases Peak Value(First Outbreak)", 4], ["Cases Valley Value(First Outbreak)", 5]] )
I_O_Deaths = np.array([["Deaths Avg. Growth Rate(First Outbreak)", 1], ["Deaths Max. Growth Rate(First Outbreak)", 1], ["Deaths Growth Length(First Outbreak)", 2], ["Deaths Avg. Submission Rate(First Outbreak)", 3], ["Deaths Max. Submission Rate(First Outbreak)", 3], ["Deaths Submission Length(First Outbreak)", 2], ["Deaths Total Length(First Outbreak)", 2], ["Deaths Peak Value(First Outbreak)", 4], ["Deaths Valley Value(First Outbreak)", 5]])
T_O_Total_Handeling = np.array([["Total Cases per Million", 1], ["Total Deaths per Million", 2], ["Total Tests per Million", 3], ["Case-Death Ratio", 4], ["Test-Case Ratio", 5], ["Case-Death Pair Peak Ratio", 6]])
T_O_Case_Handeling = np.array([["Number of Cases Peaks", 1], ["Cases Avg. Growth Rate", 2], ["Cases Max. Growth Rate", 2], ["Cases Growth Length", 3], ["Cases Avg. Submission Rate", 4], ["Cases Max. Submission Rate", 4], ["Cases Submission Length", 3], ["Cases Total Length", 3], ["Cases Peak Value", 5], ["Cases Valley Value", 6]])
T_O_Death_Handeling = np.array([["Number of Deaths Peaks", 1], ["Deaths Avg. Growth Rate", 2], ["Deaths Max. Growth Rate", 2], ["Deaths Growth Length", 3], ["Deaths Avg. Submission Rate", 4], ["Deaths Max. Submission Rate", 4], ["Deaths Submission Length", 3], ["Deaths Total Length", 3], ["Deaths Peak Value", 5], ["Deaths Valley Value", 6]])
G_R_Severity = np.array([["Max Stringency", 1], ["Max Government Response", 2], ["Max Containment Health", 3], ["Max Economic Support", 4]])
G_R_Economic= np.array([["Economic Support to Containment Health Ratio", 1], ["Economic Support to Stringency Ratio", 2], ["Economic Support to Government Response Ratio", 3]])
G_R_Length = np.array([["Num Days Some Sectors Closed", 1], 
["Num Days All Non-Essential Sectors Closed", 2], 
["Num Days Public Transport Closed", 3], 
["Num Days Stay-at-Home except for Essential Trips", 4], 
["Num Days Stay-at-Home Total Lockdown", 5], 
["Num Days Internal Movement Restricted", 6], 
["Num Days Int'l Bans for some Countries", 7], 
["Num Days Int'l Bans Total Border Closure", 8], 
["Num Days Public Information Campaign", 9], 
["Num Days Facial Coverings in Some Public Places", 10], 
["Num Days Facial Coverings in All Public Places", 11], 
["Num Days Facial Coverings Mandate", 12], 
["Num Days Some Schools Closed", 13], 
["Num Days All Schools Closed", 14], 
["Num Days Public Events Cancelled", 15], 
["Num Days Restricting Gatherings (<1000)", 16], 
["Num Days Restricting Gatherings (<100)", 17], 
["Num Days Restricting Gatherings (<10)", 18]])
O_S_Strict_Cases = np.array([["Num Cases Before Some Sectors Closed", 1], 
["Avg Case Rate Before Some Sectors Closed", 2], 
["Max Case Rate Before Some Sectors Closed", 2], 
["Num Cases Before Stay-at-Home except for Essential Trips", 3], 
["Avg Cases Rate Before Stay-at-Home except for Essential Trips", 4], 
["Max Case Rate Before Stay-at-Home except for Essential Trips", 4], 
["Num Cases Before Int'l Bans for some Countries", 5], 
["Avg Cases Rate Before Int'l Bans for some Countries", 6], 
["Max Case Rate Before Int'l Bans for some Countries", 6], 
["Num Cases Before Public Information Campaign", 7], 
["Avg Case Rate Before Public Information Campaign", 8], 
["Max Case Rate Before Public Information Campaign", 8], 
["Num Cases Before Facial Coverings in Some Public Places", 9], 
["Avg Case Rate Before Facial Coverings in Some Public Places", 10], 
["Max Case Rate Before Facial Coverings in Some Public Places", 10], 
["Num Cases Before Some Schools Closed", 11], 
["Avg Case Rate Before Some Schools Closed", 12], 
["Max Case Rate Before Some Schools Closed", 12], 
["Num Cases Before Restricting Gatherings (<1000)", 13], 
["Avg Case Rate Before Restricting Gatherings (<1000)", 14], 
["Max Case Rate Before Restricting Gatherings (<1000)", 14]])
O_S_Strict_Deaths = np.array([["Num Deaths Before Some Sectors Closed", 1], 
["Avg Death Rate Before Some Sectors Closed", 2], 
["Max Death Rate Before Some Sectors Closed", 2], 
["Num Deaths Before Stay-at-Home except for Essential Trips", 3], 
["Avg Deaths Rate Before Stay-at-Home except for Essential Trips", 4], 
["Max Death Rate Before Stay-at-Home except for Essential Trips", 4], 
["Num Deaths Before Int'l Bans for some Countries", 5], 
["Avg Deaths Rate Before Int'l Bans for some Countries", 6], 
["Max Death Rate Before Int'l Bans for some Countries", 6], 
["Num Deaths Before Public Information Campaign", 7], 
["Avg Death Rate Before Public Information Campaign", 8], 
["Max Death Rate Before Public Information Campaign", 8], 
["Num Deaths Before Facial Coverings in Some Public Places", 9], 
["Avg Death Rate Before Facial Coverings in Some Public Places", 10], 
["Max Death Rate Before Facial Coverings in Some Public Places", 10], 
["Num Deaths Before Some Schools Closed", 11], 
["Avg Death Rate Before Some Schools Closed", 12], 
["Max Death Rate Before Some Schools Closed", 12], 
["Num Deaths Before Restricting Gatherings (<1000)", 13], 
["Avg Death Rate Before Restricting Gatherings (<1000)", 14], 
["Max Death Rate Before Restricting Gatherings (<1000)", 14]])
O_S_Most_Strict_Cases = np.array([["Num Cases Before All Non-Essential Sectors Closed", 1], 
["Avg Case Rate Before All Non-Essential Sectors Closed", 2], 
["Max Case Rate Before All Non-Essential Sectors Closed", 2], 
["Num Cases Before Public Transport Closed", 3], 
["Avg Cases Rate Before Public Transport Closed", 4], 
["Max Case Rate Before Public Transport Closed", 4], 
["Num Cases Before Stay-at-Home Total Lockdown", 5], 
["Avg Cases Rate Before Stay-at-Home Total Lockdown", 6], 
["Max Case Rate Before Stay-at-Home Total Lockdown", 6], 
["Num Cases Before Internal Movement Restricted", 7], 
["Avg Cases Rate Before Internal Movement Restricted", 8], 
["Max Case Rate Before Internal Movement Restricted", 8], 
["Num Cases Before Int'l Bans Total Border Closure", 9], 
["Avg Cases Rate Before Int'l Bans Total Border Closure", 10], 
["Max Case Rate Before Int'l Bans Total Border Closure", 10], 
["Num Cases Before Facial Coverings in All Public Places", 11], 
["Avg Case Rate Before Facial Coverings in All Public Places", 12], 
["Max Case Rate Before Facial Coverings in All Public Places", 12], 
["Num Cases Before Facial Coverings Mandate", 11], 
["Avg Case Rate Before Facial Coverings Mandate", 12], 
["Max Case Rate Before Facial Coverings Mandate", 12], 
["Num Cases Before All Schools Closed", 13], 
["Avg Case Rate Before All Schools Closed", 14], 
["Max Case Rate Before All Schools Closed", 14], 
["Num Cases Before Public Events Cancelled", 15], 
["Avg Case Rate Before Public Events Cancelled", 16], 
["Max Case Rate Before Public Events Cancelled", 16], 
["Num Cases Before Restricting Gatherings (<100)", 17], 
["Avg Case Rate Before Restricting Gatherings (<100)", 18], 
["Max Case Rate Before Restricting Gatherings (<100)", 18], 
["Num Cases Before Restricting Gatherings (<10)", 17], 
["Avg Case Rate Before Restricting Gatherings (<10)", 18], 
["Max Case Rate Before Restricting Gatherings (<10)", 18]])
O_S_Most_Strict_Deaths = np.array([["Num Deaths Before All Non-Essential Sectors Closed", 1], 
["Avg Death Rate Before All Non-Essential Sectors Closed", 2], 
["Max Death Rate Before All Non-Essential Sectors Closed", 2], 
["Num Deaths Before Public Transport Closed", 3], 
["Avg Deaths Rate Before Public Transport Closed", 4], 
["Max Death Rate Before Public Transport Closed", 4], 
["Num Deaths Before Stay-at-Home Total Lockdown", 5], 
["Avg Deaths Rate Before Stay-at-Home Total Lockdown", 6], 
["Max Death Rate Before Stay-at-Home Total Lockdown", 6], 
["Num Deaths Before Internal Movement Restricted", 7], 
["Avg Deaths Rate Before Internal Movement Restricted", 8], 
["Max Death Rate Before Internal Movement Restricted", 8], 
["Num Deaths Before Int'l Bans Total Border Closure", 9], 
["Avg Deaths Rate Before Int'l Bans Total Border Closure", 10], 
["Max Death Rate Before Int'l Bans Total Border Closure", 10], 
["Num Deaths Before Facial Coverings in All Public Places", 11], 
["Avg Death Rate Before Facial Coverings in All Public Places", 12], 
["Max Death Rate Before Facial Coverings in All Public Places", 12], 
["Num Deaths Before Facial Coverings Mandate", 11], 
["Avg Death Rate Before Facial Coverings Mandate", 12], 
["Max Death Rate Before Facial Coverings Mandate", 12], 
["Num Deaths Before All Schools Closed", 13], 
["Avg Death Rate Before All Schools Closed", 14], 
["Max Death Rate Before All Schools Closed", 14], 
["Num Deaths Before Public Events Cancelled", 15], 
["Avg Death Rate Before Public Events Cancelled", 16], 
["Max Death Rate Before Public Events Cancelled", 16], 
["Num Deaths Before Restricting Gatherings (<100)", 17], 
["Avg Death Rate Before Restricting Gatherings (<100)", 18], 
["Max Death Rate Before Restricting Gatherings (<100)", 18], 
["Num Deaths Before Restricting Gatherings (<10)", 17], 
["Avg Death Rate Before Restricting Gatherings (<10)", 18], 
["Max Death Rate Before Restricting Gatherings (<10)", 18]])
O_S_First_Action = np.array([["Num Cases Before Peak Stringency", 1], ["Avg Case Growth Before Peak Stringency", 2], ["Max Case Growth Before Peak Stringency", 2], ["First Max Stringency to Preceding Cases Ratio", 3], ["Num Deaths Before Peak Stringency", 4], ["Avg Death Growth Before Peak Stringency", 5], ["Max Death Growth Before Peak Stringency", 5], ["First Max Stringency to Preceding Deaths Ratio", 6], ["Num Cases Before Peak Containment Health", 7], ["Avg Case Growth Before Peak Containment Health", 8], ["Max Case Growth Before Peak Containment Health", 8], ["First Max Containment Health to Preceding Cases Ratio", 9], ["Num Deaths Before Peak Containment Health", 10], ["Avg Death Growth Before Peak Containment Health", 11], ["Max Death Growth Before Peak Containment Health", 11], ["First Max Containment Health to Preceding Deaths Ratio", 12], ["Num Cases Before Peak Economic Support", 13], ["Avg Case Growth Before Peak Economic Support", 14], ["Max Case Growth Before Peak Economic Support", 14], ["First Max Economic Support to Preceding Cases Ratio", 15], ["Num Deaths Before Peak Economic Support", 16], ["Avg Death Growth Before Peak Economic Support", 17], ["Max Death Growth Before Peak Economic Support", 17], ["First Max Economic Support to Preceding Deaths Ratio", 18], ["Num Cases Before Peak Government Response", 19], ["Avg Case Growth Before Peak Government Response", 20], ["Max Case Growth Before Peak Government Response", 20], ["First Max Government Response to Preceding Cases Ratio", 21], ["Num Deaths Before Peak Government Response", 22], ["Avg Death Growth Before Peak Government Response", 23], ["Max Death Growth Before Peak Government Response", 23], ["First Max Government Response to Preceding Deaths Ratio", 24]])

#Calculate the sub-indexes
sub_index_calc(Composite_Index, I_O_Cases, Scaled_Metrics, "Initial Outbreak: Cases")
sub_index_calc(Composite_Index, I_O_Deaths, Scaled_Metrics, "Initial Outbreak: Deaths")
sub_index_calc(Composite_Index, T_O_Total_Handeling, Scaled_Metrics, "Total Outbreak: Total Handeling")
sub_index_calc(Composite_Index, T_O_Case_Handeling, Scaled_Metrics, "Total Outbreak: Case Handeling")
sub_index_calc(Composite_Index, T_O_Death_Handeling, Scaled_Metrics, "Total Outbreak: Death Handeling")
sub_index_calc(Composite_Index, G_R_Severity, Scaled_Metrics, "Response Severity: Max Severity")
sub_index_calc(Composite_Index, G_R_Economic, Scaled_Metrics, "Response Severity: Relative Economic Support")
sub_index_calc(Composite_Index, G_R_Length, Scaled_Metrics, "Response Severity: Length of Response")
sub_index_calc(Composite_Index, O_S_Strict_Cases, Scaled_Metrics, "Risk Tolerance: Cases before Strict Measures")
sub_index_calc(Composite_Index, O_S_Strict_Deaths, Scaled_Metrics, "Risk Tolerance: Deaths before Strict Measures")
sub_index_calc(Composite_Index, O_S_Most_Strict_Cases, Scaled_Metrics, "Risk Tolerance: Cases before Most Strict Measures")
sub_index_calc(Composite_Index, O_S_Most_Strict_Deaths, Scaled_Metrics, "Risk Tolerance: Deaths before Most Strict Measures")
sub_index_calc(Composite_Index, O_S_First_Action, Scaled_Metrics, "Risk Tolerance: Outbreak Severity before Peak Action")

#Change NULL values in the Initial Outbreaks: Deaths column to 1 (these instances are when no deaths are recorded)
Composite_Index.loc[:,"Initial Outbreak: Deaths"].fillna(1, inplace=True)

#This function calcultes the values of the bigger indexes based on the smaller indexes
def large_index_calc(Comp_Index, Index_Name, Sub_Indexes):
    #loop through every country
    for i in Comp_Index.index.values:
        #Calcuate the index value
        avg = Comp_Index.loc[i, Sub_Indexes].mean()
        #record the vlaue in the main dataframe
        Comp_Index.loc[i, Index_Name] = avg
    return

#Create the larger indexes    
large_index_calc(Composite_Index, "Risk Tolerance: Strict Measures", ["Risk Tolerance: Cases before Strict Measures", "Risk Tolerance: Deaths before Strict Measures"])
large_index_calc(Composite_Index, "Risk Tolerance: Most Strict Measures", ["Risk Tolerance: Cases before Strict Measures", "Risk Tolerance: Deaths before Strict Measures"])
large_index_calc(Composite_Index, "Total Outbreak", ["Total Outbreak: Total Handeling", "Total Outbreak: Case Handeling", "Total Outbreak: Death Handeling"])
large_index_calc(Composite_Index, "Initial Outbreak", ["Initial Outbreak: Cases", "Initial Outbreak: Deaths"])
large_index_calc(Composite_Index, "Outbreak Severity", ["Initial Outbreak", "Total Outbreak"])
large_index_calc(Composite_Index, "Response Severity", ["Response Severity: Max Severity", "Response Severity: Relative Economic Support", "Response Severity: Length of Response"])
large_index_calc(Composite_Index, "Risk Tolerance: Outbreak Severity before Actions", ["Risk Tolerance: Strict Measures", "Risk Tolerance: Most Strict Measures"])
large_index_calc(Composite_Index, "Risk Tolerance", ["Risk Tolerance: Outbreak Severity before Actions", "Risk Tolerance: Outbreak Severity before Peak Action"])
large_index_calc(Composite_Index, "Composite Index", ["Outbreak Severity", "Response Severity", "Risk Tolerance"])

#save the composite index file as an excel file
Composite_Index.to_excel('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/Composite_Index.xlsx')