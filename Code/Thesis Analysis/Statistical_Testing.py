# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:32:35 2021

@author: theod
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import pingouin as pg

#The Shapiro Test tests for the null hypothesis that the data was drawn from a Noraml Distrubution. It reads in a column.
def Shapiro_Test(df_Col, data_Type):
    #The Value label indicates that the cell in the data frame only contians a single value
    if data_Type == 'Value' or data_Type == 'Interval' or data_Type == 'Ratio':
        #find the values
        x = df_Col.values
        #remove instances of null or empty values
        x = x[~np.isnan(x)]
        #perfrom the shapiro test
        shapiro_test = stats.shapiro(x)
        return shapiro_test
    #The Array label indicates that the cell in the dataframe can contian multiple values (i.e. an array)
    elif data_Type == 'Array':
        #rip all the values form all the cells in the dataframe column
        x = np.concatenate(df_Col)
        #delte instances where "None" was reorted
        x = np.delete(x, np.where(x.astype(str) == 'None'))
        #perform the sahpiro test
        shapiro_test = stats.shapiro(x)
        return shapiro_test
    else:
        return "N/A", 'N/A'

def Spearman_Test(Test_Col, Base_Col, Test_Col_dTpye):
    if Test_Col_dTpye == 'Value':
        data = pd.concat([Base_Col.replace(0, np.nan), Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        results = spearmanr(data)
        return results, data.iloc[:, 0], data.iloc[:,1]
    elif Test_Col_dTpye == 'Array':
        data = pd.concat([Base_Col.replace(0, np.nan), Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        x = []
        y = []
        #loop through every country
        for n in data.index.values:
            #find the y-value array
            y_new = data.loc[n, Test_Col]
                     
            #if the array is empty, do nothing
            if len(y_new) == 0:
                pass
            #otherwise...
            else:
                #save all the values that aren't 'None
                y_add = np.delete(np.asarray(y_new), np.where(np.asarray(y_new).astype(str) == 'None'))
                #make an array of the corresponding x_value of the sma elength as y_add
                x_add = [Base_Col[n]] * len (y_add)
                #add y_add and x_add to the x and y value array
                x = np.append(x, x_add)
                y = np.append(y, y_add)
        results = spearmanr(x,y)
        return results, x, y

def Kendall_Test(Test_Col, Base_Col, Test_Col_dTpye):
    if Test_Col_dTpye == 'Value':
        data = pd.concat([Base_Col.replace(0, np.nan), Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        results = kendalltau(data.iloc[:, 0], data.iloc[:, 1])
        return results, data.iloc[:, 0], data.iloc[:,1]
    elif Test_Col_dTpye == 'Array':
        data = pd.concat([Base_Col.replace(0, np.nan), Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        x = []
        y = []
        #loop through every country
        for n in data.index.values:
            #find the y-value array
            y_new = data.loc[n, Test_Col]
                     
            #if the array is empty, do nothing
            if len(y_new) == 0:
                pass
            #otherwise...
            else:
                #save all the values that aren't 'None
                y_add = np.delete(np.asarray(y_new), np.where(np.asarray(y_new).astype(str) == 'None'))
                #make an array of the corresponding x_value of the sma elength as y_add
                x_add = [Base_Col[n]] * len (y_add)
                #add y_add and x_add to the x and y value array
                x = np.append(x, x_add)
                y = np.append(y, y_add)
        results = kendalltau(x,y)
        return results, x, y

def Pearson_Test(Test_Col, Base_Col, Test_Col_dTpye):
    if Test_Col_dTpye == 'Value':
        data = pd.concat([Base_Col.replace(0, np.nan), Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        results = pearsonr(data)
        return results, data.iloc[:, 0], data.iloc[:,1]
    elif Test_Col_dTpye == 'Array':
        data = pd.concat([Base_Col.replace(0, np.nan), Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        x = []
        y = []
        #loop through every country
        for n in data.index.values:
            #find the y-value array
            y_new = data.loc[n, Test_Col]
                     
            #if the array is empty, do nothing
            if len(y_new) == 0:
                pass
            #otherwise...
            else:
                #save all the values that aren't 'None
                y_add = np.delete(np.asarray(y_new), np.where(np.asarray(y_new).astype(str) == 'None'))
                #make an array of the corresponding x_value of the sma elength as y_add
                x_add = [Base_Col[n]] * len (y_add)
                #add y_add and x_add to the x and y value array
                x = np.append(x, x_add)
                y = np.append(y, y_add)
        results = pearsonr(x,y)
        return results, x, y
    
def ANOVA_Test(Test_Col, Base_Col, Test_Col_dTpye):
    if Test_Col_dTpye == 'Value':
        data = pd.concat([Base_Col, Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        results = pg.anova(dv = data.columns.values[1], between = data.columns.values[0], data = data, detailed = True)
        return results, data
    elif Test_Col_dTpye == 'Array':
        data = pd.concat([Base_Col, Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        x = []
        y = []
        #loop through every country
        for n in data.index.values:
            #find the y-value array
            y_new = data.loc[n, Test_Col]
                     
            #if the array is empty, do nothing
            if len(y_new) == 0:
                pass
            #otherwise...
            else:
                #save all the values that aren't 'None
                y_add = np.delete(np.asarray(y_new), np.where(np.asarray(y_new).astype(str) == 'None'))
                #make an array of the corresponding x_value of the sma elength as y_add
                x_add = [Base_Col[n]] * len (y_add)
                #add y_add and x_add to the x and y value array
                x = np.append(x, x_add)
                y = np.append(y, y_add)
        new_data = pd.DataFrame(data = {'x': x, 'y': y.astype(float)})
        results = pg.anova(dv = 'y', between = 'x', data = new_data, detailed = True)
        return results, new_data

def Kruskal_Wallis_Test(Test_Col, Base_Col, Test_Col_dType):
    if Test_Col_dType == 'Value':
        data = pd.concat([Base_Col, Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        results = pg.kruskal(dv = data.columns.values[1], between = data.columns.values[0], data = data, detailed = True)
        return results, data
    elif Test_Col_dType == 'Array':
        data = pd.concat([Base_Col, Country_Metrics.loc[:, Test_Col]], axis = 1, sort = False)
        data.dropna(inplace = True)
        x = []
        y = []
        #loop through every country
        for n in data.index.values:
            #find the y-value array
            y_new = data.loc[n, Test_Col]
                     
            #if the array is empty, do nothing
            if len(y_new) == 0:
                pass
            #otherwise...
            else:
                #save all the values that aren't 'None
                y_add = np.delete(np.asarray(y_new), np.where(np.asarray(y_new).astype(str) == 'None'))
                #make an array of the corresponding x_value of the sma elength as y_add
                x_add = [Base_Col[n]] * len (y_add)
                #add y_add and x_add to the x and y value array
                x = np.append(x, x_add)
                y = np.append(y, y_add)
        new_data = pd.DataFrame(data = {'x': x, 'y': y.astype(float)})
        results = pg.kruskal(dv = 'y', between = 'x', data = new_data)
        return results, new_data

#This function inserts linebreaks in string when they exceed a character limit. It is used in figure labeling.
def insert_linebreak(string, leng_Label):
    #Create array to hold the location of spaces in the string
    spaces = []
    #find the location of every space in the string
    for i in range(0, len(string)):
        if string[i] == " ":
            spaces.append(i)
    
    #Create an array to store where line breaks need to be added
    add_NL = []
    level = leng_Label
    #find the position of all the necessary linebreaks
    for i in range(0, len(spaces)-1):
        if spaces[i] == spaces[-1]:
            if len(string) > level:
                add_NL.append(spaces[i])
        elif spaces[i] <= level and spaces[i+1] > level:
            add_NL.append(spaces[i])
            level = level + leng_Label
            
    #add the linebreakd to the string
    add_NL.reverse()
    for i in add_NL:
        string = string[:i] + " \n" + string[i:]
    
    #return the string with line breaks
    return string   
    
    
    

#htis funciton perform correlaiton statsicial tests between two sets of metrics. it is imperaitve that the base and est columns follow the format of the Confoudning base and test lists
def Correlation_Testing(Base_Metric_Columns, Test_Metric_Columns, Graphs_Name, Graphs_On_Off, Excel_Name, Excel_On_Off, Metric_df):
    #perform all the Shapiro Tests
    n = 0
    for i in Test_Metric_Columns:
        results = Shapiro_Test(Metric_df.loc[:, i[0]], i[1])
        Test_Metric_Columns[n].append(results)
        n = n + 1
        
    n = 0
    for i in Base_Metric_Columns:
        results = Shapiro_Test(i[0], i[1])
        Base_Metric_Columns[n].append(results)
        n = n + 1
    
    #Initalize the Grpahs file if graphing is turned on
    if Graphs_On_Off == True:
        graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + Graphs_Name + '.pdf')
    
    #Initialize the list that will be turned into dataframes to be written into an excel file recording the test results
    Ratio_Integer_List = []
    Ordinal_List = []
    Nominal_List = []
    
    #loop through all the base metrics
    for i in Base_Metric_Columns:
        #Loop through all the test metrics
        for j in Test_Metric_Columns:
            #Initilize the graph if graphins is turned on
            if Graphs_On_Off == True:
                print(i[2] + " vs. " + j[0])
                fig = plt.figure(constrained_layout = True)
                gs = fig.add_gridspec(3,2)
            
            #intitalize an x and y vairbales for data storage
            x = []
            y = []
            #if the test metric is a ratio or interval scale...
            if i[1] == 'Ratio' or i[1] == 'Interval':
                #if both the base and test metric are normall distrubuted...
                if j[2][1] >= 0.05 and i[3][1] >= 0.05:
                    #Perfrom a Pearson Correlation Test
                    PT, x, y = Pearson_Test(j[0], i[0], j[1])
                    
                    #Make the grpah if grpahing is truned on
                    if Graphs_On_Off == True:
                        #Initialize the axes
                        ax = fig.add_subplot(gs[:-1, :])
                        ax2 = fig.add_subplot(gs[2,0])
                        ax3 = fig.add_subplot(gs[2,1])
                        
                        #Set the axis titles and formatting
                        ax.set_title(insert_linebreak(i[2] + ' vs ' + j[0], 60))
                        ax2.set_title("Shapiro-Wilk Test")
                        ax2.axis('off')
                        ax3.set_title("Pearson Correlation Test")
                        ax3.axis('off')
                        
                        #Graph the x and y data
                        ax.plot(x, y, marker = "o", ls = "")
                        ax.set_xlabel(i[2])
                        ax.set_ylabel(insert_linebreak(j[0], 24))
                        
                        #Make the Shapiro Test results into a table
                        Shapiro_df = pd.DataFrame({'Metric': [i[2], j[0]], 'Score': [round(i[3][0], 3), round(j[2][0], 3)], 'P-Value': [round(i[3][1], 3), round(j[2][1], 3)]})
                        table = ax2.table(cellText = Shapiro_df.values, colLabels=Shapiro_df.columns)
                        
                        #Make the Mpearson test rsults into a dataframe
                        Pearson_df= pd.DataFrame({'Output': ['Score', 'P-Value'], 'Results': [round(PT[0], 3), round(PT[1], 3)], 'Outcome':[0, 0]})
                
                        #Fill in the significance of the test based on the p-value
                        if PT[1] <= 0.05:
                            Pearson_df.iloc[1, 'Outcome'] = 'Significant'
                        else: 
                            Pearson_df.loc[1, 'Outcome'] = 'Not Significant'
                            
                        #Fill in the relationship f the test based on the correlation coefficent
                        if abs(PT[0]) < .10:
                            Pearson_df.loc[0,'Outcome'] = 'Negligible'
                        elif abs(PT[0]) < .20:
                            Pearson_df.loc[0,'Outcome'] = 'Weak'
                        elif abs(PT[0]) < .40:
                            Pearson_df.loc[0,'Outcome'] = 'Moderate'
                        elif abs(PT[0]) < .60:
                            Pearson_df.loc[0,'Outcome'] = 'Relatively Strong'
                        elif abs(PT[0]) < .80:
                            Pearson_df.loc[0,'Outcome'] = 'Strong'
                        elif abs(PT[0]) <= 1:
                            Pearson_df.loc[0,'Outcome'] = 'Very Strong'
                        
                        #turn the pearson results into a table
                        table2 = ax3.table(cellText = Pearson_df.values, colLabels = Pearson_df.columns, loc = 'center')
                        
                        #Make the axes of the graphs into a log scale if the data is large
                        if np.amax(x) > 1000:
                            ax.set_xscale('log')
                        if np.amax(y) > 1000:
                            ax.set_yscale('log')
                            
                    #If graphing is truned on...
                    if Excel_On_Off == True: 
                        #Initialize teh significance / relationship indicators
                        Base_Shapiro_Sig = 'NNNN'
                        Test_Shapiro_Sig = 'NNNN'
                        Relationship = 'NNNN'
                        Significance = 'NNNN'
                        
                        #Determine the sahe of the metic disturbution based on the Shapiro p-value
                        if i[3][1] >= 0.05:
                            Base_Shapiro_Sig = 'Normally Distributed'
                        else:
                            Base_Shapiro_Sig = 'Not Normall Distributed'
                        if j[2][1] >= 0.05:
                            Test_Shapiro_Sig = 'Normally Distributed'
                        else:
                            Test_Shapiro_Sig = 'Not Normall Distributed'
                            
                        #Fill in the singificane of the test absed on the p-value
                        if PT[1] <= 0.05:
                            Significance = 'Significant'
                        else: 
                            Significance = 'Not Significant'
                        
                        #Fill in the relaitonship of the test based on the correlation coeficent
                        if abs(PT[0]) < .10:
                            Relationship = 'Negligible'
                        elif abs(PT[0]) < .20:
                            Relationship = 'Weak'
                        elif abs(PT[0]) < .40:
                            Relationship = 'Moderate'
                        elif abs(PT[0]) < .60:
                            Relationship = 'Relatively Strong'
                        elif abs(PT[0]) < .80:
                            Relationship = 'Strong'
                        elif abs(PT[0]) <= 1:
                            Relationship = 'Very Strong'
                        
                        #add all of the information to the ratio/integer list
                        Ratio_Integer_List.append([i[2], j[0], i[3][0], i[3][1], Base_Shapiro_Sig, j[2][0], j[2][1], Test_Shapiro_Sig, "Pearson", PT[0], Relationship, PT[1], Significance])
                
                #Otherwise...
                else:
                    #Perform a Spearman Correlation Test
                    ST, x, y = Spearman_Test(j[0], i[0], j[1])
                    
                    #If graphing is tunred on...
                    if Graphs_On_Off == True:
                        #Intitialize the axes
                        ax = fig.add_subplot(gs[:-1, :])
                        ax2 = fig.add_subplot(gs[2,0])
                        ax3 = fig.add_subplot(gs[2,1])
                        
                        #Set teh titels and axes formats
                        ax.set_title(insert_linebreak(i[2] + ' vs ' + j[0], 60))
                        ax2.set_title("Shapiro-Wilk Test")
                        ax2.axis('off')
                        ax3.set_title("Spearman Correlation Test")
                        ax3.axis('off')
                        
                        #graph the x and y data
                        ax.plot(x, y, marker = "o", ls = "")
                        ax.set_xlabel(i[2])
                        ax.set_ylabel(insert_linebreak(j[0], 24))
                        
                        #tunr the sahpiro test results into a table
                        Shapiro_df = pd.DataFrame({'Metric': [i[2], j[0]], 'Score': [round(i[3][0], 3), round(j[2][0], 3)], 'P-Value': [round(i[3][1], 3), round(j[2][1], 3)]})
                        table = ax2.table(cellText = Shapiro_df.values, colLabels=Shapiro_df.columns, loc = 'center')
                        
                        #tunr the spearman test results into a table
                        Spearman_df = pd.DataFrame({'Output': ['Score', 'P-Value'], 'Results': [round(ST[0], 3), round(ST[1], 3)], 'Outcome':[0, 0]})
                
                        #Deteremine the test significance based on the p-value
                        if ST[1] <= 0.05:
                            Spearman_df.loc[1, 'Outcome'] = 'Significant'
                        else: 
                            Spearman_df.loc[1, 'Outcome'] = 'Not Significant'
                            
                        #determine the realtionship based on the correlation coefficent
                        if abs(ST[0]) < .10:
                            Spearman_df.loc[0,'Outcome'] = 'Negligible'
                        elif abs(ST[0]) < .20:
                            Spearman_df.loc[0,'Outcome'] = 'Weak'
                        elif abs(ST[0]) < .40:
                            Spearman_df.loc[0,'Outcome'] = 'Moderate'
                        elif abs(ST[0]) < .60:
                            Spearman_df.loc[0,'Outcome'] = 'Relatively Strong'
                        elif abs(ST[0]) < .80:
                            Spearman_df.loc[0,'Outcome'] = 'Strong'
                        elif abs(ST[0]) <= 1:
                            Spearman_df.loc[0,'Outcome'] = 'Very Strong'
                            
                        #add the spearman results to teh graph
                        table2 = ax3.table(cellText = Spearman_df.values, colLabels = Spearman_df.columns, loc = 'center')
                        
                        #scale teh axes if the data is large
                        if np.amax(x) > 1000:
                            ax.set_xscale('log')
                        if np.amax(y) > 1000:
                            ax.set_yscale('log')
                    
                    #if the excel is turned on....
                    if Excel_On_Off == True:
                        #intialize the signifcance and relationship variables
                        Base_Shapiro_Sig = 'NNNN'
                        Test_Shapiro_Sig = 'NNNN'
                        Relationship = 'NNNN'
                        Significance = 'NNNN'
                        
                        #determine the shape of the metrics distrbuton based on the shappiro p-value
                        if i[3][1] >= 0.05:
                            Base_Shapiro_Sig = 'Normally Distributed'
                        else:
                            Base_Shapiro_Sig = 'Not Normall Distributed'
                        if j[2][1] >= 0.05:
                            Test_Shapiro_Sig = 'Normally Distributed'
                        else:
                            Test_Shapiro_Sig = 'Not Normall Distributed'
                        
                        #determine the spearman test significant base don it's p-value
                        if ST[1] <= 0.05:
                            Significance = 'Significant'
                        else: 
                            Significance = 'Not Significant'
                        
                        #determine the relationship based on the correlation coefficent
                        if abs(ST[0]) < .10:
                            Relationship = 'Negligible'
                        elif abs(ST[0]) < .20:
                            Relationship = 'Weak'
                        elif abs(ST[0]) < .40:
                            Relationship = 'Moderate'
                        elif abs(ST[0]) < .60:
                            Relationship = 'Relatively Strong'
                        elif abs(ST[0]) < .80:
                            Relationship = 'Strong'
                        elif abs(ST[0]) <= 1:
                            Relationship = 'Very Strong'
                        
                        #add the data tot eh ratio/integer list
                        Ratio_Integer_List.append([i[2], j[0], i[3][0], i[3][1], Base_Shapiro_Sig, j[2][0], j[2][1], Test_Shapiro_Sig, "Spearman", ST[0], Relationship, ST[1], Significance])
            
            #if the data is ordinal (i.e. ranked)
            elif i[1] == 'Ordinal':
                #perfom the kendall test
                KT, x, y = Kendall_Test(j[0], i[0], j[1])
                
                #if grpahing is turned on...
                if Graphs_On_Off == True:
                    #initnialzie the xes
                    ax = fig.add_subplot(gs[:-1, :])
                    ax2 = fig.add_subplot(gs[2,0])
                    ax3 = fig.add_subplot(gs[2,1])
                    
                    #set up the titles and formatting
                    ax.set_title(insert_linebreak(i[2] + ' vs ' + j[0], 60))
                    ax2.set_title("Shapiro-Wilk Test")
                    ax2.axis('off')
                    ax3.set_title("Kendall Correlation Test")
                    ax3.axis('off')
                    
                    #make the plot
                    ax.plot(x, y, marker = "o", ls = "")
                    ax.set_xlabel(i[2])
                    ax.set_ylabel(insert_linebreak(j[0], 24))
                    
                    #turn the shapiro test results as a table
                    Shapiro_df = pd.DataFrame({'Metric': [i[2], j[0]], 'Score': ['N/A, Ordinal', round(j[2][0], 3)], 'P-Value': ['N/A, Ordinal', round(j[2][1], 3)]})
                    table = ax2.table(cellText = Shapiro_df.values, colLabels=Shapiro_df.columns, loc = 'center')
                    
                    #make the results of the kendall test into a dataframe
                    Kendall_df = pd.DataFrame({'Output': ['Score', 'P-Value'], 'Results': [round(KT[0], 3), round(KT[1], 3)], 'Outcome':[0, 0]})
            
                    #determin the test significane base on it's p-value
                    if KT[1] <= 0.05:
                        Kendall_df.loc[1, 'Outcome'] = 'Significant'
                    else: 
                        Kendall_df.loc[1, 'Outcome'] = 'Not Significant'
                    
                    #deremine the relationship based on it's correlation coefficent
                    if abs(KT[0]) < .30:
                        Kendall_df.loc[0,'Outcome'] = 'Weak Agreement'
                    elif abs(KT[0]) < .50:
                        Kendall_df.loc[0,'Outcome'] = 'Moderate Agreement'
                    elif abs(KT[0]) < .70:
                        Kendall_df.loc[0,'Outcome'] = 'Good Agreement'
                    elif abs(KT[0]) <= 1:
                        Kendall_df.loc[0,'Outcome'] = 'Strong Agreement'
                        
                    #add the kendall test results ot the graph
                    table2 = ax3.table(cellText = Kendall_df.values, colLabels = Kendall_df.columns, loc = 'center')
                    
                    #scale teh axes if the data is large
                    if np.amax(x) > 1000:
                        ax.set_xscale('log')
                    if np.amax(y) > 1000:
                        ax.set_yscale('log')
                        
                #if the excel is turned on
                if Excel_On_Off == True:
                    #intilaize the signifcance and relaitonship indicators
                    Test_Shapiro_Sig = 'NNNN'
                    Relationship = 'NNNN'
                    Signficance = 'NNNN'
                    
                    #determine the shape of the disturbution based on teh shapiro p-value
                    if j[2][1] >= 0.05:
                        Test_Shapiro_Sig = 'Normally Distributed'
                    else: 
                        Test_Shapiro_Sig = 'Not Normally Distributed'
                    
                    #detemriene thesignificance of the test based on its p-value
                    if KT[1] <= 0.05:
                        Significance = 'Significant'
                    else: 
                        Significance = 'Not Significant'
                    
                    #determin eht erealtionship based on teh correlaiton coefficent
                    if abs(KT[0]) < .30:
                        Relationship = 'Weak Agreement'
                    elif abs(KT[0]) < .50:
                        Relationship = 'Moderate Agreement'
                    elif abs(KT[0]) < .70:
                        Relationship = 'Good Agreement'
                    elif abs(KT[0]) <= 1:
                        Relationship = 'Strong Agreement'
                    
                    #add the resutls to the ordinal list
                    Ordinal_List.append([i[2], j[0], j[2][0], j[2][1], Test_Shapiro_Sig, "Kendall", KT[0], Relationship, KT[1], Significance])
            
            #if the data is nominal (categrocial)
            elif i[1] == 'Nominal':
                if j[2][1] >= 0.05:
                    #perform an ANOVa analysis
                    ANOVA, data = ANOVA_Test(j[0], i[0], j[1])
                
                    #if grpahing is turned on
                    if Graphs_On_Off == True:
                        #Initialize the axes
                        ax = fig.add_subplot(gs[:-1, :])
                        ax2 = fig.add_subplot(gs[2,:])
                        ax.set_title(insert_linebreak(i[2] + ' vs ' + j[0], 60))
                    
                        #label the table with it's significance
                        if ANOVA.iloc[0, 5] <= 0.05:
                            ax2.set_title("ANOVA - " + "Significant")
                        else:
                            ax2.set_title("ANOVA - " + "Not Significant")
                    
                        #turn the axies off for the table
                        ax2.axis('off')
                    
                        #add the table to the graph
                        table = ax2.table(cellText = ANOVA.values, colLabels = ANOVA.columns, loc = 'center')
                    
                        #turn the data into a historgram
                        categories = np.unique(data.iloc[:,0])
                        #make sure very categorsi is a different color on the historgram
                        for k in range(0, len(categories)):
                            grouped_data = data.loc[data.iloc[:,0] == categories[k]]
                            ax.hist(grouped_data.iloc[:, 1], histtype = 'step', stacked = True, fill = False, label = grouped_data.iloc[0,0] + '-' + str(round(grouped_data.iloc[:, 1].mean(), 3)))
                        ax.legend()
                
                    #if the excel is truned on...
                    if Excel_On_Off == True:
                        #make a signifncae variable
                        Significance = 'NNNN'
                    
                        #determine the significane of the test
                        if ANOVA.iloc[0, 5] <= 0.05:
                            Significance = "Significant"
                        else:
                            Significance = "Not Significant"
                    
                        #output the results of the test
                        Nominal_List.append([i[2], j[0], j[2][0], j[2][1], "Significant", "ANOVA", ANOVA.iloc[0, 4], ANOVA.iloc[0, 5], Significance])
                
                #Otherwise, perform the Kruskal-Willis Test
                else:
                    #perform the Kruskall-Willis Test
                    KWT, data = Kruskal_Wallis_Test(j[0], i[0], j[1])
                    
                    #if grpahing is turned on
                    if Graphs_On_Off == True:
                        #Initialize the axes
                        ax = fig.add_subplot(gs[:-1, :])
                        ax2 = fig.add_subplot(gs[2,:])
                        ax.set_title(insert_linebreak(i[2] + ' vs ' + j[0], 60))
                    
                        #label the table with it's significance
                        if KWT.iloc[0, 3] <= 0.05:
                            ax2.set_title("Kruskal-Willis - " + "Significant")
                        else:
                            ax2.set_title("Kruskal-Willis - " + "Not Significant")
                    
                        #turn the axies off for the table
                        ax2.axis('off')
                    
                        #add the table to the graph
                        table = ax2.table(cellText = KWT.values, colLabels = KWT.columns, loc = 'center')
                    
                        #turn the data into a historgram
                        categories = np.unique(data.iloc[:,0])
                        #make sure very categorsi is a different color on the historgram
                        for k in range(0, len(categories)):
                            grouped_data = data.loc[data.iloc[:,0] == categories[k]]
                            ax.hist(grouped_data.iloc[:, 1], histtype = 'step', stacked = True, fill = False, label = grouped_data.iloc[0,0] + '-' + str(round(grouped_data.iloc[:, 1].median(), 3)))
                        ax.legend()
                    
                    #if the excel is truned on...
                    if Excel_On_Off == True:
                        #make a signifncae variable
                        Significance = 'NNNN'
                    
                        #determine the significane of the test
                        if KWT.iloc[0, 3] <= 0.05:
                            Significance = "Significant"
                        else:
                            Significance = "Not Significant"
                    
                        #output the results of the test
                        Nominal_List.append([i[2], j[0], j[2][0], j[2][1], "Not Significant", "Kruskal-Wallis", KWT.iloc[0, 2], KWT.iloc[0, 3], Significance])
                
            #only save the figure if grpahing is on
            if Graphs_On_Off == True:
                graphs_File.savefig(fig)
                        
    #ony close the graph file if grpahing is on
    if Graphs_On_Off == True:
        graphs_File.close() 
    
    #if the excel file is on...
    if Excel_On_Off == True:
        #turn the lists into dataframes
        Ratio_Integer_df = pd.DataFrame(Ratio_Integer_List, columns = ['Base Metric', 'Test Metric', 'Base Shapiro Score', 'Base Shapiro P-Valie', 'Base Shapiro Significant?', 'Test Shapiro Score', 'Test Shapiro P-Valie', 'Test Shapiro Significant?', 'Test Performed', 'Score', 'Relationship', 'P-Value', 'Significant?'])
        Ordinal_df = pd.DataFrame(Ordinal_List, columns = ['Base Metric', 'Test Metric', 'Test Shapiro Score', 'Test Shapiro P-Valie', 'Test Shapiro Significant?', 'Test Performed', 'Score', 'Relationship', 'P-Value', 'Significant?'])
        Nominal_df = pd.DataFrame(Nominal_List, columns = ['Base Metric', 'Test Metric', 'Test Shapiro Score', 'Test Shapiro P-Valie', 'Test Shapiro Significant?', 'Test Performed', 'Score', 'P-Value', 'Significant?'])
        
        #write the dataframes to excel files in their own sheets
        with pd.ExcelWriter('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + Excel_Name + '.xlsx') as writer:
            Ratio_Integer_df.to_excel(writer, sheet_name='Ratio and Integer Metrics')
            Ordinal_df.to_excel(writer, sheet_name = 'Ordinal Metrics')
            Nominal_df.to_excel(writer, sheet_name = 'Nominal Metrics')
    
    return

#make the list of counfounding variables test data
COVID_19_Metrics = [['Total Cases per Million', 'Value'], ['Total Deaths per Million', 'Value'], ['Total Tests per Million', 'Value'], ['Case-Death Ratio', 'Value'], ['Test-Case Ratio', 'Value'],  
                            ['Cases Avg. Growth Rate', 'Array'], ['Cases Max. Growth Rate', 'Array'], ['Cases Growth Length', 'Array'], ['Cases Avg. Submission Rate', 'Array'], ['Cases Max. Submission Rate', 'Array'], 	
                            ['Cases Submission Length', 'Array'], ['Cases Total Length', 'Array'], ['Cases Peak Value', 'Array'], ['Cases Valley Value', 'Array'], 
                            ["Cases Avg. Growth Rate(First Outbreak)", "Value"], ["Cases Max. Growth Rate(First Outbreak)", "Value"], 
                            ["Cases Growth Length(First Outbreak)", "Value"], ["Cases Avg. Submission Rate(First Outbreak)", "Value"], ["Cases Max. Submission Rate(First Outbreak)", "Value"], 
                            ["Cases Submission Length(First Outbreak)", "Value"], ["Cases Total Length(First Outbreak)", "Value"], ["Cases Peak Value(First Outbreak)", "Value"], 
                            ["Cases Valley Value(First Outbreak)", "Value"], ['Deaths Avg. Growth Rate', 'Array'], 
                            ['Deaths Max. Growth Rate', 'Array'], ['Deaths Growth Length', 'Array'], ['Deaths Avg. Submission Rate', 'Array'], ['Deaths Max. Submission Rate', 'Array'], ['Deaths Submission Length', 'Array'], 
                            ['Deaths Total Length', 'Array'], ['Deaths Peak Value', 'Array'], ['Deaths Valley Value', 'Array'], 
                            ["Deaths Avg. Growth Rate(First Outbreak)", "Value"], ["Deaths Max. Growth Rate(First Outbreak)", "Value"], 
                            ["Deaths Growth Length(First Outbreak)", "Value"], ["Deaths Avg. Submission Rate(First Outbreak)", "Value"], ["Deaths Max. Submission Rate(First Outbreak)", "Value"], 
                            ["Deaths Submission Length(First Outbreak)", "Value"], ["Deaths Total Length(First Outbreak)", "Value"], ["Deaths Peak Value(First Outbreak)", "Value"], 
                            ["Deaths Valley Value(First Outbreak)", "Value"], ['Case-Death Pair Peak Ratio', 'Array']]

#make the list of the confounding vairbales base data
Confounding_Base_Columns_List = [[population_Data.loc[:, 'Population Density (per sq km)'], 'Ratio', 'Population Density (per sq km)'], [population_Data.loc[:, 'Median Age'], 'Ratio', 'Median Age'], [economic_IMF_File.loc[:, 'General government net lending/borrowing (Percent of GDP)'], 'Interval', 'Government Net Lending/Borrowing (% of GDP)'],
                                 [economic_IMF_File.loc[:, 'Gross domestic product, current prices (Purchasing power parity; international dollars: Billions)'], 'Ratio', 'GDP'], [economic_IMF_File.loc[:, 'Unemployment rate (Percent of total labor force)'], 'Ratio', 'Unemployment Rate'],
                                 [healthcare_WHO_File.loc[:, 'Rank'], 'Ordinal', 'WHO Healthcare Rank'], [healthcare_WHO_File.loc[:, 'Index'], 'Interval', 'WHO Healthcare Index Score'], [press_freedom_File.loc[:, 'Rank'], 'Ordinal', 'Press Freedom Rank'], [press_freedom_File.loc[:, 'Score'], 'Interval', 'Press Freedom Score'], [press_freedom_File.loc[:, 'Categorical Rank'], 'Nominal', 'Press Freedom Category'], 
                                 [pandemic_prepardness_JH_File.loc[:, 'Region'], 'Nominal', 'Geographic Region'], [pandemic_prepardness_JH_File.loc[:, 'Income'], 'Nominal', 'Categorical Income Level']]

#make the list of the GHS rankigs metric base data
Pandemic_Prepardness_Base_Columns_List = [[pandemic_prepardness_JH_File.loc[:, 'Overall Score'], 'Ratio', 'Overall Pandemic Prepardness'], [pandemic_prepardness_JH_File.loc[:, 'Overal Prepardness'], 'Nominal', 'Overall Prepardness'], 
                                          [pandemic_prepardness_JH_File.loc[:, 'Prevention Score'], 'Ratio', 'Prevention Score'], [pandemic_prepardness_JH_File.loc[:, 'Prevention Prepardness'], 'Nominal', 'Prevention Prepardness'], 
                                          [pandemic_prepardness_JH_File.loc[:, 'Detect Score'], 'Ratio', 'Detect Score'], [pandemic_prepardness_JH_File.loc[:, 'Detection Prepardness'], 'Nominal', 'Detection Prepardness'], 
                                          [pandemic_prepardness_JH_File.loc[:, 'Respond Score'], 'Ratio', 'Respond Score'], [pandemic_prepardness_JH_File.loc[:, 'Respond Prepardness'], 'Nominal', 'Respond Prepardness'], 
                                          [pandemic_prepardness_JH_File.loc[:, 'Health Score'], 'Ratio', 'Health Score'], [pandemic_prepardness_JH_File.loc[:, 'Health Prepardness'], 'Nominal', 'Health Prepardness'], 
                                          [pandemic_prepardness_JH_File.loc[:, 'Norms Score'], 'Ratio', 'Norms Score'], [pandemic_prepardness_JH_File.loc[:, 'Norms Prepardness'], 'Nominal', 'Norms Prepardness'], 
                                          [pandemic_prepardness_JH_File.loc[:, 'Risk Score'], 'Ratio', 'Risk Score'], [pandemic_prepardness_JH_File.loc[:, 'Risk Prepardness'], 'Nominal', 'Risk Prepardness']]

Oxford_Indices_Test_Column_List = [["Max Stringency", "Value"], 
["Max Government Response", "Value"], 
["Max Containment Health", "Value"], 
["Max Economic Support", "Value"], 
["Economic Support to Containment Health Ratio", "Value"], 
["Economic Support to Stringency Ratio", "Value"], 
["Economic Support to Government Response Ratio", "Value"], 
["Num Days Some Sectors Closed", "Value"], 
["Num Days All Non-Essential Sectors Closed", "Value"], 
["Num Days Public Transport Closed", "Value"], 
["Num Days Stay-at-Home except for Essential Trips", "Value"], 
["Num Days Stay-at-Home Total Lockdown", "Value"], 
["Num Days Internal Movement Restricted", "Value"], 
["Num Days Int'l Bans for some Countries", "Value"], 
["Num Days Int'l Bans Total Border Closure", "Value"], 
["Num Days Public Information Campaign", "Value"], 
["Num Days Facial Coverings in Some Public Places", "Value"], 
["Num Days Facial Coverings in All Public Places", "Value"], 
["Num Days Facial Coverings Mandate", "Value"], 
["Num Days Some Schools Closed", "Value"], 
["Num Days All Schools Closed", "Value"], 
["Num Days Public Events Cancelled", "Value"], 
["Num Days Restricting Gatherings (<1000)", "Value"], 
["Num Days Restricting Gatherings (<100)", "Value"], 
["Num Days Restricting Gatherings (<10)", "Value"]]

Risk_Tolerance = [["Num Cases Before Some Sectors Closed", "Value"], 
["Avg Case Rate Before Some Sectors Closed", "Value"], 
["Max Case Rate Before Some Sectors Closed", "Value"], 
["Num Cases Before All Non-Essential Sectors Closed", "Value"], 
["Avg Case Rate Before All Non-Essential Sectors Closed", "Value"], 
["Max Case Rate Before All Non-Essential Sectors Closed", "Value"], 
["Num Deaths Before Some Sectors Closed", "Value"], 
["Avg Death Rate Before Some Sectors Closed", "Value"], 
["Max Death Rate Before Some Sectors Closed", "Value"], 
["Num Deaths Before All Non-Essential Sectors Closed", "Value"], 
["Avg Death Rate Before All Non-Essential Sectors Closed", "Value"], 
["Max Death Rate Before All Non-Essential Sectors Closed", "Value"], 
["Num Cases Before Public Transport Closed", "Value"], 
["Avg Cases Rate Before Public Transport Closed", "Value"], 
["Max Case Rate Before Public Transport Closed", "Value"], 
["Num Deaths Before Public Transport Closed", "Value"], 
["Avg Deaths Rate Before Public Transport Closed", "Value"], 
["Max Death Rate Before Public Transport Closed", "Value"], 
["Num Cases Before Stay-at-Home except for Essential Trips", "Value"], 
["Avg Cases Rate Before Stay-at-Home except for Essential Trips", "Value"], 
["Max Case Rate Before Stay-at-Home except for Essential Trips", "Value"], 
["Num Deaths Before Stay-at-Home except for Essential Trips", "Value"], 
["Avg Deaths Rate Before Stay-at-Home except for Essential Trips", "Value"], 
["Max Death Rate Before Stay-at-Home except for Essential Trips", "Value"], 
["Num Cases Before Stay-at-Home Total Lockdown", "Value"], 
["Avg Cases Rate Before Stay-at-Home Total Lockdown", "Value"], 
["Max Case Rate Before Stay-at-Home Total Lockdown", "Value"], 
["Num Deaths Before Stay-at-Home Total Lockdown", "Value"], 
["Avg Deaths Rate Before Stay-at-Home Total Lockdown", "Value"], 
["Max Death Rate Before Stay-at-Home Total Lockdown", "Value"], 
["Num Cases Before Internal Movement Restricted", "Value"], 
["Avg Cases Rate Before Internal Movement Restricted", "Value"], 
["Max Case Rate Before Internal Movement Restricted", "Value"], 
["Num Deaths Before Internal Movement Restricted", "Value"], 
["Avg Deaths Rate Before Internal Movement Restricted", "Value"], 
["Max Death Rate Before Internal Movement Restricted", "Value"], 
["Num Cases Before Int'l Bans for some Countries", "Value"], 
["Avg Cases Rate Before Int'l Bans for some Countries", "Value"], 
["Max Case Rate Before Int'l Bans for some Countries", "Value"], 
["Num Deaths Before Int'l Bans for some Countries", "Value"], 
["Avg Deaths Rate Before Int'l Bans for some Countries", "Value"], 
["Max Death Rate Before Int'l Bans for some Countries", "Value"], 
["Num Cases Before Int'l Bans Total Border Closure", "Value"], 
["Avg Cases Rate Before Int'l Bans Total Border Closure", "Value"], 
["Max Case Rate Before Int'l Bans Total Border Closure", "Value"], 
["Num Deaths Before Int'l Bans Total Border Closure", "Value"], 
["Avg Deaths Rate Before Int'l Bans Total Border Closure", "Value"], 
["Max Death Rate Before Int'l Bans Total Border Closure", "Value"], 
["Num Cases Before Public Information Campaign", "Value"], 
["Avg Case Rate Before Public Information Campaign", "Value"], 
["Max Case Rate Before Public Information Campaign", "Value"], 
["Num Cases Before Facial Coverings in Some Public Places", "Value"], 
["Avg Case Rate Before Facial Coverings in Some Public Places", "Value"], 
["Max Case Rate Before Facial Coverings in Some Public Places", "Value"], 
["Num Cases Before Some Schools Closed", "Value"], 
["Avg Case Rate Before Some Schools Closed", "Value"], 
["Max Case Rate Before Some Schools Closed", "Value"], 
["Num Cases Before Restricting Gatherings (<1000)", "Value"], 
["Avg Case Rate Before Restricting Gatherings (<1000)", "Value"], 
["Max Case Rate Before Restricting Gatherings (<1000)", "Value"], 
["Num Deaths Before Public Information Campaign", "Value"], 
["Avg Death Rate Before Public Information Campaign", "Value"], 
["Max Death Rate Before Public Information Campaign", "Value"], 
["Num Deaths Before Facial Coverings in Some Public Places", "Value"], 
["Avg Death Rate Before Facial Coverings in Some Public Places", "Value"], 
["Max Death Rate Before Facial Coverings in Some Public Places", "Value"], 
["Num Deaths Before Some Schools Closed", "Value"], 
["Avg Death Rate Before Some Schools Closed", "Value"], 
["Max Death Rate Before Some Schools Closed", "Value"], 
["Num Deaths Before Restricting Gatherings (<1000)", "Value"], 
["Avg Death Rate Before Restricting Gatherings (<1000)", "Value"], 
["Max Death Rate Before Restricting Gatherings (<1000)", "Value"], 
["Num Cases Before Facial Coverings in All Public Places", "Value"], 
["Avg Case Rate Before Facial Coverings in All Public Places", "Value"], 
["Max Case Rate Before Facial Coverings in All Public Places", "Value"], 
["Num Cases Before Facial Coverings Mandate", "Value"], 
["Avg Case Rate Before Facial Coverings Mandate", "Value"], 
["Max Case Rate Before Facial Coverings Mandate", "Value"], 
["Num Cases Before All Schools Closed", "Value"], 
["Avg Case Rate Before All Schools Closed", "Value"], 
["Max Case Rate Before All Schools Closed", "Value"], 
["Num Cases Before Public Events Cancelled", "Value"], 
["Avg Case Rate Before Public Events Cancelled", "Value"], 
["Max Case Rate Before Public Events Cancelled", "Value"], 
["Num Cases Before Restricting Gatherings (<100)", "Value"], 
["Avg Case Rate Before Restricting Gatherings (<100)", "Value"], 
["Max Case Rate Before Restricting Gatherings (<100)", "Value"], 
["Num Cases Before Restricting Gatherings (<10)", "Value"], 
["Avg Case Rate Before Restricting Gatherings (<10)", "Value"], 
["Max Case Rate Before Restricting Gatherings (<10)", "Value"], 
["Num Deaths Before Facial Coverings in All Public Places", "Value"], 
["Avg Death Rate Before Facial Coverings in All Public Places", "Value"], 
["Max Death Rate Before Facial Coverings in All Public Places", "Value"], 
["Num Deaths Before Facial Coverings Mandate", "Value"], 
["Avg Death Rate Before Facial Coverings Mandate", "Value"], 
["Max Death Rate Before Facial Coverings Mandate", "Value"], 
["Num Deaths Before All Schools Closed", "Value"], 
["Avg Death Rate Before All Schools Closed", "Value"], 
["Max Death Rate Before All Schools Closed", "Value"], 
["Num Deaths Before Public Events Cancelled", "Value"], 
["Avg Death Rate Before Public Events Cancelled", "Value"], 
["Max Death Rate Before Public Events Cancelled", "Value"], 
["Num Deaths Before Restricting Gatherings (<100)", "Value"], 
["Avg Death Rate Before Restricting Gatherings (<100)", "Value"], 
["Max Death Rate Before Restricting Gatherings (<100)", "Value"], 
["Num Deaths Before Restricting Gatherings (<10)", "Value"], 
["Avg Death Rate Before Restricting Gatherings (<10)", "Value"], 
["Max Death Rate Before Restricting Gatherings (<10)", "Value"], 
["Num Cases Before Peak Stringency", "Value"], 
["Avg Case Growth Before Peak Stringency", "Value"], 
["Max Case Growth Before Peak Stringency", "Value"], 
["First Max Stringency to Preceding Cases Ratio", "Value"], 
["Num Deaths Before Peak Stringency", "Value"], 
["Avg Death Growth Before Peak Stringency", "Value"], 
["Max Death Growth Before Peak Stringency", "Value"], 
["First Max Stringency to Preceding Deaths Ratio", "Value"], 
["Num Cases Before Peak Containment Health", "Value"], 
["Avg Case Growth Before Peak Containment Health", "Value"], 
["Max Case Growth Before Peak Containment Health", "Value"], 
["First Max Containment Health to Preceding Cases Ratio", "Value"], 
["Num Deaths Before Peak Containment Health", "Value"], 
["Avg Death Growth Before Peak Containment Health", "Value"], 
["Max Death Growth Before Peak Containment Health", "Value"], 
["First Max Containment Health to Preceding Deaths Ratio", "Value"], 
["Num Cases Before Peak Economic Support", "Value"], 
["Avg Case Growth Before Peak Economic Support", "Value"], 
["Max Case Growth Before Peak Economic Support", "Value"], 
["First Max Economic Support to Preceding Cases Ratio", "Value"], 
["Num Deaths Before Peak Economic Support", "Value"], 
["Avg Death Growth Before Peak Economic Support", "Value"], 
["Max Death Growth Before Peak Economic Support", "Value"], 
["First Max Economic Support to Preceding Deaths Ratio", "Value"], 
["Num Cases Before Peak Government Response", "Value"], 
["Avg Case Growth Before Peak Government Response", "Value"], 
["Max Case Growth Before Peak Government Response", "Value"], 
["First Max Government Response to Preceding Cases Ratio", "Value"], 
["Num Deaths Before Peak Government Response", "Value"], 
["Avg Death Growth Before Peak Government Response", "Value"], 
["Max Death Growth Before Peak Government Response", "Value"], 
["First Max Government Response to Preceding Deaths Ratio", "Value"]]
        

Correlation_Testing(Confounding_Base_Columns_List, COVID_19_Metrics, "Confounding_Variable_COVID_19_Metrics", False, "Confounding_Variable_COVID_19_Metrics", False, Country_Metrics)
Correlation_Testing(Pandemic_Prepardness_Base_Columns_List, COVID_19_Metrics, "Pandemic_Prepardness_COVID_19_Metrics", False, "Pandemic_Prepardness_COVID_19_Metrics", False, Country_Metrics)
Correlation_Testing(Confounding_Base_Columns_List, Oxford_Indices_Test_Column_List, "Confounding_Variables_vs_Oxford_Indices_Graphs", False, "Confounding_Variables_vs_Oxford_Indices_Tables", False, Country_Metrics)
Correlation_Testing(Pandemic_Prepardness_Base_Columns_List, Oxford_Indices_Test_Column_List, "Pandemic_Prepardness_Rankings_vs_Oxford_Indice_Metrics_Graphs", False, "Pandemic_Prepardness_Rankings_vs_Oxford_Indice_Metrics_Tables", False, Country_Metrics) 
Correlation_Testing(Confounding_Base_Columns_List, Risk_Tolerance, "Confounding_Variables_vs_Risk_Tolerance", False, "Confounding_Variables_vs_Risk_Tolerance_Tables", False, Country_Metrics)
Correlation_Testing(Pandemic_Prepardness_Base_Columns_List, Risk_Tolerance, "Pandemic_Prepardness_Rankings_vs_Risk_Tolerance", False, "Pandemic_Prepardness_Rankings_vs_Risk_Tolerance_Tables", False, Country_Metrics) 
