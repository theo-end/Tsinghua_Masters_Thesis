# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:46:55 2020

@author: theod
"""
import time
import matplotlib.pyplot as plt
start_time = time.time()

#turn off autoamtic plot generation in spyder
plt.ioff()

exec(open("Read_Data.py").read())
print("Done Reading Data")
exec(open("ISO_Tagger.py").read())
print("Done Tagging Data")
exec(open("Combine_Data.py").read())
print("Done Combining Data")
exec(open("Reindex_Data.py").read())
print("Done Reindexing Data")
exec(open("Convert_Data.py").read())
print("Done Converting Data")
exec(open("Outlier_Removal.py").read())
print("Done Removing Outliers")
exec(open("Create_MA.py").read())
print("Done Creating Moving Averages")
exec(open("Smoothing_Function.py").read())
print("Done Smoothing Data")
exec(open("Peak_Finding.py").read())
print("Done Finding Peaks")
exec(open("interpolation_Funct.py").read())
print("Done Interpolating Data")
exec(open("Metric_Generation.py").read())
print("Done Generating Metrics")
exec(open("Plot_Generation.py").read())
print("Done Making Plots")
exec(open("Statistical_Testing.py").read())
print("Done Performing Statistical Tests")
exec(open("Index_Creation.py").read())
print("Done Creating Indexes")

print("Run Time (sec): ", (time.time() - start_time))




