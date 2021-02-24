# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:01:45 2020

@author: theod
"""
import pandas as pd
import numpy as np
import scipy.interpolate as interp
from scipy.signal import find_peaks, peak_widths, peak_prominences, find_peaks_cwt
import scipy.signal
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter as svg
from scipy.interpolate import splev, splrep
from scipy.interpolate import BSpline, make_lsq_spline

case_keypoints = np.sort(np.append(cases_Extrema.loc['ABW', 'Mins Final'], cases_Extrema.loc['ABW', 'Maxes Final']))

case_x = np.arange(0, len(cases_smoothed_15_3.loc['ABW', :]))
case_y = cases_smoothed_15_3.loc['ABW', :].values

x_extra = []
for i in range(0, len(case_keypoints)-1):
    #print(case_keypoints[i], case_keypoints[i+1], "--------------------------------------")
    direction = 0
    if case_y[case_keypoints[i]] < case_y[case_keypoints[i+1]]:
        direction = 1
    elif case_y[case_keypoints[i]] > case_y[case_keypoints[i+1]]:
        direction = -1
    
    if case_y[case_keypoints[i]] == 0 and case_y[case_keypoints[i+1]] == 0:
        k=0
    else:
        x = np.array(np.arange(case_keypoints[i], case_keypoints[i+1]+1), int)
        y = np.array(case_y[x], float)
        model = np.poly1d(np.polyfit(x,y,3))
        idx = np.argwhere(np.diff(np.sign(y-model(x)))).flatten()
        x_add = idx + case_keypoints[i]
        
        #print(x_add)
        x_remove = []
        x_analysis = np.sort(np.append(x_add, [case_keypoints[i], case_keypoints[i+1]]))
        for j in range(0, len(x_analysis)-1):
            #print(x_analysis[j], x_analysis[j+1], direction,  "-----------")
            if direction == 1:
                #print(x_analysis[np.argwhere(case_y[x_analysis[j+1:-1]] < case_y[x_analysis[j]])+j+1])
                x_remove = np.append(x_remove, x_analysis[np.argwhere(case_y[x_analysis[j+1:-1]] < case_y[x_analysis[j]])+j+1])
            elif direction == -1:
                #print(x_analysis[np.argwhere(case_y[x_analysis[j+1:-1]] > case_y[x_analysis[j]])+j+1])
                x_remove = np.append(x_remove, x_analysis[np.argwhere(case_y[x_analysis[j+1:-1]] > case_y[x_analysis[j]])+j+1])
        
        x_extra = np.append(x_extra, np.setdiff1d(x_analysis, x_remove))


x_final = np.array(np.sort(np.unique(np.append(case_keypoints, x_extra))), int)
print(x_final)

PChip = interp.PchipInterpolator(x_final, case_y[x_final], extrapolate=False)

graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/' + 'Spline_Smoothing_Testing.pdf')

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(case_x, case_y, label = "Data")
ax.scatter(case_keypoints, case_y[case_keypoints], label = "Keypoints")
ax.plot(case_x, PChip(case_x), label = "Spline")
ax.scatter(x_final, case_y[x_final], zorder = 10, label = 'Pts')
for i in range(0, len(case_keypoints)-1):
    x = np.array(np.arange(case_keypoints[i], case_keypoints[i+1]+1), int)
    y = np.array(case_y[x], float)
    model = np.poly1d(np.polyfit(x,y,3))
    idx = np.argwhere(np.diff(np.sign(y-model(x)))).flatten()
    ax.plot(x, model(x))
    x_add = idx+case_keypoints[i]
    ax.scatter(x_add, y[idx])

ax.legend()

graphs_File.savefig(fig)
graphs_File.close()
