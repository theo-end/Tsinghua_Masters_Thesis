# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:09:36 2020

@author: theod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import scipy.interpolate as interp
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths, peak_prominences, find_peaks_cwt
from scipy.signal import argrelextrema


# =============================================================================
# graphs_File = PdfPages('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/fitting_Testing.pdf')
# 
# for i in cases_MA.index.values:
#     y = cases_MA.loc[i, :]
#     x = range(0, len(y))
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     fig.suptitle(i)
#     
#     ax.plot(x, y, label = "MA")
#     ax.plot(x, savgol_filter(y, 31, 3), label = "SG 31,3")
#     ax.plot(x, savgol_filter(y, 15, 3), label = "SG 15,3")
#     #ax.plot(x, savgol_filter(y, 31, 6), label = "SG 31,6")
#     ax.plot(x, savgol_filter(y, 61, 3), label = "SG 61,3")
#     
#     ax.legend()
#     
#     graphs_File.savefig(fig)
# =============================================================================

peaks_valleys = pd.DataFrame(data = None, index = cases_MA.index.values, columns = ['MA', 'SG 15,3', 'SG 31,3', 'SG 61,3'])
for i in cases_MA.index.values:
# =============================================================================
#     peaks_valleys.loc[i, 'MA'] = len(find_peaks(cases_MA.loc[i, :])[0]) + len(find_peaks(-1 * cases_MA.loc[i, :])[0])
#     peaks_valleys.loc[i, 'SG 15,3'] = len(find_peaks(savgol_filter(cases_MA.loc[i, :], 15, 3))[0]) + len(find_peaks(savgol_filter(-1 * cases_MA.loc[i, :], 15, 3))[0])
#     peaks_valleys.loc[i, 'SG 31,3'] = len(find_peaks(savgol_filter(cases_MA.loc[i, :], 31, 3))[0]) + len(find_peaks(savgol_filter(-1 * cases_MA.loc[i, :], 31, 3))[0])
#     peaks_valleys.loc[i, 'SG 61,3'] = len(find_peaks(savgol_filter(cases_MA.loc[i, :], 61, 3))[0]) + len(find_peaks(savgol_filter(-1 * cases_MA.loc[i, :], 61, 3))[0])
# 
# =============================================================================
    y = cases_MA.loc[i, :].values
    peaks_valleys.loc[i, 'MA'] = len(argrelextrema(y, np.greater, mode = 'wrap')[0]) + len(argrelextrema(y, np.less,  mode = 'wrap')[0]) 
    print(i, argrelextrema(y, np.greater, mode = 'wrap')[0], argrelextrema(y, np.less, mode = 'wrap')[0])
    
peaks_valleys.to_excel('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/peak_valleys_Testing.xlsx')
#graphs_File.close()  