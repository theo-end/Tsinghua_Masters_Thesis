# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:15:10 2020

@author: theod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline
from sklearn.svm import SVC



y1 = np.array(cases_MA.loc['USA',:].values)
x1 = np.arange(len(y1))
max_data1 = np.array(argrelextrema(y1, np.greater, order = 30))
min_data1 = np.array(argrelextrema(y1, np.less, order = 30))
sub_data1 = np.append(max_data1, min_data1)
sub_data1 = np.append(sub_data1, [0, (len(y1)-1)])

print(y1)
print(x1)
print(max_data1)
print(min_data1)


i = 0
j = 0
for i in range (len(y1)):
    j = y1[i]
    if j > 1:
        break
print(i)
sub_data1 = np.append(sub_data1, [i])
sub_data1.sort()
print(sub_data1)

y2 = np.array(cases_MA.loc['BRA',:].values)
x2 = np.arange(len(y2))
max_data2 = np.array(argrelextrema(y2, np.greater, order = 30))
min_data2 = np.array(argrelextrema(y2, np.less, order = 30))
sub_data2 = np.append(max_data2, min_data2)
sub_data2 = np.append(sub_data2, [0, (len(y2)-1)])

k = 0
m = 0
for k in range (len(y2)):
    m = y2[k]
    if m > 1:
        break
print(k)
print(m)

sub_data2 = np.append(sub_data2, [k])
sub_data2.sort()

fig, ax = plt.subplots(2,4, sharex = True, sharey=True, figsize=(15,8), gridspec_kw={'hspace':0, 'wspace':0})

i1d1 = interp.interp1d(sub_data1, y1[sub_data1])
ax[0,0].plot(x1, i1d1(x1), label = "interp1d")
ax[0,0].plot(x1, y1, '.', label = "MA")
ax[0,0].set_title('Interp 1D')
ax[0,0].set(ylabel='USA New Cases per Million')


PcI1 = interp.PchipInterpolator(sub_data1, y1[sub_data1])
ax[0,1].plot(x1, PcI1(x1), label = "PchipInterpolator")
ax[0,1].plot(x1, y1, '.', label = "MA")
ax[0,1].set_title('Pchip Interpolator')


A1DI1 = interp.Akima1DInterpolator(sub_data1, y1[sub_data1])
ax[0,2].plot(x1, A1DI1(x1), label = "Akima1DInterpolator")
ax[0,2].plot(x1, y1, '.', label = "MA")
ax[0,2].set_title('Akima 1D Interpolator')


CS1 = interp.CubicSpline(sub_data1, y1[sub_data1])
ax[0,3].plot(x1, CS1(x1), label = "CubicSpline")
ax[0,3].plot(x1, y1, '.', label = "MA")
ax[0,3].set_title('Cubic Spline')


print(i1d1)
print(PcI1)
print(A1DI1)
print(CS1)



i1d2 = interp.interp1d(sub_data2, y2[sub_data2])
ax[1,0].plot(x2, i1d2(x2), label = "interp1d")
ax[1,0].plot(x2, y2, '.', label = "MA")
ax[1,0].set(ylabel='BRA New Cases per Million')
ax[1,0].set(xlabel = 'Days')


PcI2 = interp.PchipInterpolator(sub_data2, y2[sub_data2])
ax[1,1].plot(x2, PcI2(x2), label = "PchipInterpolator")
ax[1,1].plot(x2, y2, '.', label = "MA")
ax[1,1].set(xlabel = 'Days')


A1DI2 = interp.Akima1DInterpolator(sub_data2, y2[sub_data2])
ax[1,2].plot(x2, A1DI2(x2), label = "Akima1DInterpolator")
ax[1,2].plot(x2, y2, '.', label = "MA")
ax[1,2].set(xlabel = 'Days')


CS2 = interp.CubicSpline(sub_data2, y2[sub_data2])
ax[1,3].plot(x2, CS2(x2), label = "CubicSpline")
ax[1,3].plot(x2, y2, '.', label = "MA")
ax[1,3].set(xlabel = 'Days')




fig.savefig("Interpolation Methods")



plt.show()







