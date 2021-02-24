# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:42:39 2020

@author: theod
"""
import pandas as pd
import numpy as np
import scipy.interpolate as interp

x = np.arange(len(cases_MA.columns.values))  
cases_first_derv = pd.DataFrame(data = None, index = case_Interpolation.index.values, columns = x)
death_first_derv = pd.DataFrame(data = None, index = death_Interpolation.index.values, columns = x)
for i in case_Interpolation.index.values:
    cases_first_derv.loc[i, :] = case_Interpolation.loc[i, 'PChip'](x, 1)
    death_first_derv.loc[i, :] = death_Interpolation.loc[i, 'PChip'](x,1)


    
#cases_first_derv.to_excel('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/cases_first_derv.xlsx')
#death_first_derv.to_excel('C:/Users/theod/Tsinghua Masters Thesis/Indices/Output Files/death_first_derv.xlsx')

for i in cases_first_derv.index.values:
    print(i, case_Interpolation.loc[i, 'PChip'](x, 1).roots(discontinuity = True))