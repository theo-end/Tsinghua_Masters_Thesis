# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:19:10 2020

@author: theod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


USA_Total = np.array(cases_MA.loc['USA', :])
USA_Pop = np.array(population_Data.loc['USA', :])
USA_perMil = USA_Total * USA_Pop

FRA_Total = np.array(cases_MA.loc['FRA', :])
FRA_Pop = np.array(population_Data.loc['FRA', :])
FRA_perMil = FRA_Total * FRA_Pop


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))


ax[1].plot(cases_MA.columns.values, USA_Total, label = "USA")
ax[1].plot(cases_MA.columns.values, FRA_Total, label = "FRA")
ax[1].set(xlabel = 'Month', ylabel = 'COVID-19 Cases per Million')
ax[1].legend()

ax[0].plot(cases_MA.columns.values, USA_perMil)
ax[0].plot(cases_MA.columns.values, FRA_perMil)
ax[0].set(xlabel = 'Month', ylabel = 'Total COVID-19 Cases')

ax[0].xaxis.set_major_locator(mdates.MonthLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax[1].xaxis.set_major_locator(mdates.MonthLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

fig.savefig("Data Equity")