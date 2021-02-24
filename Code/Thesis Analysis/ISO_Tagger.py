# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:35:18 2020

@author: theod
"""
import numpy as np
import pandas as pd

#join ISO_Lookup and the case and death dataframes to add the ISO correctly
cases_File = pd.merge(cases_File, ISO_Lookup, how='left', left_on = ['Province/State', 'Country/Region'], right_on = ['Province_State', 'Country_Region'])
death_File = pd.merge(death_File, ISO_Lookup, how='left', left_on = ['Province/State', 'Country/Region'], right_on = ['Province_State', 'Country_Region'])

#rename iso3 columns to ISO for later consistency
cases_File.rename(columns={"iso3": "ISO"}, inplace=True)
death_File.rename(columns={"iso3": "ISO"}, inplace=True)


#delete unecessary columns from the case and death dataframes
del cases_File["Province/State"]
del cases_File["Country/Region"]
del cases_File["Province_State"]
del cases_File["Country_Region"]

del death_File["Province/State"]
del death_File["Country/Region"]
del death_File["Province_State"]
del death_File["Country_Region"]

#delete rows that don't have ISO codes (ex. the cruise ships)
cases_File.dropna(inplace=True)
death_File.dropna(inplace=True)

