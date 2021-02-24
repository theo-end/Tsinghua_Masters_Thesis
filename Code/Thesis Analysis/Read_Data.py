# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:25:45 2020

@author: theod
"""
import numpy as np
import pandas as pd

#Reads in all data files as DataFrames
testing_File = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/covid-testing-all-observations.csv')
cases_File = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/Case_Data.csv')
death_File = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/Death_Data.csv')
pandemic_prepardness_JH_File = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/GHS Rankings.csv')
economic_IMF_File = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/IMF Economic Rankings.csv')
press_freedom_File = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/Press Freedom Rankings.csv')
healthcare_WHO_File = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/WHO Healthcare Rankings.csv')
population_Data = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/Population_Data.csv')
ISO_Lookup = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/ISO_Lookup.csv')
Known_Outliers = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/Known_Outliers.csv')
Manual_Outliers = pd.read_csv('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/Manual_Outliers.csv')

xls = pd.ExcelFile('C:/Users/theod/Tsinghua Masters Thesis/Indices/Read Files/OxCGRT_timeseries_all.xlsx')
government_Response_Index = pd.read_excel(xls, 'government_response_index')
stringency_Index = pd.read_excel(xls, 'stringency_index')
containment_Health_Index = pd.read_excel(xls, 'containment_health_index')
economic_Support_Index = pd.read_excel(xls, 'economic_support_index')
c1_schoolclosing = pd.read_excel(xls, 'c1_school_closing')
c2_workplaceclosing = pd.read_excel(xls, 'c2_workplace_closing')
c3_cancelpublicevents = pd.read_excel(xls, 'c3_cancel_public_events')
c4_restrictionsongatherings = pd.read_excel(xls, 'c4_restrictions_on_gatherings')
c5_closepublictransport = pd.read_excel(xls, 'c5_close_public_transport')
c6_stayathomerequirements = pd.read_excel(xls, 'c6_stay_at_home_requirements')
c7_movementrestrictions = pd.read_excel(xls, 'c7_movementrestrictions')
c8_internationaltravel = pd.read_excel(xls, 'c8_internationaltravel')
e1_incomesupport = pd.read_excel(xls, 'e1_income_support')
e2_debtrelief = pd.read_excel(xls, 'e2_debtrelief')
h1_publicinfocampaign = pd.read_excel(xls, 'h1_public_information_campaigns')
h2_testingpolicy = pd.read_excel(xls, 'h2_testing_policy')
h3_contacttracing = pd.read_excel(xls, 'h3_contact_tracing')
h6_facialcoverings = pd.read_excel(xls, 'h6_facial_coverings')

#Deletes all unecesary Columns from all DataFrames
del testing_File["Source URL"]
del testing_File["Source label"]
del testing_File["Notes"]
del testing_File["Cumulative total per thousand"]
del testing_File["Daily change in cumulative total per thousand"]
del testing_File["7-day smoothed daily change"]
del testing_File["7-day smoothed daily change per thousand"]
del testing_File["Short-term tests per case"]
del testing_File["Short-term positive rate"]
del testing_File["Entity"]


del cases_File["Lat"]
del cases_File["Long"]


del death_File["Lat"]
del death_File["Long"]

del pandemic_prepardness_JH_File["Country"]

del press_freedom_File["Country"]

del government_Response_Index["country_name"]
del stringency_Index["country_name"]
del containment_Health_Index["country_name"]
del economic_Support_Index["country_name"]
del c1_schoolclosing["country_name"]
del c2_workplaceclosing["country_name"]
del c3_cancelpublicevents["country_name"]
del c4_restrictionsongatherings["country_name"]
del c5_closepublictransport["country_name"]
del c6_stayathomerequirements["country_name"]
del c7_movementrestrictions["country_name"]
del c8_internationaltravel["country_name"]
del e1_incomesupport["country_name"]
del e2_debtrelief["country_name"]
del h1_publicinfocampaign["country_name"]
del h2_testingpolicy["country_name"]
del h3_contacttracing["country_name"]
del h6_facialcoverings["country_name"]

del healthcare_WHO_File["Country"]

del population_Data["Country"]
del population_Data["Code"]

del ISO_Lookup["UID"]
del ISO_Lookup["iso2"]
del ISO_Lookup["code3"]
del ISO_Lookup["FIPS"]
del ISO_Lookup["Admin2"]
del ISO_Lookup["Lat"]
del ISO_Lookup["Long_"]
del ISO_Lookup["Combined_Key"]
del ISO_Lookup["Population"]

