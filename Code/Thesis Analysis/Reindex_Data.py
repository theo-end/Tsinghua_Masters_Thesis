# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:11:22 2020

@author: theod```
"""
import numpy as np
import pandas as pd

#We reindex all the usable DataFrames (i.e. all but testing_File) to have the ISO as the index and if it's a timeseries, the date at the column headers
cases_File.set_index('ISO', inplace = True)
death_File.set_index('ISO', inplace = True)
economic_IMF_File.set_index('ISO', inplace = True)
healthcare_WHO_File.set_index('ISO', inplace = True)
pandemic_prepardness_JH_File.set_index('Country Code', inplace = True)
press_freedom_File.set_index('ISO', inplace = True)
population_Data.set_index('ISO', inplace = True)
government_Response_Index.set_index('country_code', inplace = True)
stringency_Index.set_index('country_code', inplace = True)
containment_Health_Index.set_index('country_code', inplace = True)
economic_Support_Index.set_index('country_code', inplace = True)
c1_schoolclosing.set_index('country_code', inplace = True)
c2_workplaceclosing.set_index('country_code', inplace = True)
c3_cancelpublicevents.set_index('country_code', inplace = True)
c4_restrictionsongatherings.set_index('country_code', inplace = True)
c5_closepublictransport.set_index('country_code', inplace = True)
c6_stayathomerequirements.set_index('country_code', inplace = True)
c7_movementrestrictions.set_index('country_code', inplace = True)
c8_internationaltravel.set_index('country_code', inplace = True)
e1_incomesupport.set_index('country_code', inplace = True)
e2_debtrelief.set_index('country_code', inplace = True)
h1_publicinfocampaign.set_index('country_code', inplace = True)
h2_testingpolicy.set_index('country_code', inplace = True)
h3_contacttracing.set_index('country_code', inplace = True)
h6_facialcoverings.set_index('country_code', inplace = True)
Known_Outliers.set_index('ISO', inplace = True)
Manual_Outliers.set_index('ISO', inplace = True)


#Makes all dates in DateTime
cases_File.columns = pd.to_datetime(cases_File.columns)
death_File.columns = pd.to_datetime(death_File.columns)
government_Response_Index.columns = pd.to_datetime(government_Response_Index.columns)
stringency_Index.columns = pd.to_datetime(stringency_Index.columns)
containment_Health_Index.columns = pd.to_datetime(containment_Health_Index.columns)
economic_Support_Index.columns = pd.to_datetime(economic_Support_Index.columns)
c1_schoolclosing.columns = pd.to_datetime(c1_schoolclosing.columns)
c2_workplaceclosing.columns = pd.to_datetime(c2_workplaceclosing.columns)
c3_cancelpublicevents.columns = pd.to_datetime(c3_cancelpublicevents.columns)
c4_restrictionsongatherings.columns = pd.to_datetime(c4_restrictionsongatherings.columns)
c5_closepublictransport.columns = pd.to_datetime(c5_closepublictransport.columns)
c6_stayathomerequirements.columns = pd.to_datetime(c6_stayathomerequirements.columns)
c7_movementrestrictions.columns = pd.to_datetime(c7_movementrestrictions.columns)
c8_internationaltravel.columns = pd.to_datetime(c8_internationaltravel.columns)
e1_incomesupport.columns = pd.to_datetime(e1_incomesupport.columns)
e2_debtrelief.columns = pd.to_datetime(e2_debtrelief.columns)
h1_publicinfocampaign.columns = pd.to_datetime(h1_publicinfocampaign.columns)
h2_testingpolicy.columns = pd.to_datetime(h2_testingpolicy.columns)
h3_contacttracing.columns = pd.to_datetime(h3_contacttracing.columns)
h6_facialcoverings.columns = pd.to_datetime(h6_facialcoverings.columns)
Known_Outliers.loc[:, 'Date'] = pd.to_datetime(Known_Outliers.loc[:, 'Date'])
Manual_Outliers.loc[:, 'Date'] = pd.to_datetime(Manual_Outliers.loc[:, 'Date'])

#*NOTE: the new_Testing DataFrame is not inlcuded becuase it has already been reindexed by nature of it's construction