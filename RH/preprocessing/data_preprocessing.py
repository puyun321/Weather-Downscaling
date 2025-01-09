# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:23:31 2024

@author: Steve
"""

import pandas as pd 
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
os.chdir("..\..")
#%% read weather info data and prep info data
skip_regionname="信義區"
zhuoshui_weather_info=pd.read_excel(r"dataset\T.xlsx",sheet_name="date_info",index_col=0).iloc[:,:3]
zhuoshui_weather_info.columns=["col0","col1","col2"]

zhuoshui_prep_info=pd.read_excel(r"dataset\Prep.xlsx",sheet_name="date_info",index_col=0).iloc[:,:3]
zhuoshui_prep_info.columns=["col0","col1","col2"]

result = pd.merge(zhuoshui_weather_info, zhuoshui_prep_info, on=["col0","col1","col2"], how="inner")
result = pd.DataFrame([result.iloc[i,:] for i in range(len(result)) if result.iloc[i,0]!=skip_regionname]).reset_index(drop=True)

#%% get same index from weather info and prep info
weather_index=[result[result.eq(zhuoshui_weather_info.iloc[i,:], axis=1).all(1)].index.tolist() for i in range(len(zhuoshui_weather_info))]
prep_index=[result[result.eq(zhuoshui_prep_info.iloc[i,:], axis=1).all(1)].index.tolist() for i in range(len(zhuoshui_prep_info))]

weather_index_=[i for i in range(len(weather_index)) if len(weather_index[i])==1]
prep_index_=[i for i in range(len(prep_index)) if len(prep_index[i])==1]

weather_index_=pd.DataFrame(weather_index_)
prep_index_=pd.DataFrame(weather_index_)

#%% save as excel
writer = pd.ExcelWriter(r'dataset\same_index.xlsx', engine='xlsxwriter')
#save same index for cwb,outdoor,indoor  
weather_index_.to_excel(writer,sheet_name="weather_index")
prep_index_.to_excel(writer,sheet_name="prep_index")

writer.save()

#%% save simulate info

import os
os.chdir(r"D:\lab\research\research_use_function")
# from find_distance import find_nearest_index
os.chdir(os.path.dirname(__file__))

simulate_info = result.iloc[:,:2].drop_duplicates(keep="first").reset_index(drop=True)
simulate_info.to_csv(r"info\simulate_zhuoshui_info.csv",encoding="big5")

#%% get unique date index from date
weather_index_=pd.read_excel(r'dataset/same_index.xlsx',sheet_name="weather_index")
all_station_info=pd.read_csv(r"info\selected_zhuoshui_info2.csv",index_col=0,encoding="big5")
all_station_info=all_station_info.iloc[weather_index_.iloc[:,0],:].reset_index(drop=True)

station_info=pd.read_csv(r"info\selected_zhuoshui_info3.csv",index_col=0,encoding="big5")
date=pd.DataFrame(result.iloc[:,2])
unique_date=date.drop_duplicates(keep="first").reset_index(drop=True)
unique_index=[date[date.iloc[:,0]==unique_date.iloc[i,0]].index for i in range(unique_date.shape[0])]
unique_index=[np.array(array) for array in unique_index]

#%% save unique date index to dataframe
max_len = max(map(len, unique_index))

combined_list = []
for i in range(len(unique_index)):
    array=unique_index[i]
    if len(unique_index[i])<max_len:        
        array2=np.array([np.nan for i in range(max_len-len(array))])
        array=np.concatenate([array,array2])
    
    combined_list.append(array)
 
unique_index=np.array(combined_list)
unique_index=pd.DataFrame(unique_index)
unique_index.to_csv(r"dataset\unique_index.csv")

#%% check the index of each date located in unique date and save as csv
unique_date_index = pd.DataFrame([unique_date[unique_date.iloc[:,0]==date.iloc[index,0]].index[0] for index in range(date.shape[0])])
unique_date_index.to_csv(r"dataset\unique_date_index.csv")


