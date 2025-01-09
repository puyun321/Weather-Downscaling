# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:53:29 2024

@author: Steve
"""

import os
import pandas as pd
import numpy as np
os.chdir(os.path.dirname(__file__))
os.chdir("..")
from others import create_file
from error_indicator import ErrorIndicator
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import  ScalarMappable
os.chdir(os.path.dirname(__file__))

#%% read data

spatiotemporal_train_result=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(T).xlsx",sheet_name="train").iloc[:-2,1:16]
spatiotemporal_test_result=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(T).xlsx",sheet_name="test").iloc[:-2,1:16]

train_index=spatiotemporal_train_result.index
test_index=spatiotemporal_test_result.index+len(train_index)

spatiotemporal_result=pd.concat([spatiotemporal_train_result,spatiotemporal_test_result]).reset_index(drop=True)
station_info=spatiotemporal_result.iloc[:,:3]
station_info=station_info.drop_duplicates(keep="first")
index=station_info.index

all_date=station_info.iloc[:,2].drop_duplicates(keep="first")
unique_date_index=[station_info[station_info.iloc[:,2]==date].index for date in all_date]
train_unique_date_index=[unique_date_index[i] for i in range(len(unique_date_index)) if np.all(unique_date_index[i]<=len(train_index)).all()]
test_unique_date_index=[unique_date_index[i] for i in range(len(unique_date_index)) if i>=len(train_unique_date_index)]

#%% get maximum value of training and testing
spatiotemporal_obs=spatiotemporal_result.iloc[:,3:9]
spatiotemporal_obs_mean=spatiotemporal_obs.mean(axis=1)

#get train max
regional_mean_max_train=np.array([spatiotemporal_obs_mean.iloc[index].mean() for index in train_unique_date_index])
regional_mean_max_train_index = np.argsort(regional_mean_max_train)[::-1]

#get test max
regional_mean_max_test=np.array([spatiotemporal_obs_mean.iloc[index].mean() for index in test_unique_date_index])
regional_mean_max_test_index = np.argsort(regional_mean_max_test)[::-1]

#%% date info
os.chdir("..")
date_info=pd.read_excel(r"dataset\T.xlsx",sheet_name="date_info",index_col=0)
same_index=pd.read_excel(r"dataset\same_index.xlsx",sheet_name="weather_index",index_col=0)
date_info=date_info.iloc[np.array(same_index.iloc[:,0]),:].reset_index(drop=True)

unique_index=pd.read_csv(r"dataset\unique_index.csv",index_col=0)

new_unique_index=[[]*1 for j in range(unique_index.shape[1])]
for j in range(unique_index.shape[1]):
    # Find indices where NaN values are present
    nan_indices = unique_index.index[unique_index.iloc[:,j].isna()].tolist()
    for i in range(unique_index.shape[0]):        
        if i in nan_indices:
            new_unique_index[j].append(int(new_unique_index[j][-1]))
        else:
            new_unique_index[j].append(int(unique_index.iloc[i,j]))

new_unique_index = np.array(new_unique_index).T

date=np.array([date_info.iloc[new_unique_index[date_index][0],2] for date_index in range(new_unique_index.shape[0])])
region_info=date_info.iloc[:,:2].drop_duplicates(keep="first").reset_index(drop=True)

#%% Read the shapefile
shapefile_path = "gis/zhuoshui_town.shp"
gdf = gpd.read_file(shapefile_path)
os.chdir(os.path.dirname(__file__))

#%% plot regional result
# Create a colormap
min_val=0; max_val=40
norm = Normalize(vmin=min_val, vmax=max_val)
cmap = plt.cm.get_cmap('jet')
sm = ScalarMappable(norm=norm, cmap=cmap)

def plot_regional_result(region_info,array,date_index, timestep, train_index=0,obs_index=0):
    
    select_region_index=pd.DataFrame([region_info[region_info.iloc[:,0]==array.iloc[index,0]].index for index in range(array.shape[0])]).iloc[:,0]
    new_array=pd.concat([region_info.iloc[select_region_index,0].reset_index(drop=True),pd.DataFrame(array.iloc[:,1]).reset_index(drop=True)],axis=1)
    new_array.columns=['TOWNNAME','value']

    # Perform the join
    joined_gdf = gdf.merge(new_array, on='TOWNNAME')
    # Plot the shapefile
    fig, ax = plt.subplots(figsize=(10, 10))
    joined_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap, norm=norm, legend=True)
    gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    
    if train_index==0:
        date_index=date_index
        create_file(r"event\train\%s"%date[date_index][0:13])
        if obs_index==0:
            plt.savefig(r"event\train\%s\obs_t+%s.png"%(date[date_index][0:13], timestep))
        else:
            plt.savefig(r"event\train\%s\pred_t+%s.png"%(date[date_index][0:13], timestep))
    else:
        date_index=date_index
        create_file(r"event\test\%s"%date[date_index][0:13])

        if obs_index==0:
            plt.savefig(r"event\test\%s\obs_t+%s.png"%(date[date_index][0:13], timestep))
        else:
            plt.savefig(r"event\test\%s\pred_t+%s.png"%(date[date_index][0:13], timestep))

    plt.close()
    
# plot regional result
for train_index_ in range(len(regional_mean_max_train_index)):
    for timestep in range(1,7):
        index=regional_mean_max_train_index[train_index_]
        plot_regional_result(region_info,pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[index],[0,2+timestep]]),index, timestep, 0,0)
        plot_regional_result(region_info,pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[index],[0,8+timestep]]),index, timestep, 0,1)

for test_index_ in range(len(regional_mean_max_test_index)):
    index= test_index_+len(regional_mean_max_train_index)
    for timestep in range(1,7):
        plot_regional_result(region_info,pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[index],[0,2+timestep]]),index, timestep, 1,0)
        plot_regional_result(region_info,pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[index],[0,8+timestep]]),index, timestep, 1,1)

#%% get maximum date for train and test
max_train_date = date[regional_mean_max_train_index]
max_test_date = date[regional_mean_max_test_index]

#%% read each subcatchment info
os.chdir("..")
btm_info=pd.read_csv(r"gis\btm.txt"); btm_info=btm_info.loc[:,'TOWNNAME']
mid_info=pd.read_csv(r"gis\mid.txt"); mid_info=mid_info.loc[:,'TOWNNAME']
top_info=pd.read_csv(r"gis\top.txt"); top_info=top_info.loc[:,'TOWNNAME']
os.chdir(os.path.dirname(__file__))

#%% calculate each event performance
r2=[[] for timestep in range(1,7)];rmse=[[] for timestep in range(1,7)];mae=[[] for timestep in range(1,7)];
region_mae=[[] for timestep in range(1,7)];top_mae=[[] for timestep in range(1,7)];mid_mae=[[] for timestep in range(1,7)];
btm_mae=[[] for timestep in range(1,7)]

i=0
for date_index in regional_mean_max_train_index:
    print(i)
    obs_info=pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[date_index],0])
    array=obs_info
    select_region_index=pd.DataFrame([region_info[region_info.iloc[:,0]==array.iloc[index,0]].index for index in range(array.shape[0])]).iloc[:,0]
    select_region_info=region_info.iloc[select_region_index,:]
    
    btm_index=[select_region_info[select_region_info.iloc[:,0]==btm_info.iloc[index_]].index[0] for index_ in range(btm_info.shape[0]) if len(select_region_info[select_region_info.iloc[:,0]==btm_info.iloc[index_]].index)>0]
    mid_index=[select_region_info[select_region_info.iloc[:,0]==mid_info.iloc[index_]].index[0] for index_ in range(mid_info.shape[0]) if len(select_region_info[select_region_info.iloc[:,0]==mid_info.iloc[index_]].index)>0]
    top_index=[select_region_info[select_region_info.iloc[:,0]==top_info.iloc[index_]].index[0] for index_ in range(top_info.shape[0]) if len(select_region_info[select_region_info.iloc[:,0]==top_info.iloc[index_]].index)>0]
    
    for timestep in range(1,7):
        obs=pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[date_index],2+timestep])
        forecast=pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[date_index],8+timestep])
        r2[timestep-1].append(ErrorIndicator.R2(obs, forecast))
        rmse[timestep-1].append(ErrorIndicator.RMSE(obs, forecast))
        mae[timestep-1].append(ErrorIndicator.np_mae(obs, forecast)[0])
        region_mae[timestep-1].append([ErrorIndicator.np_mae(obs.iloc[index,0], forecast.iloc[index,0]) for index in range(obs.shape[0])])
        top_mae[timestep-1].append(ErrorIndicator.np_mae(obs.iloc[top_index,0], forecast.iloc[top_index,0]))
        mid_mae[timestep-1].append(ErrorIndicator.np_mae(obs.iloc[mid_index,0], forecast.iloc[mid_index,0]))
        btm_mae[timestep-1].append(ErrorIndicator.np_mae(obs.iloc[btm_index,0], forecast.iloc[btm_index,0]))
        
    i+=1
    
r2_=pd.DataFrame(r2).T; 
r2_=pd.concat([pd.DataFrame(date[regional_mean_max_train_index]).reset_index(drop=True),r2_],axis=1)
rmse_=pd.DataFrame(rmse).T
rmse_=pd.concat([pd.DataFrame(date[regional_mean_max_train_index]).reset_index(drop=True),rmse_],axis=1)
mae_=pd.DataFrame(mae).T
mae_=pd.concat([pd.DataFrame(date[regional_mean_max_train_index]).reset_index(drop=True),mae_],axis=1)
top_mae_=pd.DataFrame(top_mae).T
top_mae_=pd.concat([pd.DataFrame(date[regional_mean_max_train_index]).reset_index(drop=True),top_mae_],axis=1)
mid_mae_=pd.DataFrame(mid_mae).T
mid_mae_=pd.concat([pd.DataFrame(date[regional_mean_max_train_index]).reset_index(drop=True),mid_mae_],axis=1)
btm_mae_=pd.DataFrame(btm_mae).T
btm_mae_=pd.concat([pd.DataFrame(date[regional_mean_max_train_index]).reset_index(drop=True),btm_mae_],axis=1)

writer = pd.ExcelWriter(r'result\event_performance(train).xlsx',engine='xlsxwriter')
r2_.to_excel(writer,sheet_name="r2")
rmse_.to_excel(writer,sheet_name="rmse")
mae_.to_excel(writer,sheet_name="mae")
top_mae_.to_excel(writer,sheet_name="top_mae")
mid_mae_.to_excel(writer,sheet_name="mid_mae")
btm_mae_.to_excel(writer,sheet_name="btm_mae")

for i in range(6):
    array=pd.DataFrame(region_mae[i])
    array=pd.concat([pd.DataFrame(date[regional_mean_max_train_index]).reset_index(drop=True),array],axis=1)
    array.to_excel(writer,sheet_name="region_mae(T+%s)"%(i+1))

writer.close()

#%% model evaluation
r2=[[] for timestep in range(1,7)];rmse=[[] for timestep in range(1,7)];mae=[[] for timestep in range(1,7)];
region_mae=[[] for timestep in range(1,7)]; top_mae=[[] for timestep in range(1,7)]; mid_mae=[[] for timestep in range(1,7)];
btm_mae=[[] for timestep in range(1,7)]

i=0
for date_index in regional_mean_max_test_index:
    print(i)
    date_index=date_index+len(regional_mean_max_train_index)
    obs_info=pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[date_index],0])
    array=obs_info
    select_region_index=pd.DataFrame([region_info[region_info.iloc[:,0]==array.iloc[index,0]].index for index in range(array.shape[0])]).iloc[:,0]
    select_region_info=region_info.iloc[select_region_index,:]
    
    btm_index=[select_region_info[select_region_info.iloc[:,0]==btm_info.iloc[index_]].index[0] for index_ in range(btm_info.shape[0]) if len(select_region_info[select_region_info.iloc[:,0]==btm_info.iloc[index_]].index)>0]
    mid_index=[select_region_info[select_region_info.iloc[:,0]==mid_info.iloc[index_]].index[0] for index_ in range(mid_info.shape[0]) if len(select_region_info[select_region_info.iloc[:,0]==mid_info.iloc[index_]].index)>0]
    top_index=[select_region_info[select_region_info.iloc[:,0]==top_info.iloc[index_]].index[0] for index_ in range(top_info.shape[0]) if len(select_region_info[select_region_info.iloc[:,0]==top_info.iloc[index_]].index)>0]
    
    for timestep in range(1,7):
        obs=pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[date_index],2+timestep])
        forecast=pd.DataFrame(spatiotemporal_result.iloc[unique_date_index[date_index],8+timestep])
        r2[timestep-1].append(ErrorIndicator.R2(obs, forecast))
        rmse[timestep-1].append(ErrorIndicator.RMSE(obs, forecast))
        mae[timestep-1].append(ErrorIndicator.np_mae(obs, forecast)[0])
        region_mae[timestep-1].append([ErrorIndicator.np_mae(obs.iloc[index,0], forecast.iloc[index,0]) for index in range(obs.shape[0])])
        top_mae[timestep-1].append(ErrorIndicator.np_mae(obs.iloc[top_index,0], forecast.iloc[top_index,0]))
        mid_mae[timestep-1].append(ErrorIndicator.np_mae(obs.iloc[mid_index,0], forecast.iloc[mid_index,0]))
        btm_mae[timestep-1].append(ErrorIndicator.np_mae(obs.iloc[btm_index,0], forecast.iloc[btm_index,0]))
        
    i+=1
    
r2_=pd.DataFrame(r2).T; 
r2_=pd.concat([pd.DataFrame(date[regional_mean_max_test_index+len(regional_mean_max_train_index)]).reset_index(drop=True),r2_],axis=1)
rmse_=pd.DataFrame(rmse).T
rmse_=pd.concat([pd.DataFrame(date[regional_mean_max_test_index+len(regional_mean_max_train_index)]).reset_index(drop=True),rmse_],axis=1)
mae_=pd.DataFrame(mae).T
mae_=pd.concat([pd.DataFrame(date[regional_mean_max_test_index+len(regional_mean_max_train_index)]).reset_index(drop=True),mae_],axis=1)
top_mae_=pd.DataFrame(top_mae).T
top_mae_=pd.concat([pd.DataFrame(date[regional_mean_max_test_index+len(regional_mean_max_train_index)]).reset_index(drop=True),top_mae_],axis=1)
mid_mae_=pd.DataFrame(mid_mae).T
mid_mae_=pd.concat([pd.DataFrame(date[regional_mean_max_test_index+len(regional_mean_max_train_index)]).reset_index(drop=True),mid_mae_],axis=1)
btm_mae_=pd.DataFrame(btm_mae).T
btm_mae_=pd.concat([pd.DataFrame(date[regional_mean_max_test_index+len(regional_mean_max_train_index)]).reset_index(drop=True),btm_mae_],axis=1)

writer = pd.ExcelWriter(r'result\event_performance(test).xlsx',engine='xlsxwriter')
r2_.to_excel(writer,sheet_name="r2")
rmse_.to_excel(writer,sheet_name="rmse")
mae_.to_excel(writer,sheet_name="mae")
top_mae_.to_excel(writer,sheet_name="top_mae")
mid_mae_.to_excel(writer,sheet_name="mid_mae")
btm_mae_.to_excel(writer,sheet_name="btm_mae")

for i in range(6):
    array=pd.DataFrame(region_mae[i])
    array=pd.concat([pd.DataFrame(date[regional_mean_max_test_index+len(regional_mean_max_train_index)]).reset_index(drop=True),array],axis=1)
    array.to_excel(writer,sheet_name="region_mae(T+%s)"%(i+1))

writer.close()

#%% plot each event performance in mae
train1=pd.read_excel(r'result\event_performance(train).xlsx',sheet_name="region_mae(T+1)",index_col=0)
train2=pd.read_excel(r'result\event_performance(train).xlsx',sheet_name="region_mae(T+2)",index_col=0)
train3=pd.read_excel(r'result\event_performance(train).xlsx',sheet_name="region_mae(T+3)",index_col=0)

test1=pd.read_excel(r'result\event_performance(test).xlsx',sheet_name="region_mae(T+1)",index_col=0)
test2=pd.read_excel(r'result\event_performance(test).xlsx',sheet_name="region_mae(T+2)",index_col=0)
test3=pd.read_excel(r'result\event_performance(test).xlsx',sheet_name="region_mae(T+3)",index_col=0)

min_val=0
max_val=pd.concat([train1,train2,train3,test1,test2,test3],axis=0).iloc[:,1:].max().mean()
unique_station_info=pd.DataFrame(station_info.iloc[:,0].drop_duplicates(keep="first")).reset_index(drop=True)

#%%  read shp file
import seaborn as sns
os.chdir("..")
shapefile_path = "gis/zhuoshui_town.shp"
gdf = gpd.read_file(shapefile_path)
os.chdir(os.path.dirname(__file__))

#%% plot mae figure
cmap = sns.color_palette("Reds", as_cmap=True)
norm = Normalize(vmin=min_val, vmax=max_val)

for timestep in range(1,7):
    array=pd.read_excel(r'result\event_performance(train).xlsx',sheet_name="region_mae(T+%s)"%timestep,index_col=0)
    for i in range(len(array)):
        new_array=pd.concat([unique_station_info,array.iloc[i,1:].astype(float).reset_index(drop=True)],axis=1)
        new_array.columns=['TOWNNAME','value']
        create_file(r"event\train_error\%s"%array.iloc[i,0][0:13])
        # Perform the join
        joined_gdf = gdf.merge(new_array, on='TOWNNAME', how='outer')
        # Plot the shapefile
        fig, ax = plt.subplots(figsize=(10, 10))
        # Create a color bar with 70 levels
        joined_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 5))
        plt.savefig(r"event\train_error\%s\t+%s_error.png"%(array.iloc[i,0][0:13], timestep))
        
for timestep in range(1,7):
    array=pd.read_excel(r'result\event_performance(test)-new.xlsx',sheet_name="region_mae(T+%s)"%timestep,index_col=0)
    for i in range(len(array)):
        
        new_array=pd.concat([unique_station_info,array.iloc[i,1:].astype(float).reset_index(drop=True)],axis=1)
        new_array.columns=['TOWNNAME','value']
        create_file(r"event\test_error\%s"%array.iloc[i,0][0:13])
        # Perform the join
        joined_gdf = gdf.merge(new_array, on='TOWNNAME')
        # Plot the shapefile
        fig, ax = plt.subplots(figsize=(10, 10))
        joined_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 5))
        plt.savefig(r"event\test_error\%s\t+%s_error.png"%(array.iloc[i,0][0:13], timestep))
