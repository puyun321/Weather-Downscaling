# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:10:26 2024

@author: Steve
"""


import os
import pandas as pd
import numpy as np
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
os.chdir(os.path.dirname(__file__))
os.chdir("..")
from others import create_file
from error_indicator import ErrorIndicator
os.chdir(os.path.dirname(__file__))

#%%

spatiotemporal_train_result=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(T).xlsx",sheet_name="train").iloc[:,1:16]
spatiotemporal_test_result=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(T).xlsx",sheet_name="test").iloc[:,1:16]
spatiotemporal_result=pd.concat([spatiotemporal_train_result,spatiotemporal_test_result])
station_info=spatiotemporal_result.loc[:,'region'].drop_duplicates(keep="first").reset_index(drop=True)

simulate_train=pd.read_excel(r'result\performance_simulate.xlsx',sheet_name='train',index_col=0)
simulate_test=pd.read_excel(r'result\performance_simulate.xlsx',sheet_name='test',index_col=0)
simulate=pd.concat([simulate_train,simulate_test],axis=0).reset_index(drop=True)

#%% date info
date=spatiotemporal_result.loc[:,'date'].drop_duplicates(keep="first").reset_index(drop=True)
new_unique_index=[[]*1 for j in range(len(date))]
for j in range(len(date)):
    new_unique_index[j]=np.array(spatiotemporal_result[spatiotemporal_result.loc[:,'date']==date[j]].index)

region_info=spatiotemporal_result.loc[:,['region','postcode']].drop_duplicates(keep="first").reset_index(drop=True)

#%% get each season indices
# Create the DataFrame
df = pd.DataFrame(date)
df.columns=['timestamp']
# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define season date ranges
winter_condition = (df['timestamp'].dt.month == 12) | (df['timestamp'].dt.month < 3)
spring_condition = (df['timestamp'].dt.month >= 3) & (df['timestamp'].dt.month < 6)
summer_condition = (df['timestamp'].dt.month >= 6) & (df['timestamp'].dt.month < 9)
autumn_condition = (df['timestamp'].dt.month >= 9) & (df['timestamp'].dt.month < 12)

# Get indices for each season
winter_indices = set(df[winter_condition].index)
spring_indices = set(df[spring_condition].index)
summer_indices = set(df[summer_condition].index)
autumn_indices = set(df[autumn_condition].index)


#%% calculate each subcatchment performance
os.chdir("..")
btm_info=pd.read_csv(r"gis\btm.txt"); btm_info=btm_info.loc[:,'TOWNNAME']
mid_info=pd.read_csv(r"gis\mid.txt"); mid_info=mid_info.loc[:,'TOWNNAME']
top_info=pd.read_csv(r"gis\top.txt"); top_info=top_info.loc[:,'TOWNNAME']

btm_index=region_info[region_info.iloc[:,0].isin(btm_info)].index
mid_index=region_info[region_info.iloc[:,0].isin(mid_info)].index
top_index=region_info[region_info.iloc[:,0].isin(top_info)].index

#%% define timestep
forecast_timestep=6 # start from 1

#%% plot 4 seasons result
season=['winter','spring','summer','autumn']
indices=[winter_indices,spring_indices,summer_indices,autumn_indices]

shapefile_path = "gis/zhuoshui_town.shp"
gdf = gpd.read_file(shapefile_path)

writer = pd.ExcelWriter(r'T\result\seasonal_performance\seasonal_performance(T+%s).xlsx'%forecast_timestep,engine='xlsxwriter')
for season_index in range(len(season)):

    #select observation data
    selected_obs_array=pd.concat([pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T for index in indices[season_index]],axis=0).mean().astype(float)
    selected_obs_array_=pd.concat([station_info,selected_obs_array],axis=1); selected_obs_array_.columns=['TOWNNAME','value']
    selected_obs_std=pd.concat([pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T for index in indices[season_index]],axis=0).std()
    selected_obs_std_=pd.concat([station_info,selected_obs_std],axis=1); selected_obs_std_.columns=['TOWNNAME','value']
    #select forecast data
    selected_forecast_array=pd.concat([pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],9+forecast_timestep-1]).reset_index(drop=True).T for index in indices[season_index]],axis=0).mean().astype(float)
    selected_forecast_array_=pd.concat([station_info,selected_forecast_array],axis=1); selected_forecast_array_.columns=['TOWNNAME','value']
    #calculate the absolute error of forecast
    selected_forecast_mae=pd.DataFrame([ErrorIndicator.np_mae(pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T,
                                                            pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],9+forecast_timestep-1]).reset_index(drop=True).T) for index in indices[season_index]]).mean()
    selected_forecast_mae_=pd.concat([station_info,selected_forecast_mae.astype(float)],axis=1); selected_forecast_mae_.columns=['TOWNNAME','value']
    
    #select simulate data
    selected_simulate_array=pd.concat([pd.DataFrame(simulate.iloc[new_unique_index[index],-1]).reset_index(drop=True).T for index in indices[season_index]],axis=0).mean().astype(float)
    selected_simulate_array_=pd.concat([station_info,selected_simulate_array],axis=1); selected_simulate_array_.columns=['TOWNNAME','value']

    #calculate the absolute error of simulation
    selected_simulate_mae=pd.DataFrame([ErrorIndicator.np_mae(pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T,
                                                            pd.DataFrame(simulate.iloc[new_unique_index[index],-1]).reset_index(drop=True).T) for index in indices[season_index]]).mean()
    selected_simulate_mae_=pd.concat([station_info,selected_simulate_mae.astype(float)],axis=1); selected_simulate_mae_.columns=['TOWNNAME','value']
    
    
    """
    Plot figures
    """
    create_file(r"T\image_result\season\T+%s"%(forecast_timestep))
    cmap = plt.cm.get_cmap('jet')
    # min_val = 20; max_val=spatiotemporal_result.iloc[:,3:].max().max()
    min_val = 0; max_val=40
    norm = Normalize(vmin=min_val, vmax=max_val)
    #plot forecast figure
    fig, ax = plt.subplots(figsize=(10, 10))
    forecast_gdf=gdf.merge(selected_forecast_array_, on='TOWNNAME')
    forecast_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
    gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 10))
    plt.savefig(r"T\image_result\season\T+%s\forecast_%s.png"%(forecast_timestep,season[season_index]))
    gc.collect()
        
    #plot obs figure
    fig, ax = plt.subplots(figsize=(10, 10))
    obs_gdf=gdf.merge(selected_obs_array_, on='TOWNNAME')
    obs_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
    gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 10))
    plt.savefig(r"T\image_result\season\T+%s\obs_%s.png"%(forecast_timestep,season[season_index]))
    gc.collect()
    
    #plot simulate figure
    fig, ax = plt.subplots(figsize=(10, 10))
    obs_gdf=gdf.merge(selected_simulate_array_, on='TOWNNAME')
    obs_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
    gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 10))
    plt.savefig(r"T\image_result\season\T+%s\simulate_%s.png"%(forecast_timestep,season[season_index]))
    gc.collect()    
    
    #plot std figure
    cmap = plt.cm.get_cmap('Greens')
    # min_val = 20; max_val=spatiotemporal_result.iloc[:,3:].max().max()
    min_val = 3; max_val=8
    norm = Normalize(vmin=min_val, vmax=max_val)
    fig, ax = plt.subplots(figsize=(10, 10))
    std_gdf=gdf.merge(selected_obs_std_, on='TOWNNAME')
    std_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
    gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 6))
    plt.savefig(r"T\image_result\season\T+%s\std_%s.png"%(forecast_timestep,season[season_index]))
    gc.collect()
    
    #plot error figure
    min_error = 0.0; max_error=2
    cmap2 = sns.color_palette("Reds", as_cmap=True)

    norm2 = Normalize(vmin=min_error, vmax=max_error)
    # norm2 = LogNorm(vmin=min_error, vmax=max_error)
    fig, ax = plt.subplots(figsize=(10, 10))
    mae_gdf=gdf.merge(selected_forecast_mae_, on='TOWNNAME')
    mae_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap2,norm=norm2)
    gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=ax, ticks=np.linspace(min_error, max_error, 10))
    plt.savefig(r"T\image_result\season\T+%s\error_%s.png"%(forecast_timestep,season[season_index]))
    gc.collect()
        
    #plot simulate error figure
    min_error = 0.0; max_error=2
    cmap2 = sns.color_palette("Reds", as_cmap=True)

    norm2 = Normalize(vmin=min_error, vmax=max_error)
    # norm2 = LogNorm(vmin=min_error, vmax=max_error)
    fig, ax = plt.subplots(figsize=(10, 10))
    mae_gdf=gdf.merge(selected_simulate_mae_, on='TOWNNAME')
    mae_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap2,norm=norm2)
    gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=ax, ticks=np.linspace(min_error, max_error, 10))
    plt.savefig(r"T\image_result\season\T+%s\simulate_error_%s.png"%(forecast_timestep,season[season_index]))
    gc.collect()
    
    #each subcatchment result
    btm_obs=selected_obs_array_.iloc[btm_index,1].mean();mid_obs=selected_obs_array_.iloc[mid_index,1].mean();top_obs=selected_obs_array_.iloc[top_index,1].mean()
    btm_std=selected_obs_std_.iloc[btm_index,1].mean();mid_std=selected_obs_std_.iloc[mid_index,1].mean();top_std=selected_obs_std_.iloc[top_index,1].mean()
    btm_forecast=selected_forecast_array_.iloc[btm_index,1].mean();mid_forecast=selected_forecast_array_.iloc[mid_index,1].mean();top_forecast=selected_forecast_array_.iloc[top_index,1].mean()
    btm_mae=selected_forecast_mae_.iloc[btm_index,1].mean();mid_mae=selected_forecast_mae_.iloc[mid_index,1].mean();top_mae=selected_forecast_mae_.iloc[top_index,1].mean()
    btm_simulate=selected_simulate_array_.iloc[btm_index,1].mean();mid_simulate=selected_simulate_array_.iloc[mid_index,1].mean();top_simulate=selected_simulate_array_.iloc[top_index,1].mean()
    btm_simulate_mae=selected_simulate_mae_.iloc[btm_index,1].mean();mid_simulate_mae=selected_simulate_mae_.iloc[mid_index,1].mean();top_simulate_mae=selected_simulate_mae_.iloc[top_index,1].mean()
    

    subcatchment_statistics=pd.DataFrame([[btm_obs,mid_obs,top_obs],[btm_forecast,mid_forecast,top_forecast],
                                          [btm_std,mid_std,top_std],[btm_mae,mid_mae,top_mae]])
    subcatchment_statistics.index=['obs','forecast','std','mae']; subcatchment_statistics.columns=['btm','mid','top']
    
    #save result into excel
    selected_obs_array_.to_excel(writer,sheet_name="%s_obs_mean"%season[season_index])
    selected_forecast_array_.to_excel(writer,sheet_name="%s_forecast_mean"%season[season_index])
    selected_obs_std_.to_excel(writer,sheet_name="%s_obs_std"%season[season_index])
    selected_simulate_array_.to_excel(writer,sheet_name="%s_simulate_mean"%season[season_index])
    selected_forecast_mae_.to_excel(writer,sheet_name="%s_forecast_mae"%season[season_index])
    selected_simulate_mae_.to_excel(writer,sheet_name="%s_simulate_mae"%season[season_index])    
    subcatchment_statistics.to_excel(writer,sheet_name="%s_statistics"%season[season_index])
    
writer.close()

#%% plot each year and each season result
zero = df['timestamp'].dt.year == 2020 
one = df['timestamp'].dt.year == 2021
two = df['timestamp'].dt.year == 2022
three = df['timestamp'].dt.year == 2023

# Get indices for each season
zero_indices = set(df[zero].index)
one_indices = set(df[one].index)
two_indices = set(df[two].index)
three_indices = set(df[three].index)

zero_season_indices = [list(set(zero_indices) & set(winter_indices)),list(set(zero_indices) & set(spring_indices)),
                       list(set(zero_indices) & set(summer_indices)),list(set(zero_indices) & set(autumn_indices))]
one_season_indices = [list(set(one_indices) & set(winter_indices)),list(set(one_indices) & set(spring_indices)),
                       list(set(one_indices) & set(summer_indices)),list(set(one_indices) & set(autumn_indices))]
two_season_indices = [list(set(two_indices) & set(winter_indices)),list(set(two_indices) & set(spring_indices)),
                       list(set(two_indices) & set(summer_indices)),list(set(two_indices) & set(autumn_indices))]
three_season_indices = [list(set(three_indices) & set(winter_indices)),list(set(three_indices) & set(spring_indices)),
                       list(set(three_indices) & set(summer_indices)),list(set(three_indices) & set(autumn_indices))]
all_indices=[zero_season_indices,one_season_indices,two_season_indices,three_season_indices]
years=[2020,2021,2022,2023]

writer = pd.ExcelWriter(r'T\result\seasonal_performance\seasonal_performance(each_year)-T+%s.xlsx'%forecast_timestep,engine='xlsxwriter')

for year_index in range(len(all_indices)):
    year_obs=[]; year_std=[]; year_forecast=[]; year_mae=[];year_simulate=[]; year_simulate_mae=[]
    for season_index in range(len(all_indices[year_index])):
        season_indices=all_indices[year_index][season_index]
        #select observation data
        selected_obs_array=pd.concat([pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T for index in season_indices],axis=0).mean()
        selected_obs_array_=pd.concat([station_info,selected_obs_array],axis=1); selected_obs_array_.columns=['TOWNNAME','value']
        selected_obs_std=pd.concat([pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T for index in indices[season_index]],axis=0).std()
        selected_obs_std_=pd.concat([station_info,selected_obs_std],axis=1); selected_obs_std_.columns=['TOWNNAME','value']

        #select forecast data
        selected_forecast_array=pd.concat([pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],9+forecast_timestep-1]).reset_index(drop=True).T for index in season_indices],axis=0).mean()
        selected_forecast_array_=pd.concat([station_info,selected_forecast_array],axis=1); selected_forecast_array_.columns=['TOWNNAME','value']
        
        #calculate the absolute error
        selected_forecast_mae=pd.DataFrame([ErrorIndicator.np_mae(pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T,pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],9+forecast_timestep-1]).reset_index(drop=True).T) for index in indices[season_index]]).mean()
        selected_forecast_mae_=pd.concat([station_info,selected_forecast_mae.astype(float)],axis=1); selected_forecast_mae_.columns=['TOWNNAME','value']

        #select simulate data
        selected_simulate_array=pd.concat([pd.DataFrame(simulate.iloc[new_unique_index[index],-1]).reset_index(drop=True).T for index in season_indices],axis=0).mean()
        selected_simulate_array_=pd.concat([station_info,selected_simulate_array],axis=1); selected_simulate_array_.columns=['TOWNNAME','value']
        
        #calculate the absolute error
        selected_simulate_mae=pd.DataFrame([ErrorIndicator.np_mae(pd.DataFrame(spatiotemporal_result.iloc[new_unique_index[index],3+forecast_timestep-1]).reset_index(drop=True).T,pd.DataFrame(simulate.iloc[new_unique_index[index],-1]).reset_index(drop=True).T) for index in indices[season_index]]).mean()
        selected_simulate_mae_=pd.concat([station_info,selected_simulate_mae.astype(float)],axis=1); selected_simulate_mae_.columns=['TOWNNAME','value']

        """
        Plot figures
        """
        cmap = plt.cm.get_cmap('jet')
        # min_val = 20; max_val=spatiotemporal_result.iloc[:,3:].max().max()
        min_val = 0; max_val=40
        norm = Normalize(vmin=min_val, vmax=max_val)
        #plot forecast figure
        fig, ax = plt.subplots(figsize=(10, 10))
        forecast_gdf=gdf.merge(selected_forecast_array_, on='TOWNNAME', how='outer')
        forecast_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 10))
        plt.savefig(r"T\image_result\season\T+%s\forecast_%s(%s).png"%(forecast_timestep,season[season_index],years[year_index]))
        gc.collect()
            
        #plot obs figure
        fig, ax = plt.subplots(figsize=(10, 10))
        obs_gdf=gdf.merge(selected_obs_array_, on='TOWNNAME', how='outer')
        obs_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 10))
        plt.savefig(r"T\image_result\season\T+%s\obs_%s(%s).png"%(forecast_timestep,season[season_index],years[year_index]))
        gc.collect()
        
        #plot simulate figure
        fig, ax = plt.subplots(figsize=(10, 10))
        obs_gdf=gdf.merge(selected_simulate_array_, on='TOWNNAME', how='outer')
        obs_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 10))
        plt.savefig(r"T\image_result\season\T+%s\simulate_%s(%s).png"%(forecast_timestep,season[season_index],years[year_index]))
        gc.collect()        
        
        #plot std figure
        cmap = plt.cm.get_cmap('Greens')
        # min_val = 20; max_val=spatiotemporal_result.iloc[:,3:].max().max()
        min_val = 3; max_val=8
        norm = Normalize(vmin=min_val, vmax=max_val)
        fig, ax = plt.subplots(figsize=(10, 10))
        std_gdf=gdf.merge(selected_obs_std_, on='TOWNNAME', how='outer')
        std_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap,norm=norm)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.linspace(min_val, max_val, 6))
        plt.savefig(r"T\image_result\season\T+%s\std_%s(%s).png"%(forecast_timestep,season[season_index],years[year_index]))
        gc.collect()
        
        #plot error figure
        min_error = 0; max_error=2
        cmap2 = sns.color_palette("Reds", as_cmap=True)
    
        norm2 = Normalize(vmin=min_error, vmax=max_error)
        fig, ax = plt.subplots(figsize=(10, 10))
        mae_gdf=gdf.merge(selected_forecast_mae_, on='TOWNNAME', how='outer')
        mae_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap2,norm=norm2)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=ax, ticks=np.linspace(min_error, max_error, 10))
        plt.savefig(r"T\image_result\season\T+%s\error_%s(%s).png"%(forecast_timestep,season[season_index],years[year_index]))
        gc.collect()

        fig, ax = plt.subplots(figsize=(10, 10))
        mae_gdf=gdf.merge(selected_simulate_mae_, on='TOWNNAME', how='outer')
        mae_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap2,norm=norm2)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=ax, ticks=np.linspace(min_error, max_error, 10))
        plt.savefig(r"T\image_result\season\T+%s\simulate_error_%s(%s).png"%(forecast_timestep,season[season_index],years[year_index]))
        gc.collect()
        
        year_obs.append(selected_obs_array)
        year_std.append(selected_obs_std)
        year_forecast.append(selected_forecast_array)
        year_mae.append(selected_forecast_mae)
        year_simulate.append(selected_simulate_array)
        year_simulate_mae.append(selected_simulate_mae)
        
    #save result into excel
    year_obs=pd.DataFrame(year_obs).T; year_obs.columns=season
    year_obs=pd.concat([station_info,year_obs],axis=1)
    year_obs.to_excel(writer,sheet_name="%s_obs_mean"%(years[year_index]))

    year_std=pd.DataFrame(year_std).T; year_std.columns=season
    year_std=pd.concat([station_info,year_std],axis=1)
    year_std.to_excel(writer,sheet_name="%s_obs_std"%(years[year_index]))
    
    year_forecast=pd.DataFrame(year_forecast).T; year_forecast.columns=season
    year_forecast=pd.concat([station_info,year_forecast],axis=1)
    year_forecast.to_excel(writer,sheet_name="%s_forecast"%(years[year_index]))
    
    year_mae=pd.DataFrame(year_mae).T; year_mae.columns=season
    year_mae=pd.concat([station_info,year_mae],axis=1)
    year_mae.to_excel(writer,sheet_name="%s_mae"%(years[year_index]))
    
    year_simulate=pd.DataFrame(year_simulate).T; year_simulate.columns=season
    year_simulate=pd.concat([station_info,year_simulate],axis=1)
    year_simulate.to_excel(writer,sheet_name="%s_simulate"%(years[year_index]))
    
    year_simulate_mae=pd.DataFrame(year_simulate_mae).T; year_simulate_mae.columns=season
    year_simulate_mae=pd.concat([station_info,year_simulate_mae],axis=1)
    year_simulate_mae.to_excel(writer,sheet_name="%s_mae"%(years[year_index]))
    
    #each subcatchment result
    btm_obs=year_obs.iloc[btm_index,1:].mean();mid_obs=year_obs.iloc[mid_index,1:].mean();top_obs=year_obs.iloc[top_index,1:].mean()
    btm_std=year_std.iloc[btm_index,1:].mean();mid_std=year_std.iloc[mid_index,1:].mean();top_std=year_std.iloc[top_index,1:].mean()
    btm_forecast=year_forecast.iloc[btm_index,1:].mean();mid_forecast=year_forecast.iloc[mid_index,1:].mean();top_forecast=year_forecast.iloc[top_index,1:].mean()
    btm_mae=year_mae.iloc[btm_index,1:].mean();mid_mae=year_mae.iloc[mid_index,1:].mean();top_mae=year_mae.iloc[top_index,1:].mean()
    btm_simulate=year_simulate.iloc[btm_index,1:].mean();mid_simulate=year_simulate.iloc[mid_index,1:].mean();top_simulate=year_simulate.iloc[top_index,1:].mean()
    btm_simulate_mae=year_simulate_mae.iloc[btm_index,1:].mean();mid_simulate_mae=year_simulate_mae.iloc[mid_index,1:].mean();top_simulate_mae=year_simulate_mae.iloc[top_index,1:].mean()
    
    subcatchment_statistics=pd.concat([pd.DataFrame([btm_obs,btm_forecast,btm_simulate,btm_std,btm_mae,btm_simulate_mae]),
                                       pd.DataFrame([mid_obs,mid_forecast,mid_simulate,mid_std,mid_mae,mid_simulate_mae]),
                                          pd.DataFrame([top_obs,top_forecast,top_simulate,top_std,top_mae,top_simulate_mae])])
    subcatchment_statistics.index=['btm_obs','btm_forecast', 'btm_simulate','btm_std','btm_mae','btm_simulate_mae',
                                   'mid_obs','mid_forecast', 'mid_simulate','mid_std','mid_mae','mid_simulate_mae',
                                   'top_obs','top_forecast', 'top_simulate' ,'top_std','top_mae','top_simulate_mae']
    
    subcatchment_statistics.to_excel(writer,sheet_name="%s_subcatchment"%(years[year_index]))
    
writer.close()

#%% merge subcatchment result
zero_stas=pd.read_excel(r'T\result\seasonal_performance\seasonal_performance(each_year)-T+%s.xlsx'%forecast_timestep,sheet_name="2020_subcatchment",index_col=0)
one_stas=pd.read_excel(r'T\result\seasonal_performance\seasonal_performance(each_year)-T+%s.xlsx'%forecast_timestep,sheet_name="2021_subcatchment",index_col=0)
two_stas=pd.read_excel(r'T\result\seasonal_performance\seasonal_performance(each_year)-T+%s.xlsx'%forecast_timestep,sheet_name="2022_subcatchment",index_col=0)
three_stas=pd.read_excel(r'T\result\seasonal_performance\seasonal_performance(each_year)-T+%s.xlsx'%forecast_timestep,sheet_name="2023_subcatchment",index_col=0)
columns_name=['%s_%s'%(year,season_) for year in years for season_ in season]
stas=pd.concat([zero_stas,one_stas,two_stas,three_stas],axis=1); stas.columns=columns_name
stas.to_csv(r'T\result\seasonal_performance\seasonal_statistics_comparison-T+%s.csv'%forecast_timestep)


