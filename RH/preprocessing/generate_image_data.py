# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:19:56 2024

@author: Steve
"""

import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
os.chdir("..\..")

#%% read unique index
# unique index is the index of all locations for each date
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

#%%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

same_index=pd.read_excel(r"dataset\same_index.xlsx",sheet_name="weather_index",index_col=0)
RH=pd.read_excel(r"dataset\RH.xlsx",sheet_name="obs_input",index_col=0)
RH=RH.iloc[np.array(same_index.iloc[:,0]),:].reset_index(drop=True)
date_info=pd.read_excel(r"dataset\RH.xlsx",sheet_name="date_info",index_col=0)
date_info=date_info.iloc[np.array(same_index.iloc[:,0]),:].reset_index(drop=True)
date=np.array([date_info.iloc[new_unique_index[date_index][0],2] for date_index in range(new_unique_index.shape[0])])
region_info=date_info.iloc[:,:2].drop_duplicates(keep="first").reset_index(drop=True)

simulate_RH=pd.read_excel(r"dataset\RH.xlsx",sheet_name="simulate_data",index_col=0)
simulate_RH=simulate_RH.iloc[np.array(same_index.iloc[:,0]),:].reset_index(drop=True)
sort_simulate_RH=np.array([simulate_RH.iloc[new_unique_index[date_index],0] for date_index in range(new_unique_index.shape[0])])

#%%
for timestep in range(6):
    sort_RH=np.array([RH.iloc[new_unique_index[date_index],timestep] for date_index in range(new_unique_index.shape[0])])
    
# Read the shapefile
    shapefile_path = "gis/zhuoshui_town.shp"
    gdf = gpd.read_file(shapefile_path)
    
    # Create a colormap
    min_val=np.min([np.min(sort_RH),np.min(sort_simulate_RH)]); max_val=np.max([np.max(sort_RH),np.max(sort_simulate_RH)])
    norm = Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.cm.get_cmap('Blues')
    sm = ScalarMappable(norm=norm, cmap=cmap)
    
    date_index=0
    def plot_obs_regional_result(region_info,array,legend=0):
        new_array=pd.concat([region_info.iloc[:,0],pd.DataFrame(array)],axis=1)
        new_array.columns=['TOWNNAME','value']
    
        # Perform the join
        joined_gdf = gdf.merge(new_array, on='TOWNNAME')
    
        # Plot the shapefile
        if legend==0:
            fig, ax = plt.subplots(figsize=(10, 10), facecolor='none')
            joined_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap, norm=norm, legend=False)
            gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
            plt.savefig(r"image_dataset\RH\obs(T-%s)\%s.png"%(timestep,date[date_index][:13]), transparent=True)
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            joined_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap, norm=norm, legend=True)
            gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
            plt.savefig(r"image_dataset\RH\obs(T-%s)\對照組.png"%timestep)
            plt.show()
        
    # 對照組
    plot_obs_regional_result(region_info,pd.DataFrame(sort_RH[date_index,:]),legend=1)
    
    for date_index in range(new_unique_index.shape[0]):
        plot_obs_regional_result(region_info,pd.DataFrame(sort_RH[date_index,:]),legend=0)

#%%

def plot_simulate_regional_result(region_info,array,legend=0):
    new_array=pd.concat([region_info.iloc[:,0],pd.DataFrame(array)],axis=1)
    new_array.columns=['TOWNNAME','value']

    # Perform the join
    joined_gdf = gdf.merge(new_array, on='TOWNNAME')

    # Plot the shapefile
    if legend==0:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='none')
        joined_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap, norm=norm, legend=False)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        plt.savefig(r"image_dataset\RH\simulate\%s.png"%date[date_index][:13], transparent=True)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        joined_gdf.plot(ax=ax, column='value', edgecolor='none', cmap=cmap, norm=norm, legend=True)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        plt.savefig(r"image_dataset\RH\simulate\對照組.png")
        plt.show()
 
# 對照組
plot_simulate_regional_result(region_info,pd.DataFrame(sort_simulate_RH[date_index,:]),legend=1)

for date_index in range(new_unique_index.shape[0]):
    plot_simulate_regional_result(region_info,pd.DataFrame(sort_simulate_RH[date_index,:]),legend=0)
