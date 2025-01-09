# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:45:01 2024

@author: Steve
"""


import os
import pandas as pd
import numpy as np
os.chdir(os.path.dirname(__file__))
os.chdir("..")
from others import create_file
from error_indicator import ErrorIndicator

#%% model 1 temperature performance
"""
We first read the temperature performance from the directories
"""

temp_path="T"
os.chdir(temp_path)
spatiotemporal_T_train=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(T).xlsx",sheet_name="train").iloc[:,1:16]
spatiotemporal_T_test=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(T).xlsx",sheet_name="test").iloc[:,1:16]

train_index=spatiotemporal_T_train.index
test_index=spatiotemporal_T_test.index+len(train_index)

spatiotemporal_T=pd.concat([spatiotemporal_T_train,spatiotemporal_T_test]).reset_index(drop=True)
station_info=spatiotemporal_T.iloc[:,:3]
station_info=station_info.drop_duplicates(keep="first")
train_station_info= station_info.iloc[train_index,:].reset_index(drop=True)
test_station_info= station_info.iloc[test_index,:].reset_index(drop=True)

index=station_info.index
T_obs = spatiotemporal_T.iloc[:,3:9]
spatiotemporal_T= spatiotemporal_T.iloc[:,9:]

#%% model 2 temperature performance
lstm_T_train=pd.read_excel("result\performance_lstm(T).xlsx",sheet_name="train").iloc[:,1:16]
lstm_T_test=pd.read_excel("result\performance_lstm(T).xlsx",sheet_name="test").iloc[:,1:16]
lstm_T=pd.concat([lstm_T_train,lstm_T_test]).reset_index(drop=True).iloc[:,9:]

#%% simulation temperature performance
simulate_T_train=pd.read_excel("result\performance_simulate.xlsx",sheet_name="train").iloc[:,1:]
simulate_T_test=pd.read_excel("result\performance_simulate.xlsx",sheet_name="test").iloc[:,1:]
simulate_T=pd.concat([simulate_T_train,simulate_T_test]).reset_index(drop=True).iloc[:,-1]

#%% model 1 rh performance

os.chdir("..")
RH_path="RH"
os.chdir(RH_path)
spatiotemporal_RH_train=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(RH).xlsx",sheet_name="train").iloc[:,1:16]
spatiotemporal_RH_test=pd.read_excel("result\performance_spatiotemporal_transformer-lstm(RH).xlsx",sheet_name="test").iloc[:,1:16]

spatiotemporal_RH=pd.concat([spatiotemporal_RH_train,spatiotemporal_RH_test]).reset_index(drop=True)
RH_obs = spatiotemporal_RH.iloc[:,3:9]
spatiotemporal_RH= spatiotemporal_RH.iloc[:,9:]

#%% model 2 rh performance
lstm_RH_train=pd.read_excel("result\performance_lstm(RH).xlsx",sheet_name="train").iloc[:,1:16]
lstm_RH_test=pd.read_excel("result\performance_lstm(RH).xlsx",sheet_name="test").iloc[:,1:16]
lstm_RH=pd.concat([lstm_RH_train,lstm_RH_test]).reset_index(drop=True).iloc[:,9:]

#%% simulation rh performance
simulate_RH_train=pd.read_excel("result\performance_simulate.xlsx",sheet_name="train").iloc[:,1:]
simulate_RH_test=pd.read_excel("result\performance_simulate.xlsx",sheet_name="test").iloc[:,1:]
simulate_RH=pd.concat([simulate_RH_train,simulate_RH_test]).reset_index(drop=True).iloc[:,-1]
os.chdir(os.path.dirname(__file__)) #change to vpd path

#%% vpd calculation
import math

def vapour_pressure_deficit(temp,rh):
    es= 6.11 * math.exp((1/461)*((1/273)-(1/temp)))
    vpd = es*(100-rh)/100
    return vpd

vpd_obs=[[]*1 for j in range(6)]
spatiotemporal_vpd=[[]*1 for j in range(6)]
lstm_vpd=[[]*1 for j in range(6)]
simulate_vpd=[[]*1 for j in range(6)]

for j in range(6): 
    for i in range(len(T_obs.iloc[:,j])):
        vpd_obs[j].append(vapour_pressure_deficit(T_obs.iloc[i,j],RH_obs.iloc[i,j]))
        spatiotemporal_vpd[j].append(vapour_pressure_deficit(spatiotemporal_T.iloc[i,j],spatiotemporal_RH.iloc[i,j]))
        lstm_vpd[j].append(vapour_pressure_deficit(lstm_T.iloc[i,j],lstm_RH.iloc[i,j]))        
        simulate_vpd[j].append(vapour_pressure_deficit(simulate_T.iloc[i],simulate_RH.iloc[i]))        
        
#%%
vpd_obs = np.array(vpd_obs).T
spatiotemporal_vpd = np.array(spatiotemporal_vpd).T
lstm_vpd = np.array(lstm_vpd).T
simulate_vpd = np.array(simulate_vpd).T

#%%
vpd_obs_train=vpd_obs[train_index,:];vpd_obs_test=vpd_obs[test_index,:]
spatiotemporal_vpd_train=spatiotemporal_vpd[train_index,:];spatiotemporal_vpd_test=spatiotemporal_vpd[test_index,:]
lstm_vpd_train=lstm_vpd[train_index,:];lstm_vpd_test=lstm_vpd[test_index,:]
simulate_vpd_train=simulate_vpd[train_index,:];simulate_vpd_test=simulate_vpd[test_index,:]

#%%

def model_evaluation(pred, obs):
    r2=[[]*1 for i in range(obs.shape[1])];rmse=[[]*1 for i in range(obs.shape[1])];
    rae=[[]*1 for i in range(obs.shape[1])];mae=[[]*1 for i in range(obs.shape[1])];
    for i in range(0,obs.shape[1]):
        rmse[i].append(ErrorIndicator.RMSE(pred[:,i], obs[:,i]))
        r2[i].append(ErrorIndicator.R2(pred[:,i],obs[:,i]))
        rae[i].append(ErrorIndicator.RAE(pred[:,i],obs[:,i]))
        mae[i].append(ErrorIndicator.MAE(pred[:,i],obs[:,i]))
        
    rmse=np.squeeze(np.array(rmse));
    r2=np.squeeze(np.array(r2));
    rae=np.squeeze(np.array(rae));
    mae=np.squeeze(np.array(mae));

    
    performance= pd.DataFrame(np.array([rmse,r2,rae,mae])).T
    performance.index=['t+%s'%i for i in range(1,7)]
    performance.columns=['rmse','r2','rae','mae']
    
    return performance

#%%
spatiotemporal_train_performance=model_evaluation(spatiotemporal_vpd_train, vpd_obs_train)
spatiotemporal_test_performance=model_evaluation(spatiotemporal_vpd_test, vpd_obs_test)
lstm_train_performance=model_evaluation(lstm_vpd_train, vpd_obs_train)
lstm_test_performance=model_evaluation(lstm_vpd_test, vpd_obs_test)
simulate_train_performance=model_evaluation(simulate_vpd_train, vpd_obs_train)
simulate_test_performance=model_evaluation(simulate_vpd_test, vpd_obs_test)

#%%

vpd_train=pd.concat([train_station_info,pd.DataFrame(vpd_obs_train),pd.DataFrame(spatiotemporal_vpd_train)],axis=1)
vpd_test=pd.concat([test_station_info,pd.DataFrame(vpd_obs_test),pd.DataFrame(spatiotemporal_vpd_test)],axis=1)
vpd_train.columns= spatiotemporal_T_train.columns; vpd_test.columns= spatiotemporal_T_test.columns; 

writer = pd.ExcelWriter(r'result\performance_spatiotemporal(VPD).xlsx',engine='xlsxwriter')
spatiotemporal_train_performance.to_excel(writer,sheet_name="train_performance")
spatiotemporal_test_performance.to_excel(writer,sheet_name="test_performance")
vpd_train.to_excel(writer,sheet_name="train")
vpd_test.to_excel(writer,sheet_name="test")
writer.close()

#%%
vpd_train=pd.concat([train_station_info,pd.DataFrame(vpd_obs_train),pd.DataFrame(lstm_vpd_train)],axis=1)
vpd_test=pd.concat([test_station_info,pd.DataFrame(vpd_obs_test),pd.DataFrame(lstm_vpd_test)],axis=1)
vpd_train.columns= lstm_T_train.columns; vpd_test.columns= lstm_T_test.columns; 

writer = pd.ExcelWriter(r'result\performance_lstm(VPD).xlsx',engine='xlsxwriter')
lstm_train_performance.to_excel(writer,sheet_name="train_performance")
lstm_test_performance.to_excel(writer,sheet_name="test_performance")
vpd_train.to_excel(writer,sheet_name="train")
vpd_test.to_excel(writer,sheet_name="test")
writer.close()

#%%
vpd_train=pd.concat([train_station_info,pd.DataFrame(vpd_obs_train),pd.DataFrame(simulate_vpd_train).iloc[:,-1]],axis=1)
vpd_test=pd.concat([test_station_info,pd.DataFrame(vpd_obs_test),pd.DataFrame(simulate_vpd_test).iloc[:,-1]],axis=1)
vpd_train.columns= simulate_T_train.columns; vpd_test.columns= simulate_T_test.columns; 

writer = pd.ExcelWriter(r'result\performance_simulate(VPD).xlsx',engine='xlsxwriter')
simulate_train_performance.to_excel(writer,sheet_name="train_performance")
simulate_test_performance.to_excel(writer,sheet_name="test_performance")
vpd_train.to_excel(writer,sheet_name="train")

vpd_test.to_excel(writer,sheet_name="test")
writer.close()