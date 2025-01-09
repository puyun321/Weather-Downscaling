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
os.chdir(os.path.dirname(__file__)) #change to heat index path

#%% HI calculation
def heat_index(temp,rh):
    c1=-8.78469475556; c2=1.6113941; c3=2.33854883889; c4=-0.14611605; c5=-0.012308094; c6=-0.0164248277778; c7=0.002211732;
    c8=0.00072546; c9=-0.000003582
    # temp=(temp*9/5)+32
    if temp<10:
        temp=10
    HI = c1+c2*temp+c3*rh+c4*temp*rh+c5*(temp**2)+c6*(rh**2)+c7*(temp**2)*rh+c8*temp*(rh**2)+c9*(temp**2)*(rh**2)
    return HI

HI_obs=[[]*1 for j in range(6)]
spatiotemporal_HI=[[]*1 for j in range(6)]
lstm_HI=[[]*1 for j in range(6)]
simulate_HI=[[]*1 for j in range(6)]

for j in range(6): 
    for i in range(len(T_obs.iloc[:,j])):
        HI_obs[j].append(heat_index(T_obs.iloc[i,j],RH_obs.iloc[i,j]))
        spatiotemporal_HI[j].append(heat_index(spatiotemporal_T.iloc[i,j],spatiotemporal_RH.iloc[i,j]))
        lstm_HI[j].append(heat_index(lstm_T.iloc[i,j],lstm_RH.iloc[i,j]))        
        simulate_HI[j].append(heat_index(simulate_T.iloc[i],simulate_RH.iloc[i]))        
        
HI_obs = np.array(HI_obs).T; 
spatiotemporal_HI = np.array(spatiotemporal_HI).T
lstm_HI = np.array(lstm_HI).T
simulate_HI = np.array(simulate_HI).T

#%%
def convert_celcuis(temp):
    temp=(temp-32)*5/9 
    return temp

#%%
HI_obs_train=HI_obs[train_index,:];HI_obs_test=HI_obs[test_index,:]
spatiotemporal_HI_train=spatiotemporal_HI[train_index,:];spatiotemporal_HI_test=spatiotemporal_HI[test_index,:]
lstm_HI_train=lstm_HI[train_index,:];lstm_HI_test=lstm_HI[test_index,:]
simulate_HI_train=simulate_HI[train_index,:];simulate_HI_test=simulate_HI[test_index,:]

#%%
"""
評估指標
"""

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
spatiotemporal_train_performance=model_evaluation(spatiotemporal_HI_train, HI_obs_train)
spatiotemporal_test_performance=model_evaluation(spatiotemporal_HI_test, HI_obs_test)
lstm_train_performance=model_evaluation(lstm_HI_train, HI_obs_train)
lstm_test_performance=model_evaluation(lstm_HI_test, HI_obs_test)
simulate_train_performance=model_evaluation(simulate_HI_train, HI_obs_train)
simulate_test_performance=model_evaluation(simulate_HI_test, HI_obs_test)

#%%
HI_train=pd.concat([train_station_info,pd.DataFrame(HI_obs_train),pd.DataFrame(spatiotemporal_HI_train)],axis=1)
HI_test=pd.concat([test_station_info,pd.DataFrame(HI_obs_test),pd.DataFrame(spatiotemporal_HI_test)],axis=1)
HI_train.columns= spatiotemporal_T_train.columns; HI_test.columns= spatiotemporal_T_test.columns; 

writer = pd.ExcelWriter(r'result\performance_spatiotemporal(HI).xlsx',engine='xlsxwriter')
spatiotemporal_train_performance.to_excel(writer,sheet_name="train_performance")
spatiotemporal_test_performance.to_excel(writer,sheet_name="test_performance")
HI_train.to_excel(writer,sheet_name="train")
HI_test.to_excel(writer,sheet_name="test")
writer.close()

#%%
HI_train=pd.concat([train_station_info,pd.DataFrame(HI_obs_train),pd.DataFrame(lstm_HI_train)],axis=1)
HI_test=pd.concat([test_station_info,pd.DataFrame(HI_obs_test),pd.DataFrame(lstm_HI_test)],axis=1)
HI_train.columns= lstm_T_train.columns; HI_test.columns= lstm_T_test.columns; 

writer = pd.ExcelWriter(r'result\performance_lstm(HI).xlsx',engine='xlsxwriter')
lstm_train_performance.to_excel(writer,sheet_name="train_performance")
lstm_test_performance.to_excel(writer,sheet_name="test_performance")
HI_train.to_excel(writer,sheet_name="train")
HI_test.to_excel(writer,sheet_name="test")
writer.close()

#%%
HI_train=pd.concat([train_station_info,pd.DataFrame(HI_obs_train),pd.DataFrame(simulate_HI_train).iloc[:,-1]],axis=1)
HI_test=pd.concat([test_station_info,pd.DataFrame(HI_obs_test),pd.DataFrame(simulate_HI_test).iloc[:,-1]],axis=1)
HI_train.columns= simulate_T_train.columns; HI_test.columns= simulate_T_test.columns; 

writer = pd.ExcelWriter(r'result\performance_simulate(HI).xlsx',engine='xlsxwriter')
simulate_train_performance.to_excel(writer,sheet_name="train_performance")
simulate_test_performance.to_excel(writer,sheet_name="test_performance")
HI_train.to_excel(writer,sheet_name="train")

HI_test.to_excel(writer,sheet_name="test")
writer.close()