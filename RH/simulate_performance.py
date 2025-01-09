# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:49:49 2024

@author: Steve
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings

os.chdir(os.path.dirname(__file__))
paths=os.listdir(r"D:\lab\research\weather downscale\zhuoshui\RH\dataset")

#%% define the model and objective
warnings.filterwarnings("ignore")

#%% read data

weather_same_index = pd.read_excel(r"dataset\same_index.xlsx",sheet_name="weather_index",index_col=0)

# temperature
factor=["T","RH"]

def read_data(factor,same_index,get_info=0):
    file_name ="%s-new.xlsx"%factor
    file_path = Path("dataset") / file_name
    
    # Read the sheets you need
    obs_input = pd.read_excel(file_path, sheet_name="obs_input", index_col=0)
    obs_output = pd.read_excel(file_path, sheet_name="obs_output", index_col=0)
    simulate_data = pd.read_excel(file_path, sheet_name="simulate_data", index_col=0)
    # select the data, see get_simulate_info&same_index.py
    obs_input = obs_input.iloc[same_index.iloc[:,0],:].reset_index(drop=True)
    obs_output = obs_output.iloc[same_index.iloc[:,0],:].reset_index(drop=True)
    simulate_data = simulate_data.iloc[same_index.iloc[:,0],:].reset_index(drop=True)
    
    if get_info==1:
    # only get date info
        date_info = pd.read_excel(file_path, sheet_name="date_info", index_col=0)
        date_info = date_info.iloc[same_index.iloc[:,0],:].reset_index(drop=True)
        return obs_input, obs_output, simulate_data, date_info
    else:
        return obs_input, obs_output, simulate_data
    
obs_input1, obs_output1, simulate_data1, date_info = read_data(factor[0],weather_same_index,get_info=1)
obs_input2, obs_output2, simulate_data2 = read_data(factor[1],weather_same_index,get_info=0)

factor2 = ["par","Wd","Ws"]
def read_data2(factor,same_index,get_info=0):
    file_name ="%s-new.xlsx"%factor
    file_path = Path("dataset") / file_name
    
    # Read the sheets you need
    obs_input = pd.read_excel(file_path, sheet_name="obs_input", index_col=0)
    obs_output = pd.read_excel(file_path, sheet_name="obs_output", index_col=0)
    # select the data, see get_simulate_info&same_index.py
    obs_input = obs_input.iloc[same_index.iloc[:,0],:].reset_index(drop=True)
    obs_output = obs_output.iloc[same_index.iloc[:,0],:].reset_index(drop=True)

    return obs_input, obs_output

obs_input3, obs_output3 = read_data2(factor2[0],weather_same_index,get_info=0)
obs_input4, obs_output3 = read_data2(factor2[1],weather_same_index,get_info=0)
obs_input5, obs_output5 = read_data2(factor2[2],weather_same_index,get_info=0)

obs_input1, obs_input2, obs_input3, obs_input4, obs_input5 = np.array(obs_input1), np.array(obs_input2), np.array(obs_input3), np.array(obs_input4), np.array(obs_input5)
simulate_data1, simulate_data2 = np.array(simulate_data1), np.array(simulate_data2)
simulate_data = np.stack([simulate_data1, simulate_data2],axis=2)

#%% merge data

# normalization   
obs_output=[obs_output1, obs_output2]
forecast_index=1 # 0 for temperature, 1 for relative humidity
y_output = np.array(obs_output[forecast_index]) #obs_output1

homogeneous_input= np.stack([obs_input1, obs_input2],axis=2)
homogeneous_input= np.concatenate([np.expand_dims(simulate_data[:,0,:],axis=1),homogeneous_input],axis=1)
heterogeneous_input = np.stack([obs_input3, obs_input4, obs_input5],axis=2)
obs_input= np.stack([obs_input1, obs_input2, obs_input3, obs_input4, obs_input5],axis=2)

#%% 切割訓練、驗證、測試資料
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

unique_index=pd.read_csv(r"dataset\unique_index.csv",index_col=0)
train_index=np.array(unique_index.iloc[:int(len(unique_index)*(train_ratio+val_ratio)),:])
train_index=train_index.flatten()
train_index=train_index[~np.isnan(train_index)].astype(int)

test_index=np.array(unique_index.iloc[int(len(unique_index)*(train_ratio+val_ratio)):,:])
test_index_index=test_index.flatten()
test_index=test_index[~np.isnan(test_index)].astype(int)

y_output_train = y_output[train_index,:]
train_date_info=np.array(date_info)[train_index]

y_output_test = y_output[test_index,:]
test_date_info=np.array(date_info)[test_index]

simulate_train = homogeneous_input[train_index,0,forecast_index]
simulate_test = homogeneous_input[test_index,0,forecast_index]

#%%
"""
評估指標
"""
def RMSE(prediction,target):
    rmse = np.sqrt(((prediction-target)**2).mean())
    return rmse
def R2(prediction,target):
    avg_pred = np.mean(prediction,axis=0)
    avg_target = np.mean(target,axis=0)
    
    numerator = np.sum((target-avg_target)*(prediction-avg_pred))
    denominator = (np.sum((target-avg_target)**2)*np.sum((prediction-avg_pred)**2))**0.5
    r2 = (numerator/denominator)**2
    return r2

def RAE(predicted,actual):

    actual = np.array(actual)
    predicted = np.array(predicted)

    mean_actual = np.mean(actual)

    absolute_errors = np.abs(actual - predicted)
    absolute_errors_sum = np.sum(absolute_errors)

    mean_absolute_deviation = np.sum(np.abs(actual - mean_actual))

    rae = absolute_errors_sum / mean_absolute_deviation

    return rae

def simulate_evaluation(simulate, obs):
    r2=[[]*1 for i in range(obs.shape[1])];rmse=[[]*1 for i in range(obs.shape[1])];rae=[[]*1 for i in range(obs.shape[1])]
    for i in range(0,obs.shape[1]):
        rmse[i].append(RMSE(simulate, obs[:,i]))
        r2[i].append(R2(simulate,obs[:,i]))
        rae[i].append(RAE(simulate,obs[:,i]))
        
    rmse=np.squeeze(np.array(rmse));
    r2=np.squeeze(np.array(r2));
    rae=np.squeeze(np.array(rae));
    
    performance= pd.DataFrame(np.array([rmse,r2,rae])).T
    performance.index=['t+%s'%i for i in range(1,7)]
    performance.columns=['rmse','r2','rae']
    
    return performance

#%%
simulate_train_performance=simulate_evaluation(simulate_train, y_output_train)
simulate_test_performance=simulate_evaluation(simulate_test, y_output_test)

#%% save forecasts and others result
# save model performance and forecast into excel
writer = pd.ExcelWriter(r'result\performance_simulate.xlsx',engine='xlsxwriter')
#save performance
simulate_train_performance.to_excel(writer,sheet_name="simulate_train_performance")
simulate_test_performance.to_excel(writer,sheet_name="simulate_test_performance")
#save forecast
columns_name=np.concatenate([['region','postcode','date'],['obs_T+%s'%i for i in range(1,7)],['simulate_T+6']])
train=pd.DataFrame(np.concatenate([np.array(train_date_info), y_output_train,np.expand_dims(simulate_train,axis=1)],axis=1))
train.columns=columns_name
train.to_excel(writer,sheet_name="train")
test=pd.DataFrame(np.concatenate([np.array(test_date_info), y_output_test, np.expand_dims(simulate_test,axis=1)],axis=1))
test.columns=columns_name
test.to_excel(writer,sheet_name="test")
writer.close() 
