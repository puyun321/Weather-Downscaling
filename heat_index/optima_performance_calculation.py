# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:45:01 2024

@author: Steve
"""


import os
import pandas as pd
from pathlib import Path
import numpy as np
os.chdir(os.path.dirname(__file__))
os.chdir("..")
from others import create_file
from error_indicator import ErrorIndicator
os.chdir(os.path.dirname(__file__))

#%%
spatiotemporal_train_result=pd.read_excel("result\performance_spatiotemporal(HI).xlsx",sheet_name="train").iloc[:,1:16]
spatiotemporal_test_result=pd.read_excel("result\performance_spatiotemporal(HI).xlsx",sheet_name="test").iloc[:,1:16]
spatiotemporal_HI_train = spatiotemporal_train_result.iloc[:,3:9]
HI_obs_train = spatiotemporal_train_result.iloc[:,9:]
spatiotemporal_HI_test = spatiotemporal_test_result.iloc[:,3:9]
HI_obs_test = spatiotemporal_test_result.iloc[:,9:]

simulate_train=pd.read_excel(r'result\performance_simulate(HI).xlsx',sheet_name='train',index_col=0)
simulate_test=pd.read_excel(r'result\performance_simulate(HI).xlsx',sheet_name='test',index_col=0)

simulate_HI_train=simulate_train.iloc[:,-1]
simulate_HI_test=simulate_test.iloc[:,-1]

train_station_info = spatiotemporal_train_result.iloc[:,:3]
test_station_info = spatiotemporal_test_result.iloc[:,:3]

#%%
def classify_data(data, temp_ranges):
    # Initialize the classes array with zeros
    classes = np.zeros_like(data, dtype=float)
    
    # Classify the data based on the given temperature ranges
    classes[data <= temp_ranges[0]] = 1
    classes[(data > temp_ranges[0]) & (data <= temp_ranges[1])] = 2
    classes[(data > temp_ranges[1]) & (data <= temp_ranges[2])] = 3
    # classes[(data > temp_ranges[2]) & (data <= temp_ranges[3])] = 4
    classes[data > temp_ranges[2]] = 4
    
    return classes

#%%
temp_ranges = [27, 32, 41, 54]
spatiotemporal_HI_train=classify_data(spatiotemporal_HI_train, temp_ranges)
spatiotemporal_HI_test=classify_data(spatiotemporal_HI_test, temp_ranges)
HI_obs_train=classify_data(HI_obs_train, temp_ranges)
HI_obs_test=classify_data(HI_obs_test, temp_ranges)
simulate_HI_train=classify_data(simulate_HI_train, temp_ranges)
simulate_HI_test=classify_data(simulate_HI_test, temp_ranges)

#%% model performance evaluation

from sklearn.metrics import confusion_matrix
def model_accuracy(pred_class,obs_class):
    matrix = []
    for i in range(0,obs_class.shape[1]):
        if len(pred_class.shape)>1:
            matrix.append(confusion_matrix(obs_class[:,i], pred_class[:,i], labels=[1,2,3,4]))
        else:
            matrix.append(confusion_matrix(obs_class[:,i], pred_class, labels=[1,2,3,4]))
    
    matrix_=pd.DataFrame(np.concatenate(matrix))
    matrix_.index=pd.concat([pd.DataFrame(['level1','level2','level3','level4']) for i in range(6)]).iloc[:,0]
    matrix_.columns=['level1','level2','level3','level4']
    return matrix_
    
spatiotemporal_train_performance=model_accuracy(spatiotemporal_HI_train, HI_obs_train)
spatiotemporal_test_performance=model_accuracy(spatiotemporal_HI_test, HI_obs_test)
simulate_train_performance=model_accuracy(simulate_HI_train, HI_obs_train)
simulate_test_performance=model_accuracy(simulate_HI_test, HI_obs_test)

#%% save as excel
HI_train=pd.concat([train_station_info,pd.DataFrame(HI_obs_train),pd.DataFrame(spatiotemporal_HI_train)],axis=1)
HI_test=pd.concat([test_station_info,pd.DataFrame(HI_obs_test),pd.DataFrame(spatiotemporal_HI_test)],axis=1)
HI_train.columns= spatiotemporal_train_result.columns; HI_test.columns= spatiotemporal_train_result.columns; 

writer = pd.ExcelWriter(r'result\performance_spatiotemporal(class).xlsx',engine='xlsxwriter')
spatiotemporal_train_performance.to_excel(writer,sheet_name="train_performance")
spatiotemporal_test_performance.to_excel(writer,sheet_name="test_performance")
HI_train.to_excel(writer,sheet_name="train")
HI_test.to_excel(writer,sheet_name="test")
writer.close()

#%%
HI_train=pd.concat([train_station_info,pd.DataFrame(HI_obs_train),pd.DataFrame(simulate_HI_train).iloc[:,-1]],axis=1)
HI_test=pd.concat([test_station_info,pd.DataFrame(HI_obs_test),pd.DataFrame(simulate_HI_test).iloc[:,-1]],axis=1)
HI_train.columns= simulate_train.columns; HI_test.columns= simulate_test.columns; 
writer = pd.ExcelWriter(r'result\performance_simulate(class).xlsx',engine='xlsxwriter')
simulate_train_performance.to_excel(writer,sheet_name="train_performance")
simulate_test_performance.to_excel(writer,sheet_name="test_performance")
HI_train.to_excel(writer,sheet_name="train")

HI_test.to_excel(writer,sheet_name="test")
writer.close()