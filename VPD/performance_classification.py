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
os.chdir(os.path.dirname(__file__))

#%%
spatiotemporal_train_result=pd.read_excel("result\performance_spatiotemporal(VPD).xlsx",sheet_name="train").iloc[:,1:16]
spatiotemporal_test_result=pd.read_excel("result\performance_spatiotemporal(VPD).xlsx",sheet_name="test").iloc[:,1:16]
spatiotemporal_vpd_train = spatiotemporal_train_result.iloc[:,3:9]
vpd_obs_train = spatiotemporal_train_result.iloc[:,9:]
spatiotemporal_vpd_test = spatiotemporal_test_result.iloc[:,3:9]
vpd_obs_test = spatiotemporal_test_result.iloc[:,9:]

simulate_train=pd.read_excel(r'result\performance_simulate(VPD).xlsx',sheet_name='train',index_col=0)
simulate_test=pd.read_excel(r'result\performance_simulate(VPD).xlsx',sheet_name='test',index_col=0)

simulate_vpd_train=simulate_train.iloc[:,-1]
simulate_vpd_test=simulate_test.iloc[:,-1]

train_station_info = spatiotemporal_train_result.iloc[:,:3]
test_station_info = spatiotemporal_test_result.iloc[:,:3]

#%%
def classify_data(data, small_threshold, huge_threshold):
    # Define thresholds
    huge_threshold = huge_threshold
    small_threshold = small_threshold
    # Classify the data
    classes = np.zeros_like(data, dtype=str)
    classes[data > huge_threshold] = 3
    classes[(data <= huge_threshold) & (data > small_threshold)] = 2
    classes[data <= small_threshold] = 1
    classes = classes.astype(float)
    return classes

#%%
small_threshold, huge_threshold = 0.8, 0.95
spatiotemporal_vpd_train=classify_data(spatiotemporal_vpd_train, small_threshold, huge_threshold)
spatiotemporal_vpd_test=classify_data(spatiotemporal_vpd_test, small_threshold, huge_threshold)
vpd_obs_train=classify_data(vpd_obs_train, small_threshold, huge_threshold)
vpd_obs_test=classify_data(vpd_obs_test, small_threshold, huge_threshold)
simulate_vpd_train=classify_data(simulate_vpd_train, small_threshold, huge_threshold)
simulate_vpd_test=classify_data(simulate_vpd_test, small_threshold, huge_threshold)

#%% model performance evaluation
from sklearn.metrics import confusion_matrix
def model_accuracy(pred_class,obs_class):
    matrix = []
    for i in range(0,obs_class.shape[1]):
        if len(pred_class.shape)>1:
            matrix.append(confusion_matrix(obs_class[:,i], pred_class[:,i], labels=[1,2,3]))
        else:
            matrix.append(confusion_matrix(obs_class[:,i], pred_class, labels=[1,2,3]))
    
    matrix_=pd.DataFrame(np.concatenate(matrix))
    matrix_.index=pd.concat([pd.DataFrame(['level1','level2','level3']) for i in range(6)]).iloc[:,0]
    matrix_.columns=['level1','level2','level3']
    return matrix_
    
spatiotemporal_train_performance=model_accuracy(spatiotemporal_vpd_train, vpd_obs_train)
spatiotemporal_test_performance=model_accuracy(spatiotemporal_vpd_test, vpd_obs_test)
simulate_train_performance=model_accuracy(simulate_vpd_train, vpd_obs_train)
simulate_test_performance=model_accuracy(simulate_vpd_test, vpd_obs_test)

#%%

vpd_train=pd.concat([train_station_info,pd.DataFrame(vpd_obs_train),pd.DataFrame(spatiotemporal_vpd_train)],axis=1)
vpd_test=pd.concat([test_station_info,pd.DataFrame(vpd_obs_test),pd.DataFrame(spatiotemporal_vpd_test)],axis=1)
vpd_train.columns= spatiotemporal_train_result.columns; vpd_test.columns= spatiotemporal_train_result.columns; 

writer = pd.ExcelWriter(r'result\performance_spatiotemporal(class).xlsx',engine='xlsxwriter')
spatiotemporal_train_performance.to_excel(writer,sheet_name="train_performance")
spatiotemporal_test_performance.to_excel(writer,sheet_name="test_performance")
vpd_train.to_excel(writer,sheet_name="train")
vpd_test.to_excel(writer,sheet_name="test")
writer.close()

#%%
vpd_train=pd.concat([train_station_info,pd.DataFrame(vpd_obs_train),pd.DataFrame(simulate_vpd_train).iloc[:,-1]],axis=1)
vpd_test=pd.concat([test_station_info,pd.DataFrame(vpd_obs_test),pd.DataFrame(simulate_vpd_test).iloc[:,-1]],axis=1)
vpd_train.columns= simulate_train.columns; vpd_test.columns= simulate_test.columns; 

writer = pd.ExcelWriter(r'result\performance_simulate(class).xlsx',engine='xlsxwriter')
simulate_train_performance.to_excel(writer,sheet_name="train_performance")
simulate_test_performance.to_excel(writer,sheet_name="test_performance")
vpd_train.to_excel(writer,sheet_name="train")

vpd_test.to_excel(writer,sheet_name="test")
writer.close()