# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:46:19 2025

@author: Steve
"""
import numpy as np

class ErrorIndicator:
    
    @staticmethod
    def RMSE(prediction, target):
        rmse = np.sqrt(((prediction - target) ** 2).mean())
        return rmse
    
    @staticmethod
    def R2(prediction, target):
        avg_pred = np.mean(prediction, axis=0)
        avg_target = np.mean(target, axis=0)
        
        numerator = np.sum((target - avg_target) * (prediction - avg_pred))
        denominator = (np.sum((target - avg_target) ** 2) * np.sum((prediction - avg_pred) ** 2)) ** 0.5
        r2 = (numerator / denominator) ** 2
        return r2
    
    @staticmethod
    def RAE(predicted, actual):
        actual = np.array(actual)
        predicted = np.array(predicted)
    
        mean_actual = np.mean(actual)
    
        absolute_errors = np.abs(actual - predicted)
        absolute_errors_sum = np.sum(absolute_errors)
    
        mean_absolute_deviation = np.sum(np.abs(actual - mean_actual))
    
        rae = absolute_errors_sum / mean_absolute_deviation
    
        return rae
    
    @staticmethod
    def MAE(prediction,target):
        rmse = np.mean((np.abs(prediction-target)))
        return rmse
    
    @staticmethod
    def np_mae(output,output_pred):
        mae=0
        try:
            output=np.asarray(output); output_pred=np.asarray(output_pred)
            for i in range(0,len(output)):
                mae=mae+np.abs((output[i]-output_pred[i]))
            mae=mae/len(output)
        except:
            mae=np.abs(output-output_pred)
    
        return mae