# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:57:22 2024

@author: Steve
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os

os.chdir(os.path.dirname(__file__))
os.chdir("..")
paths=os.listdir(r"dataset")

#%% read data
weather_same_index = pd.read_excel(r"dataset\same_index.xlsx",sheet_name="weather_index",index_col=0)

# temperature
factor=["T","RH"]

def read_data(factor,same_index,get_info=0):
    file_name ="%s.xlsx"%factor
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
    file_name ="%s.xlsx"%factor
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

homogeneous_input= np.stack([obs_input1, obs_input2],axis=2)
homogeneous_input= np.concatenate([np.expand_dims(simulate_data[:,0,:],axis=1),homogeneous_input],axis=1)

heterogeneous_input = np.stack([obs_input3, obs_input4, obs_input5],axis=2)

#%% define all function
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, Dense, LSTM, Input, LayerNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from error_indicator import ErrorIndicator

def normalize_3d(n_array,array):
    # Calculate the max and mean along the third dimension
    f_array= array.reshape((array.shape[0]*array.shape[1],array.shape[2]))
    min_ = np.min(f_array, axis=0, keepdims=True)
    max_ = np.max(f_array, axis=0, keepdims=True)
        
    # Normalize the array along the third dimension
    normalized_array = (n_array - min_) / (max_-min_)
    return normalized_array

# Define the positional encoding function
def positional_encoding(max_len, d_model):
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    encoding = np.zeros((max_len, d_model))
    if d_model % 2 == 0:
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        encoding[:, 0::2] = np.sin(position / div_term)
        encoding[:, 1::2] = np.cos(position / div_term)
    else:
        div_term = np.exp(np.arange(d_model) * -(np.log(10000.0) / d_model))
        encoding = np.sin(position / div_term)
    return encoding

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    pos_encoding = positional_encoding(inputs.shape[1], d_model=inputs.shape[2])
    # Normalization and Attention  
    x = inputs + pos_encoding[np.newaxis,:,:] 
    # x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization()(res)
    x = TimeDistributed(Dense(units=ff_dim, activation='relu'))(x)
    return x + res

def transformer_decoder(inputs2,encoder_output, head_size, num_heads, ff_dim, dropout=0):
    
    pos_encoding = positional_encoding(inputs2.shape[1], d_model=inputs2.shape[2])
    x = inputs2 + pos_encoding[np.newaxis,:,:]
    # Normalization and Attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, encoder_output)
    x = LayerNormalization()(x)
    res = x + inputs2

    # Feed Forward Part
    x = LayerNormalization()(res)
    x = TimeDistributed(Dense(units=ff_dim, activation='relu'))(x)
    return x + res

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

#%% read code dataset
train_code1=pd.read_excel("code_dataset/T/encoded_value.xlsx",sheet_name="train_code_y(obs)",index_col=0)
train_code2=pd.read_excel("code_dataset/T/encoded_value.xlsx",sheet_name="train_code_y(simulate)",index_col=0)
test_code1=pd.read_excel("code_dataset/T/encoded_value.xlsx",sheet_name="test_code_y(obs)",index_col=0)
test_code2=pd.read_excel("code_dataset/T/encoded_value.xlsx",sheet_name="test_code_y(simulate)",index_col=0)
code_1=np.array(pd.concat([train_code1,test_code1])); code_2=np.array(pd.concat([train_code2,test_code2]))
code_3=np.array(pd.read_excel("code_dataset/T/encoded_value(T-1-T-5).xlsx",sheet_name="obs_T1_code",index_col=0))
code_4=np.array(pd.read_excel("code_dataset/T/encoded_value(T-1-T-5).xlsx",sheet_name="obs_T2_code",index_col=0))
code_5=np.array(pd.read_excel("code_dataset/T/encoded_value(T-1-T-5).xlsx",sheet_name="obs_T3_code",index_col=0))
code_6=np.array(pd.read_excel("code_dataset/T/encoded_value(T-1-T-5).xlsx",sheet_name="obs_T4_code",index_col=0))
code_7=np.array(pd.read_excel("code_dataset/T/encoded_value(T-1-T-5).xlsx",sheet_name="obs_T5_code",index_col=0))

code=np.stack([code_7,code_6,code_5,code_4,code_3,code_2,code_1],axis=1)
unique_date_index=pd.read_csv("dataset/unique_date_index.csv",index_col=0)
all_code=np.array([code[unique_date_index.iloc[index,0],:,:] for index in range(unique_date_index.shape[0])])

#%% define x and y 
forecast_index=0 # 0 for temperature, 1 for relative humidity
x_heterogeneous = all_code # we did not use heterogeneous input in this model
x_homogeneous = homogeneous_input
obs_output=[obs_output1, obs_output2]
y_output = np.array(obs_output[forecast_index]) #obs_output1

#%% split into training, validation and testing
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

#%% 
x_homogeneous_train = x_homogeneous[train_index,:,:]
x_heterogeneous_train = x_heterogeneous[train_index,:,:]
y_output_train = y_output[train_index,:]
train_date_info=np.array(date_info)[train_index]

x_homogeneous_test = x_homogeneous[test_index,:,:]
x_heterogeneous_test = x_heterogeneous[test_index,:,:]
y_output_test = y_output[test_index,:]
test_date_info=np.array(date_info)[test_index]

#%% data concatenate
x_train=[x_heterogeneous_train, x_homogeneous_train]
x_test=[x_heterogeneous_test, x_homogeneous_test]

time_steps1, num_features1 = x_heterogeneous_train.shape[1], x_heterogeneous_train.shape[2]
time_steps2, num_features2 = x_homogeneous_train.shape[1], x_homogeneous_train.shape[2]

#%% model construction
model_name = 'spatiotemporal_transformer-lstm(%s)'%factor[forecast_index]
learning_rate = 0.01
patience = 20
epochs = 50
batch_size = 64
lstm_hidden_layer = 3 # see tuner result

K.clear_session()       
# set input shapes
time_steps1, num_features1 = x_heterogeneous_train.shape[1], x_heterogeneous_train.shape[2]
time_steps2, num_features2 = x_homogeneous_train.shape[1], x_homogeneous_train.shape[2]
# set input placeholders
inputs1 = Input(shape=(time_steps1,num_features1))
inputs2 = Input(shape=(time_steps2,num_features2))
# define layers
encoder_output = transformer_encoder(inputs1, head_size=num_features1, num_heads=time_steps1, ff_dim=num_features1, dropout=0)
encoder_output = Dense(num_features2)(encoder_output)
decoder_output = transformer_decoder(inputs2, encoder_output=encoder_output, head_size=num_features1+num_features2, num_heads=time_steps2, ff_dim=num_features2, dropout=0)
for _ in range(lstm_hidden_layer):  # Add specified number of hidden layers
    decoder_output = LSTM(num_features1+num_features2, return_sequences=True)(decoder_output)
# define model
output = Flatten()(decoder_output)
output = Dense(units=y_output_train.shape[1])(output)
model = Model(inputs=[inputs1, inputs2] , outputs=output)
# define other parameters
adam = Adam(learning_rate = learning_rate)
model.compile(optimizer=adam,loss="mse")
earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=0) 
model.summary()    

# 訓練模式
checkpoint = ModelCheckpoint("T/model/"+ model_name+ ".keras", save_best_only = True)
callback_list = [earlystopper,checkpoint]     
# callback_list = [earlystopper,checkpoint]        
model.fit(x_train, y_output_train, epochs=epochs, batch_size=batch_size, validation_split=val_ratio/(train_ratio+val_ratio),callbacks=callback_list,shuffle=True)

#%% model prediction
model_name = 'spatiotemporal_transformer-lstm(%s)'%factor[forecast_index]
batch_size = 64
model = load_model("T/model/"+ model_name+ ".keras")
train_y = model.predict(x_train,batch_size=batch_size)
test_y = model.predict(x_test,batch_size=batch_size)

#%% model performance evaluation
train_performance=model_evaluation(train_y, y_output_train)
test_performance=model_evaluation(test_y, y_output_test)

#%% save forecasts and others result
# save model performance and forecast into excel
writer = pd.ExcelWriter(r'T\result\performance_%s.xlsx'%(model_name.split(".")[0]),engine='xlsxwriter')
#save performance
train_performance.to_excel(writer,sheet_name="train_performance")
test_performance.to_excel(writer,sheet_name="test_performance")
#save forecast
columns_name=np.concatenate([['region','postcode','date'],['obs_T+%s'%i for i in range(1,7)],['pred_T+%s'%i for i in range(1,7)]])
train=pd.DataFrame(np.concatenate([np.array(train_date_info), y_output_train,train_y],axis=1))
train.columns=columns_name
train.to_excel(writer,sheet_name="train")
test=pd.DataFrame(np.concatenate([np.array(test_date_info), y_output_test, test_y],axis=1))
test.columns=columns_name
test.to_excel(writer,sheet_name="test")
writer.close() 


