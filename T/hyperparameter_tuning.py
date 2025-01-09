# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:57:22 2024

@author: Steve
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

os.chdir(os.path.dirname(__file__))
os.chdir("..")

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
obs_input = np.stack([obs_input1, obs_input2, obs_input3, obs_input4, obs_input5],axis=2)

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
from tensorflow.keras.layers import TimeDistributed, Reshape, Flatten, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from error_indicator import ErrorIndicator

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

"""
建構模型之函數
"""

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    
    pos_encoding = positional_encoding(inputs.shape[1], d_model=inputs.shape[2])
    # Normalization and Attention  
    x = inputs + pos_encoding[np.newaxis,:,:] 
    # x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = x + inputs

    # Feed Forward Part
    # x = LayerNormalization(epsilon=1e-3)(res)
    x = LayerNormalization()(res)
    x = TimeDistributed(Dense(units=ff_dim, activation='relu'))(x)
    return x + res

def transformer_decoder(inputs2,encoder_output, head_size, num_heads, ff_dim, dropout=0):
    
    pos_encoding = positional_encoding(inputs2.shape[1], d_model=inputs2.shape[2])
    x = inputs2 + pos_encoding[np.newaxis,:,:]
    # Normalization and Attention
    # x = LayerNormalization(epsilon=1e-3)(inputs2)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, encoder_output)
    # x = LayerNormalization(epsilon=1e-3)(x)
    x = LayerNormalization()(x)
    res = x + inputs2

    # Feed Forward Part
    x = LayerNormalization()(res)
    x = TimeDistributed(Dense(units=ff_dim, activation='relu'))(x)
    return x + res

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

def model_evaluation(pred, obs):
    r2=[[]*1 for i in range(obs.shape[1])];rmse=[[]*1 for i in range(obs.shape[1])];rae=[[]*1 for i in range(obs.shape[1])]
    for i in range(0,obs.shape[1]):
        rmse[i].append(RMSE(pred[:,i], obs[:,i]))
        r2[i].append(R2(pred[:,i],obs[:,i]))
        rae[i].append(RAE(pred[:,i],obs[:,i]))
        
    rmse=np.squeeze(np.array(rmse));
    r2=np.squeeze(np.array(r2));
    rae=np.squeeze(np.array(rae));
    
    performance= pd.DataFrame(np.array([rmse,r2,rae])).T
    performance.index=['t+%s'%i for i in range(1,7)]
    performance.columns=['rmse','r2','rae']
    
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

#%% normalization 
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

#%% define the model and objective
import warnings
warnings.filterwarnings("ignore")

#%% keras tuner
num_features1,num_features2 = x_heterogeneous_train.shape[2], x_homogeneous_train.shape[2]

from kerastuner.tuners import RandomSearch

def create_model(hp):

    # 清理權重參數記憶體
    K.clear_session()       
    inputs = Input(shape=(84,))

    inputs1 = Lambda(lambda x: tf.expand_dims(x[:, :70],-1))(inputs)
    inputs2 = Lambda(lambda x: tf.expand_dims(x[:, 70:],-1))(inputs)
    
    inputs1 = Reshape((7,10))(inputs1)
    inputs2 = Reshape((7,2))(inputs2)

    encoder_output = transformer_encoder(inputs1, head_size=num_features1, num_heads=time_steps1, ff_dim=num_features1, dropout=0)
    encoder_output = Dense(num_features2)(encoder_output)
    decoder_output = transformer_decoder(inputs2, encoder_output=encoder_output, head_size=num_features1+num_features2, num_heads=time_steps2, ff_dim=num_features2, dropout=0)

    
    lstm_hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=3, step=1)
    for _ in range(lstm_hidden_layers):  # Add specified number of hidden layers
        decoder_output = LSTM(num_features1+num_features2, return_sequences=True)(decoder_output)
    
    output = Flatten()(decoder_output)
    output = Dense(units=y_output_train.shape[1])(output)

    # define model
    output = Flatten()(decoder_output)
    output = Dense(units=y_output_train.shape[1])(output)
    model = Model(inputs=inputs , outputs=output)
    # define other parameters
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    # define other parameters
    adam = Adam(learning_rate = learning_rate)
    model.compile(optimizer=adam,loss="mse")
    model.summary()    

    return model

# Instantiate the RandomSearch tuner
tuner = RandomSearch(create_model,
                     objective='val_loss',
                     max_trials=5,
                     executions_per_trial=3,
                     directory='my_dir',
                     project_name='my_project')


# Reshape x_heterogeneous_train to (44431, 28)
x_heterogeneous_train_reshaped = np.reshape(x_heterogeneous_train, (x_heterogeneous_train.shape[0], -1))
# Reshape x_homogeneous_train to (44431, 30)
x_homogeneous_train_reshaped = np.reshape(x_homogeneous_train, (x_homogeneous_train.shape[0], -1))
# Concatenate along axis 1
x_train_ = np.concatenate([x_heterogeneous_train_reshaped, x_homogeneous_train_reshaped], axis=1)

# Perform the hyperparameter search
tuner.search(x_train_,  y_output_train,
             epochs=10,
             validation_split=val_ratio/(train_ratio+val_ratio))

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:", best_hyperparameters.values)
print("Best Model Summary:", best_model.summary())

#%%
os.chdir(os.path.dirname(__file__))
# Get the results of the tuning trials
parameter = [tuner.get_best_hyperparameters(num_trials=5)[i].values for i in range(5)]
best_trials = tuner.oracle.get_best_trials(num_trials=5)
result = [trial.score for trial in best_trials]
tuner_results = [pd.DataFrame(parameter),pd.DataFrame(result)]
results_df = pd.concat(tuner_results, axis=1)
results_df.columns=['hidden_layers', 'learning_rate','mse']
results_df.index=[i for i in range(results_df.shape[0])]

# Save the results to a CSV file
results_df.to_csv(r'result\tuner_results\tuner_results.csv', index=False)


#%% Visualize the results (example: using matplotlib)
import matplotlib.pyplot as plt

# Create a figure object with specified DPI
plt.figure(dpi=300)
# Plot the loss over trials
plt.bar(results_df.index, results_df['mse'], align='center', alpha=0.5)
plt.xlabel('Trial ID')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error over Trials')

plt.grid(True)
plt.savefig(r'result\tuner_results.jpg')
plt.show()



