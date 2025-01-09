# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:22:55 2024

@author: Steve
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(__file__))
os.chdir("../..")
#%%
obs_path=r"image_dataset/RH/obs(T-0)"
obs_path_dir=os.listdir(obs_path)[:-1]

simulate_path=r"image_dataset/RH/osimulate"
simulate_path_dir=os.listdir(simulate_path)[:-1]

#%% read image

import cv2

obs_rh=[cv2.resize(cv2.cvtColor(cv2.imread(obs_path+'\\'+image_path), cv2.COLOR_BGR2GRAY), (180, 180)) for image_path in obs_path_dir]
simulate_rh=[cv2.resize(cv2.cvtColor(cv2.imread(simulate_path+'\\'+image_path), cv2.COLOR_BGR2GRAY), (180, 180)) for image_path in simulate_path_dir]

#%% 切割訓練、驗證、測試資料
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

train_index=np.array([i for i in range(int(len(obs_rh)*(train_ratio+val_ratio)))])
test_index=np.array([i for i in range(int(len(obs_rh)*(train_ratio+val_ratio)),len(obs_rh))])

train_data_1=[obs_rh[index] for index in train_index]
train_data_2=[simulate_rh[index] for index in train_index]
train_data=np.concatenate([train_data_1,train_data_2])

test_data_1=[obs_rh[index] for index in test_index]
test_data_2=[simulate_rh[index] for index in test_index]
test_data=np.concatenate([test_data_1,test_data_2])


#%% define the model and objective

import tensorflow as tf

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print('GPU is available and will be used.')
else:
    print('GPU is not available. TensorFlow will use CPU.')
    
    
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Lambda,MaxPooling2D,UpSampling2D,Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

""" Define DNN model """
            
def AE_model(input_shape):
    
    inputs = Input(shape=(input_shape[1],input_shape[2]))
    encoder = Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    encoder = Conv2D(filters=16, kernel_size=3, activation='relu', padding="same")(encoder)
    encoder = MaxPooling2D((2, 2), padding="same")(encoder)
    encoder = Conv2D(filters=32, kernel_size=3, activation='relu', padding="same")(encoder)
    encoder = MaxPooling2D((2, 2), padding="same")(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(10)(encoder)

    
    decoder = Dense(units=45*45*32, activation='relu')(encoder)
    decoder = Reshape(target_shape=(45, 45, 32))(decoder)
    decoder= Conv2D(filters=32, kernel_size=3, activation='relu', padding="same")(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder= Conv2D(filters=16, kernel_size=3, activation='relu', padding="same")(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2D(1, (2, 2), padding='same')(decoder)
    model = Model(inputs=inputs, outputs=decoder)

    print(model.summary())
    return model 


#%% change to correct wd
os.chdir(os.path.dirname(__file__))
os.chdir("..")

#%% model construction
model=AE_model(train_data.shape)
learning_rate=1e-3
adam = Adam(learning_rate=learning_rate)
model.compile(optimizer=adam,loss='mse')
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=0)        
save_path=r"model\autoencoder.keras"
checkpoint =ModelCheckpoint(save_path,save_best_only=True)
callback_list=[earlystopper,checkpoint]        
model.fit(train_data, train_data, epochs=50, batch_size=8,validation_split=0.1,callbacks=callback_list, shuffle= True)

#%% model prediction
save_path=r"model\autoencoder.keras"
K.clear_session()
model = load_model(save_path)
batch_size=8
train_y = model.predict(train_data,batch_size=batch_size)
test_y = model.predict(test_data,batch_size=batch_size)

#%% get model code 
# Define a new model that outputs the desired layer's output
get_code = Model(inputs=model.input, outputs=model.layers[7].output)

# Get the output of the desired layer using the predict method
train_code_y = get_code.predict(train_data)
test_code_y = get_code.predict(test_data)

#%% save encoded data to dataframe (see code dataset)
train_code_y = pd.DataFrame(train_code_y)
test_code_y = pd.DataFrame(test_code_y)

writer = pd.ExcelWriter(r"code_dataset\RH\encoded_value.xlsx",engine='xlsxwriter')
train_code_y.iloc[train_index,:].reset_index(drop=True).to_excel(writer,sheet_name="train_code_y(obs)")
train_code_y.iloc[len(train_index):,:].reset_index(drop=True).to_excel(writer,sheet_name="train_code_y(simulate)")
test_code_y.iloc[:len(test_index),:].reset_index(drop=True).to_excel(writer,sheet_name="test_code_y(obs)")
test_code_y.iloc[len(test_index):,:].reset_index(drop=True).to_excel(writer,sheet_name="test_code_y(simulate)")
# Save the Excel file
writer.close()

#%%
"""
The code below is to generate the autoencoder prediction, run if only needed (need to spent a lot of time)
"""

#%% save figures 
for i in range(len(train_y)):
    if i <len(train_index):
        file_path="image_result\ae_predict\obs\%s"%obs_path_dir[i]
        # Save the image
        cv2.imwrite(file_path, train_y[i])
        
        file_path2="image_result\\ae_predict\\real data\\obs_%s"%obs_path_dir[i]
        cv2.imwrite(file_path2, train_data[i])      
        
    else:
        index=i-len(train_index)
        file_path="image_result\ae_predict\simulate\%s"%simulate_path_dir[index]
        # Save the image
        cv2.imwrite(file_path, train_y[i]) 
        
        file_path2="image_result\\ae_predict\\real data\\simulate_%s"%obs_path_dir[index]
        cv2.imwrite(file_path2, train_data[i])
    
for i in range(len(test_y)):
    if i <len(test_index):
        index=i+len(train_index)
        file_path="image_result\ae_predict\obs\%s"%obs_path_dir[index]
        # Save the image
        cv2.imwrite(file_path, test_y[i])
        
        file_path2="image_result\\ae_predict\\real data\\obs_%s"%obs_path_dir[index]
        cv2.imwrite(file_path2, test_data[i])
    else:
        index=i-len(test_index)+len(train_index)
        file_path="image_result\ae_predict\simulate\%s"%simulate_path_dir[index]
        # Save the image
        cv2.imwrite(file_path, test_y[i]) 
        
        file_path2="image_result\\ae_predict\\real data\\simulate_%s"%obs_path_dir[index]
        cv2.imwrite(file_path2, test_data[i])
        
#%% output reshape
train_y_flatten=train_y.reshape((train_y.shape[0]*train_y.shape[1]*train_y.shape[2]))
train_data_flatten=train_data.reshape((train_data.shape[0]*train_data.shape[1]*train_data.shape[2]))

#testing#
test_y_flatten=test_y .reshape((test_y .shape[0]*test_y .shape[1]*test_y .shape[2]))
test_data_flatten=test_data.reshape((test_data.shape[0]*test_data.shape[1]*test_data.shape[2]))

#%% model performance evaluation 
os.chdir("..")
from error_indicator import ErrorIndicator

train_ae_R2=ErrorIndicator.R2(train_y_flatten, train_data_flatten)
train_ae_rmse=ErrorIndicator.RMSE(train_y_flatten, train_data_flatten)
train_ae_RAE=ErrorIndicator.RAE(train_y_flatten, train_data_flatten)

test_ae_R2=ErrorIndicator.R2(test_y_flatten, test_data_flatten)
test_ae_rmse=ErrorIndicator.RMSE(test_y_flatten, test_data_flatten)
test_ae_RAE=ErrorIndicator.RAE(test_y_flatten, test_data_flatten)

train=pd.DataFrame([train_ae_R2,train_ae_rmse,train_ae_RAE])
train.index = ['train_ae_R2','train_rmse','train_RAE']
test=pd.DataFrame([test_ae_R2,test_ae_rmse,test_ae_RAE])
test.index = ['test_ae_R2','test_rmse','test_RAE']
ae_performance = pd.concat([train,test],axis=0)

#%%
writer = pd.ExcelWriter(r"RH\result\autoencoder(performance).xlsx",engine='xlsxwriter')
ae_performance.to_excel(writer,sheet_name="ae_performance")
train_code_y.iloc[train_index,:].reset_index(drop=True).to_excel(writer,sheet_name="train_code_y(obs)")
train_code_y.iloc[len(train_index):,:].reset_index(drop=True).to_excel(writer,sheet_name="train_code_y(simulate)")
test_code_y.iloc[:len(test_index),:].reset_index(drop=True).to_excel(writer,sheet_name="test_code_y(obs)")
test_code_y.iloc[len(test_index):,:].reset_index(drop=True).to_excel(writer,sheet_name="test_code_y(simulate)")
# Save the Excel file
writer.close()
