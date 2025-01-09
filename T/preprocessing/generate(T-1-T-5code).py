# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:00:39 2024

@author: Steve
"""

import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
os.chdir("../..")

#%%
import cv2

obs_path1="image_dataset\\T\\obs(T-1)"
obs_path2="image_dataset\\T\\obs(T-2)"
obs_path3="image_dataset\\T\\obs(T-3)"
obs_path4="image_dataset\\T\\obs(T-4)"
obs_path5="image_dataset\\T\\obs(T-5)"
obs_path_dir=os.listdir(obs_path1)[:-1]

obs_T1=np.array([cv2.resize(cv2.cvtColor(cv2.imread(obs_path1+'\\'+image_path), cv2.COLOR_BGR2GRAY), (180, 180)) for image_path in obs_path_dir])
obs_T2=np.array([cv2.resize(cv2.cvtColor(cv2.imread(obs_path2+'\\'+image_path), cv2.COLOR_BGR2GRAY), (180, 180)) for image_path in obs_path_dir])
obs_T3=np.array([cv2.resize(cv2.cvtColor(cv2.imread(obs_path3+'\\'+image_path), cv2.COLOR_BGR2GRAY), (180, 180)) for image_path in obs_path_dir])
obs_T4=np.array([cv2.resize(cv2.cvtColor(cv2.imread(obs_path4+'\\'+image_path), cv2.COLOR_BGR2GRAY), (180, 180)) for image_path in obs_path_dir])
obs_T5=np.array([cv2.resize(cv2.cvtColor(cv2.imread(obs_path5+'\\'+image_path), cv2.COLOR_BGR2GRAY), (180, 180)) for image_path in obs_path_dir])

#%% 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

save_path=r"T\model\autoencoder.keras"
K.clear_session()
model = load_model(save_path)
get_code1 = Model(inputs=model.layers[1].input, outputs=model.layers[7].output)

# Get the output of the desired layer using the predict method
obs_T1_code = get_code1.predict(obs_T1)
obs_T2_code = get_code1.predict(obs_T2)
obs_T3_code = get_code1.predict(obs_T3)
obs_T4_code = get_code1.predict(obs_T4)
obs_T5_code = get_code1.predict(obs_T5)

#%%
writer = pd.ExcelWriter(r"code_dataset\T\encoded_value(T-1-T-5).xlsx",engine='xlsxwriter')
pd.DataFrame(obs_T1_code).to_excel(writer,sheet_name="obs_T1_code")
pd.DataFrame(obs_T2_code).to_excel(writer,sheet_name="obs_T2_code")
pd.DataFrame(obs_T3_code).to_excel(writer,sheet_name="obs_T3_code")
pd.DataFrame(obs_T4_code).to_excel(writer,sheet_name="obs_T4_code")
pd.DataFrame(obs_T5_code).to_excel(writer,sheet_name="obs_T5_code")
writer.close()
