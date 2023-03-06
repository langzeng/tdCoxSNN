import os, sys
dirc_server='directory to tdCoxSNN'
os.chdir(dirc_server)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.backend import clear_session

# import local modules
from Tensorflow.loss_tdCoxSNN_Tensorflow import loss_tdCoxSNN_Tensorflow
from funcs_util.funcs import baseline_hazard, survprob

tf.keras.backend.set_floatx('float32')

# Load Data
pbc2_train_test = pd.read_pickle("Data/Python/train_test_idxs_pbc2.pkl")
pbc2 = pd.read_pickle("Data/Python/pbc2long.pkl")
# tstart, tstop, tstop_final: time in days
# xt: time-dependent variables
# x: baseline variables

index_train = np.array([x in pbc2_train_test[0] for x in pbc2.id])
index_test = np.array([x in pbc2_train_test[1] for x in pbc2.id])

# Landmarking
# if landmarkmonth = 0, all subjects will be used to train the model
landmarkmonth = 3
# Select subjects survived beyond landmarkmonth
index_landmark_id = np.array([((x/30)>landmarkmonth) for x in pbc2.tstop_final])
# Select the visits after landmark time
index_landmark_visit = np.array([((x/30)>landmarkmonth) for x in pbc2.tstart])

pbc2_train = pbc2[index_train & index_landmark_id].copy()
# For each subject in test dataset, use their first visit after landmark time to predict
# The rest visits are treated as truth for validation
pbc2_test = pbc2[index_test & index_landmark_id & index_landmark_visit].copy().groupby('id').first().reset_index()

feature = ['xt'+str(y+1) for y in range(20)]+['x'+str(y+1) for y in range(7)]
feature_tobe_scaled = ['xt'+str(y+1) for y in range(7)]+['x7']
feature_tobe_scaled_index = np.array([feature.index(f) for f in feature_tobe_scaled]).reshape(-1)

# normalization of train input
scalerx = StandardScaler()  # To standardize the inputs

# predictors
x_train = pd.DataFrame.to_numpy(pbc2_train[feature])
x_train[:,feature_tobe_scaled_index] = scalerx.fit_transform(x_train[:,feature_tobe_scaled_index])

x_test = pd.DataFrame.to_numpy(pbc2_test[feature])
x_test[:,feature_tobe_scaled_index] = scalerx.transform(x_test[:,feature_tobe_scaled_index])

# survival outcome
y_train = pbc2_train[['tstart','tstop','event']].values
y_test = pbc2_test[['tstart','tstop','event']].values

### set up DNN parameters ###
num_nodes = 30             # number of nodes per hidden layer
num_lr = 0.01              # learning rate
num_dr = 0.2               # dropout rate
num_epoch = 20            # number of epoches for optimization
batch_size = 50            # number of batch size for optimization

clear_session()

demo = layers.Input(shape=(x_train.shape[1],), name='demo_input')
layer1 = layers.Dense(num_nodes, activation='selu')(demo)
layer1 = layers.BatchNormalization()(layer1)
layer1 = layers.Dropout(num_dr)(layer1)
target = layers.Dense(1, activation='linear')(layer1)
model = models.Model(inputs=demo, outputs=target)
optimizers.Adam(learning_rate=num_lr)
model.compile(optimizer='Adam', loss=loss_tdCoxSNN_Tensorflow, metrics=None)

model.fit(x_train, y_train, epochs=num_epoch, batch_size = batch_size,verbose=1)
print("Training Complete")

rs_train = model.predict(x_train)
rs_test = model.predict(x_test)

# calculate baseline hazard function
base_haz = baseline_hazard(np.column_stack((y_train,rs_train)))
# prepare the test dataset
test_rs = np.column_stack((pbc2_test[['id','tstart']],rs_test)) # id time predicted_risk_score

# calculate the survival probability at (time_of_interest+last_obs_time) for each subject
S = survprob(time_of_interest = [1,30,60,180,365], # in days
             haz = base_haz, 
             test_risk_score = test_rs)
