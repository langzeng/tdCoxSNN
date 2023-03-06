import os
dirc_server = 'directory to tdCoxSNN'
os.chdir(dirc_server)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PyTorch.loss_tdCoxSNN_PyTorch import loss_tdCoxSNN_PyTorch
from funcs_util.funcs import baseline_hazard, survprob

# Load Data
pbc2_train_test = pd.read_pickle("Data/Python/train_test_idxs_pbc2.pkl")
pbc2 = pd.read_pickle("/ihome/yding/laz52/AMDprediction/Github/Data/Python/pbc2long.pkl")
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
# For each subject in test dataset, use the information of their first visit after landmark time to predict
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
torch.set_default_dtype(torch.float32)
num_nodes = 30             # number of nodes per hidden layer
num_lr = 0.01              # learning rate
num_dr = 0.2               # dropout rate
num_epoch = 20             # number of epoches for optimization
batch_size = 50            # number of batch size for optimization

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

# Instantiate training and test data
train_data = Data(x_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(x_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


class tdCoxSNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(tdCoxSNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=num_dr)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = F.selu(self.layer_1(x))
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        return x

model_torch = tdCoxSNN(x_train.shape[1], num_nodes, 1)
learning_rate = num_lr
loss_fn = loss_tdCoxSNN_PyTorch
optimizer = torch.optim.Adam(model_torch.parameters(), lr=learning_rate)

epoch_loss = []
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model_torch(inputs)
        loss = loss_fn(outputs, labels)
        if loss != torch.tensor(0.):
            loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss.append(running_loss / (i+1))
    print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {running_loss / (i+1):.5f}')
print("Training Complete")

with torch.no_grad():
    rs_train = model_torch(torch.tensor(x_train,dtype=torch.float32)).clone().detach()
    rs_test = model_torch(torch.tensor(x_test,dtype=torch.float32)).clone().detach()

# calculate baseline hazard function
base_haz = baseline_hazard(np.column_stack((y_train,rs_train)))
# prepare the test dataset
test_rs = np.column_stack((pbc2_test[['id','tstart']],rs_test)) # id time predicted_risk_score

# calculate the survival probability at (time_of_interest+last_obs_time) for each subject
S = survprob(time_of_interest = [1,30,60,180,365], # in days
             haz = base_haz, 
             test_risk_score = test_rs)
