"""
Created on Mon Oct 5 14:31:24 2020

@author: Zhao Chenwei
"""
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score #for accuracy calculation
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier


#Define ANN Function
class ANN(nn.Module):
    def __init__(self,n1,n2,n3):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n1, out_features=n2)
        self.fc2 = nn.Linear(in_features=n2, out_features=n3)
        self.output = nn.Linear(in_features=n3, out_features=2)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.output(x)
        x = F.softmax(x)
        return x

def transform(array):
    for i in range(len(array)):
        if array[i]==2:
            array[i]=0
    return array
def inverse_t(array):
    for i in range(len(array)):
         if array[i]==0:
            array[i]=2
    return array

# Load Dataset And Preprocessing Dataset
df = pd.read_csv('new_data.csv')
df_target=pd.read_csv('test_set.csv')
col_names = ['gameId', 'creationTime', 'gameDuration', 'winner', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
             'firstDragon', 'firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills',
             't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
#split dataset in features and target variable
feature_cols = ['gameId', 'creationTime', 'gameDuration',  'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
             'firstDragon', 'firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills',
             't1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
X = df[feature_cols] # Features
X_Test=df_target[feature_cols]
Y = df.winner # Target variabl
Y=transform(Y)
Y_Test=df_target.winner
Y_Test=transform(Y_Test)


#Apply ANN Method
#Data Preprocessing
#Delete some attributes with small correlation
X = X.drop(['gameId', 'creationTime', 'gameDuration'], axis=1).values
X_Test = X_Test.drop(['gameId', 'creationTime', 'gameDuration'], axis=1).values

Y=Y.values
Y_Test=Y_Test.values
X_train = torch.FloatTensor(X)
X_Test = torch.FloatTensor(X_Test)
Y_train = torch.LongTensor(Y)
Y_Test = torch.LongTensor(Y_Test)

#Record the time of running process
start=time.time()
model = ANN(16,35,11)
criterion = nn.CrossEntropyLoss()
#learning_rate=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
epochs = 100
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(X_train)
    loss = criterion(y_hat, Y_train)
    loss_arr.append(loss)
    if i % 2 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


predict_out = model(X_Test)
_,predict_y = torch.max(predict_out, 1)
print("The accuracy of ANN after deduction is ", accuracy_score(Y_Test, predict_y) )

#Record the end of running process
end=time.time()
print("The time for ANN running test data is",end-start,"s")
#Transform results labels as "1" and "2" result
Y_Test=inverse_t(Y_Test)


