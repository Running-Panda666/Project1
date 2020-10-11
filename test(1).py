"""
Created on Mon Oct 5 14:31:24 2020

@author: Zhao Chenwei
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import accuracy_score #for accuracy calculation
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier



# Load Dataset
# Preprocessing Dataset
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

Y = df.winner # Target variable

#Data Cleaning
#Calculate the correlation efficient
Data=X.values
pccs=[]
for i in range(Data.shape[1]):
    a=np.corrcoef(Data[:,i], Y)
    pccs.append(a[1][0])
item=feature_cols
#create a figure object
figure = plt.figure()
axes1 = figure.add_subplot(1, 1, 1)
#Set colour and type of lines
axes1.plot(item, pccs, 'ro--')
#Set Basic Information
axes1.set_xlabel('Features')#x axesl label
axes1.set_ylabel('Correlation')#y axesl label
axes1.set_title("The Figure of Correlation Efficient")#title
pl.xticks(rotation=90)
plt.show()
#Normalization
mm = MinMaxScaler()
X_train=X
Y_train=Y
X_test=df_target[feature_cols] # Features
Y_test=df_target.winner # Target variable
#Delete some attributes with small correlation
X_train = X_train.drop(['gameId', 'creationTime', 'gameDuration'], axis=1).values
X_test = X_test.drop(['gameId', 'creationTime', 'gameDuration'], axis=1).values

#Normalization
X_train = mm.fit_transform(X_train)
X_test = mm.fit_transform(X_test)

#Apply Decision Tree Method
# Create Decision Tree classifer object
#Record the time of running process
start=time.time()
clf1 = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf1 = clf1.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = clf1.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("The Accuracy of DT for tast data is",accuracy_score(Y_test, Y_pred))
end=time.time()
print("The time for DT running test data is",end-start,"s")


#Apply SVM Method
# Create Support Vector Machine classifer object
#Record the time of running process
start=time.time()
clf2=SVC(C=200,kernel= 'rbf')
# Train Support Vector Machine Classifer
clf2= clf2.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = clf2.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("The Accuracy of SVM for test data is:",accuracy_score(Y_test, Y_pred))
end=time.time()
print("The time for SVM running test data is",end-start,"s")


#Apply KNN Method
# Create Parameter number of neighbour of KNN classifer object
#Record the time of running process
start=time.time()
clf3=KNeighborsClassifier(5)
# Train KNN Classifer
clf3 = clf3.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = clf3.predict(X_test)
print("The Accuracy of KNN for test data is:",accuracy_score(Y_test, Y_pred))
end=time.time()
print("The time for KNN running test data is",end-start,"s")


#Apply MLP Method
#Record the time of running process
start=time.time()
clf4=MLPClassifier(activation='logistic',solver='adam',alpha=0.001,max_iter=100)
# Train MLP Classifer
clf4 = clf4.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = clf4.predict(X_test)
end=time.time()


#Apply Fusion Method
# Create Parameter number of neighbour of Ensemble classifer object
#Record the time of running process
start=time.time()
voting_clf=VotingClassifier(
    estimators=[('SVM',clf2),('DT',clf1),('KNN',clf3),('MLP',clf4)],voting='hard'
)
voting_clf.fit(X_train,Y_train)
Y_pred = voting_clf.predict(X_test)
b=accuracy_score(Y_test, Y_pred)
print("The Accuracy of Voting Fusion for test data is:",b)
end=time.time()
print("The time for Ensemble running test data is",end-start,"s")





















