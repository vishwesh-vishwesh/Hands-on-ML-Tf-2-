# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 06:51:37 2020

@author: VISHWESH
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#%% import mnist
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

X = mnist["data"]
y = mnist["target"]
'''
y = y.reshape(-1,1)
one_hot = OneHotEncoder()
one_hot_target=one_hot.fit_transform(y)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

#%%
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

svm_rbf_clf = SVC(kernel="rbf",gamma=5,C=0.001,probability=True)    
    
#svm_rbf_clf.fit(X_train_scaled[:10000],y_train[:10000])
#%%
from sklearn.model_selection import cross_val_score, cross_val_predict
#cv_score=cross_val_score(svm_rbf_clf,X_train_scaled[:10000],y_train[:10000],cv=5,scoring="accuracy")

from sklearn.metrics import confusion_matrix,precision_score,recall_score
#%%
def precision_finder(classifier,training_data, training_label):
    cv_predict=cross_val_predict(classifier,training_data,training_label,cv=5)

    cnf_mx = confusion_matrix(training_label,cv_predict)

    precision = precision_score(training_label,cv_predict)
    recall = recall_score(training_label,cv_predict)
    return precision,recall,cnf_mx
#%%
#training_results_svc = precision_finder(svm_rbf_clf,X_train_scaled[:10000],y_train[:10000])

#%% grid_search, trade off , roc curve
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'gamma' : [0.1,0.2], 'C' : [3,4,5]},
    
    ]

grid_search = GridSearchCV(svm_rbf_clf,param_grid,cv=5,scoring = "accuracy",return_train_score=True,verbose=2)
grid_search.fit(X_train[:1000],y_train[:1000])

print(grid_search.best_params_)

cvres = grid_search.cv_results_

#pred = grid_search.predict_proba(X_train_scaled[:1000],y_train[:1000])

print(grid_search.best_score_)
#%%
svc_grid_results = precision_finder(grid_search.best_estimator_,X_train_scaled[:1000],y_train[:1000])
#%% random forest
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=42,verbose=0,oob_score=True)
rnd_clf.fit(X_train_scaled,y_train)

#%%
training_results_rnd = precision_finder(rnd_clf,X_train,y_train)
#%% oob_score - this determines what would be accuracy on testing set (like a form of validation)

print(rnd_clf.oob_score_)
oob_decision_function = rnd_clf.oob_decision_function_ #score per class
#%% importance of each features
importance = rnd_clf.feature_importances_
import matplotlib.pyplot as plt 
plt.plot(importance)
plt.show()
#%% reduce dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced =pca.fit_transform(X_train_scaled) #remove the unwanted features / dimensionality reduction

#print(pca.explained_variance_ratio_)
#%% train rnd with new reduced dataset
rnd_reduced_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=42,verbose=0,oob_score=True)
rnd_reduced_clf.fit(X_reduced,y_train)

training_results_rnd_reduced = precision_finder(rnd_reduced_clf,X_reduced,y_train)
#%% use kernel pca and tsne also reconstruction









