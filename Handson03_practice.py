# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:45:31 2020

@author: VISHWESH
"""
#%% importing 
import numpy as np
import scipy
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier

#%% import titanic data

dataset = pd.read_csv('titanic')

dataset = dataset.drop(columns=["PassengerId","Ticket","Cabin","Embarked"])
labels = dataset["Survived"]
Passenger_name = dataset["Name"]
dataset = dataset.drop(columns=["Name"])
cat_data = dataset["Sex"]
#dataset_no_sex = dataset.drop("Sex",axis=1)
dataset_no_label = dataset.drop("Survived",axis=1)
#%%

X_train, X_test, y_train, y_test = train_test_split(dataset_no_label,labels)

dataset_num = X_train.drop("Sex",axis=1)
testset_num = X_test.drop("Sex",axis=1)
#%% Visualization
print(y_train.value_counts())
print(X_train.describe())
#corr_mx = np.c_[y_train,X_train]
#corr_mx = pd.DataFrame(corr_mx,columns=.columns,index=dataset_num.index)
corr_mx = dataset.corr()

X_train.plot(kind="scatter",x="Age",y="Fare", alpha=0.4, s=X_train["Pclass"]*10,label="class",
             c=y_train,cmap=plt.get_cmap("jet"),colorbar=True)
plt.xlabel("Age")
plt.legend()
#%% Data Cleansing
#imputer = SimpleImputer(strategy = "median")

#imputer.fit(dataset_num)
#X=imputer.transform(dataset_num)

#dataset_tr = pd.DataFrame(X,columns=dataset_num.columns,index=dataset_num.index)

#%%
"""
cat_data = cat_data.values.reshape(-1,1)
one_hot = OneHotEncoder()
sex=one_hot.fit_transform(cat_data) """ #gives a sparse marix
#one_hot.categories_   -- for checking the categories representing values of attribute sex
#%%Standardize
"""
scaler = StandardScaler()
dataset_num_scaled = scaler.fit_transform(dataset_num)
"""

#%% Pipeline

num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
                ])
dataset_num_tr = num_pipeline.fit_transform(dataset_num)

num_attribs = list(dataset_num)
cat_attribs = ["Sex"]
full_pipeline = ColumnTransformer([
                ("num",num_pipeline,num_attribs),
                ("cat",OneHotEncoder(),cat_attribs)
                ])

dataset_prepared = full_pipeline.fit_transform(X_train)

#%% training/fit SGD
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(dataset_prepared,y_train)
#%% SVM svc
from sklearn.svm import SVC
svc_clf = SVC(kernel="rbf",random_state=42)
svc_clf.fit(dataset_prepared,y_train)

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(dataset_prepared,y_train)

#%% cross val
from sklearn.model_selection import cross_val_score, cross_val_predict
cv_score=cross_val_score(svc_clf,dataset_prepared,y_train,cv=5,scoring="accuracy")

from sklearn.metrics import confusion_matrix,precision_score,recall_score


#%% cross validation

def precision_finder(classifier,training_data, training_label):
    cv_predict=cross_val_predict(classifier,training_data,training_label,cv=5)

    cnf_mx = confusion_matrix(training_label,cv_predict)

    precision = precision_score(training_label,cv_predict)
    recall = recall_score(training_label,cv_predict)
    return precision,recall,cnf_mx
#%% results

training_results_svc = precision_finder(svc_clf,dataset_prepared,y_train)

import joblib
joblib.dump(svc_clf,"svc_clf_titanic.pkl")
joblib.dump(training_results_svc,"training_results_for_svc_clf.pkl")
#%% grid_search, trade off , roc curve
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'gamma' : [0.01,0.1,1], 'C' : [1,10,30], 'coef0' : [0,1,2], 'degree' : [0,1,2] },
    
    ]

grid_search = GridSearchCV(svc_clf,param_grid,cv=5,scoring = "accuracy",return_train_score=True)
grid_search.fit(dataset_prepared,y_train)

print(grid_search.best_params_)

cvres = grid_search.cv_results_

#%% cross val prediction for best grid search result model

svc_grid_results = precision_finder(grid_search.best_estimator_,dataset_prepared,y_train)

#%%testing pipeline


testset_prepared = full_pipeline.fit_transform(X_test)


#test_results = train_results(knn_clf,testset_prepared[0:1],y_test[0:1],cv=2)


#%%
pred = svc_clf.predict(testset_prepared[0:10,:])

#%%
def tester(classifier,testing_data, testing_label):
    cv_predict_test = classifier.predict(testing_data)

    cnf_mx_test = confusion_matrix(testing_label,cv_predict_test)

    precision_test = precision_score(testing_label,cv_predict_test)
    recall_test = recall_score(testing_label,cv_predict_test)
    return precision_test,recall_test,cnf_mx_test

test_predictions = tester(grid_search.best_estimator_,testset_prepared,y_test)


#%%

















