# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:12:48 2020

@author: VISHWESH
"""
import numpy as np
import os
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#%%
iris = datasets.load_iris()
X = iris["data"][:,2:4]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
                    ("scale", StandardScaler()),   
                    ("linear_SVC",LinearSVC(C=1,loss="hinge"))  # or use Linear kernel of SVC class
    
    ])

svm_clf.fit(X,y)
#%%
print(svm_clf.predict([[5.5,1.7]]))
#%% Non-linear svm
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X,y = make_moons(n_samples=100, noise=0.15)

poly_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler",StandardScaler()),
    ("svm_clf", LinearSVC(C=10,loss="hinge",random_state=42))
    ])
poly_svm_clf.fit(X,y)
#%%
print(poly_svm_clf.predict([[5.5,1.7]]))
#%% with kernel = poly
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)

#%%
print(poly_svm_clf.predict([[5.5,1.7]]))
#%% also there is rbf kernel
'''
--use rbf kernel
--use LinearSVR - for regression
-- aso use svm.SVR
'''
#%%





















