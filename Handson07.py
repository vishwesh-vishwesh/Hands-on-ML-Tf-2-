# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 00:55:33 2020

@author: VISHWESH
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
#%%
log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42, probability=True)   #probabibily = true, to produce predic_proba - use for soft voting

X,y = make_moons(n_samples=500, noise=0.30)
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=42)

#%%
voting_clf = VotingClassifier(
    estimators= [('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
    voting = 'soft'  #or hard
    )

voting_clf.fit(X,y)
#%%   accuracy of each clf

from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict((X_test))
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))
    
#%%    bagging 
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
bag_clf=BaggingClassifier(
        DecisionTreeClassifier(),n_estimators = 500,
        max_samples = 100, bootstrap = True, n_jobs =-1
        )
bag_clf.fit(X_train,y_train)
y_pred = bag_clf.predict(X_test)
y_prob = bag_clf.predict_proba(X_test)
y_accuracy = accuracy_score(y_test,y_pred)
#%% with oob - out of the box evaluation (like cross validation)

bag_clf=BaggingClassifier(
        DecisionTreeClassifier(),n_estimators = 500,
        max_samples = 100, bootstrap = True, n_jobs =-1, oob_score=True
        )
bag_clf.fit(X_train,y_train)
oob_score = bag_clf.oob_score_
y_pred_oob = bag_clf.predict(X_test)
y_prob_oob = bag_clf.predict_proba(X_test) #probabaility of each instance of x_test
y_accuracy_oob = accuracy_score(y_test,y_pred_oob)
decision_function_oob = bag_clf.oob_decision_function_ # proba for training )
#%% Random forest
from sklearn.ensemble import RandomForestClassifier #or ExtraTreesClassifier

rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1)
rnd_clf.fit(X_train,y_train)
y_pred_rf = rnd_clf.predict(X_test)
y_accuracy_rf = accuracy_score(y_test,y_pred_rf)
#%% ada boost
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm ="SAMME.R",learning_rate=0.5
        )
ada_clf.fit(X_train,y_train)
y_pred_ada = ada_clf.predict(X_test)
y_accuracy_ada = accuracy_score(y_test,y_pred_ada)
#%% GBRT or Gradient boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=3,learning_rate=1.0)
gbrt.fit(X,y)
y_pred_gbrt = gbrt.predict(X_test)
y_accuracy_ada = mean_squared_error(y_test,y_pred_gbrt)
#%% gbrt with early stopping and training again with optimum number of trees
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train,X_val,y_train,y_val = train_test_split(X,y, random_state=42)
gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=120)
gbrt.fit(X_train,y_train)

errors = [mean_squared_error(y_val,y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train,y_train)
#%%
min_error = np.min(errors)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.ylabel("Error", fontsize=16)
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plt.plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.xlabel("$x_1$", fontsize=16)

plt.show()
#%% XG Boosting
import xgboost










































