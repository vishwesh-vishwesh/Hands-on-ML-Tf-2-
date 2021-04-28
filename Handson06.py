# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:43:57 2020

@author: VISHWESH
"""
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

#%%
iris = load_iris()
X = iris.data[:,2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(X,y)
#%%
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file=os.path.join("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
# convert .dot file to png online

#%%
print(tree_clf.predict_proba([[5,1.5]]))