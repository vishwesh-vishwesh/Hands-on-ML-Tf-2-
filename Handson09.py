# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:51:06 2020

@author: VISHWESH
"""
from sklearn.datasets import load_iris
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
#%%
data = load_iris()
X = data.data
y = data.target
print(data.target_names)
#%%
plt.figure(figsize=(9, 3.5))

plt.subplot(121)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(122)
plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
plt.xlabel("Petal length", fontsize=14)
plt.tick_params(labelleft=False)

plt.show()
#%% k means
from sklearn.datasets import make_blobs
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
#%%
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    
plt.figure(figsize=(8, 4))
plot_clusters(X)
plt.show()
#%%
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

y_pred is kmeans.labels_
#%%
print(kmeans.cluster_centers_)
print(kmeans.labels_)

X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
print(kmeans.predict(X_new))   #hard clustering

print(kmeans.transform(X_new)) #soft clustering

#%% centroid init
print(kmeans.inertia_) #lowest inertia is best model in random initialization
print(kmeans.score(X))
#%% minibatch
from sklearn.cluster import MiniBatchKMeans
mini_batch_kmeans = MiniBatchKMeans(n_clusters=5)
mini_batch_kmeans.fit(X)

#%% optimal no. of clusters , k. and silhouette
from sklearn.metrics import silhouette_score
sil=silhouette_score(X, kmeans.labels_)
#%%
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]
#%%
plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
plt.show()
#%% image segmentation using clustering
# Download the ladybug image
import os
import urllib
images_path = os.path.join("images", "unsupervised_learning")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "ladybug.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/unsupervised_learning/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))
#%%
from matplotlib.image import imread
image = imread(os.path.join(images_path, filename))
print(image.shape)
#%%
X = image.reshape(-1,3)
kmeans = KMeans(n_clusters=15).fit(X)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

plt.imshow(segmented_img)
#%% clustering for preprocessing
from sklearn.datasets import load_digits
X_digits,y_digits = load_digits(return_X_y=(True))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_digits,y_digits, random_state=42)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
log_reg.fit(X_train,y_train)  #not sclaed data

score=log_reg.score(X_test,y_test)  #bseline model accuracy
#%% 
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("kmeans",KMeans(n_clusters=50)),
    ("log_reg",LogisticRegression())
    ])

pipeline.fit(X_train,y_train)

pip_score=pipeline.score(X_test,y_test)

#%% use gridsearch
from sklearn.model_selection import GridSearchCV

param_grid =[ 
    dict(kmeans__n_clusters=range(2,100))
 
    ]

grid_clf = GridSearchCV(pipeline, param_grid, cv=3,verbose=2)
grid_clf.fit(X_train,y_train)

print(grid_clf.best_params_)
print(grid_clf.best_estimator_.score(X_test,y_test))
#%%










































