# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:30:18 2020

@author: VISHWESH
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

train,test= keras.datasets.mnist.load_data()

X_train = train[0]/255.0
y_train = train[1]
X_test = test[0]/255.0
y_test = test[1]


'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
#%%
import matplotlib.pyplot as plt
plt.imshow(X_train[1], cmap="binary")
plt.axis('off')
plt.show()

#%%
'''
from sklearn.decomposition import PCA
X_flat = X_train.reshape(60000,784)
pca = PCA(n_components=0.95)
X_reduced_pca = pca.fit_transform(X_flat)
'''
#%%
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = X_train.shape[1:]),
    keras.layers.Dense(300,activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
        ])

model.summary()
#%%
opt = keras.optimizers.SGD(learning_rate=2e-1)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics="accuracy"
              )
#%%
import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") + " lr=2e-1"
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir,histogram_freq=1)
#%%
#checkpoint_cb = keras.callbacks.ModelCheckpoint("my_mnist_model.h5", save_best_only=True) creates checkpoint and saves best model
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
history = model.fit(X_train,y_train, epochs=100, validation_split = 0.2,
                    callbacks=[early_stopping_cb,tensorboard_cb])

model.save("seq_model_mnist_1.h5")
#%% tsne
'''
from sklearn.manifold import TSNE
np.random.seed(42)
X_flat = X_train.reshape(60000,784)

m = 10000
idx = np.random.permutation(60000)[:m]

X = X_flat[idx]
y = y_train[idx]
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)


plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()'''

#%%
from sklearn.model_selection import cross_val_score, cross_val_predict
#cv_score=cross_val_score(svm_rbf_clf,X_train_scaled[:10000],y_train[:10000],cv=5,scoring="accuracy")

from sklearn.metrics import confusion_matrix,precision_score,recall_score

#%%
import pandas as pd
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
#%% training acc
pred = model.predict(X_train) 
pred_classes =(np.argmax(pred,axis=1))

cnf_matrix = confusion_matrix(y_train,pred_classes)
precision_training = precision_score(y_train,pred_classes,average="weighted")
recall_training = recall_score(y_train,pred_classes,average="weighted")
#%% its wrong way!!!
'''
X_flat_test = X_test.reshape(10000,784)
pca = PCA(n_components=154)
X_test_reduced_pca = pca.fit_transform(X_flat_test)
'''
model.evaluate(X_test, y_test)

# %tensorboard --logdir=./my_logs --port=6006



























