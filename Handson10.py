# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 22:51:18 2020

@author: VISHWESH
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

#%%
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full), (X_test,y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)

X_valid,X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid,y_train = y_train_full[:5000], y_train_full[5000:]

class_names = fashion_mnist.class_names

#%%
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()
#%%
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(class_names[y_train[0]])

#%% create a sequential model
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28,28]),
    keras.layers.Dense(300,activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
        ])

model.summary()
#%% 
print(model.layers)
hidden1 = model.layers[1]
print(hidden1.name)
model.get_layer("dense") is hidden1
# find weights and biases initialized
weights,biases = hidden1.get_weights()

#keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
#%% compiling
opt = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics="accuracy"
              )

#%% class weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train) #use above weights to class_weight method inside fit
#%%training
history = model.fit(X_train,y_train, epochs=30, validation_data=(X_valid,y_valid))
# try using class_weight and sample_weight
# class_weight is used when datasets are skewed - some classes are over represented and some are under
# also use sample_weight for validation_data as third parameter
#%% plot losses using DataFrame  - learning curves
print(history.params)
#print(history.epoch)
#print(history.history)
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) #set vertical range to 0-1
plt.show()
#%%evaluations

model.evaluate(X_test,y_test)
#%% make predictions
X_new = X_test[:50]
y_proba = model.predict(X_new)
print(y_proba.round(2))

#%%
y_pred = np.argmax(model.predict(X_test[:3]),axis=1) # instead of predict_classes
print(np.array(class_names)[y_pred])

y_new = y_test[:3]
#%%
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

#%%































