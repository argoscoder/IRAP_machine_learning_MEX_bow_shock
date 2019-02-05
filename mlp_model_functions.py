#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:03:52 2018

@author: thibault
"""

"""
File dedicated to the creating, compiling, training the model and getting predictions from the test dataset
"""
import pandas as pd
import keras as ks  

"""
Creating the neural network itself
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
"""
Creates a keras model 
Arguments:
    - lay_s : ex : layer_sizes = [11,11,6] will give an ANN with layers (11, 11, 6)
    - act : ex : act = ['relu', 'relu', 'softmax']
    - dropout : dropout proportion, default to 0
"""
def create_model(lay_s, act, dropout=0.0):
    #initializing the model
    model = Sequential()
    #adding the input layer
    model.add(Dense(lay_s[0], activation = act[0], input_shape=(lay_s[0],)))
    if dropout>0:
        model.add(Dropout(dropout))
    #adding the other layers
    for i in range(1,len(lay_s)):
        model.add(Dense(lay_s[i], activation = act[i]))
    return model


"""
Saving the model to a dedicated file
"""
def save_model(filepath, model):
    model.save(filepath)
    
"""
Loading the model from a specific file
"""
def load_model(filepath):
    model = ks.models.load_model(filepath, custom_objects={'jaccard_distance': jaccard_distance})
    return model
"""
Training the model
"""
import keras.backend as K
"""
Defines the jaccard distance as a custom metrics for keras
CODE COPIE COLLE DEPUIS https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
"""
"""
Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
This loss is useful when you have unbalanced numbers of pixels within an image
because it gives all classes equal weight. However, it is not the defacto
standard for image segmentation.
For example, assume you are trying to predict if each pixel is cat, dog, or background.
You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
The loss has been modified to have a smooth gradient as it converges on zero.
This has been shifted so it converges on 0 and is smoothed to avoid exploding
or disappearing gradient.
Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
= sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
# References
Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
What is a good evaluation measure for semantic segmentation?.
IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
https://en.wikipedia.org/wiki/Jaccard_index
"""

"""
Explication du Jaccard
Les ensembles X et Y à considérer sont:
    Xc : éléments de classe c dans les prédictions
    Yc : éléments de classe c dans le set de test

Pour la précision, les ensembles considérés sont différents:
    Xc : éléments de classe c dans les prédictions
    Y : ensemble des éléments du set de test
    
Dans les 2 cas on somme ensuite sur l'ensemble des classes c

A noter que le jaccard et la précision sont donc identiques dans le cas d'une classification binaire
"""
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
# VERIFIER
"""
Compiling and fitting the model

We will now compile the model and train it to fit our training data. This happens
thanks to the Keras .compile() and .fit() methods. The code is as follows:
    
    #for classification problems, the metrics used will be metrics=['accurracy']
    model.compile(optimizer=, loss=, metrics=)    
    model.fit(X_train, y_train, epochs=, batch_size=, verbose=)
    
The fit functions returns the history, which can be saved as a variable and used
for results visualization.
    
Keras offers a lot of different possible loss functions. A few examples:
    - mean_squared_error
    - categorical_crossentropy
    - poisson
    - hinge
    - cosine_proximity
    ...

Same for optimizers:
    - SGD
    - RMSprop
    - Adagrad
    - Adam
    ...

The batch_size parameter defines how many samples will be averaged before tuning 
the parameters.
the epochs parameter sets the number of consecutive trainings.



CALLBACK functions:
    Keras enables the user to use callback arguments, which allow to take a look 
    at the state of the network during the training. Callbacks can be passed to
    the model as follows : 
        model.fit(X_train, y_train, epochs=, batch_size=, verbose=, callbacks=)
"""

"""
Arguments:
    - model : model to compile and train
    - X_train : train set to feed to the network
    - y_train : labels corresponding to the X_train data set
    - n_epochs : number of epochs of training
    - b_s : batch size during the training
    - loss_name : loss to use (default: jaccard_distance)
Returns:
    training is the history of the training
"""
def compile_and_fit(model, X_train, y_train, n_epochs, b_s, val_size=0, loss_name = jaccard_distance):
    model.compile(optimizer = 'adam', loss = [loss_name], metrics = ['acc'])
    training = model.fit(X_train, y_train, validation_split = val_size, epochs = n_epochs, batch_size = b_s, verbose = 1)
    return training

"""
Get an untimed vector of predictions from a test set
"""
def get_pred(model, X_test):
    y_pred = model.predict_classes(X_test)
    return y_pred

"""
Get a timed vector of predictions from the test set
"""
from sklearn.preprocessing import StandardScaler

def get_pred_timed(model, X_test_timed, scale_data_timed):
    y_pred_timed = pd.DataFrame()
    y_pred_timed['epoch'] = X_test_timed['epoch']
    
    scale_data = scale_data_timed.copy() 
    del scale_data['epoch']
    
    X_test = X_test_timed.copy() 
    del X_test['epoch']
        
    scaler = StandardScaler().fit(scale_data)
    X_test = scaler.transform(X_test)
    
    y_pred = model.predict_classes(X_test)
    y_pred_timed['label'] = y_pred
    
    return y_pred_timed
"""
Get the probability with which the model predicted each class
"""
def get_prob_timed(model, X_test_timed, X_train_timed):
    y_prob_timed = pd.DataFrame()
    y_prob_timed['epoch'] = X_test_timed['epoch']
    
    X_train = X_train_timed.copy() 
    del X_train['epoch']
    
    X_test = X_test_timed.copy() 
    del X_test['epoch']
        
    scaler = StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    
    y_prob = model.predict(X_test)
    y_prob_timed['prob'] = [max(y_prob[i]) for i in range(X_test.shape[0])]
    
    return y_prob_timed

