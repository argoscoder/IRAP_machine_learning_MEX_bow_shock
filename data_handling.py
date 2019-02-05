#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:46:09 2018

@author: thibault
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils 

"""
File dedicated to handling the data that we will feed our MLP with

It provides functions for data loading, pre-processing, transformations, post-processing (loading data,
extracting variations from predictions, modifying the number of classes, 
extracting the train and test set, appending new data to existing DataFrames...)
"""

"""
Loads the data from a given path
ex : path = data_folder + data_file
treat_na = 'FILL' fills the NaN with 0
treat_na = 'DROP' drops the rows containing NaN values
"""
def load_data(path, treat_na = 'FILL'):
    #loading the file with pd.read_csv
    timed_data = pd.read_csv(path, sep =',')
    if treat_na == 'FILL':
        timed_data = timed_data.fillna(timed_data.median())
    if treat_na == 'DROP':
        timed_data = timed_data.dropna()
    return timed_data

"""
Converts the 6 'year','month',...,'second' columns to a single 'epoch' column
data : DataFrame to convert
cols : concerned columns
"""
def conv_date(data, cols):
    time_table = data[cols]
    time_table.columns = ['year','month','day','hour','minute','second']
    new_data = data.drop(cols, axis=1)
    
    time_date = pd.to_datetime(time_table)
    time_epoch = (time_date - pd.Timestamp('1970-01-01'))//pd.Timedelta('1s')

    new_data['epoch'] = time_epoch
        
    return new_data
"""
Get the external params:
    EUV flux
    ...
"""
def get_external_params(filepath):
    timed_data = pd.read_csv(filepath, sep='\s+',comment='#')
    timed_data.columns = ['epoch', 'euv_flux','pdyn']
    return timed_data


"""
Re samples the data at a lower sampling rate, expressed in s, multiple of 4
"""
def resample(data, t_sample):
    return data.loc[data.index%(t_sample/4)==0]

"""
Takes a DataFrame containing raw data and returns the shocks i to n+i in 
chronological order as a new DataFrame
"""
def n_first_shocks(data, n, start_i):
    indexes = []
    nb_int = 0
    i = 0
    start_epoch = data['epoch'].iloc[i]
    while nb_int<n+start_i:
        j = 0
        curr_epoch = data['epoch'].iloc[i+1]
        start_epoch = data['epoch'].iloc[i]
        while curr_epoch - start_epoch<=5:
            start_epoch = curr_epoch
            curr_epoch = data['epoch'].iloc[i+j]
            j+=1
        
        if(nb_int>=start_i):
            indexes.extend([k for k in range(i,i+j)])
        i+=j
        nb_int+=1
    del indexes[-1]
    return data.iloc[indexes]
    
    

"""
Computes the totels 6 and 8 for a given data set containing only the totels 1
def of totels_1 : mean_(mex_els_spec_[20,200],20)
def of totels_8 : shiftT(sliding_mean(totels_1,300),120) - shiftT(sliding_mean(totels_1,300),-120) 
def of totels_6 : same with range [25,37] for energies (eV)
-> intermediate totels_16 : mean_(mex_els_spec_[25,37],20)

Argument given to the function : totels_1 or totels_16 calculated by amda

Args : 
    data : pandas.DataFrame where to compute the totels
    totels_col : column where to find the source totels
    new_totels : name of the additional totels column
"""
#NE MARCHE PAS ACTUELLEMENT
def compute_totel(data, totels_col, new_totels):
    totels_1 = data[totels_col]
    roll_tot = totels_1.rolling(75).mean() #moyenne glissante sur 300 secondes donc 75 points de donnée
    shift_pos = roll_tot.shift(30)
    shift_neg = roll_tot.shift(-30) #shift de 120 secondes donc 30 points de donnée
    comp_totels = shift_pos + shift_neg
    new_data = data.copy()
    new_data[new_totels] = comp_totels.dropna()
    return new_data

"""
Argument:
    Takes in a pandas.DataFrame() which has to contain a 'epoch' and a 'label' columns
The function plots the labels of the given dataset on a time-axis
"""
def show_labeled_data(timed_data):
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Category')
    ax1.set_ylim(1.5,4.5)
    ax1.set_xlabel('Time')
    ax1.plot(pd.to_datetime(timed_data['epoch'],unit='s'), timed_data['label'], label = 'Set1', color='r')
    plt.show()
    return 

def graph_data(timed_data, col_name):
    fig, ax1 = plt.subplots()
    ax1.set_ylabel(col_name)
    ax1.set_xlabel('Time')
    ax1.plot(pd.to_datetime(timed_data['epoch'],unit='s'), timed_data[col_name])
    return
   
""" 
Deals with missing data by filling NaN with the median
BUT depending on the class

ATTENTION: need to sort values by epoch after applying this function
"""
def fillna_by_class(timed_data):
    labels = timed_data['label'].unique()
    new_data = pd.DataFrame(columns = timed_data.columns)
    for i in range(len(labels)):
        class_data = timed_data.loc[timed_data['label']==labels[i]]
        class_data = class_data.fillna(class_data.median())
        new_data = new_data.append(class_data)
    new_data.sort_values('epoch')
    new_data.reset_index(drop=True)
    return new_data
        

"""
In our problem the classes are initially the following ones:
    - solar wind : 4
    - close environment : 2
    - in bound bow shock : 2.7
    - out bound bow shock : 3.3
    
We might want to reduce the number of classes to 3 as follows:
    - solar wind : 4
    - bow shock : 3
    - close environment : 2

Arguments:
    list of labels to transform
"""
def to_3_classes(labels):
    for i in range(len(labels)):
        if labels[i] == 2.7 or labels[i] == 3.3:
            labels[i] = 3
    return 0

"""
Useful functions to switch between labels representations
    The 2 different representations are represented here for an example with 3 distinct classes
    
    labels = [0, 0, 1, 2, 1, 0, 2, 2, 0]
    
    y = [[1, 0, 0],
         [1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 1, 0],
         [1, 0, 0],
         [0, 0, 1],
         [0, 0, 1],
         [1, 0, 0]]

"""
from sklearn.preprocessing import LabelEncoder  
 
def one_hot_encode(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded)
    return y

def one_hot_decode(y):
    labels = []
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j] == 1:
                labels.append(j)
    return labels


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Returns the formatted inputs to train/test the ANN from a raw dataFrame
Also takes as an argument the scaling data to use on the considered data
"""
def get_inputs(data, scale_data):
   y = one_hot_encode(data['label'])
   
   X = data.drop(['label','epoch'], axis=1)
   
   
   scl = scale_data.drop(['label','epoch'], axis=1)
   
   scaler = StandardScaler().fit(scl)    
   # Scale the train and test set
   X = scaler.transform(X)
   return X,y

def get_timed_inputs(data, scale_data):
    X,y = get_inputs(data, scale_data)
    X = pd.DataFrame(X)
    X['epoch'] = data['epoch']
    y = pd.DataFrame(y)
    y['epoch'] = data['epoch']
   
   
"""
Getting the train and test sets
Arguments:
    timed_data : a given pandas.DataFrame() to convert to test and train sets, containing at least a 'label' and an 'epoch' columns
    ordered : defines if the data sets created are shuffled (default: ordered=False) or chronological (ordered=True).
    test_size : proportion of samples included in the test set (default: 0.3)
    
    nb_class : defines if the problem has to be downgraded to 3 classes instead of 4

Return:
    X_train_timed, y_train_timed, X_test_timed, y_test_timed
    (These variables are called '_timed' because they still contain the 'epoch' information of the initial data)
"""
def get_timed_train_test(timed_data, ordered=False, test_size = 0.3, start_index = 0, nb_class = 3, downsampling_ext = 0):
    
    ###########################
    #this part is specific to the martian problem
    ###########################
    if nb_class == 3 :
        labels = np.ravel(timed_data['label'])
        to_3_classes(labels)
        timed_data['label'] = labels
    elif nb_class != 4 :
        print("WARNING: wrong number of classes")
    ############################
    y = pd.DataFrame()
    y['epoch'] = timed_data['epoch']
    y['label'] = timed_data['label']
        
    X = timed_data.copy()
    del X['label']
        
    if ordered :
        if start_index == 0:
            split_index = int(test_size*timed_data.count()[0])
            
            X_test_timed = X.iloc[0:split_index,:]
            y_test_timed = y.iloc[0:split_index,:]
            
            X_train_timed = X.iloc[split_index:,:]
            y_train_timed = y.iloc[split_index:,:]
            
        else :
            split_index1 = start_index
            split_index2 = start_index + int(test_size*timed_data.count()[0])
            
            X_test_timed = X.iloc[split_index1:split_index2,:]
            y_test_timed = y.iloc[split_index1:split_index2,:]
            
            X_train_timed = pd.concat([X.iloc[0:split_index1,:], X.iloc[split_index2:,:]], ignore_index=True)
            y_train_timed = pd.concat([y.iloc[0:split_index1,:], y.iloc[split_index2:,:]], ignore_index=True)
        
        if downsampling_ext>0:
            new_Xtrain = X_train_timed.copy()
            new_Xtrain['label'] = y_train_timed['label']
            
            X_train_ev = new_Xtrain.loc[new_Xtrain['label']==2.0]
            X_train_sw = new_Xtrain.loc[new_Xtrain['label']==4.0]
            X_train_sh = new_Xtrain.loc[~new_Xtrain['label'].isin([2.0,4.0])]
            
            n_ev = int(downsampling_ext*X_train_ev.count()[0])
            n_sw = int(downsampling_ext*X_train_sw.count()[0])
            
            new_Xtrain = pd.concat([X_train_ev.sample(n_ev), X_train_sw.sample(n_sw), X_train_sh])
            X_train_timed = new_Xtrain.sort_values('epoch')
            y_train_timed = X_train_timed[['epoch','label']]
            del X_train_timed['label']            
        
    else:
        X_train_timed, X_test_timed, y_train_timed, y_test_timed = train_test_split(X,y,test_size = test_size)
        
    return X_train_timed, X_test_timed, y_train_timed, y_test_timed
"""
Getting the train and test sets without any time info and apply the scaling here
"""
def get_train_test_sets(X_train_timed, X_test_timed, y_train_timed, y_test_timed):
    X_train = X_train_timed.copy()
    X_test = X_test_timed.copy()
    
    y_train = one_hot_encode(y_train_timed['label'].tolist())
    y_test =  one_hot_encode(y_test_timed['label'].tolist())
    
    del X_train['epoch']
    del X_test['epoch']  
    
    scaler = StandardScaler().fit(X_train)    
    # Scale the train and test set
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

"""
Computes the azimutal angle (relative only to the x axis)
It comes down to calculating the angle between ((0,0),(x,rho)) and ((0,0),(1,0))
in the (x,rho) plane 
Takes a DataFrame with columns x,rho and returns this DataFrame with a new column azim 
"""
def append_sza(y_timed):
    x = y_timed['x']
    rho = y_timed['rho']
    sza = []
    for i in range(y_timed.count()[0]):
        if x.iloc[i]>0:
            sza.append(np.arctan(rho.iloc[i]/x.iloc[i]))
        else:
            sza.append(np.pi + np.arctan(rho.iloc[i]/x.iloc[i]))
    new = y_timed.copy()
    new['sza'] = sza
    return new
"""
Appends new data to an existing DataFrame from a bigger one
Typical use : appending all the parameters from the test set to the labeled predictions or computed variations

y_timed : pandas.DataFrame with at least an 'epoch' column
data_to_add : pandas.DataFrame with an 'epoch' column containing y_timed['epoch']
cols : columns to add from data_to_add (all of them if not provided)
"""

def append_data_to_timed(y_timed, data_ref, cols=None):
    new_y = y_timed.copy()
    if cols==None:
        cols = data_ref.columns
    to_add = data_ref.loc[data_ref['epoch'].isin(new_y['epoch'])]
    for i in range(len(cols)):
        new_y[cols[i]] = np.array(to_add[cols[i]])
    return new_y
    
"""
Adds a column 'amda_date' to a pandas.DataFrame with at least an 'epoch' column.
This 'amda_date' corresponds to the standard time format supported by AMDA.
"""
def append_amda_date(y_timed):
    y_new =y_timed.copy()
    amda_date = []
    for i in range(y_timed.count()[0]):
        t = y_timed['epoch'].iloc[i]
        date_t = pd.to_datetime(t, unit='s')
        ymd = str(date_t)[0:10]
        hms = str(date_t)[11:]
        amda_date.append(ymd+'T'+hms)
    y_new['amda_date'] = amda_date
    return y_new

"""
Writes a file adapted from y_timed, to be uploaded to AMDA.

file_dir  : string for file directory
file_name : string for file name
y_timed : pandas.DataFrame with columns 'amda_date','label','x','y','z'
""" 
def write_amda_file(file_dir, file_name, y_timed):
    data = pd.DataFrame()
    data['time'] = y_timed['amda_date']
    data['label'] = y_timed['label']
#    data[['x','y','z']] = y_timed[['x','y','z']]
    
    data.to_csv(file_dir+file_name, index = False, sep = ' ')

"""
New version of the functions that corrects labels
Idea : instead of correcting variations, we correct labels directly by taking 
for each of them a sliding window and assigning the most represented class in this sliding window
"""
def get_corrected_pred(var_ref, y_timed,timed_proba, Dt):
    corr_y = y_timed.copy()
    corr_lab = []
    n = y_timed.count()[0]
    last_account_index = 0
    
    for i in range(n):
        if(i%10000==0):
            print("Corrected ",i,"/",n)
        curr_t = y_timed.iloc[i]['epoch']
        curr_var_t = var_ref.iloc[last_account_index]['epoch']
        
        if abs(curr_t - curr_var_t) < Dt - Dt/4 :
            
            window = y_timed.loc[(y_timed['epoch'] > curr_t - Dt/2) & ((y_timed['epoch'] < curr_t + Dt/2))]
            
            sw_window = window.loc[window['label']==2]
            sh_window = window.loc[window['label']==1]
            ev_window = window.loc[window['label']==0]
            
            sw_prob = timed_proba.loc[timed_proba['epoch'].isin(sw_window['epoch'])]['prob']
            sh_prob = timed_proba.loc[timed_proba['epoch'].isin(sh_window['epoch'])]['prob']
            ev_prob = timed_proba.loc[timed_proba['epoch'].isin(ev_window['epoch'])]['prob']
            
            n_sw = sum(sw_prob)
            n_sh = sum(sh_prob)
            n_ev = sum(ev_prob)
            
            if n_sh>0:
                if n_sw >= max(n_sh, n_ev):
                    corr_lab.append(2)
                elif n_sh >= max(n_sw, n_ev):
                    corr_lab.append(1)
                else:
                    corr_lab.append(0)
            #si les 2 seules classes sont les classes externes on réintroduit le choc
            #A FAIRE
            else:
                if n_sw > n_ev :
                    if n_sw < n_ev*3 : 
                        corr_lab.append(1)
                    else: corr_lab.append(2)
                elif n_sw <= n_ev :
                    if n_ev < n_sw*3 : 
                        corr_lab.append(1)
                    else: corr_lab.append(0)
        else:
            corr_lab.append(y_timed.iloc[i]['label'])
            
        if(curr_t>curr_var_t + Dt - Dt/4):
            last_account_index+=1;
            if last_account_index>var_ref.count()[0] - 2:
                last_account_index = var_ref.count()[0] - 1;
        
    corr_y['label'] = corr_lab
    return corr_y    
    

"""
Returns a list of variations associated to a y_timed DataFrame (typically y_test or a prediction set)
y_timed : pandas.DataFrame with at least a 'label' and an 'epoch' columns
"""
def get_var(y_timed):
    y_timed = y_timed.sort_values(by='epoch')
    y_timed.index = y_timed.epoch
    var = pd.DataFrame() #cette fois on stocke les variations dans une dataframe avec les classes précédente et suivante
    curr_state = y_timed['label'].iloc[0]
    prec = []
    follow = []
    t = []
    for i in range(y_timed.count()[0] - 1):
        new_state = y_timed['label'].iloc[i+1]
        dt = y_timed['epoch'].iloc[i+1] - y_timed['epoch'].iloc[i]
        if (curr_state != new_state) and dt<60: #si 2 états successifs sont séparés de plus de 1 minute, ce n'est pas une variation mais un trou de données
            t.append(y_timed.index[i])
            prec.append(curr_state)
            follow.append(new_state)
        curr_state = new_state
            
    var['epoch'] = t
    var['prec_class'] = prec
    var['follow_class'] = follow
    var = var.sort_values(by='epoch')
    var.index = var.epoch
    #for console printing
    print('Total nb. variations: ', var.count()[0])
    return var

""""
Adds a 'category' column to a variations list, defined as follows from the 'prec_class' and 'follow_class' columns:
    - [0,1] and [1,0] = 0
    - [1,2] and [2,1] = 1
    - [0,2] and [2,0] = 0.5 (non physical variation)
    
Syntax when used : var_with_cat = get_category(var)
"""
def outdated_get_category(var):
    cat = []
    for i in range(var.count()[0]):
        classes = [var['prec_class'].iloc[i], var['follow_class'].iloc[i]]
        classes.sort()
        if classes == [0,1]:
            cat.append(0)
        elif classes == [1,2]:
            cat.append(1)
        elif classes == [0,2]:
            cat.append(0.5)
    new_var = var.copy()
    new_var['category'] = cat
    return new_var

"""
Same function as get_category but for any number of classes
Number of combinations for n variations = sum(k=[1,n])(k) = n*(n+1)/2

var_cat = -1 for non physical variations
"""
def get_category(var, nb_class = 3):
    cat = []
    n = var.count()[0]
    for i in range(n):
        classes = [var['prec_class'].iloc[i], var['follow_class'].iloc[i]]
        classes.sort()
        if classes[0]+1!=classes[1]:
            var_cat = -1
        else:
            var_cat = nb_class*classes[0] + classes[1]
        cat.append(var_cat)       
    new_var = var.copy()
    new_var['category'] = cat
    return new_var

"""
Returns a copy of pred_var with the added column 'dt_to_closest' 
This new column represents, for each variation in pred_var, how far the closest variation of same category in true_var is.
For the variations with a .5 category (non physical var.), it justs represents how far the closest true variation is. 

true_var : pandas.DataFrame with columns 'epoch', 'category'
pred_var : pandas.DataFrame with columns 'epoch', 'category'

Syntax : 
    new_pred_var = get_closest_var_by_cat(true_var, pred_var)
"""
def get_closest_var_by_cat(true_var,pred_var):
    dt_list = []
    pn = pred_var.count()[0]
    tn = true_var.count()[0]
    for i in range(pn):
        min_dt = float('inf')
        dt = float('inf')
        for j in range(tn):
            if pred_var['category'].iloc[i] == true_var['category'].iloc[j]:
                dt = pred_var['epoch'].iloc[i] - true_var['epoch'].iloc[j]
            elif pred_var['category'].iloc[i] == 0.5:
                dt = pred_var['epoch'].iloc[i] - true_var['epoch'].iloc[j]
            if abs(dt)<abs(min_dt):
                min_dt = dt
        dt_list.append(min_dt)
    new_pred_var = pred_var.copy()
    new_pred_var['dt_to_closest'] = dt_list
    return new_pred_var    

"""
Corrects a variations list to reduce the number of quick oscillations by applying the following process:
    Get a variation var_i
    Define a time interval [var_i[t], var_i[t] + Dt]
    For all following variations in this interval, check if they cancel each other
        ex : 2->1->0->2
    Delete all the variations that satisfy this condition
    
var : pandas.DataFrame with at least the columns 'epoch', 'prec_class' and 'follow_class'
Dt  : time interval to consider in seconds
"""
def corrected_var(var, Dt):
    epoch_to_skip = []
    i = 0
    while(i<var.count()[0] - 1):
#    for i in range(var.count()[0]-1):
        t = var['epoch'].iloc[i]
        t_it = var['epoch'].iloc[i+1]
        start_class = var['prec_class'].iloc[i]
        
        furthest = 0 #indice de la variation la plus lointaine a supprimer (a partir de i)
        j = 0 #nb de tours de boucle effectués
        while(t_it<t+Dt and i+j<var.count()[0]-2):
            curr_class = var['follow_class'].iloc[i+1+j]
            if curr_class == start_class:
                furthest = j+1
            j += 1
            t_it = var['epoch'].iloc[i+1+j]
        to_skip = []
        if furthest>0:    
            to_skip = var['epoch'].iloc[i:i+furthest+1]
        
        epoch_to_skip.extend(to_skip)
        i = i+1+furthest
    clean_var = var.loc[~var['epoch'].isin(epoch_to_skip)]
    return clean_var

"""
Prend une liste de variations et renvoie les labels aux epochs demandées
Process:
    On parcourt les epoch t
    On itère sur les variations vi dès que t>vi[t]
    On vérifie si vi-1[follow_class] == vi[prec_class]
    (Si oui t prend cette classe
    Sinon il prend une des 2 selon quelle variation est la plus proche)
    en fait on prend directement la classe la plus proche

Takes a pandas.DataFrame representing class variations and transforms it back to a list of labels based on a list of
epochs when the labels have to be predicted.

Process:
    Set current variation to var[0]
    For each t in epoch_list:
        if t > current_variation[t]
            current_variation becomes the next variation
        Tests which variation is closest to t between the current one and the previous one
        Appends the class corresponding to the closest variation ('follow_class' if previous var, 'prec_class' if curr_var)
"""
def label_from_var(var, epoch_list):
    lab = []
    var_index = 0
    for t in epoch_list:
        curr_var = var.iloc[var_index]
        if t>curr_var['epoch'] and var_index<var.count()[0]-1:
            var_index+=1
        
        if var_index == 0: #si on n'a pas dépassé la première variation, on prend sa classe précédente
            lab.append(curr_var['prec_class'])
        elif var_index == var.count()[0] -1:
            lab.append(curr_var['follow_class'])
        else:
            prec_var = var.iloc[var_index - 1]
            dt0 = abs(t-prec_var['epoch'])
            dt1 = abs(t-curr_var['epoch'])
            if dt0 < dt1:
                lab.append(prec_var['follow_class'])
            else:
                lab.append(curr_var['prec_class'])
    return lab

"""
Takes a list of variations and returns a list of shock crossings associated
Process:
    if non physical var : just add the epoch of the variation
    if shock detected : takes the middle of the detected shock
'category' needed
Returns a dataframe of crossings with their epoch and direction : 0 for inbound, 1 for outbound
"""
def crossings_from_var(var):
    epochs = []
    direction = []
    for i in range(var.count()[0]-1):
        v = var.iloc[i]
        v_next = var.iloc[i+1]
        if v['category'] == 0.5:
            epochs.append(v['epoch'])
            if v['follow_class'] == 0:
                direction.append(0)
            else: 
                direction.append(1)
        else:
            dt = v_next['epoch'] - v['epoch']
            if v['follow_class'] == v_next['prec_class'] and v['category']!=v_next['category'] and dt<1200:
                t = v['epoch'] + dt/2
                epochs.append(t)
                if v_next['follow_class'] == 0:
                    direction.append(0)
                else: 
                    direction.append(1)
    crossings = pd.DataFrame()
    crossings['epoch'] = epochs
    crossings['direction'] = direction
    return crossings
"""
For every predicted crossing, associates the value of the time interval to the closest true crossing
"""
def get_closest_cross(true_cross, pred_cross):
    dt_list = []
    pn = pred_cross.count()[0]
    tn = true_cross.count()[0]
    for i in range(pn):
        min_dt = float('inf')
        for j in range(tn):
            dt = pred_cross['epoch'].iloc[i] - true_cross['epoch'].iloc[j]
            if abs(dt) < abs(min_dt):
                min_dt = dt
        dt_list.append(min_dt)
    new_pred_cross = pred_cross.copy()
    new_pred_cross['dt_to_closest'] = dt_list
    return new_pred_cross

"""
Returns the crossings that two crossings list have in common
It returns shocks from cross_ref that also appear in cross_other, with the timings they have in shock_ref
A crossing is considered common if it appears in the two lists with dates separated by less than dt
If a crossing from a list corresponds to more than one other in the other list, it will appear twice in the common list
"""
def get_common_cross(cross_ref, cross_other, dt):
    common = pd.DataFrame(columns = cross_ref.columns)
    for i in range(cross_ref.count().max()):
        curr_ref = cross_ref.iloc[i]
        candidates = cross_other.loc[abs(cross_other['epoch']-curr_ref['epoch'])<dt]
        if candidates.count().max()>0:
            common.append(curr_ref)
    return common

"""
If some crossings are separated by less than Dt in crossings, they become a new one at Dt/2
The crossings have to be merged only if they have the same direction
"""
"""
OUTDATED
def group_crossings_on_dir(cross_1dir,Dt):
    new_cross = []    
    ep = cross_1dir['epoch'].tolist()
    i=0
    while i<len(ep):
        j = 1
        t_lim = ep[i] + Dt
        curr_t = ep[i]
        to_add = [curr_t]
        while i+j<len(ep)-2 and (ep[i+j] < t_lim) :
            to_add.append(ep[i+j])
            j = j+1
        if(len(to_add)>0):
            new_cross.append(sum(to_add)/len(to_add))
        i = i+j
    return new_cross

def group_cross(cross, Dt, on_dir = True):
    if on_dir:
        inbound = cross.loc[cross['direction'] == 0]
        outbound = cross.loc[cross['direction'] == 1]
        new_ep_i = group_crossings_on_dir(inbound, Dt)
        new_ep_o = group_crossings_on_dir(outbound, Dt)
        dir_i = [0]*len(new_ep_i)
        dir_o = [1]*len(new_ep_o)
        new_cross = pd.DataFrame()
        new_cross['epoch'] = new_ep_i + new_ep_o
        new_cross['direction'] = dir_i + dir_o
        new_cross = new_cross.sort_values(by='epoch')
    else:
        new_cross = pd.DataFrame()
        new_cross['epoch'] = group_crossings_on_dir(cross, Dt)
        new_cross['direction'] = new_cross['epoch'].count()*[0]
    return new_cross
"""

"""
Based on a list of crossings and a Dt (sliding window),
we define a crossings density based on the number of crossings in each window
Returns a list of crossings with a new columns

point_rate < Dt/4 pour etre sur de ne pas en rater

Résolution temporelle des chocs
"""
def crossings_density(y_timed, cross, Dt, points_rate):
    new_y = y_timed.copy()
    density = []
    for i in range(int(y_timed.count()[0]/points_rate)+1):
        mod = y_timed.count()[0]- points_rate*i - 1;
        if(mod>points_rate):
            curr_t = y_timed['epoch'].iloc[points_rate*i]
            n = cross.loc[(cross['epoch']>curr_t - Dt/2) & (cross['epoch']<curr_t + Dt/2)].count()[0]
            density.extend([n]*points_rate)
#            print(points_rate*i)
        else:
            curr_t = y_timed['epoch'].iloc[i+mod]
            n = cross.loc[(cross['epoch']>curr_t - Dt/2) & (cross['epoch']<curr_t + Dt/2)].count()[0]
            density.extend([n]*(1+mod))
            
    new_y['density'] = density
    return new_y

"""
Based on a density of crossings, returns a list of dates associated to the peaks and
a list of "degrees" of crossings (number of multiple crossings around the said date)
"""
def final_list(y_density):
    dates = []
    degrees = []
    to_consider = y_density.loc[y_density['density']>0]
    i=0
    n = to_consider.count()[0]
    while i<n-1:
        start_t = to_consider['epoch'].iloc[i]
        curr_t = to_consider['epoch'].iloc[i+1]
        j = 1
        interval_dates = []
        interval_degree = 1
        while(curr_t - start_t < 10) & (i+j<n):
            deg = to_consider['density'].iloc[i+j]
            interval_dates.extend([curr_t]*deg)
            start_t = curr_t
            curr_t = to_consider['epoch'].iloc[i+j]
            if deg>interval_degree:
                interval_degree = deg
            j+=1
        if len(interval_dates)>0:
            dates.append(sum(interval_dates)/len(interval_dates))
        else:
            dates.append(curr_t)
        degrees.append(interval_degree)
        i+=j
#        print(i)
    final = pd.DataFrame()
    final['epoch'] = dates
    final['degree'] = degrees
    return final

"""
Return a DataFrame based on crossings_timed with additional data from data_ref
Works only for crossings list
For variations or predictions, cf append_data_to_timed()
"""
def append_data_to_crossings(crossings_timed, data_ref, cols=None):
    new_crossings = crossings_timed.copy()
    if cols==None:
        cols = data_ref.columns
    
    new_epoch = []
    for k in range(len(new_crossings)):
        lines = data_ref.loc[abs(data_ref['epoch']-new_crossings['epoch'][k])<3]
        if(lines.count()[0]>1):
            new_epoch.append(lines['epoch'].iloc[0])
        else:
            new_epoch.append(lines['epoch'].iloc[0])
        
    new_crossings['epoch'] = new_epoch
    to_add = data_ref.loc[data_ref['epoch'].isin(new_crossings['epoch'])]
    for i in range(len(cols)):
        new_crossings[cols[i]] = np.array(to_add[cols[i]])
    return new_crossings    

"""
Similar to the previous one but for solar parameters

"""
def append_ext_to_crossings(crossings_timed, data_ref, cols=None):
    data_ref = data_ref.copy()
    new_crossings = crossings_timed.copy()
    if cols==None:
        cols = data_ref.columns
    
    epoch_to_keep = []
    for k in range(len(new_crossings)):
        lines = data_ref.loc[abs(data_ref['epoch']-new_crossings['epoch'][k])<24*3600]
        epoch_to_keep.append(lines.iloc[0]['epoch'])
    print(len(epoch_to_keep))
#    new_data = data_ref.loc[data_ref['epoch'].isin(epoch_to_keep)]
#    print(new_data.count()[0])
    for i in range(len(cols)):
        data_col = []
        for j in range(len(epoch_to_keep)):
            data = data_ref.loc[data_ref['epoch']==epoch_to_keep[j]][cols[i]]
            data_col.append(data.tolist())
#        print(data_col)
        new_crossings[cols[i]] = np.array(data_col)
    return new_crossings    

"""
Function to be used with graph_pred_from_var
Adds the crossings to the plots in blue vertical lines
"""
def show_crossings(cross):
    for i in range(cross.count()[0]):
        plt.axvline(x = pd.to_datetime(cross['epoch'].iloc[i], unit='s'), c='orange', linewidth=2)

def show_valid_crossings(cross, Dt):
    for i in range(cross.count()[0]):
        c = 'r'
        if abs(cross['dt_to_closest'].iloc[i])<Dt:
            c='g'
        plt.axvline(x = pd.to_datetime(cross['epoch'].iloc[i], unit='s'), c=c, linewidth=2)

    


            
#"""
#Takes a list of epochs identifying crossings and a Dt to merge crossings 
#Returns a DataFrame with the crossings, mean crossing time, max Dt between last and first crossings merged, and standard deviation
#"""
#def shocks_from_crossing(crossings, Dt):
#    shocks = pd.DataFrame()
#    same_shock = []
#    ep = []
#    shock_Dt = []
#    nvar = []
#    curr_t = crossings[0]
#    for i in range(0,len(crossings)-1):
#        same_shock.append(crossings[i])
#        if (crossings[i+1] - curr_t) > Dt:
#            t = sum(same_shock)/len(same_shock)
#            shock_len = crossings[i+1] - same_shock[0]            
#            ep.append(t)
#            shock_Dt.append(shock_len)
#            nvar.append(len(same_shock))
#            same_shock = [] 
#        curr_t = crossings[i+1]
#    shocks['epoch'] = ep
#    shocks['Dt'] = shock_Dt
#    shocks['var_number'] = nvar
#    return shocks
#
#def shocks_from_var(var,Dt):
#    return shocks_from_crossing(crossings_from_var(var),Dt)
#        
        
    

def new_data_from_var(var, epoch_list):
    y_timed = pd.DataFrame()
    y_timed['epoch'] = epoch_list
    labels = []
    
    first_lab = var['prec_class'].iloc[0]
    yf = y_timed.loc[y_timed['epoch'] <= var['epoch'].iloc[0]]
    labels.extend([first_lab]*yf.count()[0])
    
    for i in range(1,var.count()[0]):
        curr_var = var.iloc[i]
        prec_var = var.iloc[i - 1]
        curr_t = curr_var['epoch']
        prec_t = prec_var['epoch']
        Dt = abs(curr_t - prec_t)
        lab1 = prec_var['follow_class']
        lab2 = curr_var['prec_class']
        
        if lab1!=lab2:
            y_timed1 = y_timed.loc[y_timed['epoch'] > prec_t]
            y_timed1 = y_timed1.loc[y_timed1['epoch'] <= prec_t+ Dt/2]
            
            y_timed2 = y_timed.loc[y_timed['epoch'] <= curr_t]
            y_timed2 = y_timed2.loc[y_timed2['epoch'] >= curr_t - Dt/2]
            
            labels.extend([lab1]*y_timed1.count()[0] + [lab2]*y_timed2.count()[0])        
        else : 
            y_timed1 = y_timed.loc[y_timed['epoch']>prec_t]
            y_timed1 = y_timed1.loc[y_timed1['epoch']<=curr_t]
            labels.extend([lab1]*y_timed1.count()[0])
        
    
    last_lab = var['follow_class'].iloc[-1]
    yl = y_timed.loc[y_timed['epoch'] > var['epoch'].iloc[-1]]
    labels.extend([last_lab]*yl.count()[0])
    
    print(len(labels))
    
    y_timed['label'] = labels
    return y_timed
    
    
    
    
    
    