#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:45:44 2018

@author: thibault
"""

"""
Executable file with the parameters hard-coded
"""
import numpy as np
import matplotlib.pyplot as plt
import eval_functions as ev
import data_handling as dat
import mlp_model_functions as mdl
import pandas as pd

import sys

nb_sys_arg = len(sys.argv)
val_sys_arg = np.array(sys.argv)

exec_file_name = val_sys_arg[0]
"""
Expected input command line : 
TO DO
"""

"""
Data parameters
"""
#For the complete Hall dataset:
data_path = '/home/thibault/Documents/stage_cesure/IRAP/TESTS_machine_learning/full_hall_dataset/indexed_dataset.txt'
#For the reduced dataset:
data_path_red = '/home/thibault/Documents/stage_cesure/IRAP/TESTS_machine_learning/reduced_dataset/full_reduced_dataset.txt'

"""
Additional data for evaluation
"""
#List of central values for each shock in Hall dataset (1 row per shock)
center_val_sh = pd.read_csv('./indexed_dataset_central_values.txt')
#External parameters file
external_params = pd.read_csv('/home/thibault/Documents/stage_cesure/IRAP/TESTS_machine_learning/full_hall_dataset/ext_params.txt')
"""
Initialization
"""
nb_shocks = 30 #number of shocks to load from the dataset
first_shock = 0 #index of the first shocks of the list when loading in order

nb_class = 3
nb_features = 11

"""
Model training parameters
Defined as global variables before running the functions below
"""
run_from_existing = False
model_load_path = '/home/thibault/Documents/stage_cesure/IRAP/models/model_trainall.h5'
model_save_path = '/home/thibault/Documents/stage_cesure/IRAP/models/model_trainall_reduced.h5'

#layers_sizes = [nb_features,44,22,11,3]
#layers_activations = ['relu','relu','relu','relu','softmax']

layers_sizes = [nb_features,66,22,3]
layers_activations = ['relu','relu','relu','softmax']

n_epochs = 20
batch_size = 256
val_size = 0
test_size = 0.2
downsampling = 0 #proportion of external classes points to keep in the dataset
dropout = 0

"""
Post-processing and evaluation parameters
"""
dt_corr_pred = 60
dt_density = 600

"""
For hyperparameter optimization:
    Define a variation range for every parameter that could be considered as a variable 
    hyperparameter that should be optimized
"""
network_depth_range = [3,8]
network_width_range = [3,121]

n_epochs_range = [10,11]
batch_size_range = [50,500]

dropout_range = [0,0.25]

dt_corr_range = [10,120]

"""
Gets a random value for each hyperparameter in their ranges (uniform distrib.)
"""
def randomize_hparam():
    network_depth = np.random.randint(network_depth_range[0], network_depth_range[1])
    network_width = np.random.randint(network_width_range[0], network_width_range[1])
    n_epochs = np.random.randint(n_epochs_range[0], n_epochs_range[1])
    batch_size = np.random.randint(batch_size_range[0], batch_size_range[1])
    dropout = np.random.random()*dropout_range[1]
    dt_corr = np.random.randint(dt_corr_range[0],dt_corr_range[1])
    
    layers_sizes = [11]
    prev_n_neuron = 11
    for k in range(network_depth-3):
        curr_n_neuron = np.random.randint(3,network_width)
        if k>0:
            curr_n_neuron = np.random.randint(3,prev_n_neuron+1)
        layers_sizes.append(curr_n_neuron)
        prev_n_neuron = curr_n_neuron
    layers_sizes.append(nb_class)
    layers_activations = ['relu']*(len(layers_sizes)-1) + ['softmax']
    
    return layers_sizes, layers_activations, n_epochs, batch_size, dropout, dt_corr    

"""
Defines a random neural network based on the random hyper parameters
"""
random_net = False
if random_net : 
    layers_sizes, layers_activations, n_epochs, batch_size, dropout, dt_corr = randomize_hparam()  

#"""
#Getting the data
#"""
data1 = dat.shock_by_index(pd.read_csv(data_path),0,1000)


#data = pd.concat(dat.shock_by_index(data,0,250),dat.shock_by_index(data,2000,2250),dat.shock_by_index(data,4000,3000),dat.shock_by_index(data,0,3000))

#data2 = pd.read_csv(data_path_red)

data1 = data1.drop('shock_ind', axis=1)

#if nb_class == 3:
#    data = data.replace('EV',0)
#    data = data.replace('S',1)
#    data = data.replace('SW',2)
#    
#if nb_class == 5:
#    data = data.replace('I',0)
#    data = data.replace('MPB',1)
#    data = data.replace('M',2)
#    data = data.replace('S',3)
#    data = data.replace('SW',4)
    
#data = pd.concat([data1, data2])
data = data1

data = dat.fillna_by_class(data)
data = data.sort_values('epoch')


"""
Training the network
"""
def run_training(layers_sizes, layers_activations, n_epochs, batch_size, val_size,test_size,downsampling,test_set_index=0, ordered=True):
    timed_sets = dat.get_timed_train_test(data, ordered = ordered, nb_class = nb_class, test_size=test_size, downsampling_ext=downsampling, start_index = test_set_index)
    X_train, X_test, y_train, y_test = dat.get_train_test_sets(timed_sets[0],timed_sets[1],timed_sets[2],timed_sets[3])
    
    timed_Xtest = timed_sets[1]
    timed_ytest = timed_sets[3]
    timed_ytest['label'] = dat.one_hot_decode(dat.one_hot_encode(timed_ytest['label']))
#    timed_ytest = dat.append_data_to_timed(timed_ytest, data, ['x', 'y', 'z', 'rho'])

    if run_from_existing:
        ANN = mdl.load_model(model_load_path)
    else:
        ANN = mdl.create_model(layers_sizes, layers_activations,dropout=dropout)
#    print('\nNetwork architecture:')
#    print(ANN.summary(), '\n\n')
    history = mdl.compile_and_fit(ANN, X_train, y_train, n_epochs, batch_size, val_size=val_size)
    mdl.save_model(model_save_path, ANN)
    
    return ANN, timed_Xtest, timed_ytest

"""
Get raw predictions from the network
"""
def get_prediction(ANN, timed_Xtest, timed_ytest):
#    timed_ypred = mdl.get_pred_timed(ANN, timed_Xtest, data.drop(['label','shock_ind'],axis=1))
    timed_ypred = mdl.get_pred_timed(ANN, timed_Xtest, data.drop(['label'],axis=1))

#    raw_proba = mdl.get_prob_timed(ANN, timed_Xtest, data.drop(['label','shock_ind'],axis=1))
    raw_proba = mdl.get_prob_timed(ANN, timed_Xtest, data.drop(['label'],axis=1))
    
#    timed_ypred = dat.append_data_to_timed(timed_ypred, data, ['x', 'y', 'z', 'rho'])  
#    raw_proba = dat.append_data_to_timed(raw_proba, data, ['x', 'y', 'z', 'rho']) 

    #variations
    pred_variations = dat.get_var(timed_ypred)
    true_variations = dat.get_var(timed_ytest)

    true_variations = dat.get_category(true_variations)
    pred_variations = dat.get_closest_var_by_cat(true_variations, dat.get_category(pred_variations))
#    
#    true_variations = dat.append_data_to_timed(true_variations, data, ['x', 'y', 'z', 'rho'])
#    pred_variations = dat.append_data_to_timed(pred_variations, data, ['x', 'y', 'z', 'rho'])
    
    #crossings reference
    true_crossings = dat.crossings_from_var(true_variations)
    
    return timed_ypred, raw_proba, true_variations, pred_variations, true_crossings

"""
Get the corrected prediction
"""
def get_corrected_prediction(timed_ypred, raw_proba, true_variations, pred_variations):
    timed_ycorr = dat.get_corrected_pred(pred_variations, timed_ypred,raw_proba, dt_corr_pred)
    corr_variations = dat.get_var(timed_ycorr)
    corr_variations = dat.get_closest_var_by_cat(true_variations, dat.get_category(corr_variations))
    
#    timed_ycorr = dat.append_data_to_timed(timed_ycorr, data, ['x', 'y', 'z', 'rho'])  
#    corr_variations = dat.append_data_to_timed(corr_variations, data, ['x', 'y', 'z', 'rho'])
    
    corr_crossings = dat.crossings_from_var(corr_variations)
    return timed_ycorr, corr_variations, corr_crossings

#timed_ycorr, corr_variations, corr_crossings = get_corrected_prediction(timed_ypred, raw_proba, true_variations, pred_variations)

"""
Run the last postprocess steps to get crossings dates and density
"""
def postprocess_crossings_list(timed_ycorr, corr_crossings):
    timed_ycorr = dat.crossings_density(timed_ycorr, corr_crossings, dt_density)
    final_crossings = dat.final_list(timed_ycorr)
#    final_crossings = dat.append_data_to_crossings(final_crossings, data, ['x', 'y', 'z', 'rho'])
    return timed_ycorr, final_crossings

"""
Running a k-fold validation with the defined parameters
Evaluating the network for the following metrics:
    - initial network f-measure
    - shocks identification f-measure at Dt = 300s
    - shocks identification f-measure at Dt = 600s
    - ratio nb. predicted variations / true variations
    - loss (jaccard)
Returns a DataFrame with the metrics
"""

def network_k_fold_validation(k, invert=False):
    test_size = 1/k
    if invert:
        test_size = 1-test_size
    
    #metrics arrays
    conf_matrices = []
    sw_f = []
    ev_f = []
    shock_acc = []
    shock_rec = []
    shock_f = []
    
    for i in range(k):
        print(i+1 ,'/',k,' folds')
        #runs a complete training and test for the i-th fold
        start_index_i = int(data.count()[0]*i/k)
        ANN, timed_Xtest, timed_ytest = run_training(layers_sizes, layers_activations,n_epochs, batch_size,0, test_size,downsampling,test_set_index = start_index_i, ordered=True)
        timed_ypred, raw_proba, true_variations, pred_variations, true_crossings = get_prediction(ANN, timed_Xtest, timed_ytest)
        
        #compute the evaluation metrics
        conf_m, conf_m_norm = ev.get_confusion_matrices(timed_ytest['label'], timed_ypred['label'])
        acc = ev.accuracy_from_cm(conf_m)
        rec = ev.recall_from_cm(conf_m)
        f = ev.f_measure_from_cm(conf_m)
        
        #Stores the metrics values
        conf_matrices.append(conf_m_norm)
        sw_f.append(f[2])
        ev_f.append(f[0])
        shock_acc.append(acc[1])
        shock_rec.append(rec[1])
        shock_f.append(f[1])
        
    #Builds a dataframe
    results = pd.DataFrame()
    results['SW_class_f'] = sw_f
    results['EV_class_f'] = ev_f
    results['SH_class_acc'] = shock_acc
    results['SH_class_rec'] = shock_rec
    results['SH_class_f'] = shock_f
        
        #plot
#        ev.graph_pred_from_var(true_variations, pred_variations, data_name = 'Fold n = ' + str(i))
    
    return conf_matrices, results

def global_k_fold_validation(k):
    test_size = 1/k

    #metrics arrays
    conf_matrices = []
    sw_f = []
    ev_f = []
    shock_acc = []
    shock_rec = []
    shock_f = []
    final_300s_acc = []
    final_600s_acc = []
    final_300s_rec = []
    final_600s_rec = []
    
    for i in range(k):
        print(i+1 ,'/',k,' folds')
        #runs a complete training and test for the i-th fold
        start_index_i = int(data.count()[0]*i/k)
        ANN, timed_Xtest, timed_ytest = run_training(layers_sizes, layers_activations,n_epochs, batch_size,0, test_size,downsampling,test_set_index = start_index_i)
        timed_ypred, raw_proba, true_variations, pred_variations, true_crossings = get_prediction(ANN, timed_Xtest, timed_ytest)

        timed_ycorr, corr_variations, corr_crossings = get_corrected_prediction(timed_ypred, raw_proba, true_variations, pred_variations)
        timed_ycorr, final_crossings = postprocess_crossings_list(timed_ycorr, corr_crossings)

        #compute the evaluation metrics
        conf_m, conf_m_norm = ev.get_confusion_matrices(timed_ytest['label'], timed_ypred['label'])
        acc = ev.accuracy_from_cm(conf_m)
        rec = ev.recall_from_cm(conf_m)
        f = ev.f_measure_from_cm(conf_m)
        
        #Stores the metrics values
        conf_matrices.append(conf_m_norm)
        sw_f.append(f[2])
        ev_f.append(f[0])
        shock_acc.append(acc[1])
        shock_rec.append(rec[1])
        shock_f.append(f[1])
        final_300s_acc.append(ev.acc_from_crossings(true_crossings, final_crossings, 300))
        final_300s_rec.append(ev.rec_from_crossings(true_crossings, final_crossings, 300))
        final_600s_acc.append(ev.acc_from_crossings(true_crossings, final_crossings, 600))
        final_600s_rec.append(ev.rec_from_crossings(true_crossings, final_crossings, 600))
        
    #Builds a dataframe
    results = pd.DataFrame()
    results['SW_class_f'] = sw_f
    results['EV_class_f'] = ev_f
    results['SH_class_acc'] = shock_acc
    results['SH_class_rec'] = shock_rec
    results['SH_class_f'] = shock_f
    results['final_list_acc_300'] = final_300s_acc
    results['final_list_rec_300'] = final_300s_rec
    results['final_list_acc_600'] = final_600s_acc
    results['final_list_rec_600'] = final_600s_rec
    
    return conf_matrices, results
        

"""
Random searching n different random networks with a k_fold cross-validation and 
keeping track of the performances
Returns a DataFrame 
"""
def network_random_search(n, k_fold):
    #Global param to modify
    global layers_sizes
    global layers_activations
    global n_epochs
    global batch_size
    global dropout
    global dt_corr
    
    #Parameters
    arch_list = []
    batch_list = []
    dropout_list = []
#    dt_corr_list = []
    
    #Metrics
    
    
    for i in range(n):
        print('Random net ',i+1,'/',n,' starting')
        layers_sizes, layers_activations, n_epochs, batch_size, dropout, dt_corr = randomize_hparam()  
        n_epochs = 10
        arch_list.append(layers_sizes)
        batch_list.append(batch_size)
        dropout_list.append(dropout)
#        dt_corr_list.append(dt_corr)
        
        conf_mat, cross_val_results = network_k_fold_validation(k_fold)
        
        if(i==0):
            results_mean = pd.DataFrame(columns = cross_val_results.columns)
            results_std = pd.DataFrame(columns = cross_val_results.columns)
        results_mean = results_mean.append(cross_val_results.mean(), ignore_index=True)
        results_std = results_std.append(cross_val_results.std(), ignore_index=True)
        
        
    
    return arch_list, batch_list, dropout_list, results_mean, results_std

def global_random_search(n, k_fold):
        #Global param to modify
    global layers_sizes
    global layers_activations
    global n_epochs
    global batch_size
    global dropout
    global dt_corr
    
    #Parameters
    arch_list = []
    batch_list = []
    dropout_list = []
    dt_corr_list = []
    
    #Metrics
    
    
    for i in range(n):
        print('Random net ',i+1,'/',n,' starting')
        layers_sizes, layers_activations, n_epochs, batch_size, dropout, dt_corr = randomize_hparam()  
        n_epochs = 10
        arch_list.append(layers_sizes)
        batch_list.append(batch_size)
        dropout_list.append(dropout)
        dt_corr_list.append(dt_corr)
        
        conf_mat, cross_val_results = global_k_fold_validation(k_fold)
        
        if(i==0):
            results_mean = pd.DataFrame(columns = cross_val_results.columns)
            results_std = pd.DataFrame(columns = cross_val_results.columns)
        results_mean = results_mean.append(cross_val_results.mean(), ignore_index=True)
        results_std = results_std.append(cross_val_results.std(), ignore_index=True)
        
        
    
    return arch_list, batch_list, dropout_list, dt_corr_list, results_mean, results_std

"""
Plots the recall and precision based on a k-fold or random search results
Param:
    DataFrame results
    string rec_col : name of recall column
    string acc_col : name of accuracy column
"""
def plot_recall_acc(results, rec_col, acc_col, label=None):
    plt.grid(True, linestyle='--')
    plt.rcParams.update({'font.size': 15})
    plt.xticks([0.7+i*0.05 for i in range(9)])
    plt.yticks([0.7+i*0.05 for i in range(9)])
    plt.scatter(results[rec_col], results[acc_col], label = label)
    plt.xlim(0.7,1)
    plt.ylim(0.7,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
"""
Running a complete training and test
"""
## =============================================================================
#ANN, timed_Xtest, timed_ytest = run_training(layers_sizes, layers_activations,n_epochs, batch_size, val_size, test_size,downsampling, test_set_index = 2500000)
#timed_ypred, raw_proba, true_variations, pred_variations, true_crossings = get_prediction(ANN, timed_Xtest, timed_ytest)
#timed_ycorr, corr_variations, corr_crossings = get_corrected_prediction(timed_ypred, raw_proba, true_variations, pred_variations)
#timed_ycorr, final_crossings = postprocess_crossings_list(timed_ycorr, corr_crossings)
## =============================================================================

"""
Visualizing with solar parameters
"""
#ext = pd.read_csv('/home/thibault/Documents/stage_cesure/IRAP/TESTS_machine_learning/full_hall_dataset/ext_params.txt')
#ext = ext.fillna(0)
#timed_ytrue = dat.append_ext_to_crossings(timed_ypred, ext, ['euv_flux','pdyn'])
#true_crossings = dat.append_ext_to_crossings(true_crossings, ext, ['euv_flux','pdyn'])
#corr_crossings = dat.append_ext_to_crossings(corr_crossings, ext, ['euv_flux','pdyn'])
#final_crossings = dat.append_ext_to_crossings(final_crossings, ext, ['euv_flux','pdyn'])


"""
A PLACER AILLEURS?
Evaluates a defined model on a dataset composed of n shocks with the specified param range
Needs data with a column 'shock_ind'

Mode:
    - 'ALL' : all steps
    - 'INIT_PRED' : only initial prediciton
    - 'VAR' : class variations after correction
    - 'CROSS' : final crossings list
"""

# =============================================================================
# A NOTER ici
# On considère trop de points puisqu'on prend tous les chocs ou au moins un 
# des points est dans la bonne plage de variations, putôt que seulement
# le point central
# A MODIFIER 
# =============================================================================
def evalModelOnParamBin(ANN_model, data, param_name, min_val, max_val, mode='INIT_PRED'):   
    test_data = data
    
    y_timed = dat.get_timed_inputs(test_data, data)[1]
    y_label = dat.one_hot_decode(dat.one_hot_encode(y_timed['label'].tolist()))
    y_timed['label'] = y_label
    
    X_timed = dat.get_timed_train_test(test_data.drop('shock_ind', axis=1), ordered=True, test_size = 0, start_index = 0)[0]
    
    
    timed_ypred, raw_proba, true_variations, pred_variations, true_crossings = get_prediction(ANN_model, X_timed, y_timed)
    
    if mode=='INIT_PRED':
        return y_timed, timed_ypred
    
    timed_ycorr, corr_variations, corr_crossings = get_corrected_prediction(timed_ypred, raw_proba, true_variations, pred_variations)
    if mode=='VAR':
        return true_variations, corr_variations
    
    timed_ycorr, final_crossings = postprocess_crossings_list(timed_ycorr, corr_crossings)
    if mode=='CROSS':
        return true_crossings, final_crossings
    
    return y_timed, timed_ycorr, true_variations, corr_variations, true_crossings, corr_crossings, final_crossings

"""
Evaluates a model on a dataset of n shocks, by creating k subsets with the same number of shocks
based on the value of param_name

Mode: 
    - 'ALL' : all steps
    - 'INIT_PRED' : only initial prediciton
    - 'VAR' : class variations after correction
    - 'CROSS' : final crossings list
"""
def evalModelOnParam(ANN_model, data, param_name, n_bins, mode='INIT_PRED'):
    center_val = center_val_sh.loc[center_val_sh['shock_ind'].isin(data['shock_ind'])]
    center_val = center_val.sort_values(param_name)
    cut_index = int(center_val.count().max()/n_bins)
    cut_values = [center_val[param_name].iloc[0]]
    for i in range(n_bins-1):
         cut_values.append(center_val[param_name].iloc[(i+1)*cut_index])
    cut_values.append(center_val[param_name].iloc[-1])
    
    true_perbin = []
    pred_perbin = []
    for k in range(n_bins):
        print('\nBin '+str(k+1))
        ind = center_val.loc[(center_val[param_name]>cut_values[k]) & (center_val[param_name]<cut_values[k+1])]['shock_ind']
        data_loc = data.loc[data['shock_ind'].isin(ind)]
        if(mode!='ALL'):
            true, pred = evalModelOnParamBin(ANN_model, data_loc, param_name, cut_values[k], cut_values[k+1], mode=mode)
            true_perbin.append(true)
            pred_perbin.append(pred)
        else:
            return
        
    if(mode=='CROSS'):
        fig, ax = plt.subplots(1,n_bins)
        for i in range(n_bins):
            plt.axes(ax[i])
            ev.plot_cross_stats(true_perbin[i],pred_perbin[i] )
            ax[i].set_title('Range '+ param_name + ': [' + str(cut_values[i].round(2))+', '+str(cut_values[i+1].round(2))+']')
            
    return cut_values, true_perbin, pred_perbin       














