#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 02:03:26 2018

@author: thibault
"""
"""
Executable file to wrap up everything
"""
import numpy as np
import eval_functions as ev
import data_handling as dat
import mlp_model_functions as mdl
import matplotlib.pyplot as plt
import keras as ks
import pandas as pd
"""
First try : build a GUI to work with everything easily
"""
"""
Parameters declared as global variable useful at any moment during the algorithm steps:final_crossings = dat.get_closest_cross(true_crossings, final_crossings)
    ---Data and model---
    string data_path
    int nb_shocks
    pandas.DataFrame data
    
    ---Training---
    boolean run_from_file
    string model_load_path
    string model_save_path
    int[] layers_sizes
    string[] layers_activations
    int nb_epoch
    int batch_size
    int nb_class
    float test_size
    float val_size
    float dropout
    
    ---Evaluation---
    pandas.DataFrame ytrue
    pandas.DataFrame var_true
    pandas.DataFrame crossings_true
    pandas.DataFrame ypred
    pandas.DataFrame var_pred
    int Dt_var_corr
    int Dt_density
    pandas.DataFrame ycorr
    pandas.DataFrame var_corr
    pandas.DataFrame final_crossings
"""

global data_path
global data
global nb_shocks
global n_epochs
global batch_size
global test_size
global val_size
global run_from_file
global model_load_path
global model_save_path
global layers_sizes
global layers_activations
global dropout

import tkinter as tk
import tkinter.ttk as ttk
 
root = tk.Tk()
root.geometry('580x360')
root.title('ANN Shock detection')

window = ttk.Notebook(root)
tab_data = ttk.Frame(window) 
tab_eval = ttk.Frame(window)
tab_train = ttk.Frame(window) 
window.add(tab_data, text='Data loading') 
window.add(tab_train, text='Model training')
window.pack(expand=1, fill='both')
window.add(tab_eval, text='Post-process. and evaluation')
window.pack(expand=1, fill='both')

global_font = ('Courier', 12)


"""
DATA AND MODEL
"""
"""
File definition
"""
file_label = tk.Label(tab_data, text="Data file path:",font=global_font)
file_label.grid(column=0, row=0, sticky='e')

#from tkinter import filedialog
#file = filedialog.askopenfilename()

file_entry = tk.Entry(tab_data,width=40, font=('Courier',10))
file_entry.grid(column=1, row=0)
file_entry.focus()
file_entry.insert(0,'/home/thibault/Documents/stage_cesure/IRAP/TESTS_machine_learning/full_hall_dataset/hall_9000shocks_45min_labeled.txt')

"""
Number of shocks to consider
"""
nb_shocks_label = tk.Label(tab_data, text="Nb. of shocks:",font=global_font)
nb_shocks_label.grid(column=0, row=1, sticky='e')

default_shocks = tk.IntVar()
default_shocks.set(500)

nb_shocks_selection = tk.Spinbox(tab_data, from_=50, to=9500, width=5, increment = 50, textvariable=default_shocks)
nb_shocks_selection.grid(column=1,row=1, sticky='w')

"""
First shock to consider
"""
first_shock_label = tk.Label(tab_data, text="First shock index:",font=global_font)
first_shock_label.grid(column=0, row=2, sticky='e')

first_shock_selection = tk.Spinbox(tab_data, from_=0, to=9500, width=5, increment = 50)
first_shock_selection.grid(column=1,row=2, sticky='w')

"""
Getting all user parameters from the widgets
"""
def set_init_params():
    """
    Getting the parameters from the interface and loading the n first shocks of the data file
    Data is immediately nan-filled with the median
    """
    global data_path
    global data
    global nb_shocks
    global start_sh_index



    data_path = file_entry.get()
    nb_shocks = int(nb_shocks_selection.get())
    start_sh_index = int(first_shock_selection.get())
    data = dat.n_first_shocks(pd.read_csv(data_path), nb_shocks, start_sh_index)
    data = dat.fillna_by_class(data)
    data.sort_values('epoch')    
    
"""
Initialization button
"""
def clicked_init():
    set_init_params()
    print('Done initializing parameters and loading data')
    
init_button = tk.Button(tab_data, text="Load", width = 10, command=clicked_init)
init_button.grid(column=1, row=9)

"""
TRAINING
"""
"""
Load and save model path 
"""
loadModel_state = tk.BooleanVar()
loadModel_state.set(False) #set check state
loadmodel_chk = tk.Checkbutton(tab_train, text='Run from existing model', var=loadModel_state, font = global_font)
loadmodel_chk.grid(column=1, row=1, sticky = 'w')

loadmodel_label = tk.Label(tab_train, text="Model file to load:",font=global_font)
loadmodel_label.grid(column = 0, row = 0, sticky='e')
loadmodel_entry = tk.Entry(tab_train,width=40, font=('Courier',10))
loadmodel_entry.grid(column=1, row=0)

savemodel_label = tk.Label(tab_train, text="Save model to:",font=global_font)
savemodel_label.grid(column = 0, row = 2, sticky='e')
savemodel_entry = tk.Entry(tab_train,width=40, font=('Courier',10))
savemodel_entry.grid(column=1, row=2)
savemodel_entry.insert(0,'/home/thibault/Documents/stage_cesure/IRAP/models/yourmodel.h5')

"""
Architecture selection
"""
architecture_label = tk.Label(tab_train, text="Architecture:",font=global_font+tuple(["bold"]))
architecture_label.grid(column = 0, row = 4)

layers_label = tk.Label(tab_train, text="Layers:",font=global_font)
layers_label.grid(column = 0, row = 5, sticky='e')

layers_expl1 = tk.Label(tab_train, text="Format :",font=('Courier', 11, "italic"))
layers_expl1.grid(column = 0, row = 6, sticky='e')

layers_expl2 = tk.Label(tab_train, text="[n_input,n_hidden,...,n_output]",font=('Courier', 11, "italic"))
layers_expl2.grid(column = 1, row = 6)


layers_entry = tk.Entry(tab_train,width=30, font=global_font)
layers_entry.grid(column=1, row=5)
layers_entry.insert(0,'[11,3]')

"""
Training parameters label
"""
train_label = tk.Label(tab_train, text="Train param.:",font=global_font+tuple(["bold"]))
train_label.grid(column = 0, row = 7)

"""
Number of epochs of training
"""
epoch_label = tk.Label(tab_train, text="Epochs:",font=global_font)
epoch_label.grid(column=0, row=8, sticky='e')

default_epochs = tk.IntVar()
default_epochs.set(5)

epoch_selection = tk.Spinbox(tab_train, from_=1, to=100, width=5, textvariable = default_epochs)
epoch_selection.grid(column=1,row=8)

"""
Batch size for the training
"""
batch_size_label = tk.Label(tab_train, text="Batch size:",font=global_font)
batch_size_label.grid(column=0, row=9, sticky='e')

default_batch = tk.IntVar()
default_batch.set(256)

batch_size_selection = tk.Spinbox(tab_train, from_=50, to=1000, width=5, textvariable = default_batch)
batch_size_selection.grid(column=1,row=9)

"""
Test set proportion
"""
test_size_label = tk.Label(tab_train, text="Test size:",font=global_font)
test_size_label.grid(column=0, row=10, sticky='e')

default_test_size = tk.DoubleVar()
default_test_size.set(0.2)

test_size_selection = tk.Spinbox(tab_train, from_=0, to=1, width=5, increment=0.05, textvariable = default_test_size)
test_size_selection.grid(column=1,row=10)


"""
Validation set proportion
"""
val_size_label = tk.Label(tab_train, text="Validation size:",font=global_font)
val_size_label.grid(column=0, row=11, sticky='e')

val_size_selection = tk.Spinbox(tab_train, from_=0, to=1, width=5, increment=0.05)
val_size_selection.grid(column=1,row=11)

"""
Downsampling for ext. classes
"""
downsampl_label = tk.Label(tab_train, text="Downsampl. SW and evt.:",font=global_font)
downsampl_label.grid(column=0, row=12, sticky='e')

default_downsampl = tk.DoubleVar()
default_downsampl.set(1)

downsampl_selection = tk.Spinbox(tab_train, from_=0, to=1, width=5, increment=0.05, textvariable = default_downsampl)
downsampl_selection.grid(column=1,row=12)


"""
Dropout
"""
dropout_label = tk.Label(tab_train, text="Dropout:",font=global_font)
dropout_label.grid(column=0, row=13, sticky='e')

dropout_selection = tk.Spinbox(tab_train, from_=0, to=1, width=5, increment=0.05)
dropout_selection.grid(column=1,row=13)

"""
Number of classes
"""
nbclass_label = tk.Label(tab_train, text="Nb. class:",font=global_font)
nbclass_label.grid(column=0, row=14, sticky='e')

nbclass_selection = ttk.Combobox(tab_train, width = 5)
nbclass_selection['values']= (3, "4 - in progress")
nbclass_selection.current(0) #set the selected item

nbclass_selection.grid(column=1, row=14)

def set_training_params():
    global n_epochs
    global batch_size
    global test_size
    global val_size
    global downsampling
    global nb_class
    global dropout
    global run_from_file
    global model_load_path
    global model_save_path
    global layers_sizes
    global layers_activations
    
    n_epochs = int(epoch_selection.get())
    batch_size = int(batch_size_selection.get())
    test_size = float(test_size_selection.get())
    val_size = float(val_size_selection.get())
    downsampling = float(downsampl_selection.get())
    nb_class = int(nbclass_selection.get())
    dropout = float(dropout_selection.get())    
    run_from_file = loadModel_state.get()
    model_load_path = loadmodel_entry.get()
    model_save_path = savemodel_entry.get()
    layers_sizes = layers_entry.get()
    layers_sizes = list(map(int,layers_sizes[1:len(layers_sizes)-1].split(',')))
    layers_activations = ['relu']*(len(layers_sizes)-1)+['softmax']

def run_training():
    global ANN
    global timed_Xtest
    global timed_ytest
    """
    Running the training
    """
    
    timed_sets = dat.get_timed_train_test(data, ordered = True, nb_class = nb_class, test_size=test_size, downsampling_ext=downsampling)
    X_train, X_test, y_train, y_test = dat.get_train_test_sets(timed_sets[0],timed_sets[1],timed_sets[2],timed_sets[3])
    
    timed_Xtest = timed_sets[1]
    timed_ytest = timed_sets[3]
    timed_ytest['label'] = dat.one_hot_decode(dat.one_hot_encode(timed_ytest['label']))
    
    if run_from_file:
        ANN = mdl.load_model(model_load_path)
    else:
        ANN = mdl.create_model(layers_sizes, layers_activations,dropout=dropout)
#    print('\nNetwork architecture:')
#    print(ANN.summary(), '\n\n')
    history = mdl.compile_and_fit(ANN, X_train, y_train, n_epochs, batch_size, val_size=val_size)
    mdl.save_model(model_save_path, ANN)
    
    return ANN
       

def clicked_training():
    set_training_params()
    run_training()
    loadModel_state.set(True)
    if(len(loadmodel_entry.get())==0):
        loadmodel_entry.insert(0,savemodel_entry.get())
    print('Done training')

"""
Run button
"""
run_button = tk.Button(tab_train, text="Run training", width = 10, command=clicked_training)
run_button.grid(column=1, row=15)


"""
EVALUATION
"""

"""
Dt for prediction correction
"""
Dt_var_label = tk.Label(tab_eval, text="Correct. pred. Dt:",font=global_font)
Dt_var_label.grid(column=0, row=2, sticky='e')

default_Dt_var = tk.IntVar()
default_Dt_var.set(30)

Dt_var_selection = tk.Spinbox(tab_eval, from_=1, to=300, width=5, increment=10, textvariable=default_Dt_var)
Dt_var_selection.grid(column=1,row=2)

"""
Dt for density definition
"""
Dt_density_label = tk.Label(tab_eval, text="Density Dt:",font=global_font)
Dt_density_label.grid(column=0, row=6, sticky='e')

default_Dt_density = tk.IntVar()
default_Dt_density.set(300)

Dt_density_selection = tk.Spinbox(tab_eval, from_=30, to=1200, width=5, increment=30, textvariable=default_Dt_density)
Dt_density_selection.grid(column=1,row=6)

"""
Get prediction button
"""
def run_prediction():
    global timed_ypred
    global raw_proba
    timed_ypred = mdl.get_pred_timed(ANN, timed_Xtest, data.drop('label',axis=1))
    raw_proba = mdl.get_prob_timed(ANN, timed_Xtest, data.drop('label',axis=1))
    
    timed_ypred = dat.append_data_to_timed(timed_ypred, data, ['x', 'y', 'z', 'rho'])  
    raw_proba = dat.append_data_to_timed(raw_proba, data, ['x', 'y', 'z', 'rho']) 
    
    global true_variations
    global pred_variations
    
    pred_variations = dat.get_var(timed_ypred)
    true_variations = dat.get_var(timed_ytest)
    
    true_variations = dat.get_category(true_variations)
    pred_variations = dat.get_closest_var_by_cat(true_variations, dat.get_category(pred_variations))
    
    true_variations = dat.append_data_to_timed(true_variations, data, ['x', 'y', 'z', 'rho'])
    pred_variations = dat.append_data_to_timed(pred_variations, data, ['x', 'y', 'z', 'rho'])
    
    global true_crossings
    true_crossings = dat.crossings_from_var(true_variations)
    
def pred_clicked():
    run_prediction()
    print('Prediction received')
    
pred_label = tk.Label(tab_eval, text="Step 1:",font=global_font + tuple(['bold']))
pred_label.grid(column=0, row=0, sticky='e')

pred_button = tk.Button(tab_eval, text="Get prediction",width = 10, command = pred_clicked)
pred_button.grid(column=2, row=0)

"""
Prediction correction button
"""
def run_correction():
    global timed_ycorr
    global corr_variations
    global Dt_corr_pred
    global corr_crossings
    
    Dt_corr_pred = int(Dt_var_selection.get())
    timed_ycorr = dat.get_corrected_pred(pred_variations,timed_ypred, raw_proba, Dt_corr_pred)
    corr_variations = dat.get_var(timed_ycorr)
    corr_variations = dat.get_closest_var_by_cat(true_variations, dat.get_category(corr_variations))
    
    timed_ycorr = dat.append_data_to_timed(timed_ycorr, data, ['x', 'y', 'z', 'rho'])  
    corr_variations = dat.append_data_to_timed(corr_variations, data, ['x', 'y', 'z', 'rho'])
    
    corr_crossings = dat.crossings_from_var(corr_variations)
    corr_crossings = dat.get_closest_cross(true_crossings, corr_crossings)
    
def corr_clicked():
    run_correction()
    print('Correction applied')
   
corr_label = tk.Label(tab_eval, text="Step 2:",font=global_font + tuple(['bold']))
corr_label.grid(column=0, row=1, sticky='e')

corr_button = tk.Button(tab_eval, text="Get correction",width = 10, command = corr_clicked)
corr_button.grid(column=2, row=2)

"""
Density computing button
"""



def run_density():
    global Dt_density
    global timed_ycorr
    global final_crossings
    Dt_density = int(Dt_density_selection.get())
    
    timed_ycorr = dat.crossings_density(timed_ycorr, corr_crossings, Dt_density,int(Dt_density/10))
    final_crossings = dat.final_list(timed_ycorr)
    final_crossings = dat.get_closest_cross(true_crossings, final_crossings)
    
def density_clicked():
    run_density()
    print('Density and final table derived')
    
density_label = tk.Label(tab_eval, text="Step 3:",font=global_font + tuple(['bold']))
density_label.grid(column=0, row=5, sticky='e')   

density_button = tk.Button(tab_eval, text="Get cross. density",width = 10, command = density_clicked)
density_button.grid(column=2, row=6)


"""
All in one button
"""
def all_steps_clicked():
    run_prediction()
    print('Prediction received')
    run_correction()
    print('Correction applied')
    run_density()
    print('Density and final table derived')
    
eval_button = tk.Button(tab_eval, text="All steps",width = 10, command = all_steps_clicked)
eval_button.grid(column=2, row=11)

#"""
#Evaluation graphs
#Possible graphs: mvat, time distribution, rho_x
#"""
#graphs_label = tk.Label(tab_eval, text="Graphs:",font=global_font + tuple(['bold']))
#graphs_label.grid(column=3, row=0, sticky='e')   
#
#"""
#Initial pred labels
#"""
#def pred_lab_clicked():
#    ev.interact_graph_pred_from_var(true_variations, pred_variations, data_name = "Initial prediction")
#    plt.show()
#    
#pred_lab_button = tk.Button(tab_eval, text="Labels",width = 2, command = pred_lab_clicked)
#pred_lab_button.grid(column=4, row=0)
#
#"""
#Initial pred histogram
#"""
#def pred_hist_clicked():
#    ev.graph_hist_by_cat(pred_variations, "Initial prediction")
#    plt.show()
#
#pred_hist_button = tk.Button(tab_eval, text="Hist.",width = 2, command = pred_hist_clicked)
#pred_hist_button.grid(column=5, row=0)
#
#"""
#Initial pred rho_x
#"""
#def pred_rhox_clicked():
#    ev.graph_rho_x(pred_variations, "Initial prediction")
#    
#pred_rhox_button = tk.Button(tab_eval, text="rho - x",width = 2, command = pred_rhox_clicked)
#pred_rhox_button.grid(column=6, row=0)

root.mainloop()
    



    
    













