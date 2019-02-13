#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:20:12 2018

@author: thibault
"""

"""
Wraper to get predictions from a set of unseen data
"""

import mlp_model_functions as mdl
import pandas as pd
import data_handling as dat

"""
Parameters:
    model (.h5 file)
    scaling data?
    data : pandas.DataFrame
    Dt_pred_corr : float
    Dt_dens_comp : float    
"""

#model_path = '/home/thibault/Documents/stage_cesure/IRAP/models/BigTrain.h5'
#ANN = mdl.load_model(model_path)
#Dt_pred_corr = 60
#Dt_density = 600

#define data here
#test_df = test_df.fillna(data.median())
#ANN = mdl.load_model(model_path)
#
#init_pred = mdl.get_pred_timed(ANN, test_df, data.drop(['label'], axis=1))
#proba = mdl.get_prob_timed(ANN, test_df, data.drop(['label'], axis=1))
#
#corr_pred = dat.get_corrected_pred(init_pred, proba, Dt_pred_corr)
#vcorr = dat.get_category(dat.get_var(corr_pred))
#corr_crossings = dat.crossings_from_var(vcorr)
#
#corr_pred = dat.crossings_density(corr_pred, corr_crossings, Dt_density,int(Dt_density/10))
#final_crossings = dat.final_list(corr_pred)

def pred_from_unseen(model, unseen_data, scale_data, dt_corr, dt_density):
    unseen_data = unseen_data.fillna(scale_data.median())
    
    init_pred = mdl.get_pred_timed(model, unseen_data, scale_data)
    proba = mdl.get_prob_timed(model, unseen_data, scale_data)
    
    init_var = dat.get_var(init_pred)
    init_var = dat.get_category(init_var)
    
    corr_pred = dat.get_corrected_pred(init_var, init_pred, proba, dt_corr)
    vcorr = dat.get_category(dat.get_var(corr_pred))
    vcorr = dat.corrected_var(vcorr, 15) #deletes variations faster than 15s
    corr_crossings = dat.crossings_from_var(vcorr)

    corr_pred = dat.crossings_density(corr_pred, corr_crossings, dt_density,int(dt_density/10))
    final_crossings = dat.final_list(corr_pred)
    return corr_pred, vcorr, final_crossings
    