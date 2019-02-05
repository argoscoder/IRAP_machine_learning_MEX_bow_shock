#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:59:42 2018

@author: thibault
"""

"""
Dedicated to reading AMDA data directly with webservice requests
"""
import urllib
import io
import numpy as np
import pandas as pd
import mlp_model_functions as mdl
import data_handling as dat
from pred_from_unseen import pred_from_unseen

import sys

nb_sys_arg = len(sys.argv)
val_sys_arg = np.array(sys.argv)

exec_file_name = val_sys_arg[0]
"""
Expected input command line : 
    with 4 args
    $ python data_from_AMDA.py model_file_path start_time end_time scale_data_path
    
    with 2 args - uses default model and data
    $ python data_from_AMDA.py start_time end_time 
"""

#token_url = 'http://amda.irap.omp.eu/php/rest/auth.php'
#get_param_url = 'http://amda.irap.omp.eu/php/rest/getParameter.php'


token_url = 'http://amda.irap.omp.eu/php/rest/auth.php'

#param_url for old amda
#get_param_url = 'http://amda-old.irap.omp.eu/php/rest/getParameter.php'
#param url for new amda
get_param_url = 'http://amda.irap.omp.eu/php/rest/getParameter.php'

"""
Username and password
"""
username = 'noel'
password = 'nirapass'

"""
Default start and end times
"""
start_time = '2008-01-03T05:30:00'
end_time = '2008-01-03T09:30:00'

#Default scaling data : reduced dataset
#A CHANGER mais set de Hall trop grand
scale_data_path = '/home/thibault/Documents/stage_cesure/IRAP/TESTS_machine_learning/reduced_dataset/full_reduced_dataset.txt'


#Default model : not fully optimized yet
model_path = '/home/thibault/Documents/stage_cesure/IRAP/models/model_trainall.h5'

#if arguments provided via command line
if(nb_sys_arg > 1 ):
    if(nb_sys_arg==3):
        start_time = val_sys_arg[1]
        end_time = val_sys_arg[2]
    elif(nb_sys_arg==5):
        model_path = val_sys_arg[1]
        start_time = val_sys_arg[2]
        end_time = val_sys_arg[3]
        scale_data_path = val_sys_arg[4]
    
    print('Prediction between ', start_time, ' and ', end_time )
    
    

IMA_qualityID = 'mex_h_qual'
#paramID = 'ws_totels_1'
#paramID2 = 'imf'
sampling=4

"""
MEX parameters
"""
#mex_params = ['mex_xyz','mex_h_dens','mex_h_vel','ws_totels_1','ws_totels_6','ws_totels_8','ws_rho_mex']
#mex_df_cols = ['epoch', 'x', 'y', 'z', 'density_IMA', 'vx_IMA', 'vy_IMA', 'vz_IMA','totels_1', 'totels_6', 'totels_8', 'rho']

"""
Returns a valid token to connect to AMDA
Uses AMDA web service
"""
def get_token():
    response = urllib.request.urlopen(token_url)
    token = response.read()
    encoding = response.headers.get_content_charset('utf-8')
    decoded_token = token.decode(encoding)
#    print(decoded_token)
    return decoded_token
"""
Builds the 'command' url to get a specific parameter between start time and end time
"""   
def buildParamURL(start_time, end_time, paramID, token):
    get_url = get_param_url + '?' + 'startTime='+start_time + '&stopTime='+end_time +'&parameterID='+paramID+'&token='+token+'&sampling='+str(sampling)+'&userID='+username+'&password='+password
    return get_url

"""
Very general, returns the response of a URL as a string
"""
def get_string_response(url):
    response = urllib.request.urlopen(url)
    encoded = response.read()
    encoding = response.headers.get_content_charset('utf-8')
    decoded = encoded.decode(encoding)
#    print(decoded)
    return decoded

"""
Returns the URL of the file for the specified parameters
"""
def get_file_URL(start_time, end_time, paramID, token, amda_old = False):
    param_url = buildParamURL(start_time, end_time, paramID, token)
    resp = get_string_response(param_url)
#    print(resp)
    if amda_old:
        return resp
    file_url = resp.split('"')[-2]
    file_url = file_url.split('\\')
    file_url = ''.join(file_url)
#    print(file_url)
    return file_url

"""
Returns a DataFrame from a string representing an entire .txt file
""" 
def get_df_from_string(file_str):
    file_str = io.StringIO(file_str)
    return pd.read_csv(file_str, comment='#', sep='\s+')

"""
Wraps up all the previous steps to download directly a DataFrame from user defined parameters
Uses AMDA web service
"""
def download_single_df(start_time, end_time, paramID, amda_old=False, columns=None):
    token = get_token()
    file_url = get_file_URL(start_time, end_time, paramID, token, amda_old=amda_old)
    file_str = get_string_response(file_url)
    df = get_df_from_string(file_str)

    if columns!=None:
        df.columns = columns
#    else:
#        df.columns = ['date']+[(paramID + '_'+ str(i)) for i in range(df.count(axis=1).max()-1)]
    return df


"""
Returns a dataframe of parameters in param_list between start_time and end_time
Uses AMDA web service
"""
def download_multiparam_df(start_time, end_time, param_list, amda_old =False, columns=None):
    dfs = []
    col_index = 0
    for i in range(len(param_list)):
        print(param_list[i] +' loading...')
        df = download_single_df(start_time, end_time, param_list[i], amda_old = amda_old)
        if i>0:
            df = df.iloc[:,1:]
        df_dim = df.count(axis=1)[0] + df.isna().sum(axis=1)[0]
        df.columns = columns[col_index:col_index+df_dim]
        col_index = col_index + df_dim
        dfs.append(df)
    complete = dfs[0].join(dfs[1:])
#    complete['date'] = (pd.to_datetime(complete['date']) - pd.to_datetime('1970-01-01 00:00:00'))/pd.Timedelta('1s')
    if columns!=None:
        complete.columns = columns
    return complete

"""
Returns a dataframe of custom solar parameters between start_time and end_time
Problem-specific
Uses AMDA web service
"""
def external_param(start_time, end_time, amda_old = False):
    ext_param = ['ws_13', 'mars_sw_pdyn']
    ext_col_names = ['epoch','euv_flux','pdyn']
    
    curr_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)    

    data = download_multiparam_df(timestamp_to_AMDAdate(curr_time),timestamp_to_AMDAdate(end_time),ext_param, amda_old=amda_old, columns = ext_col_names)
    data['epoch'] = (pd.to_datetime(data['epoch']) - pd.to_datetime('1970-01-01 00:00:00'))/pd.Timedelta('1s')
    return data

"""
Returns a dataframe of AMDA-defined features for MEX
Problem-specific
Uses AMDA web service
"""
def mex_data_from_AMDA(start_time, end_time, amda_old=True):
    #MEX params in old AMDA
#    mex_params = ['mex_xyz','mex_h_dens','mex_h_vel','ws_totels_1','ws_totels_6','ws_totels_8','ws_rho_mex','mex_h_qual']
    #MEX params in new AMDA
    #to look for in your workspace parameters, these are the paramID from AMDA
    mex_params = ['mex_xyz_mso','mex_h_dens','mex_h_vel','ws_0','ws_1','ws_12','ws_4','mex_h_qual']
    #Columns names
    mex_df_cols = ['epoch', 'x', 'y', 'z', 'density_IMA', 'vx_IMA', 'vy_IMA', 'vz_IMA','totels_1', 'totels_6', 'totels_8', 'rho', 'IMA_flag']
    
    curr_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    data = download_multiparam_df(timestamp_to_AMDAdate(curr_time),timestamp_to_AMDAdate(end_time),mex_params, amda_old=amda_old, columns = mex_df_cols)
    data['epoch'] = (pd.to_datetime(data['epoch']) - pd.to_datetime('1970-01-01 00:00:00'))/pd.Timedelta('1s')
    
    return data
    
"""
Uses a model to predict the variations and shock crossings between start_time and end_time
scale_data : representative data (reduced dataset for example, possible with array of tailored values)
dt_corr : for correction
dt_dens : for post processing
time_window : duration of the time window for each download of the data 

For each window, the data where ELS (totels1) is missing is dropped
A CHANGER surement
"""
def mex_pred_from_AMDA(model,scale_data,dt_corr, dt_dens, start_time, end_time, time_window, amda_old=True):
    #MEX params in old AMDA
#    mex_params = ['mex_xyz','mex_h_dens','mex_h_vel','ws_totels_1','ws_totels_6','ws_totels_8','ws_rho_mex','mex_h_qual']
    #MEX params in new AMDA
    mex_params = ['mex_xyz_mso','mex_h_dens','mex_h_vel','ws_0','ws_1','ws_12','ws_4','mex_h_qual']
    mex_df_cols = ['epoch', 'x', 'y', 'z', 'density_IMA', 'vx_IMA', 'vy_IMA', 'vz_IMA','totels_1', 'totels_6', 'totels_8', 'rho', 'IMA_flag']
    
    curr_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    all_label = []
    all_var = []
    all_crossings = []
    i=0
    while curr_time<end_time :
        next_time = min(curr_time + pd.to_timedelta(time_window, 's'), end_time)
        curr_data = download_multiparam_df(timestamp_to_AMDAdate(curr_time),timestamp_to_AMDAdate(next_time),mex_params, amda_old=amda_old, columns = mex_df_cols)
        curr_data['epoch'] = (pd.to_datetime(curr_data['epoch']) - pd.to_datetime('1970-01-01 00:00:00'))/pd.Timedelta('1s')
        #AJOUTER ICI
        # if (critere de validité des données):
        
        curr_data = curr_data.loc[~curr_data['totels_1'].isna()]
        
        if curr_data.count().max() > 0:
            
            label_pred, var_pred, cross_pred = pred_from_unseen(model, curr_data.drop('IMA_flag',axis=1), scale_data, dt_corr, dt_dens)
            var_pred = dat.append_data_to_timed(var_pred, curr_data, ['x','y','z','rho','IMA_flag'])
#            var_pred['IMA_flag'] = var_pred['IMA_flag'].fillna(-1)
            all_label.append(label_pred)
            all_var.append(var_pred)
            all_crossings.append(cross_pred)
            
        else:
            print('SKIP WINDOW')
        curr_time = next_time
        i+=1
        print(i, " subwindows evaluated")
    label_DF = pd.concat(all_label)
    var_DF = pd.concat(all_var)
    cross_DF = pd.concat(all_crossings)
    return label_DF, var_DF, cross_DF
    
"""
Function to attach data to a crossings list returned by mex_pred_from_AMDA 
and write it to an AMDA file
"""
def crossings_to_AMDA_file(ref_crossings, filepath):
    crossings = ref_crossings.copy()
#    crossings = dat.append_data_to_crossings(crossings, ref_data)
    crossings = dat.append_amda_date(crossings)
    crossings['epoch'] = crossings['amda_date']
    crossings = crossings.rename(index=str, columns={"epoch": "time"})
    crossings = crossings.drop('amda_date',axis=1)
    crossings.to_csv(filepath, sep = ' ', index=False)

"""
Function to to attach data to a variations list and transforms it to a class list with columns [start_time, end_time, label]
"""
def var_to_AMDA_file(ref_var, filepath):
    var = ref_var.copy()
    new_class = pd.DataFrame()
    t_start = var['epoch']
    t_end = var['epoch'].shift(-1)
    labels = var['follow_class'].drop(var.index[var.count()[0]-1])
    new_class['t_start'] = t_start.drop(t_start.index[t_start.count()-1])
    
    new_class['t_start'] = pd.to_datetime(new_class['t_start'],unit='s')
    new_class['t_end'] = pd.to_datetime(t_end,unit='s')

    new_class['t_start'] = [timestamp_to_AMDAdate(new_class['t_start'].iloc[i]) for i in range(new_class.count().max())]
    new_class['t_end'] = [timestamp_to_AMDAdate(new_class['t_end'].iloc[i]) for i in range(new_class.count().max())]
    new_class['label'] = labels
    return new_class
"""
Transforms a timestamp to a date readable in AMDA format
"""
def timestamp_to_AMDAdate(timestamp):
    ymd = str(timestamp)[0:10]
    hms = str(timestamp)[11:]
    return ymd+'T'+hms

#"""
#TEST RUN
#"""
#start_time = '2008-01-03T05:30:00'
#end_time = '2008-01-04T09:30:00'

#scl_data = pd.read_csv(scale_data_path)
#ANN = mdl.load_model(model_path)
#label_AMDA, var_AMDA, cross_AMDA = mex_pred_from_AMDA(ANN,scl_data.drop('label',axis=1),120,600,start_time,end_time,3600*24, amda_old=False)
#
#crossings_to_AMDA_file(cross_AMDA,'./TEST_amdacrossfile.txt')
















    
    
    
    