
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:32:05 2018

@author: thibault
"""

"""
File dedicated to all evaluations functions, independently from the prediction algorithm
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from fastdtw import fastdtw

"""
Returns the confusion matrix and the normalized confusion matrix from a prediction compared to a test set

y_test, y_pred : type list
"""
def get_confusion_matrices(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    m=confusion_matrix(y_test, y_pred)
    normalized_m = m.astype('float') / m.sum(axis=1)[:, np.newaxis]
    return m, normalized_m

"""
Returns the accuracy, recall and f-measure based on the confusion matrix

matrix : type matrix (defined by the return of get_confusion_matrices)
"""
#Pour se rappeler rapidement de la signification de la précision et du recall:
#Precision:
#    Nombre d'éléments correctement identifiés parmi ceux existants (quelle proportion de prédictions c correspondent a un élément réel c)
#Recall:
#    Nombre d'éléments corrects parmi ceux identifiés (quelle proportion d'éléments réels c sont classifiés c)
    
def accuracy_from_cm(matrix):
    acc = []
    for i in range(matrix.shape[0]):
        p = matrix[i][i]
        den = 0
        for j in range(matrix.shape[0]):
            den+=matrix[j][i]
        if(den>0):
            acc.append(p/den)
        else:
            acc.append(0)
    return acc

def recall_from_cm(matrix):
    recall = []
    for i in range(matrix.shape[0]):
        p = matrix[i][i]
        den = 0
        for j in range(matrix.shape[1]):
            den+=matrix[i][j]
        if(den>0):
            recall.append(p/den)
        else:
            recall.append(0)
    return recall

def f_measure_from_cm(matrix):
    p = accuracy_from_cm(matrix)
    r = recall_from_cm(matrix)
    f_mes = []
    for i in range(len(p)):
        f_mes.append(2*p[i]*r[i]/(p[i] + r[i]))
    return f_mes

"""
Evaluates the different metrics above for a test set and a prediction

This function is really the basics of evaluating the performance of an algorithm based on its predictions,
it is almost independent from the problem itself and uses generic metrics that are widely used in all machine 
learning problems.

y_test, y_pred : type pandas.DataFrame, containing at least a 'label' column
"""
def basic_evaluate(y_test, y_pred, verbose=0):
    m = get_confusion_matrices(y_test['label'], y_pred['label'])[0]
    norm_m = get_confusion_matrices(y_test['label'], y_pred['label'])[1]
    p = accuracy_from_cm(m)
    r = recall_from_cm(m)
    f = f_measure_from_cm(m)
    
    """
    For now, the global p, r and f are valid only for the 3-classes case
    """
    nb0 = y_test.loc[y_test['label']==0].count()[0]
    nb1 = y_test.loc[y_test['label']==1].count()[0]
    nb2 = y_test.loc[y_test['label']==2].count()[0]
    
    gp = (p[0]*nb0 + p[1]*nb1 + p[2]*nb2)/y_pred.count()[0] 
    gr = (r[0]*nb0 + r[1]*nb1 + r[2]*nb2)/y_pred.count()[0] 
    gf = (f[0]*nb0 + f[1]*nb1 + f[2]*nb2)/y_pred.count()[0]  
    """
    """
    if(verbose==1):
        print('\nClass indices :   0 = Close Environment  or  0 = Close Environment')
        print('                  1 = Bow Shock              1 = inward Bow Shock')
        print('                  2 = Solar Wind             2 = outward Bow Shock')
        print('                                             3 = Solar Wind')
        
        print('\nPrecisions : ', p)
        print('Recalls    : ', r)
        print('F-measures : ', f)
        
        print('\nGlobal precision : ', gp)
        print('Global recall    : ', gr)
        print('Global f-measure : ', gf)    
        
        print('\n')
        print('Confusion matrix:\n')
        for i in range(len(norm_m)):
            print(norm_m[i])
    
    return norm_m, p, r, f

#"""
#Plots the classes variations graph based directly on the labels
#PREFER THE graph_pred_from_var VERSION IT IS MUCH LIGHTER  
#"""
#def graph_predictions(y_true, y_pred, prob):
#    y_true = y_true.sort_values(by='epoch')
#    y_pred = y_pred.sort_values(by='epoch')
#    
#    fig, ax = plt.subplots()
#    ax.set_ylim(-0.5,2.5)
#    ax.set_xlabel('Time (epoch in s)')
#    ax.set_ylabel('Class')
##    ax.plot(pd.to_datetime(y_true['epoch'], unit='s'), one_hot_decode(one_hot_encode(y_true.label)), linewidth = 1.5, label = 'Test data', linestyle='--', color='green')
##    ax.plot(pd.to_datetime(y_pred['epoch'], unit='s'), y_pred.label, linewidth = 1.0, label = 'Prediction', color='red')
#    ax.plot(y_true['epoch'], one_hot_decode(one_hot_encode(y_true.label)), linewidth = 1.5, label = 'Test data', linestyle='--', color='blue')
#    ax.plot(y_pred['epoch'], y_pred.label, linewidth = 1.0, label = 'Prediction', color='red')
#    ax.scatter(y_pred['epoch'], y_pred.label, s = 10, c='red', label = 'Data Point')
#    ax.legend()
#    
#    ax2 = ax.twinx()
#    ax2.set_ylabel('Trust level')
#    ax2.set_ylim(0.2,1.2)
#    ax2.grid(linestyle='--')
#    ax2.plot(prob['epoch'], prob['prob'], linewidth = 0.5, color='blue')
#    return

"""
Returns the euclidean distance between two label time series
It is the square root of the squared errors point by point
Widely used for time series comparison
"""
def euclidean_dist(y_true, y_pred):
    lab_true = y_true['label']
    lab_pred = y_pred['label']
    
    dist = 0
    comp_dist = 0
    for i in range(len(lab_true)):
        dist += (lab_true[i] - lab_pred[i])**2
        comp_dist += lab_true[i]**2
    return np.sqrt(dist)

def dtw_dist(y_true, y_pred):
    y_standard = [0]*y_true.count()[0]
    standard_dist = fastdtw(y_true['label'], y_standard, dist = 2)[0]
    distance, path = fastdtw(y_true['label'], y_pred['label'], dist=2)
    return distance


"""
We want to study the influence of parameters that are external to the training on the results of
predictions for the shock. We subdivide this parameter's value range in bins and then compute
the accuracy and recall for each one of those bins

Arguments:
    param_name : str : name of the param to study (included as a column in the shock_data DataFrame?)
    shock_data : DataFrame : basically y_true or y_pred where label==shock
    nb_bins : number of bins to divide the parameters value range into
    show_dist : if True, plots the distribution of the data along the parameter
"""
def split_data_on_param(param_name, data, nb_bins, show_dist=False):
    p_min = data[param_name].min()
    p_max = data[param_name].max()
    bin_width = (p_max - p_min)/nb_bins
    bin_center_vals = [(p_min+(i+0.5)*bin_width) for i in range(nb_bins)]
    
    sub_data = [[]]*nb_bins
    for i in range(nb_bins):
        curr_shocks = data.loc[data[param_name] > p_min + i*bin_width]
        curr_shocks = curr_shocks.loc[curr_shocks[param_name] < p_min + (i+1)*bin_width]
        sub_data[i] = curr_shocks
    #at this point sub_data is a list of DataFrame corresponding to each bin of the parameter
    #we can then compute evaluation metrics on those subsets
    
    if show_dist:
        fig, ax = plt.subplots()
        data_dist = [sub_data[i].count()[0] for i in range(len(sub_data))]
#        ax = sns.distplot(data_dist, bins=nb_bins)
        ax.grid(False)
        ax.set_ylabel('Number of data points')
        ax.set_xlabel(param_name)
        ax = plt.plot(bin_center_vals,data_dist)
    
    ###
    return bin_center_vals,sub_data

"""
Classifies the data in sub datasets (cf previous function) and evaluates each one of them
"""
def perf_on_param(param_name, true_data, pred_data, nb_bins):
    bins,split_tdata = split_data_on_param(param_name, true_data, nb_bins, show_dist=True)
    split_pdata = split_data_on_param(param_name, pred_data, nb_bins)[1]
    acc = []
    rec = []
    f_meas = []
    for i in range(nb_bins):
        if split_tdata[i].loc[split_tdata[i]['label'] == 1.0].count()[0]>0:
            m,p,r,f = basic_evaluate(split_tdata[i], split_pdata[i])
            acc.append(p[1])
            rec.append(r[1])
            f_meas.append(f[1])  #on prend les stats uniquement pour la classe 1 (choc)
#        elif i>0:
#            acc.append(acc[i-1])
#            rec.append(rec[i-1])
#            f_meas.append(f_meas[i-1])
        else:
            acc.append(0) #2 as impossible value
            rec.append(0)
            f_meas.append(0)
    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    ax.set_xlabel(param_name)
    ax.scatter(bins, acc, label = 'Accuracy')
    ax.scatter(bins, rec, label = 'Recall')
    ax.scatter(bins, f_meas, label = 'F-Measure')
    ax.legend()
    return

def bis_perf_on_param(param_name, true_data, pred_data):
    f, ax = plt.subplots();
    



"""
##################################################################################################################
Attention : The following evaluation functions are based on variations lists instead of labels lists !!
##################################################################################################################
"""

"""
Arguments:
    true_var, pred_var, Dt the considered interval around true variations
    true_var, pred_var : epochs list [t0,t1,...,tn]
Return:
    Returns the average number of predicted variations around true variations in an interval true_var[t] +- Dt

"""
def mean_var_around_true(true_var, pred_var, Dt):
    avg = 0
    
    for i in range(len(true_var)):
        count_pred = 0
        for j in range(len(pred_var)):
            if abs(true_var[i] - pred_var[j])<Dt:
                count_pred+=1
        avg += count_pred
    return avg/len(true_var)
#a modifier par type d'environnement et + ou -

"""
Plots the results of mean_var_around_true for various time intervals, with a time resolution res

true_var, pred_var : pandas.DataFrame with at least an 'epoch' column
res : int for time resolution (in seconds)
data_name : name of the dataset to give the plot a title
"""
def graph_mvat(true_var, pred_var, res, data_name=''):
    t = []
    to_plot = []
    for i in range(1200//res):
        to_plot.append(mean_var_around_true(true_var['epoch'].tolist(), pred_var['epoch'].tolist(),i*res))
        t.append(i*res)
    fig = plt.figure()
    plt.plot(t, to_plot, linewidth = 0.8)
    fig.suptitle(data_name+'\nMean number of predicted variations around true variations')
    plt.xlim(0,1200)
    plt.ylim(0,3.5)
    plt.xlabel('Dt (seconds)')
    plt.ylabel('n_var')
    plt.show()
    return


from scipy.stats import norm #gaussian distribution
"""
Plots the time distribution of time difference between predicted variation and the closest true variation

pred_var : pandas.DataFrame with columns 'epoch'
true_var : pandas.DataFrame with columns 'epoch'
data_name : name of the prediction dataset for plot title
"""
def graph_hist(true_var, pred_var, data_name=''):    
    to_plot = []
    pred_var = pred_var['epoch'].tolist()
    true_var = true_var['epoch'].tolist()
    for i in range(len(pred_var)):
        min_Dt = float('inf')
        sg_min = 1
        for j in range(len(true_var)):
            Dt = pred_var[i] - true_var[j]
            if abs(Dt)<min_Dt:
                min_Dt = abs(Dt)
                if abs(Dt)>0:
                    sg_min = Dt/abs(Dt)
        to_plot.append(min_Dt*sg_min)
    sns.set(style="whitegrid")
    f, ax = plt.subplots()
    sns.despine()
    plt.xlim(-1200,1200)
    plt.ylim(0,0.005)
    ax = sns.distplot(to_plot, color = 'g')
    plt.xlabel('Dt (seconds)')
    stats = 'Mean : '+ str(norm.fit(to_plot)[0].round(0)) + 's\nStd deviation : '+str(norm.fit(to_plot)[1].round(0))+'s' 
    plt.text(-1190,0.004, stats)
    f.suptitle(data_name+'\nEcarts des variations prédites par rapport aux variations réelles')
    plt.show()
    return

"""
Plots the time distribution of time difference between predicted variation and the closest true variation, differenciated on categories

pred_var : pandas.DataFrame with columns 'epoch'
data_name : name of the prediction dataset for plot title
"""
def graph_hist_by_cat(var_pred, data_name = ''):
    """
    Même chose que sns_hist en séparant par catégorie de transition
    var_pred doit contenir la colonne dt_to_closest
    """
    ws_to_shock = var_pred.loc[var_pred['category']==1]['dt_to_closest']
    ev_to_shock = var_pred.loc[var_pred['category']==0]['dt_to_closest']
    non_phys = var_pred.loc[var_pred['category']==0.5]['dt_to_closest']
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3,1, sharey = True, sharex = True)
    plt.xlim(-1200,1200)
    plt.ylim(0,0.005)
    #ajouter fit=norm dans les options de distplot pour également plotter un fit gaussien
    sns.distplot(ws_to_shock, color='b', ax = axes[0], axlabel='Solar wind to shock; m = '+ str(norm.fit(ws_to_shock)[0].round(0)) + ', std_dev = ' + str(norm.fit(ws_to_shock)[1].round(0)) , norm_hist=True)
    sns.distplot(ev_to_shock, color = 'g', ax = axes[1], axlabel = 'Close env. to shock; m = '+ str(norm.fit(ev_to_shock)[0].round(0)) + ', std_dev = ' + str(norm.fit(ev_to_shock)[1].round(0)) , norm_hist=True)
    sns.distplot(non_phys, color = 'r', ax=axes[2], axlabel = 'Non physical var.; m = '+ str(norm.fit(non_phys)[0].round(0)) + ', std_dev = ' + str(norm.fit(non_phys)[1].round(0)) +'\nDt(seconds)', norm_hist=True)
    fig.suptitle(data_name+'\nVariations distribution by category')
    return

"""
Plots the spatial distribution of a variations list in the (x,rho) plane
Reminder: rho = (y^2 + z^2)^(1/2)

var : variations list with at least columns 'category','x','rho'
data_name : name of the dataset for plot title
"""
def graph_rho_x(var, data_name = ''):    
    sns.set_style('whitegrid')
    
    plot_color = []
    for i in range(var.count()[0]):
        if var['category'].iloc[i] == 0:
            plot_color.append('green')
        if var['category'].iloc[i] == 1:
            plot_color.append('blue')
        if var['category'].iloc[i] == 0.5:
            plot_color.append('red')
    fig = plt.figure()
    plt.xlim(-3,3)
    plt.ylim(0,6)
    sns.despine()
    cmap = sns.light_palette((0,0,0),n_colors=5, as_cmap = True)
    sns.kdeplot(var['x'], var['rho'], cmap=cmap, alpha = 0.8)
    plt.scatter(var['x'], var['rho'], alpha = 0.5, color = plot_color)
    plt.suptitle(data_name+'\nDistribution of class variations in the (x, rho) plane' )
    plt.show()
    return

"""
Plots the graph of classes transitions from a true variations list and a predicted variations list.

true_var, pred_var : pandas.DataFrame with columns 'epoch', 'pred_class', 'follow_class'
data_name : name of the prediction dataset for the plot title
"""
def graph_label_from_var(var):
    fig, ax = plt.subplots()
    ax.set_ylim(-0.5,2.5)
    ax.set_xlabel('Time (epoch in s)')
    ax.set_ylabel('Class')
    
    colors = ['lightblue','crimson','palegreen']
    toplot = [[],[]]
    #first list for the epoch, second for the classes
    for i in range(var.count()[0]-1):
        epoch = var['epoch'].iloc[i]
        next_epoch = var['epoch'].iloc[i+1]
        prec_class, follow_class = var['prec_class'].iloc[i], var['follow_class'].iloc[i]
        toplot[0].append(epoch)
        toplot[0].append(epoch)
        toplot[1].append(prec_class)
        toplot[1].append(follow_class)        
        rect = patches.Rectangle((epoch,0), next_epoch - epoch, 2, facecolor=colors[int(follow_class)], alpha = 0.5)
        ax.add_patch(rect)
#    ax.plot(pd.to_datetime(toplot[0], unit='s'),toplot[1], linewidth = 2, color = 'red')
    ax.plot(toplot[0],toplot[1], linewidth = 1, color = 'red')
    ax.set_ylim(-0.5,2.5)
    fig.suptitle('\nLabel representation')
    plt.show()
    return

import matplotlib.patches as patches
def patches_from_var(var):
    n = var.count()[0]
    
    colors = ['lightblue','red','steelblue']
    
    for i in range(n-1):
        curr_epoch = var['epoch'].iloc[i]
        next_epoch = var['epoch'].iloc[i+1]
        follow_class = var['follow_class'].iloc[i]
        rect = patches.Rectangle((curr_epoch,0), next_epoch-curr_epoch, 2, facecolor=colors[int(follow_class)], alpha = 0.2)
        plt.axis.add_patch(rect)
        

def graph_pred_from_var(true_var, pred_var, data_name=''):
    fig, ax = plt.subplots()
    ax.set_ylim(-0.5,2.5)
    ax.set_xlabel('Time (epoch in s)')
    ax.set_ylabel('Class')
    
    true_toplot = [[],[]]
    #first list for the epoch, second for the classes
    for i in range(true_var.count()[0]):
        epoch = true_var['epoch'].iloc[i]
        prec_class, follow_class = true_var['prec_class'].iloc[i], true_var['follow_class'].iloc[i]
        true_toplot[0].append(epoch)
        true_toplot[0].append(epoch)
        true_toplot[1].append(prec_class)
        true_toplot[1].append(follow_class)
        
    pred_toplot = [[],[]]
    #first list for the epoch, second for the classes, third for dt_to_closest var
    for i in range(pred_var.count()[0]):
        epoch = pred_var['epoch'].iloc[i]
        prec_class, follow_class = pred_var['prec_class'].iloc[i], pred_var['follow_class'].iloc[i]
        pred_toplot[0].append(epoch)
        pred_toplot[0].append(epoch)
        pred_toplot[1].append(prec_class)
        pred_toplot[1].append(follow_class)

    
    ax.plot(pd.to_datetime(true_toplot[0], unit='s'), true_toplot[1], linestyle='--', linewidth = 2, color = 'green')
    ax.plot(pd.to_datetime(pred_toplot[0], unit='s'), pred_toplot[1], color = 'red', linewidth=0.9)
    ax.set_ylim(-0.5,2.5)
    fig.suptitle(data_name + '\nClass representation')
    plt.show()
    return
"""
Interactive plot
Every time the mouse is pressed, the plot focuses on the next predicted variation in a
smaller window (45 minutes?)
window_width in hours
"""
def interact_graph_pred_from_var(true_var, pred_var,window_width=1.5, start_var = None, data_name = ''):
    window_width = window_width*3600
    if start_var == None:
        start_var = 0        
    
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_xlabel('Time (epoch in s)')
    ax.set_ylabel('Class')
    
    j=start_var
    
    class WindowMover:
        def __init__(self, minj, maxj, j, varplot):
            self.varplot = varplot
            self.minj = minj
            self.maxj = maxj
            self.cid = varplot.figure.canvas.mpl_connect('button_release_event', self)
            self.j = j
        def __call__(self, event):
            if event.button == 1:
                self.j = min(self.j+2,self.maxj)
            elif event.button == 3:
                self.j = max(self.j-2,self.minj)
            update_axes(self)
            self.varplot.figure.canvas.draw()
    
    date = pd.to_datetime(true_var['epoch'].iloc[0], unit = 's')
    def update_axes(wm0):
        date = pd.to_datetime(true_var['epoch'].iloc[wm0.j] + 150, unit = 's')
        ax.set_ylim(-0.25,2.25)
        ax.set_xlim(date - pd.Timedelta(window_width/2, unit='s'), date + pd.Timedelta(window_width/2, unit='s'))
        fig.suptitle(data_name + '\nClass representation\n' + str(date))
    
    true_toplot = [[],[]]
    #first list for the epoch, second for the classes
    for i in range(true_var.count()[0]):
        epoch = true_var['epoch'].iloc[i]
        prec_class, follow_class = true_var['prec_class'].iloc[i], true_var['follow_class'].iloc[i]
        true_toplot[0].append(epoch)
        true_toplot[0].append(epoch)
        true_toplot[1].append(prec_class)
        true_toplot[1].append(follow_class)
        
    pred_toplot = [[],[]]
    #first list for the epoch, second for the classes, third for dt_to_closest var
    for i in range(pred_var.count()[0]):
        epoch = pred_var['epoch'].iloc[i]
        prec_class, follow_class = pred_var['prec_class'].iloc[i], pred_var['follow_class'].iloc[i]
        pred_toplot[0].append(epoch)
        pred_toplot[0].append(epoch)
        pred_toplot[1].append(prec_class)
        pred_toplot[1].append(follow_class)    
    
    ax.set_ylim(-0.5,2.5)
    
    trueplot, = ax.plot(pd.to_datetime(true_toplot[0], unit='s'), true_toplot[1], linestyle='--', linewidth = 2, color = 'green')
    predplot, = ax.plot(pd.to_datetime(pred_toplot[0], unit='s'), pred_toplot[1], color = 'red', linewidth=0.9)
    wm0 = WindowMover(0,true_var.count()[0],j,trueplot)
    
    fig.suptitle(data_name + '\nClass representation\n' + str(date))
    print(wm0.j)
    
    
#    plt.show()
    return



"""
Now that we have clearly identified the crossings and that we can filter them if they are to close,
we can define new metrics directly on the crossings themselves, compared to the real ones.
This will be similar to what was previously done, by defining a +-dt window around crossings

Accuracy : each time at least one real crossing is in the window around a predicted, adds 1

Recall : each time exactly one predicted crossing is in the window around a real, adds 1
"""
def acc_from_crossings(cross_true, cross_pred, Dt):
    acc = 0
    non_acc = 0
    ep_pred = cross_pred['epoch'].tolist()
    ep_true = cross_true['epoch'].tolist()
    for i in range(len(ep_pred)):
        t = ep_pred[i]
        found=0
        for j in range(len(ep_true)):
            if ep_true[j]<t+Dt and ep_true[j]>t-Dt:
                found+=1
        if found>0:
            acc+=1
        else:
            non_acc+=1
    return acc/(acc+non_acc)

def rec_from_crossings(cross_true, cross_pred, Dt):
    rec = 0
    non_rec = 0
    ep_pred = cross_pred['epoch'].tolist()
    ep_true = cross_true['epoch'].tolist()
    for i in range(len(ep_true)):
        t = ep_true[i]
        found=0
        for j in range(len(ep_pred)):
            if ep_pred[j]<t+Dt and ep_pred[j]>t-Dt:
                found+=1
        if found==1:
            rec+=1
        else:
            non_rec+=1
    return rec/(rec+non_rec)

def f_from_crossings(cross_true, cross_pred, Dt):
    acc = acc_from_crossings(cross_true, cross_pred, Dt)
    rec = rec_from_crossings(cross_true, cross_pred, Dt)
    if acc==0 and rec == 0:
        return 0
    return 2*acc*rec/(acc+rec)
        
def plot_cross_stats(cross_true, cross_pred, data_name=''):
    x_data = []
    acc = []
    rec = []
    f_m = []
    for i in range(41):
        x_data.append(i*30)
        a = acc_from_crossings(cross_true, cross_pred, i*30)
        r = rec_from_crossings(cross_true, cross_pred, i*30)
        f = f_from_crossings(cross_true, cross_pred, i*30)
        acc.append(a)
        rec.append(r)
        f_m.append(f)
    fig, ax = plt.subplots()
    ax.plot(x_data, acc, color='r', label = 'Accuracy')
    ax.plot(x_data, rec, color='g', label = 'Recall')
    ax.plot(x_data, f_m, color='orange', label = 'F-measure')
    ax.set_xlabel('Dt (seconds)')
    ax.set_xlim(0,1200)
    ax.set_ylim(0,1)
    ax.legend()
    fig.suptitle(data_name + '\nBasic redefined metrics')

"""
Plots a 2D histogram of shocks locations
Interesting cmap : 
    'rocket'
    'mako'
    'gray'/'gray_r'
    'afmhot'
    'gist_stern_r'
    'BuGn'
    'CMR_Map_r'

norm=
    mcolors.PowerNorm(0.4)
"""
def graph_shock_2dhist(crossings):
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True) 
    plt.rc('font', family='serif')
    gr = ax.hist2d(crossings['x'], crossings['rho'], bins = 100, cmap = 'gray_r')
    fig.colorbar(gr[3], ax=ax, spacing = 'uniform')
    mars = plt.Circle((0,0),1, edgecolor='black', facecolor='whitesmoke')
    plt.axis('equal')
    plt.text(0,0.1,'Mars',horizontalalignment='center')
    ax.add_artist(mars)
    ax.set_xlim(-2.5,2)
    ax.set_ylim(0,4)
    ax.set_xlabel('$x$ (Rm)', usetex=True)
    ax.set_ylabel('$\\rho$ (Rm)', usetex=True)
    
"""
Plots violin distributions of the parameter param_name,
with each violin corresponding to a variability degree
"""
def violin_param_on_degree(final_crossings, param_name):
    fig, ax = plt.subplots()
    cmap = sns.cubehelix_palette(start=0.5, light=1)
    hue = final_crossings['degree']

    sns.violinplot(final_crossings['degree'], final_crossings[param_name], bw=0.15, scale='width',inner='stick',palette=cmap)
    max_deg = final_crossings['degree'].max()
    deg_list = [i for i in range(max_deg)]
    sh_count = [] 
    plt.text(-1,final_crossings[param_name].min(),"Nb. shocks")
    for i in range(max_deg):
        sh_count=final_crossings.loc[final_crossings['degree']==i+1].count()[0]
        plt.text(i-0.3,final_crossings[param_name].min(),str(sh_count))
#    plt.plot(deg_list, sh_count)
    ax.set_xlabel('Variability degree')

"""
Sorts the list by 'dt_to_closest' ie. the timing-gap to the closest real crossing 
Splits the list in nb_samples equal samples 
Plots the distribution of param_name for each sample
"""
def param_distrib_on_closest(final_crossings, param_name, nb_samples, val_abs=True):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sorted_crossings = final_crossings.copy()
    if val_abs:
        sorted_crossings['dt_to_closest'] = np.abs(sorted_crossings['dt_to_closest'])
    sorted_crossings = sorted_crossings.sort_values('dt_to_closest')    
    sample_assignation = []
    nb_cross = final_crossings.count().max()
    for i in range(nb_cross):
        sample_assignation.append(np.floor(i*nb_samples/nb_cross))
    sorted_crossings['sample'] = sample_assignation
#    pal = sns.cubehelix_palette(10, rot=-.8, light=.7)
    pal = sns.dark_palette('grey', n_colors = nb_samples)
    g = sns.FacetGrid(sorted_crossings, row='sample', hue='sample', palette = pal)
    g.map(sns.kdeplot, param_name, shade=True, alpha = 0.9,lw = 2, bw=0.1)
    g.map(sns.kdeplot, param_name, color='w',lw = 2.5, bw = 0.1)
#    g.map(sns.jointplot, param_name[0], param_name[1], kind='kde')
    g.fig.subplots_adjust(hspace=-0.7)

"""
Similar to histogram with variations, but with shocks
"""
def graph_hist_shocks(sh_true, sh_pred, data_name = ''):
    to_plot = []
    sh_pred = sh_pred['epoch'].tolist()
    sh_true = sh_true['epoch'].tolist()
    for i in range(len(sh_pred)):
        min_Dt = float('inf')
        sg_min = 1
        for j in range(len(sh_true)):
            Dt = sh_pred[i] - sh_true[j]
            if abs(Dt)<min_Dt:
                min_Dt = abs(Dt)
                if abs(Dt)>0:
                    sg_min = Dt/abs(Dt)
        to_plot.append(min_Dt*sg_min)
    sns.set(style="whitegrid")
    f, ax = plt.subplots()
    sns.despine()
    plt.xlim(-1200,1200)
    plt.ylim(0,0.005)
    ax = sns.distplot(to_plot, color = 'firebrick')
    plt.xlabel('Dt (seconds)')
    stats = 'Mean : '+ str(norm.fit(to_plot)[0].round(0)) + 's\nStd deviation : '+str(norm.fit(to_plot)[1].round(0))+'s' 
    plt.text(-1190,0.004, stats)
    f.suptitle(data_name+'\nEcart des chocs prédits par rapport aux chocs réels')
    plt.show()
    return





