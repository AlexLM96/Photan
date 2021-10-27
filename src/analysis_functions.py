# -*- coding: utf-8 -*-
"""
Functions to analy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import core

#%%

"""
Function to find a perievent-histogram for a continuous variable

Parameters
----------
data: photometry class object. 
    Signal to be analyzed

ref_ts: array-like. 
    Reference timestamps or reference index in the signal

min_max: tuple
    Range around the events.
idx: Boolean (defaults to False) 
    If true it considers ref_ts as the reference timestamps, otherwise it considers ref_ts as indexes.
    of the data.signal when reference events occurred.
    
Returns
-------
trials_df: pd.DataFrame
    Dataframe with rows as trials and columns as timepoints.

"""

def cont_var_peh(data, ref_ts, min_max, name = None, idx = False):
    trials = []
    
    if not idx:
        for event in ref_ts:
            idx_evt = np.where(data.timestamps == event)[0][0]
            start = idx_evt - (abs(min_max[0]) * data.sr)
            end = idx_evt + (abs(min_max[1]) * data.sr)
            c_trial = data.signal[int(start):int(end)]
            trials.append(list(c_trial))
            
    else:
        for event in ref_ts:
            start = event - (abs(min_max[0]) * data.sr)
            end = event + (abs(min_max[1]) * data.sr)
            c_trial = data.signal[int(start):int(end)]
            trials.append(list(c_trial))
            
    trials_df = pd.DataFrame(trials)
    
    return trials_df
    
#%%

"""
Function to plot peri-event histograms.

This function returns a figure with a heatmap of all the trials, and a line plot with the average response.
It is designed to take the output of "cont_var_peh" as a input.

Parameters
----------
trials_df: pd.DataFrame

Returns
-------
fig, ax: plt figure

"""

def plot_peh(trials_df):
    fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [2, 3]})
    sns.heatmap(trials_df, axis = ax[0])
    ax[1].plot(trials_df.mean())
    
    return fig, ax
