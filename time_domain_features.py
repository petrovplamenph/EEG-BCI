# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:23:37 2019

@author: Plamen
"""
import numpy as np
import scipy
from collections import Counter


def autocorr(x):
    """
    Calculate Autoregressive coefs.
    """
    result = np.correlate(x, x, mode='full')
    return list(result[len(result)//2:])


def calculate_mean_vars(data):

    mean_vars = []
    for sig in data:
        var = np.var(sig, axis=0).mean()
        mean_vars.append(var)
    return np.array(mean_vars)


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(values):
    n5 = np.nanpercentile(values, 5)
    n25 = np.nanpercentile(values, 25)
    n75 = np.nanpercentile(values, 75)
    n95 = np.nanpercentile(values, 95)
    median = np.nanpercentile(values, 50)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    var = np.nanvar(values)
    rms = np.nanmean(np.sqrt(values**2))
    skew = scipy.stats.skew(values,axis=0)
    return [n5, n25, n75, n95, median, mean, std, var, rms, skew]

def calculate_crossings_per_hz(values, freq):
    
    mean_val = np.mean(values)
    crossings = (((values[:-1]-mean_val) * (values[1:]-mean_val)) < 0).sum()
    mean_crossings = crossings*(freq/values.shape[0])
    return [mean_crossings]

def get_chanel_features(values, freq):
    entropy = calculate_entropy(values)
    crossings = calculate_crossings_per_hz(values, freq)
    statistics = calculate_statistics(values)
    autocorr_cfs =  autocorr(values)[:16]
    return np.array([entropy] + crossings + statistics+autocorr_cfs)


def make_fets(data, freq):
    """
    Itarate over the data and for each segment
    calculate statsitstical proparties
    """
    result = []
    for sig in data:
        _, n_ch = sig.shape
        
        all_chanels_feats = []
        for ch_idx  in range(n_ch):
            ch_data = sig[:,ch_idx]
            chanel_features = get_chanel_features(ch_data, freq)
            all_chanels_feats.append(chanel_features)
        result.append(np.array(all_chanels_feats).T)
    return np.array(result)